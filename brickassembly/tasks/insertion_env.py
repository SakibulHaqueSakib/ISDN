"""WP5 insertion environment in the MuJoCo twin -- master_report §5.1-§5.5.

    env = InsertionEnv(stage=0, mode="residual", reward="sparse", vision=True, seed=0)
    obs = env.reset(); obs, r, done, info = env.step(action)

§5.1 asks for an Isaac Lab DirectRLEnv. There is no GPU here, so this is the
CPU twin's reduction of the same MDP, built so that PPO is affordable on four
cores (~1 s of wall time per 10 s episode):

  * the placer is a 1.5 kg floating HAND body under the arm's impedance law
    (control.KP_POS/KP_ROT, D = 2 zeta sqrt(K m), gravity compensated) -- the
    Panda's operational-space inertia at the insertion pose is 1-3 kg -- with
    the brick a rigid child of it (the grasp). The wrist F/T is the hand's
    free-joint constraint force, low-passed like runtime.FT_TAU. The same bricks, baseplate and breakable
    clutch joint model (sim/joint_model/clutch.py) as the full cell.
  * actions are §5.2's 6-D EE-frame pose increments (+-2 mm / +-2 deg, tanh,
    integrated by the policy-level action integrator) plus a 7th channel: the
    press force. A 2x4 needs 8 x 8.9 = 71 N to seat and the §2.5 impedance
    gives 6 N for the 10 mm the integrator's leash allows, so without it an
    end-to-end policy cannot insert at all (v3.1 amendment).
  * mode="residual": the action is added to the §WP4.4 scripted controller
    (ScriptedBase, the same state machine as motion/skills.insert) [Marougkas];
    mode="e2e": the policy alone.

Observation groups (§5.2, adapted where there is no arm):
  proprio 16   hand position rel. believed target (3), rotation 6D (6),
               linear/angular velocity (6), gripper width (1)
  ft_hist 60   last 10 policy steps x 6D wrench, EE frame, / (30 N, 2 N*m)
  context 21   believed-target delta pose (3 + 6D rot), brick type one-hot (8),
               engagement estimate (1), n_studs / 8 (1), reserved (2)
  vision   9   a fine target estimate (sigma 0.3 mm / 0.3 deg) -- the stand-in
               for the wrist-camera embedding; ZEROED when vision=False (R4, A3)
  brace    7   stabilizer contact wrench (6) + active flag (1): zero here, the
               placement is onto the baseplate or an isolated support
  budget   1   force budget / 100 N
  prev_act 7
  => 121 (112 without vision content; the slot is kept, zeroed)
Critic privileges (+26), each inferable from force history per [InformedAAC]:
  true delta pose (9), clutch state one-hot (3), engaged fraction (1),
  contact count / 10 (1), lateral error true (2), sliding speed (1),
  DR friction / f_break / gate (3), time fraction (1), peak force / budget (1),
  max joint utilisation (1), zeros (3).

The "believed" target is the true socket plus the handoff estimate error
(sigma 0.8 mm / 1 deg per axis): what the plan and a calibrated cell give
without looking. Curriculum (§5.4) sets the INITIAL error of the start pose from the
true target; the believed target is what the controller aims at.
"""

import math

import mujoco
import numpy as np

import planner as P
from sim.joint_model import clutch as CL
from sim.joint_model.capacity import F_INSERT_PER_STUD
from sim.mj import bricks as Bk
from sim.mj import control as C
from sim.mj import scene as S

POLICY_HZ = 20
PHYS_PER_POLICY = int(round(1.0 / POLICY_HZ / S.TIMESTEP))       # 50
CTRL_EVERY = 2
MAX_STEPS = 200
HAND_MASS = 1.5
HAND_I = 0.004
TCP_ABOVE_BOTTOM = 0.012
PRE_Z = 0.025                     # start above the seat (§2.5 pre-insertion pose)
DPOS, DROT = 0.002, math.radians(2.0)
LEASH = 0.010
KP_POS = np.array([1200.0, 1200.0, 600.0])
KP_ROT_INSERT = np.array([150.0, 150.0, 10.0])   # roll/pitch skills.INSERT_KP_ROT; yaw
                                  # compliant: in the real grasp the brick twists in the
                                  # pads, and a 2x4 more than 0.4 deg off in yaw wedges on
                                  # diagonal studs (0.1 mm clearance over 16 mm)
ZETA = 1.0
BELIEF_SIGMA = (0.0008, math.radians(1.0))   # a calibrated cell's absolute accuracy
FT_TAU = 0.005                    # s, the wrist sensor's low-pass (runtime.FT_TAU)
VISION_SIGMA = (0.0003, math.radians(0.3))
TYPES = ["1x1", "1x2", "1x4", "1x6", "2x2", "2x3", "2x4", "2x6"]
OBS_DIM = 121
PRIV_DIM = 26
ACT_DIM = 7

# §5.4 curriculum: (pos error m, rot error rad, neighbours, brick types)
STAGES = [
    (0.002, math.radians(2), 0, ["2x4"]),
    (0.005, math.radians(5), 0, ["2x4", "2x2"]),
    (0.010, math.radians(8), 1, ["2x2", "2x3", "2x4", "2x6"]),
    (0.015, math.radians(12), 3, ["2x2", "2x3", "2x4", "2x6"]),
]


def _rot6(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def _yaw_rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class ScriptedBase:
    """§WP4.4 as a per-policy-step controller (the residual's base). The same
    phases and constants as motion/skills.insert: align above the believed
    target, descend (fast, then 5 mm/s), on contact press or search (spiral
    2 mm @ 3 mm/s, 8 N down, +-1.5 deg wiggle @ 2 Hz), stuck -> search."""

    def __init__(self, press):
        self.press = press
        self.phase = "align"
        self.t = 0.0
        self.centre = None
        self.hist = []

    def __call__(self, env):
        """Increments act on the impedance TARGET x_d (as skills.insert sets
        x_d directly); computing them from the lagging hand position let the
        target run away along the spiral."""
        dt = 1.0 / POLICY_HZ
        self.t += dt
        x, R = env.hand_pose()
        xd = env.x_d
        tb, Rb = env.believed
        dp = np.zeros(3)
        drot = np.zeros(3)
        press = 0.0
        z_rel = x[2] - tb[2]              # brick bottom above the believed seat (both TCP-level)
        f_up = env.ft_world()[2]
        if self.phase == "align":
            dp[:2] = np.clip(tb[:2] - xd[:2], -DPOS, DPOS)
            drot = C.rot_error(Rb, env.R_d)
            if np.linalg.norm(tb[:2] - x[:2]) < 0.0003 and np.linalg.norm(C.rot_error(Rb, R)) < math.radians(0.3):
                self.phase = "descend"
        elif self.phase == "descend":
            v = 0.005 if z_rel < Bk.STUD_H + 0.003 else 0.020
            dp[2] = -v * dt
            dp[:2] = np.clip(tb[:2] - xd[:2], -0.0005, 0.0005)
            if f_up > 3.0:
                self.centre = x.copy()
                self.phase = "search" if z_rel > 0.00148 + 0.00015 else "press"
                self.t = 0.0
                self.hist = []
        elif self.phase == "press":
            press = min(1.0, self.t / 0.5) * self.press
            drot = C.rot_error(Rb, env.R_d)          # level again after the wiggle
            dp[:2] = x[:2] - xd[:2]       # laterally free: the chamfer centres the brick
            dp[2] = min(0.0, x[2] - 0.001 - xd[2])
            self.hist.append(x[2])
            if len(self.hist) > 8 and self.hist[-9] - self.hist[-1] < 0.00003 and self.t > 0.9:
                self.phase, self.t, self.centre = "search", 0.0, x.copy()
        elif self.phase == "search":
            s = 0.003 * self.t
            a = 0.001 / (2 * math.pi)
            th = math.sqrt(2 * s / a) if s > 0 else 0.0
            r = min(a * th, 0.002)
            goal = self.centre[:2] + r * np.array([math.cos(th), math.sin(th)])
            dp[:2] = np.clip(goal - xd[:2], -DPOS, DPOS)
            dp[2] = min(0.0, x[2] - 0.001 - xd[2])
            press = 8.0
            # the wiggle is absolute about the aligned orientation (as
            # increments, the target integrated the sine into a standing tilt)
            w = math.radians(1.5) * math.sin(2 * math.pi * 2.0 * self.t)
            drot = C.rot_error(C.rotvec_to_mat([w, 0.7 * w, 0.0]) @ Rb, env.R_d)
            if z_rel < 0.00148 or self.t > 5.0:
                self.phase, self.t, self.hist = "press", 0.0, []
        # to the policy's units: EE-frame increments in [-1, 1], press in [0, 1]
        a = np.zeros(ACT_DIM)
        a[:3] = np.clip(R.T @ dp / DPOS, -1, 1)
        a[3:6] = np.clip(R.T @ drot / DROT, -1, 1)
        a[6] = np.clip(press / env.budget, 0, 1) * 2 - 1
        return a


class InsertionEnv:
    def __init__(self, stage=0, mode="residual", reward="sparse", vision=True, dr=True,
                 seed=0, force_budget=None, joint="calibrated"):
        """joint: "calibrated" (§2.3.3 / WP2: 8.9 N insertion, 11.3 N break per
        stud, 1.2 mm gate) or "handtuned" (ablation A10: the light, lenient
        joint a simulation-first build would have used -- 3 N / 5 N per stud,
        2 mm gate, as BrickSim's default gate)."""
        self.stage, self.mode, self.reward_kind = stage, mode, reward
        self.joint = joint
        self.vision, self.dr = vision, dr
        self.rng = np.random.default_rng(seed)
        self.fixed_budget = force_budget
        self.m = None

    # --- episode construction --------------------------------------------------
    def _layout(self):
        pos_err, rot_err, neighbours, types = STAGES[self.stage]
        btype = types[self.rng.integers(len(types))]
        nx, ny = P.BRICKS[btype]
        i0, j0 = 2, 2
        bricks = [{"id": "target", "type": btype, "grid_pos": [i0, j0, 0], "yaw_index": 0}]
        # neighbours in the same layer: along y (both sides) and along x
        slots = [(i0, j0 - 2, "2x2"), (i0, j0 + ny, "2x2"), (i0 + nx, j0, "1x2")]
        for n in range(min(neighbours, len(slots))):
            i, j, t = slots[n]
            if j < 0:
                continue
            bricks.append({"id": "nb%d" % n, "type": t, "grid_pos": [i, j, 0], "yaw_index": 0})
        seq = [{"step": k, "brick_id": b["id"], "mating_studs": []} for k, b in enumerate(bricks)]
        seq = seq[1:] + seq[:1]          # target last
        return btype, nx, ny, bricks, seq, pos_err, rot_err

    def reset(self):
        btype, nx, ny, bricks, seq, pos_err, rot_err = self._layout()
        self.btype, self.n_studs = btype, nx * ny
        ni = max(b["grid_pos"][0] + P.BRICKS[b["type"]][0] for b in bricks) + 2
        nj = max(b["grid_pos"][1] + P.BRICKS[b["type"]][1] for b in bricks) + 2
        origin = (-ni * P.PITCH / 2, -nj * P.PITCH / 2, S.Z0)
        plan = {"bricks": bricks, "sequence": seq, "voxel_origin": list(origin)}
        tgt = np.array([origin[0] + (2 + nx / 2) * P.PITCH, origin[1] + (2 + ny / 2) * P.PITCH,
                        S.Z0])
        # start pose: true target + curriculum error, PRE_Z above the seat
        r = pos_err * math.sqrt(self.rng.uniform())
        th = self.rng.uniform(0, 2 * math.pi)
        yaw = self.rng.uniform(-1, 1) * rot_err
        tilt_ax = self.rng.normal(size=2)
        tilt = self.rng.uniform(0, 0.3) * rot_err * tilt_ax / (np.linalg.norm(tilt_ax) + 1e-9)
        R0 = C.rotvec_to_mat([tilt[0], tilt[1], 0.0]) @ _yaw_rot(yaw)
        start = tgt + [r * math.cos(th), r * math.sin(th), PRE_Z]
        hand0 = start + R0 @ np.array([0, 0, TCP_ABOVE_BOTTOM])
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, R0.ravel())

        def pre(spec):
            h = spec.worldbody.add_body(name="hand", pos=hand0.tolist(), quat=q0.tolist())
            h.add_freejoint(name="hand/free")
            h.mass = HAND_MASS
            h.inertia = [HAND_I] * 3


        # the target brick hangs from the hand, TCP_ABOVE_BOTTOM below it
        attach = {"target": ("hand", [0, 0, -TCP_ABOVE_BOTTOM], [1, 0, 0, 0])}
        pre_ids = [b["id"] for b in bricks if b["id"] != "target"]
        self.cell = S.build_cell(plan, preplaced=set(pre_ids), with_arms=False, pre=pre,
                                 attach=attach)
        m, d = self.cell.model, self.cell.data
        self.m, self.d = m, d
        self.hand = m.body("hand").id
        self.bid = m.body("target").id
        self.cm = CL.ClutchModel(self.cell)
        for b in pre_ids:
            self.cm.mate_now(b)
        self.hdof = m.jnt_dofadr[m.joint("hand/free").id]
        # domain randomisation (§5.5, the subset that exists in this twin)
        self.dr_params = np.array([0.6, 11.3, 1.2])
        f_ins, f_brk, gate0 = (8.9, 11.3, 0.0012) if self.joint == "calibrated" else \
            (3.0, 5.0, 0.0020)
        self.cm.p.f_insert_per_stud = f_ins
        self.cm.p.gate_lateral = gate0
        for st in self.cm.conn_states:
            st.patch.coef *= st.patch.f_break / f_brk
            st.patch.f_break = f_brk
        self.f_ins = f_ins
        if self.dr:
            mu = self.rng.uniform(0.3, 0.9)
            for g in range(m.ngeom):
                if m.body(m.geom_bodyid[g]).name.startswith(("target", "nb")) or "stud" in m.geom(g).name:
                    m.geom_friction[g, 0] = mu
            gate = gate0 * self.rng.uniform(0.75, 1.25)
            self.cm.p.gate_lateral = gate
            fb = f_brk * self.rng.uniform(0.8, 1.2)   # narrowed to the WP2 range (§5.5 note)
            for st in self.cm.conn_states:
                st.patch.coef *= st.patch.f_break / fb
                st.patch.f_break = fb
            self.gain = self.rng.uniform(0.8, 1.25)
            self.dr_params = np.array([mu, fb, gate * 1000])
            self.ft_bias = self.rng.normal(0, [0.15, 0.15, 0.15, 0.01, 0.01, 0.01])
        else:
            self.gain = 1.0
            self.ft_bias = np.zeros(6)
        n = self.n_studs
        # §5.5's U(20, 45) N cannot seat a 2x4 (8 x 8.9 = 71 N): the budget is
        # drawn relative to the seating force instead (v3.1 amendment)
        self.budget = self.fixed_budget or (self.rng.uniform(1.55, 2.0) if self.dr else 1.7) \
            * n * f_ins
        self.press_nom = 1.3 * n * f_ins
        # targets: true, believed (handoff estimate), fine (vision)
        self.true_tgt = (tgt + [0, 0, TCP_ABOVE_BOTTOM], np.eye(3))
        eb = self.rng.normal(0, BELIEF_SIGMA[0], 2)
        yb = self.rng.normal(0, BELIEF_SIGMA[1])
        self.believed = (self.true_tgt[0] + [eb[0], eb[1], 0.0], _yaw_rot(yb))
        ev = self.rng.normal(0, VISION_SIGMA[0], 2)
        yv = self.rng.normal(0, VISION_SIGMA[1])
        self.fine = (self.true_tgt[0] + [ev[0], ev[1], 0.0], _yaw_rot(yv))
        mujoco.mj_forward(m, d)
        self.ft_f = np.zeros(6)
        x, R = self.hand_pose()
        self.x_d, self.R_d = x.copy(), R.copy()
        self.f_press = 0.0
        self.k = 0
        self.steps = 0
        self.ft_hist = np.zeros((10, 6))
        self.prev_action = np.zeros(ACT_DIM)
        self.peak = 0.0
        self.impulse = 0.0
        self.base = ScriptedBase(min(self.press_nom, 0.85 * self.budget))
        self.t0 = d.time
        self.init_err = self._true_err()
        self.done_reason = None
        return self._obs()

    # --- state -----------------------------------------------------------------
    def hand_pose(self):
        return self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def ft_world(self):
        """Environment wrench on the brick, as the wrist feels it (world
        frame, force N, torque N*m about the TCP): low-passed, with bias."""
        return self.ft_f + self.ft_bias

    def _ft_raw(self):
        """The environment's wrench on hand + brick: the hand free joint's
        constraint force (contacts, clutch tendons and welds alike). A MuJoCo
        force sensor between hand and brick misses the clutch tendons -- they
        are joint-space forces, absent from cfrc_ext -- and read 0.6 N while
        the interference held 7 N. Force world frame; torque about the TCP."""
        f = self.d.qfrc_constraint[self.hdof:self.hdof + 6]
        R = self.d.xmat[self.hand].reshape(3, 3)
        return np.concatenate([f[:3], R @ f[3:]])

    def _true_err(self):
        x, R = self.hand_pose()
        tx, tR = self.true_tgt
        dyaw = math.atan2(R[1, 0], R[0, 0])
        sym = math.pi / 2 if P.BRICKS[self.btype][0] == P.BRICKS[self.btype][1] else math.pi
        dyaw = (dyaw + sym / 2) % sym - sym / 2
        return (float(np.linalg.norm((x - tx)[:2])), abs(math.degrees(dyaw)))

    # --- control ---------------------------------------------------------------
    def _impedance(self):
        m, d = self.m, self.d
        x, R = self.hand_pose()
        v = d.cvel[self.hand]                    # [ang(3), lin(3)] at the body's com frame
        w, vl = v[:3], v[3:]
        mass = HAND_MASS + m.body_mass[self.bid]
        K = KP_POS * self.gain
        if self.mode == "residual" and self.base.phase == "search":
            K = K.copy()
            K[:2] = 4000.0 * self.gain            # skills.SEARCH_KP_XY
        D = 2 * ZETA * np.sqrt(K * mass)
        F = K * (self.x_d - x) - D * vl
        F[2] += -self.f_press
        F -= mass * m.opt.gravity
        Kr = KP_ROT_INSERT * self.gain                   # in the hand frame
        Dr = 2 * ZETA * np.sqrt(Kr * HAND_I)
        e = R.T @ C.rot_error(self.R_d, R)
        T = R @ (Kr * e - Dr * (R.T @ w))
        d.xfrc_applied[self.hand, :3] = F
        d.xfrc_applied[self.hand, 3:] = T

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1, 1)
        base = self.base(self) if self.mode == "residual" else np.zeros(ACT_DIM)
        if self.mode == "residual":
            a = base.copy()
            a[:6] = np.clip(base[:6] + action[:6], -1, 1)
            a[6] = np.clip(base[6] + 0.5 * action[6], -1, 1)
        else:
            a = action
        x, R = self.hand_pose()
        # policy-level action integrator (IndustReal PLAI): EE-frame increments
        self.x_d = self.x_d + R @ (a[:3] * DPOS)
        off = self.x_d - x
        if np.linalg.norm(off) > LEASH:
            self.x_d = x + off / np.linalg.norm(off) * LEASH
        self.R_d = C.rotvec_to_mat(R @ (a[3:6] * DROT)) @ self.R_d
        self.f_press = (a[6] + 1) / 2 * self.budget
        m, d = self.m, self.d
        fz_max = 0.0
        a_f = CTRL_EVERY * m.opt.timestep / (FT_TAU + CTRL_EVERY * m.opt.timestep)
        for k in range(PHYS_PER_POLICY):
            if k % CTRL_EVERY == 0:
                self._impedance()
            mujoco.mj_step(m, d)
            self.cm.update(d)
            if k % CTRL_EVERY == CTRL_EVERY - 1:
                self.ft_f += a_f * (self._ft_raw() - self.ft_f)
                fz_max = max(fz_max, float(np.linalg.norm(self.ft_f[:3])))
        if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            self.done_reason = "diverged"
            return self._obs(), -1.0, True, self._info()
        self.steps += 1
        f = self.ft_world()
        self.peak = max(self.peak, fz_max)
        self.impulse += max(0.0, f[2]) / POLICY_HZ
        self.ft_hist = np.roll(self.ft_hist, 1, axis=0)
        x, R = self.hand_pose()
        self.ft_hist[0] = np.concatenate([R.T @ f[:3] / 30.0, R.T @ f[3:] / 2.0])
        self.prev_action = action
        # termination (§5.2): snap success, force violation, drop, timeout
        state = self.cm.bricks["target"].state
        done, r = False, 0.0
        if state == CL.MATED:
            done, r, self.done_reason = True, 1.0, "success"
        elif fz_max > self.budget:
            done, r, self.done_reason = True, -1.0, "force_violation"
        elif self.cm.breaks():
            done, r, self.done_reason = True, -1.0, "joint_break"
        elif self.steps >= MAX_STEPS:
            done, self.done_reason = True, "timeout"
        if self.reward_kind == "dense" and not done:
            exy, edeg = self._true_err()
            z = x[2] - self.true_tgt[0][2]
            r += -0.02 * exy / 0.005 - 0.01 * edeg / 5.0 - 0.02 * max(0.0, z) / 0.01
            r += 0.05 * (state == CL.ENGAGING)
            r -= 0.02 * max(0.0, fz_max / self.budget - 0.8)
        return self._obs(), r, done, self._info()

    # --- observation ---------------------------------------------------------------
    def _delta(self, target):
        x, R = self.hand_pose()
        tx, tR = target
        return np.concatenate([(tx - x) / 0.01, _rot6(R.T @ tR)])

    def _obs(self):
        x, R = self.hand_pose()
        v = self.d.cvel[self.hand]
        bx, bR = self.believed
        proprio = np.concatenate([(x - bx) / 0.01, _rot6(R), v[3:] / 0.05, v[:3], [0.0079 / 0.04]])
        onehot = np.zeros(8)
        onehot[TYPES.index(self.btype)] = 1
        depth = np.clip((bx[2] + Bk.STUD_H - x[2]) / Bk.STUD_H, -1, 1)
        context = np.concatenate([self._delta(self.believed), onehot, [depth, self.n_studs / 8.0],
                                  np.zeros(2)])
        vision = self._delta(self.fine) if self.vision else np.zeros(9)
        brace = np.zeros(7)
        actor = np.concatenate([proprio, self.ft_hist.ravel(), context, vision, brace,
                                [self.budget / 100.0], self.prev_action]).astype(np.float32)
        st = self.cm.bricks["target"].state
        exy_vec = (x - self.true_tgt[0])[:2] / 0.005
        conns = self.cm.bricks["target"].conns
        priv = np.concatenate([
            self._delta(self.true_tgt),
            [st == CL.DISENGAGED, st == CL.ENGAGING, st == CL.MATED],
            [np.mean([c.state == CL.MATED for c in conns]) if conns else 0.0],
            [min(self.d.ncon, 10) / 10.0], exy_vec,
            [np.linalg.norm(self.d.cvel[self.bid][3:5]) / 0.01],
            self.dr_params / np.array([1.0, 10.0, 1.0]),
            [self.steps / MAX_STEPS, self.peak / self.budget, self.cm.max_util()],
            np.zeros(3)]).astype(np.float32)
        return {"actor": actor, "critic": np.concatenate([actor, priv])}

    def _info(self):
        exy, edeg = self._true_err()
        return {"success": self.done_reason == "success", "reason": self.done_reason,
                "steps": self.steps, "peak_force_N": self.peak, "impulse_Ns": self.impulse,
                "budget_N": self.budget, "init_err_mm": self.init_err[0] * 1000,
                "init_err_deg": self.init_err[1], "final_err_mm": exy * 1000,
                "final_err_deg": edeg, "n_studs": self.n_studs, "type": self.btype,
                "stage": self.stage}
