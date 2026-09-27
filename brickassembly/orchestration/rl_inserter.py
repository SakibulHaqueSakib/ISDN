"""WP7: the WP5 policy as the behaviour tree's insertion skill -- master_report §WP7.

    ins = RLInserter("results/wp5/R1/policy.pt")
    twin_executor.assemble("S3", "weakest_joint", seed, inserter=ins, policy_name="R1")

"RL-first with scripted fallback" (§WP7.1): the tree's nominal branch runs
this; its Recovery branch runs the scripted inserter (§WP4.4), so after a
failed RL attempt the next attempt is scripted.

The policy was trained in tasks/insertion_env.py's reduced cell (a floating
hand, hand frame z up, wrench about the TCP). Here the same observation is
built from arm B of the full cell: the TCP frame is flipped to z up, the
wrist wrench is moved from the F/T site to the TCP, the believed target is the
plan's (ground-truth poses, §WP4), and the vision slot carries the same
sigma 0.3 mm estimate the policy was trained with (zero for an R4 policy).
Every pose, velocity and wrench is expressed in a task frame turned by the
seated hand's yaw: the training cell's world frame has the hand at yaw ~0, and
handing the policy world-frame poses of a hand at the plan's 90 deg made its
success head read 0.002 on every G6 attempt.
The brace-state slot stays zero, as in training: feeding the stabilizer's
wrench would put the policy outside its training distribution.

The success head's probability after 1 s is logged as
insertion_episode.predicted_success_prob (§5.6 / §WP7.2). Verification stays
the joint model's MATED state: the head is logged, not trusted, until its AUC
on held-out episodes is known (results report).
"""

import numpy as np

from motion import skills as K
from sim.mj import bricks as Bk
from sim.mj import control as C
from tasks import insertion_env as E
from tasks.ppo import Policy

FLIP = np.diag([1.0, -1.0, -1.0])        # TCP frame (z down) <-> policy hand frame (z up)


class _CellView:
    """The subset of InsertionEnv's interface that ScriptedBase reads."""

    def __init__(self, ins):
        self.ins = ins

    def hand_pose(self):
        return self.ins.hand_pose()

    @property
    def x_d(self):
        return self.ins.T.T @ self.ins.arm.x_d

    @property
    def R_d(self):
        return self.ins.T.T @ self.ins.arm.R_d @ FLIP

    @property
    def believed(self):
        return self.ins.believed

    @property
    def budget(self):
        return self.ins.budget

    def ft_world(self):
        return self.ins.ft_tcp()


class RLInserter:
    def __init__(self, policy_path, seed=0, max_steps=E.MAX_STEPS):
        self.pol = Policy(policy_path)
        self.cfg = self.pol.cfg
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps

    # --- the cell, in the policy's conventions -----------------------------------------
    def world_hand(self):
        x, R = self.arm.tcp(self.sim.d)
        return x, R @ FLIP

    def hand_pose(self):
        """Hand pose in the task frame (world turned by the seated yaw)."""
        x, R = self.world_hand()
        return self.T.T @ x, self.T.T @ R

    def ft_world(self):
        w = self.sim.ft("B")
        f, t = w[:3], w[3:]
        x, _ = self.arm.tcp(self.sim.d)
        r = self.sim.d.site_xpos[self.arm.ft_site] - x      # moment arm F/T site -> TCP
        return np.concatenate([f, t + np.cross(r, f)])

    def ft_tcp(self):
        """Wrench about the TCP, task frame."""
        w = self.ft_world()
        return np.concatenate([self.T.T @ w[:3], self.T.T @ w[3:]])

    def _delta(self, target):
        x, R = self.hand_pose()
        tx, tR = target
        return np.concatenate([(tx - x) / 0.01, E._rot6(R.T @ tR)])

    def _obs(self):
        x, R = self.hand_pose()
        v = self.arm.tcp_vel(self.sim.d)
        v = np.concatenate([self.T.T @ v[:3], self.T.T @ v[3:]])
        bx, _ = self.believed
        proprio = np.concatenate([(x - bx) / 0.01, E._rot6(R), v[:3] / 0.05, v[3:],
                                  [self.arm.opening(self.sim.d) / 0.04]])
        onehot = np.zeros(8)
        onehot[E.TYPES.index(self.btype)] = 1
        depth = np.clip((bx[2] + Bk.STUD_H - x[2]) / Bk.STUD_H, -1, 1)
        context = np.concatenate([self._delta(self.believed), onehot,
                                  [depth, self.n_studs / 8.0], np.zeros(2)])
        vision = self._delta(self.fine) if self.cfg.get("vision", True) else np.zeros(9)
        return np.concatenate([proprio, self.ft_hist.ravel(), context, vision, np.zeros(7),
                               [self.budget / 100.0], self.prev]).astype(np.float32)

    # --- the skill --------------------------------------------------------------------------
    def __call__(self, ctx, s, tgt, press, flog):
        from orchestration import twin_executor as X
        self.sim, self.arm = ctx.sim, ctx.sim.arms["B"]
        bid = s["brick_id"]
        b = next(x for x in ctx.plan["bricks"] if x["id"] == bid)
        self.btype = b["type"]
        self.n_studs = ctx.n_studs(s)
        self.budget = X.BUDGET_MARGIN * self.n_studs * E.F_INSERT_PER_STUD
        # the seated TCP pose: where the TCP is when the brick sits on its target
        off = K.in_hand_offset(self.sim, "B", bid)
        _, R_seat = self.arm.tcp(self.sim.d)       # transport set the planned yaw
        seat = K.tcp_for_brick(off, np.asarray(tgt), R_seat)
        Rb = R_seat @ FLIP
        self.T = E._yaw_rot(float(np.arctan2(Rb[1, 0], Rb[0, 0])))
        self.believed = (self.T.T @ seat, self.T.T @ Rb)
        ev = self.rng.normal(0, E.VISION_SIGMA[0], 2)
        self.fine = (self.believed[0] + [ev[0], ev[1], 0.0], self.believed[1])
        self.ft_hist = np.zeros((10, 6))
        self.prev = np.zeros(E.ACT_DIM)
        view = _CellView(self)
        base = E.ScriptedBase(min(press, 0.85 * self.budget)) if self.cfg["mode"] == "residual" else None
        arm, sim = self.arm, self.sim
        kp_rot0 = arm.kp_rot.copy()
        arm.kp_rot[:] = E.KP_ROT_INSERT            # yaw compliant, as trained
        clutch = ctx.cm
        t0 = sim.t
        peak, impulse, p_succ, steps = 0.0, 0.0, None, 0
        phase = "rl"
        sim.noslip(True)
        dt = 1.0 / E.POLICY_HZ
        while steps < self.max_steps:
            if clutch.bricks[bid].state == "MATED":
                phase = "mated"
                break
            obs = self._obs()
            a_pol, p = self.pol(obs)
            if steps == 20:
                p_succ = p
            if base is not None:
                a = base(view)
                a[:6] = np.clip(a[:6] + a_pol[:6], -1, 1)
                a[6] = np.clip(a[6] + 0.5 * a_pol[6], -1, 1)
                arm.kp_pos[:2] = C.KP_POS[:2] * (4000.0 / 1200.0 if base.phase == "search" else 1.0)
            else:
                a = a_pol
            x, Rh = self.world_hand()                  # hand-frame actions -> world
            xd = arm.x_d + Rh @ (a[:3] * E.DPOS)
            off_ = xd - x
            if np.linalg.norm(off_) > E.LEASH:
                xd = x + off_ / np.linalg.norm(off_) * E.LEASH
            arm.x_d = xd
            arm.R_d = C.rotvec_to_mat(Rh @ (a[3:6] * E.DROT)) @ arm.R_d
            arm.v_d[:] = 0
            arm.a_d[:] = 0
            f_press = (a[6] + 1) / 2 * self.budget
            arm.f_ff[:] = 0
            arm.f_ff[2] = -f_press
            fz_max = 0.0
            # the policy acts at 20 Hz, but the cell stops the press within
            # 2 ms of the joint seating or the budget being crossed, as the
            # scripted skill does: pressing on for the rest of a 50 ms step
            # after a snap-through overshot the budget and tore a 2x2 column
            # off the baseplate (G6, S3 step 2)
            for _ in range(int(round(dt / sim.m.opt.timestep)) // 2):
                sim.step()
                sim.step()
                f = self.ft_tcp()
                fz_max = max(fz_max, float(np.linalg.norm(f[:3])))
                if clutch.bricks[bid].state == "MATED" or fz_max > self.budget:
                    arm.f_ff[:] = 0
                    break
            steps += 1
            f = self.ft_world()
            peak = max(peak, fz_max)
            impulse += max(0.0, f[2]) * dt
            if flog is not None:
                flog.append((sim.t - t0, f[2], 0.0, phase))
            _, Rh = self.world_hand()
            self.ft_hist = np.roll(self.ft_hist, 1, axis=0)
            self.ft_hist[0] = np.concatenate([Rh.T @ f[:3] / 30.0, Rh.T @ f[3:] / 2.0])
            self.prev = a_pol
            if fz_max > self.budget:
                phase = "force_violation"
                break
        arm.f_ff[:] = 0
        arm.kp_pos[:] = C.KP_POS
        arm.kp_rot[:] = kp_rot0
        # hold the pose the hand is in: the policy's rotation actions accumulate
        # in R_d, and with the yaw gain back at its stiff value a few degrees of
        # leftover target twisted the brick it had just seated off its support
        # (G6: S1 and S3's base joints broke right after an RL seat)
        x, R = arm.tcp(sim.d)
        arm.x_d = x.copy()
        arm.R_d = R.copy()
        sim.noslip(False)
        return {"success": clutch.bricks[bid].state == "MATED", "peak_force_N": round(float(peak), 2),
                "impulse_Ns": round(float(impulse), 3), "duration_s": round(sim.t - t0, 3),
                "phase": phase, "searches": None, "predicted_success_prob": p_succ}
