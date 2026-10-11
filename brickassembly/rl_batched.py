"""Many worlds in one Newton model: the batched version of BrickReachEnv (rl_env.py), for RL at an order of magnitude more env steps per second than subprocess workers.

    BatchedCells         N copies of the cell (dual_arm_sim.build_cell) in one Model / SolverMuJoCo / CollisionPipeline, one captured CUDA graph per
                         frame, batched IK for all 2N hands, per-world reset (start step, welds, collision groups) and per-world weld toggles
    BatchedBreaker       dual_arm_sim.JointBreaker's rule in every world, from the solver's weld forces
    BatchedCameras       one SensorTiledCamera update per camera for all N worlds; the sensor noise (cell/cameras.sense) on the GPU with one rng per world
    BatchedReachEnv      BrickReachEnv x N with rl_vec.NewtonVecEnv's interface (torch tensors on `device`, obs groups, autoreset): RslRlVecEnv, skrl_env
                         and RlGamesVecEnv take it as it is

    bash scripts/run.sh rl_batched.py --demo          # the scripted controller in N worlds: ticks to success per world, throughput

Same task, spaces and semantics as BrickReachEnv: action Box(10) of [-1, 1] (dx, dy, dz, dyaw, grip of arm A then B), observation "policy" = proprio (54),
bricks (16 x 20), plate (8), task (8) flat = 390, "images" (optional) per camera <cam>_rgb / <cam>_depth. One definition each (rl_env.py, cell/liveview.py):
ObsSpec.encode (the observation), liveview.clamp_target / follow (the hand-target math, ManualCtlBatch), ReachReward (the reward and its weights),
pose_errors (the gate numbers). Observations only from the cameras (top_detect on each world's top image: detect -> remember -> track_held), proprioception
(joint state and its batched FK) and the plan; simulator ground truth (body_q, contacts) only in the reward and extras["privileged"].

What differs from BrickReachEnv (each is a deliberate simplification of the batched version):
  - the scene is built once; a reset restores a world's state, welds and collision groups in place (no Example rebuild: 0.2 ms instead of 2 s)
  - the camera key light is one per env (a light lives in the renderer, not in a world); the hidden extrinsic error, exposure gain and noise stream are per world
    and drawn again at every reset. Wrist cameras have no co-located ring light (also one light for all worlds)
  - the sensor noise is cell/cameras.sense in torch on the GPU, from a per-world generator (not numpy's), so the noise is statistically, not bitwise, the same
  - all frames of a tick run in every world: a world whose brick snaps keeps stepping to the end of its tick (BrickReachEnv stops the frame loop there)
  - the joint-break rule is BatchedBreaker's: JointBreaker's break condition on the same weld wrenches, without its u2 / peak bookkeeping (it costs one
    download of the solver's constraint arrays per frame while some world has a welded brick, and about 30 us of python per weld and frame)
"""

import contextlib
import io
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as D  # noqa: E402
import newton  # noqa: E402
import newton.geometry  # noqa: E402
import newton.ik as ik  # noqa: E402
import planner as P  # noqa: E402
import rl_detect  # noqa: E402
import rl_env as R  # noqa: E402
import rl_vec as V  # noqa: E402
from cell import cameras as CAM  # noqa: E402
from cell import liveview as LV  # noqa: E402
from newton.sensors import SensorTiledCamera  # noqa: E402

wp.config.quiet = True
SUBSTEPS = 16                                 # Example.substeps
FRAME_DT = 1.0 / R.FPS
STRUCT, NOTIFY = D.STRUCT_GROUP, newton.solvers.SolverNotifyFlags.CONSTRAINT_PROPERTIES


def quiet(on=True):
    return contextlib.redirect_stdout(io.StringIO()) if on else contextlib.nullcontext()


# --- kernels ----------------------------------------------------------------------------------------------------------
@wp.kernel
def scatter_targets(q: wp.array2d[float], grip: wp.array[float], dpw: int, tgt: wp.array[float]):
    """IK problem p = 2 * world + arm: its 7 arm joints and the finger target -> the arm's drive targets in the world's dof block."""
    p = wp.tid()
    w = p // 2
    k = p - 2 * w
    for j in range(7):
        tgt[w * dpw + 9 * k + j] = q[p, j]
    tgt[w * dpw + 9 * k + 7] = grip[p]
    tgt[w * dpw + 9 * k + 8] = grip[p]


@wp.kernel
def arm_arm_pairs(nacon: wp.array[wp.int32], dist: wp.array[float], geom: wp.array[wp.vec2i], world: wp.array[wp.int32], g2s: wp.array2d[wp.int32],
                  shape_body: wp.array[wp.int32], bpw: int, nab: int, flags: wp.array[wp.uint8]):
    """flags[world, a * nab + (b - nab)] = 1 for every penetrating contact (dist < 0, as cell/contacts.ContactMonitor counts them) between body a of arm A
    and body b of arm B (world-local body indices)."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    if dist[i] >= 0.0:
        return
    w = world[i]
    s0 = g2s[w, geom[i][0]]
    s1 = g2s[w, geom[i][1]]
    if s0 < 0 or s1 < 0:
        return
    b0 = shape_body[s0] - w * bpw
    b1 = shape_body[s1] - w * bpw
    if b0 >= nab and b0 < 2 * nab and b1 >= 0 and b1 < nab:       # order the pair: A's body, then B's
        t = b0
        b0 = b1
        b1 = t
    if b0 >= 0 and b0 < nab and b1 >= nab and b1 < 2 * nab:
        flags[w * nab * nab + b0 * nab + (b1 - nab)] = wp.uint8(1)


# --- vectorised pose helpers (xyzw quaternions, as dual_arm_sim) ---------------------------------------------------------
def rot_z(v, yaw):
    """Rotate vectors v (..., 3) about z by yaw (...), D.rotate(D.qz(yaw), v) in bulk."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def hand_target_pose(cmd_p, cmd_yaw):
    """Example.step's IK target of the arms in direct (rl_control) mode, in bulk: cmd_p (..., 2, 3) world, cmd_yaw (..., 2) world yaw ->
    (pos in each arm's base frame (..., 2, 3), rotation quaternion xyzw (..., 2, 4)): pos = Rz(-base yaw) (p - base), rot = Rz(-base yaw) Rz(yaw) Q_DOWN."""
    bases = np.array([a[0] for a in (D.ARM_A, D.ARM_B)])
    byaw = np.array([a[1] for a in (D.ARM_A, D.ARM_B)])
    half = (cmd_yaw - byaw) / 2
    return rot_z(cmd_p - bases, -byaw), np.stack([np.cos(half), np.sin(half), np.zeros_like(half), np.zeros_like(half)], -1)   # Rz(a) * (1, 0, 0, 0)


def hand_world_pose(t):
    """FK output t (n, 2, 7) in each arm's base frame -> the TCPs [x y z qx qy qz qw] in the world (n, 2, 7): ex.hand_fk's last line, in bulk."""
    bases = np.array([a[0] for a in (D.ARM_A, D.ARM_B)])
    byaw = np.array([a[1] for a in (D.ARM_A, D.ARM_B)])
    q = (Rot.from_euler("z", np.tile(byaw, len(t))[:, None]) * Rot.from_quat(t[..., 3:].reshape(-1, 4))).as_quat().reshape(t.shape[:-1] + (4,))
    return np.concatenate([rot_z(t[..., :3], byaw) + bases, q], -1)


def gate_ok(lateral, dz, dyaw, tilt):
    """Example.gate's pass rule (D.GATE) on R.pose_errors' numbers (lateral [m], dz [m], dyaw [rad], tilt [rad]), in bulk."""
    return ((lateral * 1e3 < D.GATE["lateral_mm"]) & (np.abs(dz) * 1e3 < D.GATE["dz_mm"]) & (np.degrees(dyaw) < D.GATE["yaw_deg"])
            & (np.degrees(tilt) < D.GATE["yaw_deg"]))


# --- the cells --------------------------------------------------------------------------------------------------------
class BatchedCells:
    """N worlds of one structure in one Model, SolverMuJoCo and CollisionPipeline, stepped by one captured CUDA graph per frame (the physics of
    dual_arm_sim.Example: same builders, solver arguments, substeps, collision groups and welds).

    Indexing is world-major: world w owns bodies [w * bpw, (w + 1) * bpw), joint coordinates [w * qpw, ...), dofs [w * dpw, ...), equality constraints
    [w * epw, ...), shapes [n_global + w * spw, ...). The two arms come first in every block (18 dofs, 2 * n_arm_bodies bodies), then one free brick per
    plan step in step order.

    The start step of a world (`reset_worlds`): bricks of earlier steps stand at their targets with their welds on and STRUCT_GROUP (as Example(start_step=n)),
    the rest wait in the feeder. The scene has all bricks in the feeder; every distinct start step has a precomputed state/weld/group table that a reset
    copies in. Sizes that were measured: the triangle-pair buffer must be >= 300k per world (contacts drop silently below), the solver's nconmax / njmax
    are per world, the pipeline's rigid_contact_max is global."""

    def __init__(self, num_worlds, structure="cube", start_steps=(0,), exe="E0", contact_max_per_world=8192, nconmax=8192, njmax=4096,
                 tri_pairs_per_world=300_000, verbose=False):
        self.N, self.structure = num_worlds, structure
        self.exe = D.EXE[exe]
        with quiet(not verbose):
            self.plan, self.name = D.make_plan(R.example_args(structure, 0, 0, True, False, None, exe))
            cell, c = D.build_cell(self.plan, self.exe, 0, SUBSTEPS, False)
        self.c, self.steps = c, self.plan["sequence"]
        self.nab = c.n_arm_bodies
        sc = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(sc)
        sc.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))      # global (world -1), shared
        sc.replicate(cell, num_worlds)
        m = self.model = sc.finalize()
        self.bpw, self.jpw, self.dpw = m.body_count // num_worlds, m.joint_count // num_worlds, m.joint_dof_count // num_worlds
        self.qpw, self.epw = m.joint_coord_count // num_worlds, m.equality_constraint_count // num_worlds
        self.n_global = int((m.shape_world.numpy() < 0).sum())
        self.spw = (m.shape_count - self.n_global) // num_worlds
        contact_max = contact_max_per_world * num_worlds
        m.rigid_contact_max = contact_max
        self.pipeline = newton.CollisionPipeline(m, reduce_contacts=True, rigid_contact_max=contact_max, broad_phase="sap",
                                                 max_triangle_pairs=max(4_000_000, tri_pairs_per_world * num_worlds))
        self.solver = newton.solvers.SolverMuJoCo(m, solver="newton", integrator="implicitfast", iterations=15, ls_iterations=100, nconmax=nconmax,
                                                  njmax=njmax, cone="elliptic", impratio=50.0, use_mujoco_contacts=False)
        self.state_0, self.state_1, self.control = m.state(), m.state(), m.control()
        self.contacts = self.pipeline.contacts()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.state_0)
        self._tables(sorted(set(start_steps)))
        self._ik()
        self.en_h = m.equality_constraint_enabled.numpy().copy()               # host mirrors of what the snap / reset toggle
        self.grp_h = m.shape_collision_group.numpy().copy()
        self.sim_time = np.zeros(num_worlds)
        self.aa_flags = wp.zeros(num_worlds * self.nab * self.nab, dtype=wp.uint8)
        self.shape_body, self.sb_h = m.shape_body, m.shape_body.numpy()
        self.graph = None
        self.simulate()                                                         # warm-up: compiles the kernels outside the capture
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            self.simulate()
        self.graph = cap.graph
        self.reset_worlds(np.arange(num_worlds), np.zeros(num_worlds, int) + self.starts[0])    # the warm-up and the capture stepped the sim

    # -- per-start-step tables ---------------------------------------------------------------------------------------
    def _tables(self, starts):
        """For each start step k: the joint state, body state, weld flags and collision groups of one world (arms parked, bricks of steps < k at
        their targets, welded and in STRUCT_GROUP; the others in the feeder)."""
        m, c, N = self.model, self.c, self.N
        assert starts and 0 <= starts[0] and starts[-1] < len(self.steps), starts
        self.starts = starts
        q0, qd0 = m.joint_q.numpy()[:self.qpw].copy(), m.joint_qd.numpy()[:self.dpw].copy()
        tgt0 = self.control.joint_target_pos.numpy()[:self.dpw].copy()
        child, qs = m.joint_child.numpy()[:self.jpw], m.joint_q_start.numpy()[:self.jpw]
        qstart = {int(child[j]): int(qs[j]) for j in range(self.jpw)}            # body -> its free joint's first coordinate
        en0 = np.zeros(self.epw, bool)
        g0 = m.shape_collision_group.numpy()[self.n_global:self.n_global + self.spw].copy()
        sb0 = m.shape_body.numpy()[self.n_global:self.n_global + self.spw]
        plate = np.array(c.plate_shapes)                                           # builder shape indices = world-local shape offsets
        self.T = {}
        for k in starts:
            q, en, g = q0.copy(), en0.copy(), g0.copy()
            g[plate] = STRUCT
            for s in self.steps[:k]:
                bid = s["brick_id"]
                pos, yaw, _ = c.target[bid]
                a = qstart[c.body[bid]]
                q[a:a + 7] = [*pos, *D.qz(yaw)]
                en[[w[0] for w in c.welds[bid]]] = True
                g[sb0 == c.body[bid]] = STRUCT
            self.state_0.joint_q.assign(np.tile(q, N))
            newton.eval_fk(m, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            bq = self.state_0.body_q.numpy().reshape(N, self.bpw, 7)[0].copy()
            self.T[k] = dict(q=q, qd=qd0, tgt=tgt0, bq=bq, bqd=np.zeros((self.bpw, 6), np.float32), en=en, g=g)

    # -- IK, drive targets ---------------------------------------------------------------------------------------------
    def _ik(self):
        """All 2N hands in one IKSolver call (24 iterations, warm-started from the last solution, as Example.step), captured as a graph."""
        P2 = 2 * self.N
        self.park_q = np.array(self.c.park_q, np.float32)
        self.ik_q = wp.array(np.tile(self.park_q, (P2, 1)), dtype=wp.float32)
        self.ik_pos, self.ik_rot = wp.zeros(P2, dtype=wp.vec3), wp.zeros(P2, dtype=wp.vec4)
        self.grip = wp.zeros(P2, dtype=wp.float32)
        mik = self.c.model_ik
        self.ik = ik.IKSolver(model=mik, n_problems=P2, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
                              objectives=[ik.IKObjectivePosition(link_index=D.EE, link_offset=wp.vec3(0, 0, 0), target_positions=self.ik_pos),
                                          ik.IKObjectiveRotation(link_index=D.EE, link_offset_rotation=wp.quat_identity(), target_rotations=self.ik_rot),
                                          ik.IKObjectiveJointLimit(joint_limit_lower=mik.joint_limit_lower, joint_limit_upper=mik.joint_limit_upper)])
        for _ in range(2):
            self.ik.step(self.ik_q, self.ik_q, iterations=24)
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            self.ik.step(self.ik_q, self.ik_q, iterations=24)
        self.graph_ik = cap.graph

    # -- the frame ----------------------------------------------------------------------------------------------------
    def simulate(self):
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, FRAME_DT / SUBSTEPS)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def frame(self, hand_pos, hand_rot, grip):
        """One physics frame in every world: IK for the hand targets (base frame: pos (N, 2, 3), rot xyzw (N, 2, 4), finger target (N, 2)), the drive
        targets, then the captured graph."""
        self.ik_pos.assign(hand_pos.reshape(-1, 3).astype(np.float32))
        self.ik_rot.assign(hand_rot.reshape(-1, 4).astype(np.float32))
        self.grip.assign(grip.reshape(-1).astype(np.float32))
        wp.capture_launch(self.graph_ik)
        wp.launch(scatter_targets, dim=2 * self.N, inputs=[self.ik_q, self.grip, self.dpw, self.control.joint_target_pos])
        wp.capture_launch(self.graph)
        self.sim_time += FRAME_DT

    def body_q(self):
        """state_0.body_q as (N, bpw, 7). Ground truth: reward, snap and extras only."""
        return self.state_0.body_q.numpy().reshape(self.N, self.bpw, 7)

    def joint_state(self):
        """(q (N, 18), qd (N, 18)): the two arms' joints (proprioception); the bricks' free joints follow in the same block."""
        return (self.state_0.joint_q.numpy().reshape(self.N, self.qpw)[:, :18], self.state_0.joint_qd.numpy().reshape(self.N, self.dpw)[:, :18])

    def arm_arm_contacts(self):
        """(N, nab * nab) uint8: which (A body, B body) pairs penetrate after the last frame (dist < 0), from the solver's contact arrays."""
        d = self.solver.mjw_data
        self.aa_flags.zero_()
        wp.launch(arm_arm_pairs, dim=d.contact.dist.shape[0], inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid,
                                                                      self.solver.mjc_geom_to_newton_shape, self.shape_body, self.bpw, self.nab, self.aa_flags])
        return self.aa_flags.numpy().reshape(self.N, -1)

    # -- reset, welds ---------------------------------------------------------------------------------------------------
    def reset_worlds(self, worlds, starts):
        """Worlds `worlds` back to the nominal start of step `starts` (one per world); the others are untouched. Joint and body state, drive targets,
        weld flags and collision groups are assigned in place (the graphs read the arrays), the IK warm start goes back to the park pose."""
        worlds, starts = np.asarray(worlds, int), np.asarray(starts, int)
        if not len(worlds):
            return
        s, N = self.state_0, self.N
        for arr, key in ((s.joint_q, "q"), (s.joint_qd, "qd"), (self.control.joint_target_pos, "tgt"), (s.body_q, "bq"), (s.body_qd, "bqd")):
            a = arr.numpy()
            a.reshape(N, -1)[worlds] = np.stack([np.ravel(self.T[k][key]) for k in starts])       # a view: rows of the worlds in place
            arr.assign(a)
        for w, k in zip(worlds, starts):
            self.en_h[w * self.epw:(w + 1) * self.epw] = self.T[k]["en"]
            self.grp_h[self.n_global + w * self.spw:self.n_global + (w + 1) * self.spw] = self.T[k]["g"]
        self._push_welds()
        q = self.ik_q.numpy().reshape(N, 2, -1)
        q[worlds] = self.park_q
        self.ik_q.assign(q.reshape(2 * N, -1))
        self.sim_time[worlds] = 0.0

    def _push_welds(self):
        self.model.equality_constraint_enabled.assign(self.en_h)
        self.model.shape_collision_group.assign(self.grp_h)
        self.solver.notify_model_changed(NOTIFY)

    def weld(self, worlds, steps):
        """Snap: weld the bricks of plan steps `steps` in `worlds` (the clutch welds on, the brick's shapes in STRUCT_GROUP), as Example.snap does."""
        for w, n in zip(worlds, steps):
            bid = self.steps[n]["brick_id"]
            self.en_h[w * self.epw + np.array([e[0] for e in self.c.welds[bid]])] = True
            lo = self.n_global + w * self.spw
            self.grp_h[lo:lo + self.spw][self.sb_h[lo:lo + self.spw] == w * self.bpw + self.c.body[bid]] = STRUCT
        self._push_welds()

    def unweld(self, w, bid, eq):
        """A weld lets go (JointBreaker.break_weld): weld `eq` (local index) of world w off, the brick's shapes back to the default group (they collide with the structure)."""
        self.en_h[w * self.epw + eq] = False
        lo = self.n_global + w * self.spw
        self.grp_h[lo:lo + self.spw][self.sb_h[lo:lo + self.spw] == w * self.bpw + self.c.body[bid]] = 1


class BatchedBreaker:
    """dual_arm_sim.JointBreaker in every world. Each frame, for every enabled weld of a world: the wrench the weld exerts on its upper brick from the
    solver's constraint forces (brick_wrench), moved to the patch centre and rotated into the nominal axes, -> the patch utilisation u (Patch.utilization:
    axial and prying load only). A weld lets go when u >= U_BREAK on BREAK_SAMPLES consecutive frames, counted from BREAK_SETTLE_S after its brick was welded
    (pre-placed bricks: from the start). One download of the solver's constraint arrays per frame, only while some world has an enabled weld; the maths per
    weld is JointBreaker.sample's, on those arrays. The u2 / peak bookkeeping of the Example is not kept."""

    def __init__(self, cells):
        import stability as ST
        self.cells, c, N = cells, cells.c, cells.N
        conns = {(k.upper, k.lower): k for k in ST.connections(P.STRUCTURES[cells.name])}
        self.weld = {(bid, w[1]): (bid, w[0]) for bid, ws in c.welds.items() for w in ws}              # (brick, support) -> (brick id, local equality index)
        assert set(self.weld) == set(conns), "weld pool vs stability.connections"
        self.conn, self.step_of = conns, {s["brick_id"]: n for n, s in enumerate(cells.steps)}
        b2n, e2n = cells.solver.mjc_body_to_newton.numpy(), cells.solver.mjc_eq_to_newton_eq.numpy()
        self.mjb = [{bid: int(np.where(b2n[w] == w * cells.bpw + b)[0][0]) for bid, b in c.body.items()} for w in range(N)]       # newton body -> mujoco body, per world
        self.mid = [{eq: int(np.where(e2n[w] == w * cells.epw + eq)[0][0]) for _, eq in self.weld.values()} for w in range(N)]     # local weld -> mujoco constraint id
        self.ts = cells.model.equality_constraint_torquescale.numpy()
        self.fired, self.count, self.failed = np.full((N, len(cells.steps)), np.inf), np.zeros((N, cells.epw), int), np.zeros(N, bool)

    def reset(self, worlds, starts):
        for w, k in zip(worlds, starts):
            self.fired[w], self.count[w], self.failed[w] = np.inf, 0, False
            self.fired[w, :k] = 0.0                                                                   # the bricks standing at the start were "fired" at t = 0

    def fire(self, worlds, steps):
        self.fired[worlds, steps] = self.cells.sim_time[worlds]

    def snapshot(self):
        """The solver's constraint arrays of every world (the last substep of the last frame), cut to the rows in use."""
        d = self.cells.solver.mjw_data
        nefc = d.nefc.numpy()
        n = int(nefc.max())
        tt = lambda a: wp.to_torch(a)[:, :n].cpu().numpy()
        return dict(nefc=nefc, typ=tt(d.efc.type), ids=tt(d.efc.id), frc=tt(d.efc.force), xpos=d.xpos.numpy(), xmat=d.xmat.numpy())

    def sample(self, w, snap):
        """{(brick, support): dict(u, F, M, Fp, Mp)} for every enabled weld of world w, JointBreaker.sample's numbers: F, M the wrench on the upper brick (world,
        about its origin), Fp, Mp at the patch centre in the nominal axes."""
        cells, xpos, xmat = self.cells, snap["xpos"][w], snap["xmat"][w]
        eqr = np.flatnonzero(snap["typ"][w, :snap["nefc"][w]] == D.EQUALITY)
        u_ids, first = np.unique(snap["ids"][w, eqr], return_index=True)
        row0, en, out = dict(zip(u_ids.tolist(), eqr[first].tolist())), cells.en_h[w * cells.epw:(w + 1) * cells.epw], {}
        for (bid, sid), (_, eq) in self.weld.items():
            r0 = row0.get(self.mid[w][eq])
            if not en[eq] or r0 is None:                                                            # disabled, or welded this very frame: no solver row yet
                continue
            f = snap["frc"][w, r0:r0 + 6].astype(float)
            up = self.mjb[w][bid]
            F, M = -f[:3], -0.5 * float(self.ts[w * cells.epw + eq]) * (xmat[up] @ f[3:6])          # brick_wrench: the wrench ON the upper brick, about its origin, world
            conn = self.conn[(bid, sid)]
            if sid is None:
                c, Rn = conn.centre, np.eye(3)
            else:
                lo, (p_nom, yaw, _) = self.mjb[w][sid], cells.c.target[sid]
                R_nom = D.rotmat(D.qz(yaw))
                c = xpos[lo] + xmat[lo] @ R_nom.T @ (conn.centre - p_nom)
                Rn = R_nom @ xmat[lo].T
            Fp, Mp = Rn @ F, Rn @ (M + np.cross(xpos[up] - c, F))
            out[(bid, sid)] = dict(u=float(conn.patch.utilization(Fp[2], Mp[0], Mp[1])), F=F, M=M, Fp=Fp, Mp=Mp)
        return out

    def update(self):
        """The break rule on every enabled weld. -> (N,) bool, the worlds in which a weld broke this frame."""
        cells, N = self.cells, self.cells.N
        en = cells.en_h.reshape(N, -1)
        live = np.flatnonzero(en.any(1) & ~self.failed)
        broke = np.zeros(N, bool)
        if not len(live):
            return broke
        snap = self.snapshot()
        for w in live:
            for (bid, sid), r in self.sample(w, snap).items():
                eq = self.weld[(bid, sid)][1]
                if cells.sim_time[w] - self.fired[w, self.step_of[bid]] < D.BREAK_SETTLE_S - 1e-9:
                    self.count[w, eq] = 0
                    continue
                self.count[w, eq] = self.count[w, eq] + 1 if r["u"] >= D.U_BREAK else 0
                if self.count[w, eq] >= D.BREAK_SAMPLES:
                    cells.unweld(w, bid, eq)
                    broke[w] = self.failed[w] = True
        if broke.any():
            cells._push_welds()
        return broke


# --- the cameras ------------------------------------------------------------------------------------------------------
def world_draw(seed):
    """What cell.cameras.Cameras(model, seed) draws once per seed, in its order: the hidden extrinsic errors (top, wrist_B, then wrist_A on its own
    stream), the key light direction, the exposure gain, and the seed of the noise stream."""
    rng = np.random.default_rng(seed)
    hidden = {name: CAM.hidden_error(name, rng) for name in CAM.CAMS if name != "wrist_A"}
    hidden["wrist_A"] = CAM.hidden_error("wrist_A", np.random.default_rng([seed, 2]))
    key = tuple(-np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)]))
    gain = float(rng.uniform(0.5, 2.0))
    return dict(hidden=hidden, key_dir=key, gain=gain, noise_seed=int(np.random.default_rng([seed, 1]).integers(2 ** 62)))


def sense_torch(depth, hdr, gain, gens, dropout=0.01, s=1.0):
    """cell.cameras.sense on the GPU, for a chunk of worlds: depth (C, H, W) clean ray distance [m], hdr (C, H, W, 3) linear, gain (C,), one
    torch.Generator per world. The same noise model (depth sigma 2e-3 r^2, dropout at depth edges, rgb sigma 2/255 after the gain and a soft clip),
    drawn in the same order. -> (depth float32 (C, H, W), rgb uint8 (C, H, W, 3))."""
    d = depth
    edge = torch.zeros_like(d, dtype=torch.bool)
    jv, jh = (d[:, 1:] - d[:, :-1]).abs() > 3 * CAM.MM * s, (d[:, :, 1:] - d[:, :, :-1]).abs() > 3 * CAM.MM * s
    edge[:, :-1] |= jv
    edge[:, 1:] |= jv
    edge[:, :, :-1] |= jh
    edge[:, :, 1:] |= jh
    n_d, n_u, n_c = torch.empty_like(d), torch.empty_like(d), torch.empty_like(hdr)
    for i, g in enumerate(gens):
        torch.randn(d.shape[1:], generator=g, device=d.device, out=n_d[i])
        torch.rand(d.shape[1:], generator=g, device=d.device, out=n_u[i])
        torch.randn(hdr.shape[1:], generator=g, device=d.device, out=n_c[i])
    noisy = torch.where(d > 0, d + n_d * CAM.sigma_depth(d), torch.zeros_like(d))
    noisy = torch.where(edge & (n_u < dropout), torch.zeros_like(d), noisy)
    rgb = (1 - torch.exp(-hdr * gain.view(-1, 1, 1, 1)) + n_c * (2 / 255)).clamp(0, 1)
    return noisy, (rgb * 255 + 0.5).to(torch.uint8)


class BatchedCameras:
    """The cameras of N worlds with one SensorTiledCamera: `capture(cam, ...)` renders every world's view in one update and applies the sensor noise on
    the GPU with one generator per world. Per world and per reseed(): the hidden extrinsic error, the exposure gain and the noise stream (the same draws as
    Cameras(seed)); the key light is one for the env (from the first seed). Outputs are torch tensors on the GPU. Rendering is an ordinary launch on the
    model's state: call it between graph launches. `res_div`: render at 1/res_div of cell.cameras.CAMS' size (the noise is then applied at that size)."""

    def __init__(self, cells, seeds, cameras=("top",), res_div=1, chunk=16):
        self.cells, self.N, self.cameras, self.chunk, self.res_div = cells, cells.N, tuple(cameras), chunk, res_div
        m = cells.model
        newton.geometry.build_bvh_shape(m, cells.state_0)
        self.sensor = SensorTiledCamera(m, config=SensorTiledCamera.RenderConfig(enable_shadows=True))
        u, self.out = self.sensor.utils, {}
        for name in self.cameras:
            c = CAM.CAMS[name]
            w, h = c["w"] // res_div, c["h"] // res_div
            self.out[name] = dict(rays=u.compute_pinhole_camera_rays(w, h, math.radians(c["vfov"])), depth=u.create_depth_image_output(w, h),
                                  hdr=u.create_hdr_color_image_output(w, h))
        self.draw = [None] * self.N
        self.gens = [torch.Generator(device="cuda") for _ in range(self.N)]
        self.gain = torch.ones(self.N, device="cuda")
        self.hidden = {n: (np.zeros((self.N, 3)), np.zeros((self.N, 3))) for n in CAM.CAMS}
        self.reseed(np.arange(self.N), seeds)
        CAM.set_lights(self.sensor, m.device, self.draw[0]["key_dir"])
        self.ms = {n: 0.0 for n in ("render", "noise")}

    def reseed(self, worlds, seeds):
        for w, sd in zip(worlds, seeds):
            d = self.draw[w] = world_draw(int(sd))
            self.gens[w].manual_seed(d["noise_seed"])
            self.gain[w] = d["gain"]
            for n in CAM.CAMS:
                self.hidden[n][0][w], self.hidden[n][1][w] = d["hidden"][n]

    def poses(self, cam, ee=None):
        """(true pos (N, 3), true Rotation (N,), nominal pos (N, 3), nominal Rotation (N,)): Cameras.poses in bulk. Wrist cameras need the hands' FK
        poses `ee` (N, 7) [x y z qx qy qz qw]."""
        dt, dr = self.hidden[cam]
        if cam == "top":
            npos, nR = CAM.nominal_pose("top")
            npos, nR = np.broadcast_to(npos, (self.N, 3)), Rot.from_quat(np.tile(nR.as_quat(), (self.N, 1)))
            return npos + dt, nR * Rot.from_rotvec(dr), npos, nR
        hR = Rot.from_quat(ee[:, 3:7])
        npos, nR = CAM.nominal_pose(cam, (ee[:, :3], hR), CAM.WRIST_DX)
        return npos + hR.apply(dt), nR * Rot.from_rotvec(dr), npos, nR

    def capture(self, cam, ee=None):
        """Render `cam` in every world (state_0 of the cells) and add the sensor noise. -> (depth (N, H, W) float32 [m, ray distance, 0 = invalid],
        rgb (N, H, W, 3) uint8, nominal camera pos (N, 3), nominal Rotation (N,)) with depth / rgb torch tensors on the GPU."""
        t0 = time.perf_counter()
        pos, R, npos, nR = self.poses(cam, ee)
        xf = wp.array(np.concatenate([pos, R.as_quat()], 1).astype(np.float32).reshape(1, self.N, 7), dtype=wp.transformf)
        o = self.out[cam]
        newton.geometry.refit_bvh_shape(self.cells.model, self.cells.state_0)
        self.sensor.update(self.cells.state_0, xf, o["rays"], depth_image=o["depth"], hdr_color_image=o["hdr"])
        wp.synchronize()
        t1 = time.perf_counter()
        depth, hdr = wp.to_torch(o["depth"])[:, 0], wp.to_torch(o["hdr"])[:, 0]
        nd, rgb = torch.empty_like(depth), torch.empty(hdr.shape, dtype=torch.uint8, device=hdr.device)
        for a in range(0, self.N, self.chunk):
            b = min(a + self.chunk, self.N)
            nd[a:b], rgb[a:b] = sense_torch(depth[a:b], hdr[a:b], self.gain[a:b], self.gens[a:b])
        torch.cuda.synchronize()
        self.ms["render"] += (t1 - t0) * 1e3
        self.ms["noise"] += (time.perf_counter() - t1) * 1e3
        return nd, rgb, npos, nR


class BatchedFK:
    """The 2N hands' poses from the measured joint angles alone (proprioception): one arm replicated 2N times, one eval_fk (Example.hand_fk, in bulk)."""

    def __init__(self, n):
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.replicate(D.build_arm(SUBSTEPS), 2 * n)
        self.model = b.finalize()
        self.state, self.n, self.nab = self.model.state(), n, self.model.body_count // (2 * n)

    def __call__(self, q):
        """q (n, 18) the two arms' joints -> TCP poses (n, 2, 7) [x y z qx qy qz qw] in the world."""
        qa = wp.array(np.ascontiguousarray(q, np.float32).reshape(-1), dtype=wp.float32)
        newton.eval_fk(self.model, qa, wp.zeros_like(qa), self.state)
        t = self.state.body_q.numpy().reshape(2 * self.n, self.nab, 7)[:, D.EE].reshape(self.n, 2, 7)
        return hand_world_pose(t)


# --- the env ----------------------------------------------------------------------------------------------------------
class BatchedReachEnv:
    """BrickReachEnv in N worlds of one Newton model (BatchedCells), with NewtonVecEnv's interface: reset(seed, options) -> (obs, extras), step(actions)
    -> (obs, reward, terminated, truncated, extras), obs = {"policy": float32 (N, 390), "images": {...}}, torch tensors on `device`; an episode that ends
    (terminated or truncated) is reset inside the same step() and the observation returned is the new episode's first one; the others continue.

    Task, action, "policy" layout, reward and snap rule: see BrickReachEnv and the module docstring. `start_steps`: the plan steps a world may start at
    (bricks before it welded on the plate), one drawn per world at every reset; reset(options={"step": int or (N,) ints}) pins them.
    extras: "log" / "episode" (Episode/return, Episode/length, Episode/success of the episodes that ended), "privileged" (ground truth per world: brick_pos,
    tcp_B, target, lateral / dz / dyaw / tilt, lifted, dist_tcp, arm_arm, step), "events" ({name: bool (N,)}, rl_env.EVENTS).
    `detect_stride`: top_detect reads the noised top image at 1/detect_stride (BrickReachEnv reads it whole: 1); `res_div`: the renderer's size divisor
    (res_div 2 with detect_stride 1 reads the same pixel grid as detect_stride 2 at full size, for a quarter of the render); `procs`: worker processes that run
    detect -> remember -> track_held over the worlds (rl_detect.py; a thread pool does not scale: the GIL), None = none up to 7 worlds, else min(16, N // 2)."""

    def __init__(self, num_envs, device="cuda:0", structure="cube", start_steps=(0,), control_every=6, max_steps=600, dpos=R.DPOS, dyaw=R.DYAW, images=False,
                 cameras=R.CAM_NAMES, image_stride=4, n_max=16, exe="E0", seed=0, detect_stride=1, res_div=2, procs=None, verbose=False,
                 w_reach=R.REACH_WEIGHTS["w_reach"], w_grasp=R.REACH_WEIGHTS["w_grasp"], w_place=R.REACH_WEIGHTS["w_place"], w_yaw=R.REACH_WEIGHTS["w_yaw"],
                 w_time=R.REACH_WEIGHTS["w_time"], w_success=R.REACH_WEIGHTS["w_success"], w_break=R.REACH_WEIGHTS["w_break"],
                 w_arm_arm=R.REACH_WEIGHTS["w_arm_arm"], w_drop=R.REACH_WEIGHTS["w_drop"], lift_dz=R.LIFT_DZ):
        kw = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.num_envs, self.device = num_envs, torch.device(device)
        self.structure, self.start_steps, self.every, self.max_steps, self.dpos, self.dyaw = structure, tuple(start_steps), control_every, max_steps, dpos, dyaw
        self.images, self.detect_stride, self.res_div = images, detect_stride, res_div
        N = num_envs
        self.cells = cells = BatchedCells(N, structure, self.start_steps, exe, verbose=verbose)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        assert not images or image_stride % res_div == 0, "image_stride must be a multiple of res_div"
        self.cams = BatchedCameras(cells, np.arange(N) + seed, tuple(dict.fromkeys(("top",) + tuple(cameras))) if images else ("top",), res_div)
        self.fk = BatchedFK(N)
        self.breaker = BatchedBreaker(cells)
        c = CAM.CAMS["top"]
        self.top_hw = (c["h"] // res_div, c["w"] // res_div)
        self.intr = dict(CAM.intrinsics("top"), width=self.top_hw[1], height=self.top_hw[0])
        procs = (0 if N < 8 else min(16, N // 2)) if procs is None else procs
        self.detector = rl_detect.Detector(N, (-(-self.top_hw[0] // detect_stride), -(-self.top_hw[1] // detect_stride)), self.intr, CAM.TOP_POS, Rot.identity(), procs)
        # the plan, per step
        c, steps = cells.c, cells.steps
        bid = [s["brick_id"] for s in steps]
        self.body_local = np.array([c.body[b] for b in bid])
        self.tgt = np.array([c.target[b][0] for b in bid])
        self.tyaw = np.array([c.target[b][1] for b in bid])
        self.square = np.array([c.target[b][2] for b in bid])
        self.slot = np.array([c.feeder[b] for b in bid])
        self.task = [dict(x=t[0], y=t[1], z=t[2], yaw=y, L=c.dims[b][0], W=c.dims[b][1], step=n, n=len(steps)) for n, (b, t, y) in enumerate(zip(bid, self.tgt, self.tyaw))]
        self.z_travel = max(s["target_pose"][2] for s in steps) + P.BRICK_H + D.TRAVEL
        # the hands: ManualCtl's targets and the command that follows them, for every world
        park = lambda base: np.array(base[0]) + D.rotate(D.qz(base[1]), D.PARK)
        self.parks = np.array([park(D.ARM_A), park(D.ARM_B)])
        self.man = LV.ManualCtlBatch(N, (D.ARM_A, D.ARM_B), self.parks)
        self.cmd_p, self.cmd_yaw, self.cmd_g = np.tile(self.parks, (N, 1, 1)), np.tile([D.ARM_A[1], D.ARM_B[1]], (N, 1)), np.full((N, 2), 0.01)
        # observation / reward
        self.obs_spec = R.ObsSpec(n_max, False)
        self.img_spec = R.ObsSpec(n_max, images, cameras, image_stride)
        self.rew = R.ReachReward(N, max_steps, lift_dz, w_reach=w_reach, w_grasp=w_grasp, w_place=w_place, w_yaw=w_yaw, w_time=w_time, w_success=w_success,
                                 w_break=w_break, w_arm_arm=w_arm_arm, w_drop=w_drop)
        sp = self.img_spec.space()
        d = {"policy": spaces.Box(-np.inf, np.inf, (sum(int(np.prod(sp[k].shape)) for k in V.POLICY_KEYS if k in sp.spaces),), np.float32)}
        if images:
            d["images"] = sp["images"]
        self.single_observation_space = spaces.Dict(d)
        self.single_action_space = spaces.Box(-1.0, 1.0, (10,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, N)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, N)
        self.max_episode_length = max_steps
        self.cfg = V.NewtonVecEnvCfg(env_name="BatchedReachEnv", decimation=control_every, step_dt=control_every / R.FPS, episode_length=max_steps, seed=seed,
                                     env_kwargs={k: v for k, v in kw.items() if k not in ("num_envs", "device")})
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self._ret = torch.zeros(N, device=self.device)
        self.step_n = np.zeros(N, int)
        self.placed, self.diverged, self.broke = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
        self.aa_prev, self.aa_count = np.zeros((N, cells.nab * cells.nab), bool), np.zeros(N, int)
        self.ms = dict(frames=0.0, obs=0.0, fk=0.0, copy=0.0, detect=0.0, ticks=0)      # wall time per phase, ms summed over the ticks since clear_timing() (the cameras' own: cams.ms)
        self._obs = None
        self._reset_worlds(np.arange(N), np.arange(N) + seed)
        self._obs = self._observe()

    # -- interface ------------------------------------------------------------------------------------------------------
    @property
    def unwrapped(self):
        return self

    def seed(self, seed=-1):
        self._seed = seed
        return seed

    def clear_timing(self):
        self.ms = {k: 0.0 for k in self.ms}
        self.cams.ms = {k: 0.0 for k in self.cams.ms}

    def reset(self, seed=None, options=None):
        """All worlds to a fresh episode. seed: re-seeds the env's rng; world w's camera seed is seed + w (the single env's, for w = 0)."""
        seed = self._seed if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self._reset_worlds(np.arange(self.num_envs), np.arange(self.num_envs) + seed, (options or {}).get("step"))
        self._obs = self._observe()
        self.episode_length_buf.zero_()
        self._ret.zero_()
        return self._obs, {"log": {}}

    def _get_observations(self):
        return self._obs

    def close(self):
        self.detector.close()

    def _reset_worlds(self, worlds, cam_seeds, step=None):
        """The physics, controller, reward and vision state of `worlds` back to a nominal start."""
        N, cells = self.num_envs, self.cells
        starts = (np.broadcast_to(step, (N,))[worlds] if step is not None else self.rng.choice(self.start_steps, len(worlds))).astype(int)
        self.step_n[worlds] = starts
        cells.reset_worlds(worlds, starts)
        mask = np.zeros(N, bool)
        mask[worlds] = True
        self.man.reset(mask)
        self.cmd_p[mask], self.cmd_yaw[mask], self.cmd_g[mask] = self.parks, [D.ARM_A[1], D.ARM_B[1]], 0.01
        self.rew.reset(mask, self.slot[self.step_n])
        self.placed[worlds], self.diverged[worlds], self.broke[worlds], self.aa_prev[worlds], self.aa_count[worlds] = False, False, False, False, 0
        self.breaker.reset(worlds, starts)
        self.cams.reseed(worlds, cam_seeds)
        self.detector.reset(worlds)
        self.pv = self._privileged(cells.body_q())

    # -- ground truth: the reward, the snap and extras only -------------------------------------------------------------------
    def _privileged(self, bq):
        """Per world: the current brick's true pose against its target, B's TCP, the brick's slot. NaN (a diverged world) becomes 0 so the reward is finite."""
        ar, n, nab = np.arange(self.num_envs), self.step_n, self.cells.nab
        brick, tcp = np.nan_to_num(bq[ar, self.body_local[n]]), np.nan_to_num(bq[:, nab + D.EE, :3])
        lat, dz, dyaw, tilt = R.pose_errors(brick, self.tgt[n], self.tyaw[n], self.square[n])
        return dict(brick=brick, brick_pos=brick[:, :3], tcp_B=tcp, target=self.tgt[n], lateral=lat, dz=dz, dyaw=dyaw, tilt=tilt, step=n.copy())

    def _snap(self, bq):
        """Example.rl_snap in every world: the current brick passes the gate -> weld it (its clutch welds on, its shapes in STRUCT_GROUP)."""
        self.pv = pv = self._privileged(bq)
        ok = gate_ok(pv["lateral"], pv["dz"], pv["dyaw"], pv["tilt"]) & ~self.placed & ~self.diverged
        if ok.any():
            w = np.flatnonzero(ok)
            self.cells.weld(w, self.step_n[w])
            self.breaker.fire(w, self.step_n[w])
            self.placed[w] = True

    def _frames(self):
        """`control_every` physics frames in every world, Example.step's direct (rl_control) path: command <- target (slew, leash on the measured hand),
        IK, drive targets, graph; then the joint-break rule, the snap gate and the arm-arm contacts."""
        cells, nab = self.cells, self.cells.nab
        bq = cells.body_q()
        for _ in range(self.every):
            self.diverged |= ~np.isfinite(bq).all((1, 2))
            ee = bq[:, [D.EE, nab + D.EE], :3]
            self.cmd_p, self.cmd_yaw, self.cmd_g = self.man.advance(self.cmd_p, self.cmd_yaw, self.cmd_g, ee, FRAME_DT)
            cells.frame(*hand_target_pose(self.cmd_p, self.cmd_yaw), self.cmd_g)
            bq = cells.body_q()
            self.broke |= self.breaker.update()                           # before the snap, as Example.step: a weld is sampled from the frame after it was made
            self._snap(bq)
            aa = cells.arm_arm_contacts().astype(bool)
            self.aa_count += (aa & ~self.aa_prev).sum(1)                  # a new (A body, B body) penetration = a new arm_arm event (ContactMonitor's fragments)
            self.aa_prev = aa
        self.diverged |= ~np.isfinite(bq).all((1, 2))
        return bq

    # -- what a policy may see ----------------------------------------------------------------------------------------------------
    def _observe(self):
        """Render, noise, detect and encode every world: -> {"policy": (N, D), ["images": {...}]} torch tensors on the device. The firewall of
        rl_env.Sim.sense, in bulk: the inputs are the top capture (-> top_detect: detect, remember, track_held), the joint state and its FK, the plan."""
        t0 = time.perf_counter()
        cells, N = self.cells, self.num_envs
        q, qd = cells.joint_state()
        fk = self.fk(q)
        hands_R = Rot.from_quat(fk[..., 3:7].reshape(-1, 4)).as_matrix().reshape(N, 2, 3, 3)
        t1 = time.perf_counter()
        top_depth, top_rgb, npos, nR = self.cams.capture("top")
        t2 = time.perf_counter()
        ds = self.detect_stride
        det = self.detector
        det.rgb_t.copy_(top_rgb[:, ::ds, ::ds])                      # straight into the workers' shared memory
        det.depth_t.copy_(top_depth[:, ::ds, ::ds])
        t3 = time.perf_counter()
        hands = [[dict(name="AB"[k], pos=fk[w, k, :3], R=hands_R[w, k], opening=q[w, 9 * k + 7:9 * k + 9]) for k in range(2)] for w in range(N)]
        dets = det.detect(hands)
        t4 = time.perf_counter()
        rows = [self.obs_spec.encode(R.ObsInputs(q[w], qd[w], [fk[w, 0], fk[w, 1]], dets[w], None, self.task[self.step_n[w]], None)) for w in range(N)]
        out = {k: np.stack([r[k] for r in rows]) for k in V.POLICY_KEYS if k in rows[0]}
        obs = {"policy": torch.as_tensor(np.concatenate([out[k].reshape(N, -1) for k in V.POLICY_KEYS if k in out], 1), dtype=torch.float32, device=self.device)}
        if self.images:
            s = self.img_spec.stride // self.res_div
            im = {}
            for cam in self.img_spec.cameras:
                depth, rgb = (top_depth, top_rgb) if cam == "top" else self.cams.capture(cam, fk[:, int(cam[-1] == "B")])[:2]
                im[cam + "_rgb"], im[cam + "_depth"] = rgb[:, ::s, ::s].contiguous().to(self.device), depth[:, ::s, ::s].contiguous().to(self.device)
            obs["images"] = im
        t5 = time.perf_counter()
        self.ms["obs"] += (t5 - t0) * 1e3
        self.ms["fk"] += (t1 - t0) * 1e3
        self.ms["copy"] += (t3 - t2) * 1e3
        self.ms["detect"] += (t4 - t3) * 1e3
        return obs

    # -- one tick -------------------------------------------------------------------------------------------------------------------
    def step(self, actions):
        N, cells = self.num_envs, self.cells
        a = actions.detach().float().cpu().numpy().reshape(N, 10)
        self.man.nudge(*R.action_targets(a, self.dpos, self.dyaw))
        t0 = time.perf_counter()
        bq = self._frames()
        self.ms["frames"] += (time.perf_counter() - t0) * 1e3
        pv = self.pv = self._privileged(bq)
        r, ev, term, trunc, dist_tcp = self.rew.update(pv["brick_pos"], pv["tcp_B"], pv["target"], pv["dyaw"], pv["tilt"], self.placed,
                                                       self.broke, self.aa_count, self.diverged)
        done = term | trunc
        dt = torch.as_tensor(done, device=self.device)
        self.episode_length_buf += 1
        rew = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        self._ret += rew
        ep = {"Episode/return": self._ret[dt].clone(), "Episode/length": self.episode_length_buf[dt].float(),
              "Episode/success": torch.as_tensor(ev["success"], dtype=torch.float32, device=self.device)[dt]}
        extras = {"log": ep, "episode": ep, "events": {k: torch.as_tensor(v, device=self.device) for k, v in ev.items()},
                  "privileged": {k: torch.as_tensor(v, device=self.device) for k, v in pv.items() if k != "brick"} | {
                      "lifted": torch.as_tensor(self.rew.lifted.copy(), device=self.device), "dist_tcp": torch.as_tensor(dist_tcp, device=self.device),
                      "arm_arm": torch.as_tensor(self.aa_count.copy(), device=self.device)}}
        self._ret[dt] = 0
        self.episode_length_buf[dt] = 0
        if done.any():
            w = np.flatnonzero(done)
            self._reset_worlds(w, self.rng.integers(2 ** 31 - 1, size=len(w)))
        self._obs = self._observe()
        self.ms["ticks"] += 1
        return self._obs, rew, torch.as_tensor(term, device=self.device), torch.as_tensor(trunc, device=self.device), extras


# --- the scripted controller, per world (ground truth lives here only, as in rl_env.ScriptedReach) -----------------------------
class WorldView:
    """What rl_env.ScriptedReach reads of a BrickReachEnv (step, plan, feeder, the hand target, privileged()), for world w of a BatchedReachEnv: the
    same controller code drives every world."""

    def __init__(self, env, w):
        self.env, self.w, self.dpos, self.dyaw = env, w, env.dpos, env.dyaw
        self.sim = self.ex = self.man = self                      # env.sim.ex.man.tgt[1] etc. all resolve here

    step_n = property(lambda s: int(s.env.step_n[s.w]))
    plan = property(lambda s: s.env.cells.plan)
    feeder = property(lambda s: s.env.cells.c.feeder)
    z_travel = property(lambda s: s.env.z_travel)
    tgt = property(lambda s: [None, dict(p=s.env.man.tgt_p[s.w, 1], yaw=float(s.env.man.tgt_yaw[s.w, 1]))])

    def privileged(self):
        pv, w = self.env.pv, self.w
        return dict(tcp_B=pv["tcp_B"][w], brick_pos=pv["brick_pos"][w], target=pv["target"][w])


class ScriptedBatch:
    """One ScriptedReach per world; `act()` -> (N, 10) torch actions. A world that starts a new episode (autoreset) gets a fresh controller."""

    def __init__(self, env):
        self.env = env
        self.pols = [R.ScriptedReach(WorldView(env, w)) for w in range(env.num_envs)]

    def reset(self, worlds):
        for w in worlds:
            self.pols[w] = R.ScriptedReach(WorldView(self.env, w))

    def act(self):
        return torch.as_tensor(np.stack([p.act() for p in self.pols]), device=self.env.device)


def run_scripted(env, max_ticks=None, seed=0):
    """The scripted controller in every world until each has finished its first episode. -> dict of per-world numpy arrays: ticks (to the first episode's end),
    ret, success, event (first event name), slip_mm (max drift of the brick in the hand: |brick - hand| against its value at the first lift, while lifted)."""
    N = env.num_envs
    env.reset(seed=seed)
    sb = ScriptedBatch(env)
    res = dict(ticks=np.zeros(N, int), ret=np.zeros(N), success=np.zeros(N, bool), event=[""] * N, slip_mm=np.zeros(N))
    done, off = np.zeros(N, bool), np.full((N, 3), np.nan)
    for t in range(max_ticks or env.max_steps):
        _, r, te, tr, ex = env.step(sb.act())
        r, te, tr = r.cpu().numpy(), te.cpu().numpy(), tr.cpu().numpy()
        live = ~done
        res["ret"][live] += r[live]
        # slip is read before the autoreset of this tick has changed the pv: env.pv is the post-reset state, so use the privileged extras
        pv = {k: v.cpu().numpy() for k, v in ex["privileged"].items() if k in ("brick_pos", "tcp_B", "lifted")}
        rel = pv["brick_pos"] - pv["tcp_B"]
        first = live & pv["lifted"] & np.isnan(off[:, 0])
        off[first] = rel[first]
        carrying = live & pv["lifted"] & ~np.isnan(off[:, 0])
        res["slip_mm"][carrying] = np.maximum(res["slip_mm"][carrying], 1e3 * np.linalg.norm(rel - off, axis=1)[carrying])
        fin = live & (te | tr)
        for w in np.flatnonzero(fin):
            res["ticks"][w], res["success"][w] = t + 1, ex["events"]["success"][w].item()
            res["event"][w] = next((k for k in R.EVENTS if ex["events"][k][w].item()), "truncated")
        sb.reset(np.flatnonzero(te.astype(bool) | tr.astype(bool)))
        done |= fin
        if done.all():
            break
    res["ticks"][~done] = max_ticks or env.max_steps
    return res


def demo(n=16):
    t0 = time.perf_counter()
    env = BatchedReachEnv(n)
    print("BatchedReachEnv(%d) built in %.1f s" % (n, time.perf_counter() - t0))
    t0 = time.perf_counter()
    res = run_scripted(env)
    dt = time.perf_counter() - t0
    print("scripted: success %d / %d, ticks to success %s, return %s, max slip %.2f mm, %.1f s (%.0f env steps/s)" % (
        res["success"].sum(), n, res["ticks"].tolist(), np.round(res["ret"], 1).tolist(), res["slip_mm"].max(), dt, env.ms["ticks"] * n / dt))
    env.close()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo(int(sys.argv[sys.argv.index("--demo") + 1]) if len(sys.argv) > sys.argv.index("--demo") + 1 else 16)
    else:
        print(__doc__)
