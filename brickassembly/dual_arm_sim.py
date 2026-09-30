"""Dual-arm LEGO assembly, live: images -> blueprint -> plan -> two Frankas.

Arm B (the placer, at +x) picks every brick from a feeder and presses it onto
the structure. Arm A (the stabilizer, at -x) presses down over the joint the
planner predicts will pry open -- before B arrives, until B lets go. Physics is
Newton (MuJoCo-Warp solver) with studded, hollow brick meshes; the viewer runs
live, in a window or a browser.

    bash scripts/run.sh dual_arm_sim.py --shape arch               # window
    bash scripts/run.sh dual_arm_sim.py --shape arch --viewer viser  # :8080
    bash scripts/run.sh dual_arm_sim.py --shape arch --strategy none
    bash scripts/run.sh dual_arm_sim.py --structure S3
    bash scripts/run.sh dual_arm_sim.py --front my.png --top my_top.png --width 8

ponytail: the clutch is a pre-allocated weld pool switched on when the gate
passes (plan §3.3, "loop joints"), and a weld never lets go -- so the brace is
executed but cannot change an outcome here. Measured: the worst pry at the
arch crown was 0.34 mm unbraced vs 0.41 mm braced. Joint failure, and with it
ablation A6, stays on BrickSim's force model (orchestration/run_plan.py).
Motion is IK on interpolated waypoints with no collision-aware planning; the
two arms work from opposite sides of the build.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
# Newton ships a single-arm brick-stacking example with LEGO-accurate meshes and
# tuned contact settings. Reused as-is rather than rebuilt (master_report §0.3);
# env.lock pins newton 1.2.1, which is what these names are checked against.
from newton.examples.contacts import example_brick_stacking as ex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blueprint as B  # noqa: E402
import planner as P  # noqa: E402

HERE = Path(__file__).resolve().parent
EE = 11                       # fr3_hand_tcp, in a single-arm builder
ARM_A = ((-0.45, 0.0, 0.0), 0.0)        # stabilizer, faces +x (plan §3.5)
ARM_B = ((0.45, 0.0, 0.0), math.pi)     # placer, faces -x
HOME_Q = [0.0, 0.024, 0.0, -2.368, 0.0, 2.392, 0.785]
PARK = np.array([0.30, 0.0, 0.35])     # hand, in its own arm's base frame
Z0 = 0.0032                   # baseplate top: a 3.2 mm plate on the table
COLORS = [(0.80, 0.10, 0.10), (0.10, 0.25, 0.80), (0.95, 0.75, 0.10),
          (0.10, 0.60, 0.20), (0.90, 0.90, 0.90), (0.95, 0.45, 0.10)]

# Calibration knobs -- geometric quantities, measure them, do not assume.
GRASP_DZ = 0.013              # TCP above a brick's bottom face while gripping; keeps the
                              # fingertips clear of the studs on the course below
FINGER_KE = 1000.0            # N/m finger drive; squeeze = FINGER_KE * 1.5 mm per finger
TIP_BELOW_TCP = 0.0089        # fingertips past the TCP (measured from the FR3 finger mesh)
PRESS = 0.001                 # placer drives this far past seated
BRACE_PRESS = 0.002           # stabilizer drives this far into what it braces

# Finger contact stiffness (plan_v4 r4 sec 2.1): Newton maps ke/kd to a MuJoCo solref of
# timeconst 2/kd, dampratio (kd/2)*sqrt(1/ke). Newton's default ShapeConfig (2500, 100) gave
# 20 ms and let the pads sink 0.9-1.6 mm. kd is set per substep count so 2/kd >= 2*dt.
FINGER_CONTACT_KE = 2.5e5
FINGER_CONTACT_KD = {16: 900.0, 8: 450.0}      # timeconst 2.2 / 4.4 ms, damping ratio 0.9 / 0.45


def finger_contact_kd(substeps, fps=60):
    kd = FINGER_CONTACT_KD[substeps]
    assert 2.0 / kd >= 2.0 / (fps * substeps), "finger contact timeconst below 2*dt"
    return kd
BRACE_TILT = math.radians(45)  # stabilizer leans away from the placer, so the
                               # two hands (63 mm thick) fit 30-40 mm apart
# Clutch welds ~100x stiffer than MuJoCo's default (0.02 s): a 2.5 g brick
# welded at the default yields tens of mm under an arm's push. solimp (0.9999, 0.9999, ...)
# (plan_v4 r5-1) stiffens it further: results/v4/p4_brace_weld/part_b_summary.json, ungrouped:
# column lateral 23-24 N/mm, vertical ~560 N/mm (the r4 (0.95, 0.99, ...) gave 0.7-1.35 N/mm).
WELD_ATTRS = {"mujoco:eq_solref": (0.002, 1.0), "mujoco:eq_solimp": (0.9999, 0.9999, 0.001, 0.5, 2.0)}
# Newton broad phase (geometry/broad_phase_common.py:133-150 test_group_pair): a negative group
# collides with everything except the same group, so the baseplate and every welded brick, all in
# -2, ignore each other and still collide with arms, the held brick and the ground (group 1).
STRUCT_GROUP = -2
TRAVEL = 0.08                 # clearance above the structure while moving
ALIGN_GAIN = 4.0              # 1/s, integral gain of the brick-on-target servo
# BrickSim's default gate. §2.3.3 asks for 1.2 mm; position-controlled IK
# without a force loop does not reliably get there yet.
GATE = {"lateral_mm": 2.0, "dz_mm": 1.5, "yaw_deg": 5.0}


# --- small pose helpers (quaternions are xyzw, as warp stores them) ----------
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def qz(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def qrot(r):
    """Quaternion from a rotation vector (axis * angle)."""
    a = float(np.linalg.norm(r))
    if a < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([*(np.asarray(r) / a * math.sin(a / 2)), math.cos(a / 2)])


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a, period=2 * math.pi):
    """a folded into (-period/2, period/2]."""
    return -((-a + period / 2) % period - period / 2)


Q_DOWN = np.array([1.0, 0.0, 0.0, 0.0])  # gripper pointing down


# --- scene -------------------------------------------------------------------
def build_arm(substeps=16, finger_ke=FINGER_CONTACT_KE, finger_kd=None):
    """One FR3 + hand, with the example's gains, gravity compensation and pads.

    The finger collision shapes (bodies 12/13) get finger_ke/finger_kd as contact
    stiffness (default FINGER_CONTACT_KE and finger_contact_kd(substeps))."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
               floating=False, enable_self_collisions=False, parse_visuals_as_colliders=False)
    b.joint_q[:9] = HOME_Q + [0.01, 0.01]
    b.joint_target_pos[:9] = b.joint_q[:9]
    # Fingers 10x stiffer than the example (100 N/m): its 0.15 N squeeze lifts a
    # brick but lets it rock in the grip the moment the press loads one edge.
    b.joint_target_ke[:9] = [400] * 7 + [FINGER_KE] * 2
    b.joint_target_kd[:9] = [40] * 7 + [2 * math.sqrt(FINGER_KE * 0.1)] * 2
    b.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
    b.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2
    attrs = b.custom_attributes
    attrs["mujoco:jnt_actgravcomp"].values = {d: True for d in range(7)}
    attrs["mujoco:gravcomp"].values = {body: 1.0 for body in range(2, 14)}
    attrs["mujoco:geom_solimp"].values = {s: (0.7, 0.95, 0.0001, 0.5, 2.0)
                                          for s, body in enumerate(b.shape_body) if body in (12, 13)}
    attrs["mujoco:geom_priority"].values = {s: 1 for s, body in enumerate(b.shape_body)
                                            if body in (12, 13)}
    kd = finger_contact_kd(substeps) if finger_kd is None else finger_kd
    for s, body in enumerate(b.shape_body):
        if body in (12, 13):
            b.shape_material_ke[s], b.shape_material_kd[s] = finger_ke, kd
    return b


BRICK_CFG = newton.ModelBuilder.ShapeConfig(
    density=ex.BRICK_DENSITY, ke=ex.BRICK_KE, kd=ex.BRICK_KD, mu=0.7,
    margin=ex.BRICK_MARGIN, gap=ex.SDF_MARGIN)
# Invisible primitive proxies (top slab, walls, studs) carry most brick-brick
# contact; the SDF mesh is what the fingers and the camera see. Same split as
# the example, generalised from 4x2 to any L x W.
PROXY_CFG = newton.ModelBuilder.ShapeConfig(
    density=0.0, ke=ex.BRICK_KE, kd=ex.BRICK_KD, mu=0.7,
    margin=ex.BRICK_MARGIN, gap=ex.SDF_MARGIN, is_visible=False)


def add_proxies(scene, body, L, W, xform):
    """Collision proxies for an L x W brick whose bottom-centre is at xform."""
    inset, pitch, H = ex.COLLIDER_INSET, ex.PITCH, ex.BRICK_HEIGHT
    ox, oy, wt = L * pitch / 2, W * pitch / 2, ex.WALL_THICKNESS
    box_hz = 0.5 * (H - ex.STUD_HEIGHT) - inset
    wall_hz = 0.5 * H - inset
    stud_hh = 0.5 * ex.STUD_HEIGHT - inset
    boxes = [((0, 0, ex.STUD_HEIGHT + inset + box_hz), (ox - inset, oy - inset, box_hz)),
             ((ox - wt / 2, 0, wall_hz + inset), (wt / 2 - inset, oy - inset, wall_hz)),
             ((-(ox - wt / 2), 0, wall_hz + inset), (wt / 2 - inset, oy - inset, wall_hz)),
             ((0, oy - wt / 2, wall_hz + inset), (ox - inset, wt / 2 - inset, wall_hz)),
             ((0, -(oy - wt / 2), wall_hz + inset), (ox - inset, wt / 2 - inset, wall_hz))]
    for pos, (hx, hy, hz) in boxes:
        scene.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=PROXY_CFG,
                            xform=wp.transform_multiply(xform, wp.transform(pos, wp.quat_identity())))
    for i in range(L):
        for j in range(W):
            pos = ((i - (L - 1) / 2) * pitch, (j - (W - 1) / 2) * pitch, H + stud_hh)
            scene.add_shape_cylinder(body, radius=ex.STUD_COLLIDER_RADIUS, half_height=stud_hh,
                                     cfg=PROXY_CFG,
                                     xform=wp.transform_multiply(xform, wp.transform(pos, wp.quat_identity())))


class Arm:
    """A queue of waypoints, eased between with smoothstep.

    A move finishes when its time is up, its wait() is true, and -- unless it
    is a contact move that is meant to stop short -- the hand has arrived.
    `bias` shifts every command, e.g. to aim a held brick rather than the hand.
    `tilt` (a rotation vector) leans the gripper off vertical.
    """

    def __init__(self, name, base, pos):
        self.name, self.base = name, base
        self.bias = np.zeros(3)
        self.cmd = (np.array(pos, float), 0.0, 0.01, np.zeros(3))
        self.moves, self.cur, self.t, self.start = [], None, 0.0, self.cmd
        self.phase = "park"

    def push(self, phase, pos, yaw, grip, dur, wait=None, then=None, contact=False,
             tilt=(0.0, 0.0, 0.0)):
        self.moves.append(dict(phase=phase, pos=np.array(pos, float), yaw=yaw, grip=grip,
                               dur=dur, wait=wait, then=then, contact=contact,
                               tilt=np.array(tilt, float)))

    def idle(self):
        return self.cur is None and not self.moves

    def update(self, dt, ee):
        if self.cur is None and self.moves:
            self.cur, self.t, self.start = self.moves.pop(0), 0.0, self.cmd
            self.phase = self.cur["phase"]
        m = self.cur
        if m is None:
            return (self.cmd[0] + self.bias,) + self.cmd[1:]
        self.t += dt
        x = min(1.0, self.t / m["dur"])
        s = x * x * (3 - 2 * x)
        p0, y0, g0, r0 = self.start
        self.cmd = (p0 + s * (m["pos"] - p0), y0 + s * wrap(m["yaw"] - y0),
                    g0 + s * (m["grip"] - g0), r0 + s * (m["tilt"] - r0))
        arrived = (m["contact"] or np.linalg.norm(ee - m["pos"] - self.bias) < 0.003
                   or self.t > m["dur"] + 2.0)
        if x >= 1.0 and arrived and (m["wait"] is None or m["wait"]()):
            self.cur = None
            if m["then"]:
                m["then"]()
        return (self.cmd[0] + self.bias,) + self.cmd[1:]


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.fps, self.substeps = 60, 16
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0

        self.plan, self.name = make_plan(args)
        bricks = {b["id"]: b for b in self.plan["bricks"]}
        steps = self.plan["sequence"]
        top = max(s["target_pose"][2] for s in steps) + P.BRICK_H
        self.z_travel = top + TRAVEL

        # --- arms ---------------------------------------------------------
        arm = build_arm(self.substeps)
        self.model_ik = arm.finalize()
        # Start parked. HOME_Q reaches 0.41 m out, so two arms 0.9 m apart
        # would begin with their hands interlocked.
        park_q = self.solve_park()
        arm.joint_q[:9] = park_q
        arm.joint_target_pos[:9] = park_q
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        for pos, yaw in (ARM_A, ARM_B):
            scene.add_builder(arm, xform=wp.transform(pos, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)))
        n_arm_bodies = arm.body_count

        # --- baseplate: a studded slab, colliding only where the build stands --
        ox, oy, _ = self.plan["voxel_origin"]
        NI = max(b["grid_pos"][0] + P.footprint(b["type"], b["yaw_index"])[0] for b in bricks.values())
        NJ = max(b["grid_pos"][1] + P.footprint(b["type"], b["yaw_index"])[1] for b in bricks.values())
        cx, cy = ox + NI * P.PITCH / 2, oy + NJ * P.PITCH / 2
        v, f = ex._make_brick_mesh(NI + 4, NJ + 4)
        scene.add_shape_mesh(-1, mesh=newton.Mesh(v, f.flatten()), color=(0.45, 0.47, 0.45),
                             xform=wp.transform((cx, cy, Z0 - ex.BRICK_HEIGHT), wp.quat_identity()),
                             cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
                                                                 has_particle_collision=False))
        self.plate_shapes = [scene.add_shape_box(-1, hx=(NI + 4) * P.PITCH / 2, hy=(NJ + 4) * P.PITCH / 2, hz=Z0 / 2,
                            xform=wp.transform((cx, cy, Z0 / 2), wp.quat_identity()), cfg=PROXY_CFG)]
        stud_hh = 0.5 * ex.STUD_HEIGHT - ex.COLLIDER_INSET
        for i in range(NI):
            for j in range(NJ):
                self.plate_shapes.append(scene.add_shape_cylinder(
                    -1, radius=ex.STUD_COLLIDER_RADIUS, half_height=stud_hh, cfg=PROXY_CFG,
                    xform=wp.transform((ox + (i + 0.5) * P.PITCH, oy + (j + 0.5) * P.PITCH,
                                        Z0 + stud_hh), wp.quat_identity())))

        # --- bricks, waiting in the placer's feeder with their final yaw ------
        solimp = scene.custom_attributes["mujoco:geom_solimp"]
        meshes, self.body, self.target, self.dims = {}, {}, {}, {}
        slots = feeder_slots(len(steps))
        for n, s in enumerate(steps):
            b = bricks[s["brick_id"]]
            nx, ny = P.footprint(b["type"], b["yaw_index"])
            L, W = max(nx, ny), min(nx, ny)
            yaw = math.pi / 2 if ny > nx else 0.0           # mesh x = long side
            if (L, W) not in meshes:
                mv, mf = ex._make_brick_mesh(L, W)
                meshes[(L, W)] = ex._build_mesh_with_sdf(mv, mf, color=COLORS[0])
            xform = wp.transform((*slots[n], 0.0005), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw))
            body = scene.add_body(xform=xform, label=b["id"])
            shape = scene.add_shape_mesh(body, mesh=meshes[(L, W)], cfg=BRICK_CFG,
                                         color=COLORS[n % len(COLORS)])
            solimp.values = solimp.values or {}
            solimp.values[shape] = (0.6, 0.95, 0.00075, 0.5, 2.5)
            add_proxies(scene, body, L, W, wp.transform_identity())
            self.body[b["id"]] = body
            self.target[b["id"]] = (np.array(s["target_pose"][:3]), yaw, L == W)
            self.dims[b["id"]] = (L * P.PITCH, W * P.PITCH)
        self.feeder = {s["brick_id"]: np.array([*slots[n], 0.0]) for n, s in enumerate(steps)}

        # --- clutch pool: one disabled weld per (brick, support) (§3.3) -------
        self.welds = {}
        for s in steps:
            bid = s["brick_id"]
            p2, y2, _ = self.target[bid]
            parents = [m[0] for m in s["mating_studs"]] or [None]   # None = baseplate
            for sid in parents:
                if sid is None:
                    rel = wp.transform(p2, qz(y2))
                else:
                    p1, y1, _ = self.target[sid]
                    rel = wp.transform(rotate(qz(-y1), p2 - p1), qz(y2 - y1))
                eq = scene.add_equality_constraint_weld(
                    body1=-1 if sid is None else self.body[sid], body2=self.body[bid],
                    relpose=rel, enabled=False, label="clutch_%s_%s" % (sid or "plate", bid),
                    custom_attributes=WELD_ATTRS)
                self.welds.setdefault(bid, []).append([eq, sid, rel, False])

        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))
        self.model = scene.finalize()
        self.n_arm_bodies = n_arm_bodies
        sb = self.model.shape_body.numpy()
        self.shapes_of = {bid: np.flatnonzero(sb == b) for bid, b in self.body.items()}
        self.set_group(self.plate_shapes)

        contact_max = 32768
        self.model.rigid_contact_max = contact_max
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=True,
                                                 rigid_contact_max=contact_max, broad_phase="sap",
                                                 # Newton's default 1M silently drops pairs past it (P0 hit ~1.02M)
                                                 max_triangle_pairs=4_000_000)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast", iterations=15,
            ls_iterations=100, nconmax=contact_max, njmax=contact_max * 2,
            cone="elliptic", impratio=50.0, use_mujoco_contacts=False)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- IK: both arms in one batched solve, each in its own base frame ---
        self.ik_q = wp.array(np.tile(park_q, (2, 1)), dtype=wp.float32)
        self.ik_pos = wp.zeros(2, dtype=wp.vec3)
        self.ik_rot = wp.zeros(2, dtype=wp.vec4)
        self.ik = ik.IKSolver(
            model=self.model_ik, n_problems=2, lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            objectives=[ik.IKObjectivePosition(link_index=EE, link_offset=wp.vec3(0, 0, 0),
                                               target_positions=self.ik_pos),
                        ik.IKObjectiveRotation(link_index=EE, link_offset_rotation=wp.quat_identity(),
                                               target_rotations=self.ik_rot),
                        ik.IKObjectiveJointLimit(joint_limit_lower=self.model_ik.joint_limit_lower,
                                                 joint_limit_upper=self.model_ik.joint_limit_upper)])

        # --- the two roles ----------------------------------------------------
        park = lambda base: np.array(base[0]) + rotate(qz(base[1]), PARK)
        self.A = Arm("A stabilizer", ARM_A, park(ARM_A))
        self.B = Arm("B placer", ARM_B, park(ARM_B))
        self.parks = {self.A: park(ARM_A), self.B: park(ARM_B)}
        self.A.push("park", park(ARM_A), 0.0, 0.0, 1.5)
        self.B.push("park", park(ARM_B), 0.0, 0.01, 1.5)
        self.next_step, self.b_step = 0, None
        self.braced, self.placed = set(), set()
        self.episodes, self.last_gate = [], {}
        self.hand_off, self.i_xy = None, np.zeros(2)
        self.done = False
        self.log_path = HERE / "results" / "proto_episodes.jsonl"
        self.log_path.parent.mkdir(exist_ok=True)
        # P0 instrumentation, all off by default (attached from __main__)
        self.monitor = None                   # cell.contacts.ContactMonitor, read-only
        self.inserts = None                   # --record-inserts: per-frame rows
        self.frame, self.place_t, self.done_t = 0, {}, None
        self.snap_gates = []                  # the gate values at every snap (dz_mm = the weld gate's dz)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.35, -0.55, 0.40), pitch=-28.0, yaw=125.0)
        self.capture()
        print("%s: %d bricks, %d braced (%s), %d clutch welds" % (
            self.name, len(steps), self.plan["validation"]["brace_required_count"],
            self.plan["validation"]["bracing_strategy"], sum(len(w) for w in self.welds.values())))

    def solve_park(self):
        """Joint angles that put the hand at PARK, pointing down."""
        q = wp.array(np.array([HOME_Q + [0.01, 0.01]]), dtype=wp.float32)
        solver = ik.IKSolver(
            model=self.model_ik, n_problems=1, lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            objectives=[ik.IKObjectivePosition(link_index=EE, link_offset=wp.vec3(0, 0, 0),
                                               target_positions=wp.array([PARK], dtype=wp.vec3)),
                        ik.IKObjectiveRotation(link_index=EE, link_offset_rotation=wp.quat_identity(),
                                               target_rotations=wp.array([Q_DOWN], dtype=wp.vec4)),
                        ik.IKObjectiveJointLimit(joint_limit_lower=self.model_ik.joint_limit_lower,
                                                 joint_limit_upper=self.model_ik.joint_limit_upper)])
        for _ in range(30):
            solver.step(q, q, iterations=24)
        return q.numpy()[0].tolist()

    # -- motion -----------------------------------------------------------------
    def schedule(self):
        """Hand the next plan step to B; send A to brace when that step needs it."""
        steps = self.plan["sequence"]
        if self.B.idle() and not self.done:
            if self.next_step < len(steps):
                self.b_step = self.next_step
                self.queue_place(steps[self.next_step])
                self.next_step += 1
            elif self.A.idle():
                self.done, self.done_t = True, self.sim_time
                self.B.push("park", self.parks[self.B], 0.0, 0.01, 1.5)
                ok = sum(e["success"] for e in self.episodes)
                print("done: %d / %d placed -> %s" % (ok, len(steps), self.log_path))
        if self.b_step is not None and self.A.idle():
            s = steps[self.b_step]
            if s["requires_brace"] and self.b_step not in self.braced:
                self.queue_brace(self.b_step, s["brace"])

    def queue_place(self, s):
        bid = s["brick_id"]
        tgt = self.target[bid][0]
        slot = self.feeder[bid]
        # the planner picks the closing axis that keeps fingers off neighbours
        yaw = math.radians(s["grasp"]["yaw_offset_deg"])
        width = s["grasp"]["grip_width_m"] + 0.0012
        g_open, g_shut = width / 2 + 0.002, width / 2 - 0.0015
        up = lambda p: np.array([p[0], p[1], self.z_travel])
        n = s["step"]
        B = self.B
        self.place_t[bid] = self.sim_time
        B.push("to feeder", up(slot), yaw, g_open, 1.5)
        B.push("descend", slot + [0, 0, GRASP_DZ], yaw, g_open, 1.0)
        B.push("grasp", slot + [0, 0, GRASP_DZ], yaw, g_shut, 0.5, contact=True)
        B.push("lift", up(slot), yaw, g_shut, 0.8)
        B.push("transport", up(tgt), yaw, g_shut, 1.5)
        B.push("pre-insert", tgt + [0, 0, 0.025 + GRASP_DZ], yaw, g_shut, 0.6,
               wait=(lambda: n in self.braced) if s["requires_brace"] else None)
        B.push("insert", tgt + [0, 0, GRASP_DZ - PRESS], yaw, g_shut, 1.2, contact=True,
               then=lambda: self.snap(s))
        B.push("release", tgt + [0, 0, GRASP_DZ - PRESS], yaw, g_open, 0.4, contact=True,
               then=lambda: self.placed.add(n))
        B.push("retract", up(tgt), yaw, g_open, 0.8)

    def queue_brace(self, n, brace):
        """Press closed fingertips on the brace spot, leaning away from the placer.

        Two vertical Franka hands need ~65 mm between them; a brace spot is
        typically 30-40 mm from the brick being placed. Tilting the stabilizer
        by BRACE_TILT, its housing on the far side, clears the placer's hand
        and the brick it carries down.
        """
        spot = np.array(brace["brace_pose"][:3]) + [0, 0, ex.STUD_HEIGHT]   # stud tops
        v = (spot - self.target[self.plan["sequence"][n]["brick_id"]][0])[:2]
        v = v / (np.linalg.norm(v) or 1.0)                   # away from the placement
        tool = np.array([*(-v * math.sin(BRACE_TILT)), -math.cos(BRACE_TILT)])  # hand -> tips
        tilt = np.array([-v[1], v[0], 0.0]) * BRACE_TILT
        yaw = math.atan2(v[1], v[0])                         # wide side of the hand across v
        press = spot - tool * (TIP_BELOW_TCP - BRACE_PRESS)
        back = press - tool * 0.05
        A = self.A
        A.push("to brace", [back[0], back[1], max(back[2], self.z_travel)], yaw, 0.0, 1.5, tilt=tilt)
        A.push("approach", back, yaw, 0.0, 0.8, tilt=tilt)
        A.push("brace", press, yaw, 0.0, 1.0, contact=True, tilt=tilt,
               then=lambda: self.braced.add(n))
        A.push("hold", press, yaw, 0.0, 0.2, contact=True, tilt=tilt,
               wait=lambda: n in self.placed)
        A.push("retract", back, yaw, 0.0, 0.6, tilt=tilt)
        A.push("park", self.parks[A], 0.0, 0.0, 1.5)

    def aim_brick(self, ee, body_q):
        """Put the held BRICK on target, not the hand (ground-truth poses, §WP4).

        The in-hand offset is measured once after the lift; an integral servo
        on the brick's own xy then removes the tracking error left over, all
        the way down, so its cavity lands over the studs (0.6 mm clearance a
        side). This is the slot the learned insertion policy (WP5) replaces.

        Tried together and reverted -- with them the cube placed 1-2 of 8, not
        8: a yaw servo, velocity feed-forward on the arm joints, integrating
        only small errors, and hovering until "settled" before the insert (the
        arm sways ~2 mm for a second after a fast move, so settling passes on
        a zero-crossing). Not bisected further.
        """
        B = self.B
        if B.phase in ("park", "to feeder", "descend", "grasp", "lift") or self.b_step is None:
            B.bias[:], self.hand_off = 0.0, None
            return
        bid = self.plan["sequence"][self.b_step]["brick_id"]
        brick = body_q[self.body[bid]][:3]
        if self.hand_off is None:
            self.hand_off, self.i_xy = ee - brick - [0, 0, GRASP_DZ], np.zeros(2)
        if B.phase in ("pre-insert", "insert"):
            err = (self.target[bid][0] - brick)[:2]
            self.i_xy = np.clip(self.i_xy + ALIGN_GAIN * self.frame_dt * err, -0.004, 0.004)
        B.bias[:] = self.hand_off + [*self.i_xy, 0.0]

    # -- clutch -----------------------------------------------------------------
    def set_group(self, shapes):
        """Put shapes in STRUCT_GROUP, in place: a captured graph reads the array contents."""
        g = self.model.shape_collision_group.numpy()
        g[shapes] = STRUCT_GROUP
        self.model.shape_collision_group.assign(g)

    def snap(self, s):
        """§2.3.3 gate on the pressed brick; if it passes, engage its clutch welds."""
        bid = s["brick_id"]
        q = self.state_0.body_q.numpy()[self.body[bid]]
        tgt, yaw, square = self.target[bid]
        lateral = float(np.hypot(*(q[:2] - tgt[:2]))) * 1000
        dz = float(q[2] - tgt[2]) * 1000
        period = math.pi / 2 if square else math.pi
        dyaw = math.degrees(abs(wrap(yaw_of(q[3:]) - yaw, period)))
        tilt = math.degrees(math.acos(np.clip(rotate(q[3:], np.array([0, 0, 1.0]))[2], -1.0, 1.0)))
        ok = (lateral < GATE["lateral_mm"] and abs(dz) < GATE["dz_mm"]
              and dyaw < GATE["yaw_deg"] and tilt < GATE["yaw_deg"])
        self.last_gate = {"brick": bid, "lateral_mm": round(lateral, 2), "dz_mm": round(dz, 2),
                          "yaw_deg": round(dyaw, 2), "tilt_deg": round(tilt, 2)}
        if ok:
            en = self.model.equality_constraint_enabled.numpy()
            for w in self.welds[bid]:
                en[w[0]] = True
                w[3] = True
            self.model.equality_constraint_enabled.assign(en)
            self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.CONSTRAINT_PROPERTIES)
            self.set_group(self.shapes_of[bid])
        brace = s["brace"] if s["requires_brace"] else None
        row = {"run_id": "%s_%s" % (self.name, self.plan["validation"]["bracing_strategy"]),
               "structure": self.name, "brick_id": bid, "step": s["step"],
               "policy": "scripted_ik", "backend": "newton_mujoco", "joint_model": "weld_pool",
               "success": ok, "final_error_mm": self.last_gate["lateral_mm"],
               "dz_mm": self.last_gate["dz_mm"], "final_error_deg": self.last_gate["yaw_deg"],
               "tilt_deg": self.last_gate["tilt_deg"], "sim_time_s": round(self.sim_time, 2),
               "brace_active": brace is not None,
               "brace_target": brace["target_brick_id"] if brace else None,
               "failure_mode": None if ok else "gate"}
        self.episodes.append(row)
        self.snap_gates.append(dict(self.last_gate, ok=ok, sim_time_s=round(self.sim_time, 3)))
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("  step %d %s %s  %s" % (s["step"], bid, "SNAP" if ok else "MISS", self.last_gate))

    # -- loop ---------------------------------------------------------------------
    def capture(self):
        self.graph = self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as c:
                self.simulate()
            self.graph = c.graph
            with wp.ScopedCapture() as c:
                self.ik.step(self.ik_q, self.ik_q, iterations=24)
            self.graph_ik = c.graph

    def simulate(self):
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise RuntimeError("simulation diverged at t=%.2f s (step %s, B %s, A %s)"
                               % (self.sim_time, self.b_step, self.B.phase, self.A.phase))
        self.schedule()
        pos, rot, grip = [], [], []
        for arm, k in ((self.A, 0), (self.B, 1)):
            ee = body_q[EE + k * self.n_arm_bodies][:3]
            if arm is self.B:
                self.aim_brick(ee, body_q)
            p, yaw, g, tilt = arm.update(self.frame_dt, ee)
            (bx, by, bz), byaw = arm.base
            pos.append(rotate(qz(-byaw), p - [bx, by, bz]))
            # a parallel gripper is 180-degree symmetric: pick the nearer wrist
            yaw = byaw + wrap(yaw - byaw, math.pi)
            world = qmul(qrot(tilt), qmul(qz(yaw), Q_DOWN))
            rot.append(qmul(qz(-byaw), world))
            grip.append(g)
        self.ik_pos.assign(np.array(pos, dtype=np.float32))
        self.ik_rot.assign(np.array(rot, dtype=np.float32))
        if self.graph_ik:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik.step(self.ik_q, self.ik_q, iterations=24)
        q = self.ik_q.numpy()
        tgt = self.control.joint_target_pos.numpy()
        for k in range(2):
            tgt[9 * k:9 * k + 7] = q[k, :7]
            tgt[9 * k + 7:9 * k + 9] = grip[k]
        self.control.joint_target_pos.assign(tgt)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.monitor:
            self.monitor.step(self)
        if self.inserts is not None and (self.B.phase == "insert" or self.A.phase in ("brace", "hold")):
            self.inserts.append((self.frame, self.state_0.joint_q.numpy()[:18].copy(), self.A.phase,
                                 self.B.phase, -1 if self.b_step is None else self.b_step))
        if (self.done and self.B.idle() and self.A.idle() and getattr(self.args, "viewer", None) == "null"):
            self.viewer.num_frames = self.viewer.frame_count + 1   # null viewer: stop when finished

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # outline where the current brick has to go, and the brace point
        starts, ends = [], []
        if self.b_step is not None and not self.done:
            bid = self.plan["sequence"][self.b_step]["brick_id"]
            (x, y, z), yaw, _ = self.target[bid]
            L, W = self.dims[bid]
            c = [np.array([x, y, z]) + rotate(qz(yaw), np.array([sx * L / 2, sy * W / 2, h]))
                 for h in (0.0, P.BRICK_H) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)]:
                starts.append(c[a])
                ends.append(c[b])
            s = self.plan["sequence"][self.b_step]
            if s["requires_brace"]:
                bx, by, bz = s["brace"]["brace_pose"][:3]
                for d in ((0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)):
                    starts.append(np.array([bx, by, bz]) - d)
                    ends.append(np.array([bx, by, bz]) + d)
        self.viewer.log_lines("/targets", wp.array(np.array(starts or [[0, 0, 0]], np.float32), dtype=wp.vec3),
                              wp.array(np.array(ends or [[0, 0, 0]], np.float32), dtype=wp.vec3),
                              (0.1, 0.9, 0.3), hidden=not starts)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        steps = self.plan["sequence"]
        ui.text("%s  (%d bricks, bracing: %s)" % (self.name, len(steps),
                                                 self.plan["validation"]["bracing_strategy"]))
        if self.b_step is not None:
            s = steps[self.b_step]
            btype = next(b["type"] for b in self.plan["bricks"] if b["id"] == s["brick_id"])
            ui.text("step %d/%d  %s %s%s" % (self.b_step + 1, len(steps), s["brick_id"], btype,
                                           "  [brace]" if s["requires_brace"] else ""))
        ui.text("B placer:     %s" % self.B.phase)
        ui.text("A stabilizer: %s" % self.A.phase)
        ok = sum(e["success"] for e in self.episodes)
        ui.text("snapped %d / %d" % (ok, len(self.episodes)))
        if self.last_gate:
            ui.text("last gate: %.2f mm, dz %.2f mm, yaw %.2f deg" % (
                self.last_gate["lateral_mm"], self.last_gate["dz_mm"], self.last_gate["yaw_deg"]))
        if self.done:
            ui.text("DONE")

    def test_final(self):
        n = len(self.plan["sequence"])
        ok = sum(e["success"] for e in self.episodes)
        assert self.done, "only reached step %s of %d" % (self.b_step, n)
        assert ok == n, "%d / %d snapped: %s" % (ok, n, [e for e in self.episodes if not e["success"]])
        body_q = self.state_0.body_q.numpy()
        for bid, (tgt, _, _) in self.target.items():   # and they stayed there
            err = np.linalg.norm(body_q[self.body[bid]][:3] - tgt) * 1000
            assert err < 3.0, "%s drifted %.1f mm after assembly" % (bid, err)

    def p0_row(self, error=None):
        """One P0 baseline row (plan_v4 P0): completion, cycle, contact events, dz at snap."""
        a, steps = self.args, self.plan["sequence"]
        model = getattr(a, "brace_model", "grasp_lp")
        starts = [self.place_t[s["brick_id"]] for s in steps if s["brick_id"] in self.place_t]
        ends = starts[1:] + [self.done_t if self.done_t is not None else self.sim_time]
        mon = self.monitor.summary() if self.monitor else {}
        events = self.monitor.events if self.monitor else []
        return {"tag": getattr(a, "tag", None) or "%s_%s_r%d" % (self.name, model, a.repeat),
                "structure": self.name, "brace_model": model, "strategy": a.strategy,
                "repeat": a.repeat, "error": error, "done": self.done,
                "placed": sum(e["success"] for e in self.episodes), "total": len(steps),
                "sim_time_s": round(self.sim_time, 2), "frames": self.frame,
                "cycle_s": [round(e - s, 2) for s, e in zip(starts, ends)],
                "event_counts": {c: v["count"] for c, v in mon.items()},
                "events": events,
                "displaced": sorted({e["brick"] for e in events if e["class"] == "displaced"}),
                "snaps": self.snap_gates,   # dz_mm: at the moment |dz| < 1.5 mm welds the brick
                "brace_required": self.plan["validation"]["brace_required_count"]}

    def finish(self, error=None):
        """After the run (also after a failed one): write --p0-out and --record-inserts."""
        if self.monitor:
            self.monitor.finalize()
        if getattr(self.args, "p0_out", None):
            out = Path(self.args.p0_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as f:
                f.write(json.dumps(self.p0_row(error)) + "\n")
        if self.inserts is not None:
            r = self.inserts
            Path(self.args.record_inserts).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.args.record_inserts,
                     frame=np.array([x[0] for x in r], int),
                     q=np.array([x[1] for x in r], np.float32).reshape(len(r), 18),
                     a_phase=np.array([x[2] for x in r], str), b_phase=np.array([x[3] for x in r], str),
                     b_step=np.array([x[4] for x in r], int))


def feeder_slots(n):
    """Pick-up spots on the placer's -y side, 70 mm apart, 0.40-0.65 m from its
    base, nearest first. The first 14 are the measured layout; the ring added
    after them serves bigger builds. (Slots on the +y side were tried: picks
    from there missed on the cube.)"""
    base = np.array(ARM_B[0][:2])
    near = lambda g: 0.40 <= np.linalg.norm(np.array(g) - base) <= 0.65
    dist = lambda g: np.linalg.norm(np.array(g) - base)
    grid = sorted((g for g in ((x, -y) for x in (0.10, 0.17, 0.24, 0.31)
                               for y in (0.22, 0.29, 0.36, 0.43, 0.50)) if near(g)), key=dist)
    grid += sorted((g for g in ((x, -y) for x in (0.03, 0.10, 0.17, 0.24, 0.31, 0.38)
                                for y in (0.22, 0.29, 0.36, 0.43, 0.50, 0.57))
                    if near(g) and g not in grid), key=dist)
    if n > len(grid):
        raise SystemExit("%d bricks but only %d feeder slots" % (n, len(grid)))
    return grid[:n]


def make_plan(args):
    """images/sample/authored structure -> assembly_plan dict in the sim's world frame."""
    if args.front:
        name = args.name or args.front.stem
        bricks = B.tile(B.carve(args.front, args.side, args.top, args.width, args.depth))
    elif args.shape:
        name, bricks = args.shape, B.load(args.shape)
    else:
        name, bricks = args.structure, P.STRUCTURES[args.structure]
    P.STRUCTURES[name] = bricks
    NI = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    NJ = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    # centre the build between the arms
    P.VOXEL_ORIGIN = (-NI * P.PITCH / 2, -NJ * P.PITCH / 2, Z0)
    model = getattr(args, "brace_model", "grasp_lp")
    plan = P.build_plan(name, args.strategy, brace_model=model)
    plan["frame"] = "dual_arm_sim world: arm A base (-0.45,0,0), arm B base (+0.45,0,0), z up"
    # a lever_press plan gets its own file: the grasp_lp one keeps the old name
    out = HERE / "plans" / ("sim_%s_%s%s.json" % (name, args.strategy,
                                                  "" if model == "grasp_lp" else "_" + model))
    out.write_text(json.dumps(plan, indent=1))
    print(B.ascii(bricks))
    print("plan -> %s" % out.relative_to(HERE))
    return plan, name


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--shape", choices=list(B.SAMPLES), help="a sample blueprint")
    src.add_argument("--structure", choices=list(P.STRUCTURES), help="a hand-authored structure")
    src.add_argument("--front", type=Path, help="front silhouette image")
    parser.add_argument("--side", type=Path)
    parser.add_argument("--top", type=Path)
    parser.add_argument("--width", type=int, help="studs across the front image")
    parser.add_argument("--depth", type=int, help="studs deep, if no side/top image")
    parser.add_argument("--name", help="structure name for an image blueprint")
    parser.add_argument("--strategy", default="weakest_joint",
                        choices=["none", "nearest", "weakest_joint"])
    parser.add_argument("--brace-model", default="grasp_lp", choices=["grasp_lp", "lever_press"],
                        help="planner.build_plan brace model")
    parser.add_argument("--monitor", action="store_true", help="attach the sec 2.4 contact monitor")
    parser.add_argument("--p0-out", type=Path, help="append one P0 baseline row (JSONL); implies --monitor")
    parser.add_argument("--repeat", type=int, default=0, help="repeat index, recorded in the P0 row")
    parser.add_argument("--tag", help="run tag for the P0 row (default shape_model_rK)")
    parser.add_argument("--record-inserts", type=Path, metavar="PATH.npz",
                        help="save both arms' joint q every frame B inserts or A braces")
    parser.set_defaults(shape="arch")
    viewer, args = newton.examples.init(parser)
    if args.front or args.structure:
        args.shape = None
    example = Example(viewer, args)
    if args.monitor or args.p0_out:
        from cell.contacts import ContactMonitor
        example.monitor = ContactMonitor(example)
    if args.record_inserts:
        example.inserts = []
    err = None
    try:
        newton.examples.run(example, args)
    except BaseException as e:      # a failed run still leaves its row
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
        raise
    finally:
        example.finish(err)
