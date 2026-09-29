"""P1 (plan_v4 §3): Newton contact physics, arm-held press and capture vs scale.

Grown from the work-reviewer's press.py (`Docs/plan_v4.md` §0 fact 6): same brick model (SDF mesh + invisible
proxies, dual_arm_sim's BRICK_CFG/PROXY_CFG, geometry x s, ke/kd/margin unchanged), same solver settings
(SolverMuJoCo newton/implicitfast/15/100/elliptic/impratio 50, Newton CollisionPipeline reduce_contacts + sap, 60 fps).
Every trial of a subcase is one Newton world of a replicated model, so a level's 20 trials run in parallel.

    bash scripts/run.sh scripts/10_newton_scale_probe.py --sub f                      # FIRST, idle GPU: fixes n_sub
    bash scripts/run.sh scripts/10_newton_scale_probe.py --sub d --scale 1,2 --nsub 16
    bash scripts/run.sh scripts/10_newton_scale_probe.py --sub all --scale 1,2        # a d b c e h (g, f any time)
    bash scripts/run.sh scripts/10_newton_scale_probe.py --summary [--chain-p95 1=0.4,2=0.7]   # Gate P1 + branch
    bash scripts/run.sh scripts/10_newton_scale_probe.py --selfcheck                  # arithmetic self-check

Subcases (plan_v4 r4, section 3 P1): a contact generation at 50 um stud-wall overlap | b capture-offset sweep at
F_press = 2 F_seat | c rest stability (10 (d) trials at F_seat, 5 s after release) | d force sweep (F_seat, F_pt, sink,
k_os) | e slip vs grip multiple m | f real-time factor (16/8/4 substeps x collide frame/sub, weld pool 210/465) |
g free-brick step body force (the ONLY body force on a brick; reproduces §0 fact 6) | h = (d)+(b) with the tube radius
cut to 0.6s mm.

Press protocol (arm-held; B = FR3 + hand from dual_arm_sim.build_arm, IK on interpolated TCP targets, prototype gains):
the brick starts gripped (per-finger N = m F/mu at the 1.5 mm squeeze via the finger drive ke, capped at 100 N) 3s mm above
the stud tops, the TCP descends at v_press; "force cap" = guarded press (§2.3): the target stops when the simulated wrist
force reaches F_cap (or FK z <= seated - 0.2s), holds 0.3 s, then the fingers open (release). Finger CONTACTS use ke_f /
kd_f(n_sub) (r4, §2.1), keeping priority 1 and solimp; brick contacts are the example's. Collision once per frame by
default (--collide frame); --collide sub repeats (d)@0 offset/20 mm/s and (e) as the cross-check.
Clutch scoring is M1's (§2.2, r4): a 1 s window opens after release when no B finger touches (contact d < 0); the first frame
in it where the brick has been at rest relative to the lower brick (speed < 1s mm/s, 6 consecutive frames) and the gate
passes (|dxy| < 1.2s, dz in [-0.5s, +0.3s] mm, tilt < 4, yaw < 5) fires. Gate passes at any other frame are `gate_transit`
and never score. The weld is scored, NOT engaged -- ponytail: a weld would freeze the brick that (c) watches.
Wrist-force sensor model: sum of the vertical contact force on the two finger shapes from the brick, sampled every substep
(Newton contact forces via SolverMuJoCo.update_contacts into a second buffer, plan §0 fact 5).
"""
import argparse
import copy
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as das  # noqa: E402  (imported for build_arm, BRICK_CFG, PROXY_CFG, poses; not edited)

ex = das.ex
FPS = 60
MU = 0.7                     # brick and finger friction (BRICK_CFG.mu)
SQUEEZE = 0.0015             # §2.3: per-finger squeeze at which the finger drive ke sets the grip
GRIP_CAP_N, HAND_N = 100.0, 70.0
SWEEP = (1, 2, 5, 10, 20, 50, 100)
TRI_PER_WORLD, TRI_F = 400000, 4_000_000       # CollisionPipeline max_triangle_pairs: per world (press batches); (f) scene (= dual_arm_sim's)
T_SETTLE, HOLD_PRESS, HOLD_LOAD, T_CARRY, D_CARRY = 0.25, 0.3, 1.0, 0.6, 0.2   # D/T -> 0.5 m/s peak
WINDOW, POST_C, REL_TIMEOUT = 1.0, 5.0, 1.0   # B3 weld window; (c) watches 5 s after release; s to wait for the pads to clear
REST_FRAMES, REST_SPEED = 6, 1.0              # B3 "at rest": speed < 1s mm/s for >= 6 consecutive frames
MESH_SOLIMP = (0.6, 0.95, 0.00075, 0.5, 2.5)          # dual_arm_sim's, per mesh shape
STUD_TOP = ex.STUD_HEIGHT
GATE_TILT, GATE_YAW = 4.0, 5.0
OUT_DEFAULT = HERE / "results" / "v4" / "p1" / "p1.jsonl"
# r4 finger contacts (§2.1): dual_arm_sim's constants when present, else the same values locally
FINGER_CONTACT_KE = getattr(das, "FINGER_CONTACT_KE", 2.5e5)
FINGER_KD = {16: 900.0, 8: 450.0, 4: 240.0}           # kd_f per n_sub; 4 only for (f) (kd <= 240, tc 8.3 ms)


def finger_kd(n_sub):
    if n_sub in getattr(das, "FINGER_CONTACT_KD", ()):
        return float(das.finger_contact_kd(n_sub))
    return FINGER_KD[n_sub]


def set_finger_contact(arm, n_sub):
    """Finger shapes (bodies 12/13) of a build_arm() builder at ke_f / kd_f(n_sub); priority 1 and solimp untouched."""
    for i, bd in enumerate(arm.shape_body):
        if bd in (12, 13):
            arm.shape_material_ke[i], arm.shape_material_kd[i] = FINGER_CONTACT_KE, finger_kd(n_sub)


# --- pure arithmetic (also what --selfcheck exercises) ---------------------------
def gate(dxy_mm, dz_mm, tilt_deg, yaw_deg, s):
    """M1 clutch gate (r4): |dxy| < 1.2s mm, dz in [-0.5s, +0.3s] mm, tilt < 4 deg, yaw < 5 deg."""
    return ((dxy_mm < 1.2 * s) & (dz_mm >= -0.5 * s) & (dz_mm <= 0.3 * s)
            & (tilt_deg < GATE_TILT) & (yaw_deg < GATE_YAW))


class Clutch:
    """B3 weld window over nw worlds (scored, never enabled). Feed update() once per frame.
    The window opens on the first frame after release with no B finger touching (contact d < 0) and lasts WINDOW s. The first
    frame in it where the brick has been at rest relative to the lower brick (speed < REST_SPEED*s mm/s for REST_FRAMES
    consecutive frames, counted from the start of the trial) and the gate passes fires. A gate pass on any other frame
    before the window closes (in transit, under the press, moving, before it opens) only counts in `transit`."""

    def __init__(self, nw):
        self.opened = np.zeros(nw, bool)
        self.t_open = np.full(nw, np.nan)
        self.run = np.zeros(nw, int)
        self.fired = np.zeros(nw, bool)
        self.transit = np.zeros(nw, int)

    def update(self, tm, act, released, touch, speed_mm_s, ok, s):
        """tm frame time; act/released/touch/speed_mm_s/ok are per-world arrays; returns the mask that fired now."""
        self.run = np.where(act & (speed_mm_s < REST_SPEED * s), self.run + 1, 0)
        new_open = act & released & ~self.opened & ~touch
        self.opened |= new_open
        self.t_open[new_open] = tm
        in_win = self.opened & (tm <= self.t_open + WINDOW + 1e-9)
        elig = in_win & (self.run >= REST_FRAMES)
        fire = act & ok & elig & ~self.fired
        self.transit += (act & ok & ~elig & ~self.fired & ~(self.opened & ~in_win)).astype(int)
        self.fired |= fire
        return fire


def rel_speed(rel, prev_rel):
    """Speed (mm/s) of the upper brick relative to its partner from two consecutive frames' relative positions (m). The
    instantaneous body_qd is not used: substep chatter aliases into a steady false velocity at 60 Hz (proxy lowers)."""
    return np.linalg.norm(rel - prev_rel, axis=1) * FPS * 1e3


def rest_fit(t, z):
    """Detrended peak-to-peak (mm), raw peak-to-peak (mm) and linear-fit slope (um/s) of a dz series (mm) over times t (s)."""
    p = np.polyfit(t, z, 1)
    return float(np.ptp(z - np.polyval(p, t))), float(np.ptp(z)), float(p[0] * 1e3)


def thr(n):
    """Pass count for 'at least 19/20' at n trials."""
    return math.ceil(0.95 * n - 1e-9)


def lvl_up(x, levels=SWEEP):
    """Smallest sweep level >= x (None if x is past the last level)."""
    return next((lv for lv in sorted(levels) if lv >= x - 1e-9), None)


def p95(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.percentile(v, 95)) if v else None


def c_held(levels, fired, n):
    """(contiguous, largest): the capture range is the largest offset up to which EVERY level fires in >= thr(n)
    trials (conservative); 'largest' is the plan's literal 'largest offset at which >= 19/20'. -1 = even the first level
    (offset 0) fails; 0.0 = only offset 0 passes."""
    ok = [f >= thr(n) for f in fired]
    contig = 0.0 if ok and ok[0] else -1.0
    for lv, k in zip(levels, ok):
        if not k:
            break
        contig = lv
    largest = max([lv for lv, k in zip(levels, ok) if k], default=-1.0)
    return contig, largest


# --- quaternion helpers, vectorised (xyzw) ------------------------------------------
def qmulv(a, b):
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qconjv(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrotv(q, v):
    u, w = q[:, :3], q[:, 3:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def rel_pose(pu, qu, pl, ql):
    """Upper relative to lower, in the lower's frame: position (m), tilt (deg), yaw (rad, unwrapped)."""
    qr = qmulv(qconjv(ql), qu)
    rel = qrotv(qconjv(ql), pu - pl)
    zax = qrotv(qr, np.broadcast_to([0.0, 0.0, 1.0], rel.shape))[:, 2]
    tilt = np.degrees(np.arccos(np.clip(zax, -1, 1)))
    x, y, z, w = qr.T
    return rel, tilt, np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gate_of(rel, tilt, yaw, g):
    """Gate inputs from a relative pose, vs the nearest lattice pose (offset rounded to the pitch, yaw to the brick's symmetry)."""
    pitch = g.pitch
    lat = np.round(rel[:, :2] / pitch) * pitch
    dxy = np.linalg.norm(rel[:, :2] - lat, axis=1) * 1e3
    dz = (rel[:, 2] - g.H) * 1e3
    period = math.pi / 2 if g.L == g.W else math.pi
    dyaw = np.degrees(np.abs(das.wrap(yaw, period)))
    return dxy, dz, tilt, dyaw


# --- brick model (press.py's, generalised to L x W, scaled) --------------------------
class Geo:
    """One brick type at one scale; the SDF mesh is built once. clear: tube radius cut to a 0.6s mm designed clearance (h)."""
    _cache = {}

    def __init__(self, brick, s, clear=False):
        self.brick, self.s, self.clear = brick, s, clear
        w_, l_ = sorted(int(x) for x in brick.split("x"))
        self.L, self.W = l_, w_                     # "2x4" -> 4 long, 2 wide
        self.H, self.pitch = ex.BRICK_HEIGHT * s, ex.PITCH * s
        old = ex.TUBE_OUTER_RADIUS
        if clear:   # tube centre to stud centre is pitch/sqrt2; minus stud radius minus 0.6 mm (all x s by the mesh scale)
            ex.TUBE_OUTER_RADIUS = ex.PITCH / math.sqrt(2) - ex.STUD_RADIUS - 0.0006
        try:
            v, f = ex._make_brick_mesh(self.L, self.W)
        finally:
            ex.TUBE_OUTER_RADIUS = old
        self.tube_r = old if not clear else ex.PITCH / math.sqrt(2) - ex.STUD_RADIUS - 0.0006
        self.mesh = ex._build_mesh_with_sdf(v, f, color=(0.8, 0.2, 0.2), scale=s)

    @classmethod
    def get(cls, brick, s, clear=False):
        k = (brick, s, clear)
        if k not in cls._cache:
            cls._cache[k] = cls(brick, s, clear)
        return cls._cache[k]

    @property
    def grasp_dz(self):
        """TCP height above the brick bottom: fingertips end 2.4s mm above the lower course's stud tops (§2.3;
        das.GRASP_DZ = 13 mm at s=1). A 13 mm rule at s=2 would leave 2 mm of the wall between the pads."""
        return 4.1e-3 * self.s + das.TIP_BELOW_TCP


def add_brick(b, body, xform, g, mesh_collide=True, proxy_collide=True):
    """Prototype brick (dual_arm_sim: add_shape_mesh + add_proxies), geometry x s. Returns the shape ids by role."""
    cfg_m, cfg_p = das.BRICK_CFG, das.PROXY_CFG
    if not mesh_collide:
        cfg_m = copy.copy(cfg_m)
        cfg_m.has_shape_collision = False
    if not proxy_collide:
        cfg_p = copy.copy(cfg_p)
        cfg_p.has_shape_collision = False
    sol = b.custom_attributes["mujoco:geom_solimp"]
    sol.values = sol.values or {}
    n0 = b.shape_count
    sh = b.add_shape_mesh(body, mesh=g.mesh, cfg=cfg_m, xform=xform)
    sol.values[sh] = MESH_SOLIMP
    s, H = g.s, g.H
    inset, pitch, wt = ex.COLLIDER_INSET * s, g.pitch, ex.WALL_THICKNESS * s
    ox, oy, sh_ = g.L * pitch / 2, g.W * pitch / 2, STUD_TOP * s
    box_hz, wall_hz, stud_hh = 0.5 * (H - sh_) - inset, 0.5 * H - inset, 0.5 * sh_ - inset
    boxes = [((0, 0, sh_ + inset + box_hz), (ox - inset, oy - inset, box_hz)),
             ((ox - wt / 2, 0, wall_hz + inset), (wt / 2 - inset, oy - inset, wall_hz)),
             ((-(ox - wt / 2), 0, wall_hz + inset), (wt / 2 - inset, oy - inset, wall_hz)),
             ((0, oy - wt / 2, wall_hz + inset), (ox - inset, wt / 2 - inset, wall_hz)),
             ((0, -(oy - wt / 2), wall_hz + inset), (ox - inset, wt / 2 - inset, wall_hz))]
    ids = dict(mesh=sh, slab=None, walls=[], studs=[])
    for k, (pos, (hx, hy, hz)) in enumerate(boxes):
        i = b.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg_p,
                            xform=wp.transform_multiply(xform, wp.transform(pos, wp.quat_identity())))
        if k == 0:
            ids["slab"] = i
        else:
            ids["walls"].append(i)
    for i in range(g.L):
        for j in range(g.W):
            pos = ((i - (g.L - 1) / 2) * pitch, (j - (g.W - 1) / 2) * pitch, H + stud_hh)
            ids["studs"].append(b.add_shape_cylinder(
                body, radius=ex.STUD_COLLIDER_RADIUS * s, half_height=stud_hh, cfg=cfg_p,
                xform=wp.transform_multiply(xform, wp.transform(pos, wp.quat_identity()))))
    ids["all"] = list(range(n0, b.shape_count))
    return ids


def add_plate(b, g, zb, ni=8, nj=6, cx=0.0, cy=0.0, margin=0):
    """The prototype baseplate: a static proxy slab (ni x nj studs wide, top at zb, centre cx, cy) with proxy studs (all but
    `margin` cells at each edge); press.py's `plate` lower. Defaults are the 8 x 6 slab of the press rigs."""
    s, pitch = g.s, g.pitch
    n0 = b.shape_count
    hz = max(zb, 0.01) / 2
    b.add_shape_box(-1, hx=ni * pitch / 2, hy=nj * pitch / 2, hz=hz, cfg=das.PROXY_CFG,
                    xform=wp.transform((cx, cy, zb - hz), wp.quat_identity()))
    stud_hh = 0.5 * STUD_TOP * s - ex.COLLIDER_INSET * s
    for i in range(margin, ni - margin):
        for j in range(margin, nj - margin):
            b.add_shape_cylinder(-1, radius=ex.STUD_COLLIDER_RADIUS * s, half_height=stud_hh, cfg=das.PROXY_CFG,
                                 xform=wp.transform((cx + (i - ni / 2 + 0.5) * pitch, cy + (j - nj / 2 + 0.5) * pitch, zb + stud_hh),
                                                    wp.quat_identity()))
    return list(range(n0, b.shape_count))


# --- warp kernels: wrist-force / penetration sensing ---------------------------------
@wp.kernel
def k_sense(n: wp.array(dtype=wp.int32), sh0: wp.array(dtype=wp.int32), sh1: wp.array(dtype=wp.int32),
            frc: wp.array(dtype=wp.spatial_vector), dist: wp.array(dtype=float), wid: wp.array(dtype=wp.int32),
            cls: wp.array(dtype=wp.int32), fing: wp.array(dtype=float), react: wp.array(dtype=float),
            bbmin: wp.array(dtype=float), fpen: wp.array(dtype=float)):
    # classes: 1 held (upper) brick, 2 lower brick, 3 finger, 4 plate. Newton stores the force ON shape0.
    i = wp.tid()
    if i >= n[0]:
        return
    a = cls[sh0[i]]
    b = cls[sh1[i]]
    fz = wp.spatial_top(frc[i])[2]
    other = int(0)
    if a == 1:
        other = b
    elif b == 1:
        other = a
        fz = -fz
    else:
        return
    # fz = force on the upper brick, z; other = the other shape's class
    w = wid[i]
    d = dist[i]
    if other == 2 or other == 4:
        wp.atomic_add(react, w, fz)
    if other == 2:
        wp.atomic_min(bbmin, w, d)
    if other == 3:
        wp.atomic_add(fing, w, -fz)      # force on the fingers from the brick: presses the wrist down
        wp.atomic_min(fpen, w, d)


@wp.kernel
def k_peak(fing: wp.array(dtype=float), react: wp.array(dtype=float), fpk: wp.array(dtype=float),
           rpk: wp.array(dtype=float)):
    w = wp.tid()
    fpk[w] = wp.max(fpk[w], fing[w])
    rpk[w] = wp.max(rpk[w], react[w])


class Sim:
    """Replicated-world model + the prototype's solver settings; frame() = `substeps` steps, graph-captured."""

    @staticmethod
    def finalize(scene, sense=False):
        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))
        m = scene.finalize()
        if sense:
            m.request_contact_attributes("force")     # before any pipe.contacts()
        return m

    def __init__(self, scene, nw, substeps, collide, cmax=12000, sense_cls=None, body_force=False, model=None, tri=None):
        assert substeps % 2 == 0, "graph capture needs an even substep count (ping-pong states)"
        self.model = m = model or Sim.finalize(scene, sense_cls is not None)
        self.nw, self.substeps, self.collide, self.body_force = nw, substeps, collide, body_force
        m.rigid_contact_max = cmax * nw
        self.tri_peak = self.tri_cap = self.con_peak = self.con_cap = self.nacon_peak = self.nacon_cap = 0
        self.pipe = newton.CollisionPipeline(m, reduce_contacts=True, rigid_contact_max=cmax * nw, broad_phase="sap",
                                             max_triangle_pairs=tri or TRI_PER_WORLD * nw)   # Newton drops pairs past it, silently
        self.solver = newton.solvers.SolverMuJoCo(
            m, solver="newton", integrator="implicitfast", iterations=15, ls_iterations=100, nconmax=cmax,
            njmax=cmax * 2, cone="elliptic", impratio=50.0, use_mujoco_contacts=False)
        self.contacts = self.pipe.contacts()
        self.dt = 1.0 / FPS / substeps
        self.control = m.control()
        self.sense = sense_cls is not None
        if self.sense:
            self.cbuf = self.pipe.contacts()
            self.cls = wp.array(sense_cls, dtype=wp.int32)
            z = lambda v: wp.array(np.full(nw, v, np.float32), dtype=float)   # noqa: E731
            self.fing, self.react, self.fpk, self.rpk, self.bbmin, self.fpen = z(0), z(0), z(0), z(0), z(1), z(1)
        self.s0 = self.s1 = self.graph = None

    def reset(self):
        m = self.model
        self.s0, self.s1 = m.state(), m.state()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.s0)

    def check_tri(self):
        """Buffer occupancy after the last collide()/step against capacity: triangle pairs, rigid contacts and MuJoCo's
        contact array. Newton/MuJoCo drop the excess and only print a warning; buf_fields() records the peaks per row.
        ponytail: with --collide sub only the last substep's counts are seen; the printed warning is the backstop."""
        npz = self.pipe.narrow_phase
        if getattr(npz, "triangle_pairs_count", None) is not None:
            self.tri_peak = max(self.tri_peak, int(npz.triangle_pairs_count.numpy()[0]))
            self.tri_cap = int(npz.triangle_pairs.shape[0])
        self.con_peak = max(self.con_peak, int(self.contacts.rigid_contact_count.numpy()[0]))
        self.con_cap = int(self.contacts.rigid_contact_max)
        mj = self.solver.mjw_data
        self.nacon_peak = max(self.nacon_peak, int(mj.nacon.numpy()[0]))
        self.nacon_cap = int(mj.naconmax)

    def buf_fields(self):
        return dict(tri_peak=self.tri_peak, tri_cap=self.tri_cap, tri_overflow=self.tri_peak > self.tri_cap > 0,
                    con_peak=self.con_peak, con_cap=self.con_cap, nacon_peak=self.nacon_peak, nacon_cap=self.nacon_cap,
                    con_overflow=self.con_peak > self.con_cap > 0 or self.nacon_peak > self.nacon_cap > 0)

    def _sense(self):
        self.fing.zero_()
        self.react.zero_()
        mj = self.solver.mjw_data
        self.solver.update_contacts(self.cbuf)
        wp.launch(k_sense, dim=mj.naconmax, inputs=[
            mj.nacon, self.cbuf.rigid_contact_shape0, self.cbuf.rigid_contact_shape1, self.cbuf.force,
            mj.contact.dist, mj.contact.worldid, self.cls, self.fing, self.react, self.bbmin, self.fpen])
        wp.launch(k_peak, dim=self.nw, inputs=[self.fing, self.react, self.fpk, self.rpk])

    def frame(self):
        if self.collide == "frame":
            self.pipe.collide(self.s0, self.contacts)
        for _ in range(self.substeps):
            if self.collide == "sub":
                self.pipe.collide(self.s0, self.contacts)
            if self.body_force:
                wp.copy(self.s0.body_f, self.fbuf)          # press.py: constant body force on the upper brick
            else:
                self.s0.clear_forces()
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            if self.sense:
                self._sense()
            self.s0, self.s1 = self.s1, self.s0

    def start(self):
        """Warm up (compile), reset the state, capture the frame graph."""
        self.reset()
        self.frame()
        self.reset()
        with wp.ScopedCapture() as cap:
            self.frame()
        self.graph = cap.graph
        if self.sense:
            for arr, v in ((self.fpk, 0), (self.rpk, 0), (self.bbmin, 1), (self.fpen, 1)):
                arr.assign(np.full(self.nw, v, np.float32))

    def step(self):
        wp.capture_launch(self.graph)


# --- arm-held press ------------------------------------------------------------------
def arm_builder(ke, cache, n_sub):
    """das.build_arm() with the finger drive set for the grip (N = ke * 1.5 mm) and the finger contacts at ke_f/kd_f(n_sub)."""
    if (ke, n_sub) not in cache:
        arm = das.build_arm()
        arm.joint_target_ke[7:9] = [ke, ke]
        arm.joint_target_kd[7:9] = [2 * math.sqrt(ke * 0.1)] * 2
        set_finger_contact(arm, n_sub)
        cache[(ke, n_sub)] = arm
    return cache[(ke, n_sub)]


def build_arm_world(g, lower, case, zb, cache, n_sub):
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    pos, yaw = das.ARM_B
    b.add_builder(arm_builder(case["grip_N"] / SQUEEZE, cache, n_sub),
                  xform=wp.transform(pos, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)))
    cls = {i: 3 for i, bd in enumerate(b.shape_body) if bd in (12, 13)}     # the two finger bodies
    q0 = wp.transform_identity()
    lb = None
    if lower == "welded":                     # M1's state: the lower brick rigid to the world
        for i in add_brick(b, -1, wp.transform((0, 0, zb), wp.quat_identity()), g)["all"]:
            cls[i] = 2
    else:                                     # first course on the plate's proxy studs
        for i in add_plate(b, g, zb):
            cls[i] = 4
        lb = b.add_body(xform=wp.transform((0, 0, zb), wp.quat_identity()), label="lower")
        for i in add_brick(b, lb, q0, g)["all"]:
            cls[i] = 2
    z_over = STUD_TOP * g.s + case["above_mm"] * 1e-3
    ub = b.add_body(xform=wp.transform((0, 0, zb + g.H + z_over), wp.quat_identity()), label="upper")
    for i in add_brick(b, ub, q0, g)["all"]:
        cls[i] = 1
    return b, dict(ub=ub, lb=lb, cls=cls, spw=b.shape_count, bpw=b.body_count)


def ik_rig(nw, model_ik, pos_base, rot):
    ik_pos = wp.array(pos_base.astype(np.float32), dtype=wp.vec3)
    ik_rot = wp.array(rot.astype(np.float32), dtype=wp.vec4)
    ee = das.EE
    solver = ik.IKSolver(
        model=model_ik, n_problems=nw, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
        objectives=[ik.IKObjectivePosition(link_index=ee, link_offset=wp.vec3(0, 0, 0), target_positions=ik_pos),
                    ik.IKObjectiveRotation(link_index=ee, link_offset_rotation=wp.quat_identity(), target_rotations=ik_rot),
                    ik.IKObjectiveJointLimit(joint_limit_lower=model_ik.joint_limit_lower,
                                             joint_limit_upper=model_ik.joint_limit_upper)])
    q = wp.array(np.tile(das.HOME_Q + [0.01, 0.01], (nw, 1)).astype(np.float32), dtype=wp.float32)
    return solver, ik_pos, ik_rot, q


def hand_targets(p_world, yaw_world):
    """das.step()'s conversion: world TCP position + yaw -> arm-B base frame position and orientation."""
    (bx, by, bz), byaw = das.ARM_B
    p = das.rotate(das.qz(-byaw), np.asarray(p_world) - [bx, by, bz])
    yaw = byaw + das.wrap(yaw_world - byaw, math.pi)      # a parallel gripper is 180-degree symmetric
    return p, das.qmul(das.qz(-byaw), das.qmul(das.qz(yaw), das.Q_DOWN))


def run_arm_batch(g, lower, cases, a, vpress=None):
    """One replicated model, one arm-held trial per world. Returns one result dict per case."""
    nw, s, zb = len(cases), g.s, a.zbase
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    cache, cls_pairs, info = {}, [], None
    for c in cases:
        b, info = build_arm_world(g, lower, c, zb, cache, a.substeps)
        off = scene.shape_count
        cls_pairs += [(off + k, v) for k, v in info["cls"].items()]
        scene.add_world(b)
    bpw, spw = info["bpw"], info["spw"]
    model_ik = das.build_arm().finalize()
    cls = np.zeros(nw * spw + 1, np.int32)                     # + the ground plane
    for k, v in cls_pairs:
        cls[k] = v
    sim = Sim(scene, nw, a.substeps, a.collide, cmax=a.cmax, sense_cls=cls)
    model = sim.model
    ee_idx = np.arange(nw) * bpw + das.EE
    up_idx = np.arange(nw) * bpw + info["ub"]
    lo_idx = np.arange(nw) * bpw + info["lb"] if info["lb"] is not None else None
    dpw = model.joint_dof_count // nw

    # per-trial parameters
    P = {k: np.array([c[k] for c in cases], float) for k in
         ("F_cap", "grip_N", "off_mm", "dir_deg", "yaw_deg", "above_mm", "v", "hold", "post")}
    fkstop = np.array([c["fkstop"] for c in cases])
    carry = np.array([c["mode"] == "carry" for c in cases])
    rel_en = np.array([c["mode"] == "press" for c in cases])    # only presses release; load/carry end at the hold
    half_w = g.W * g.pitch / 2                                  # finger joint value when the pads touch the brick
    g_shut, g_open = half_w - SQUEEZE, half_w + 0.002           # das.queue_place's g_shut / g_open
    gdz = g.grasp_dz
    off_xy = P["off_mm"][:, None] * 1e-3 * np.stack([np.cos(np.radians(P["dir_deg"])), np.sin(np.radians(P["dir_deg"]))], 1)
    z_b0 = zb + g.H + STUD_TOP * s + P["above_mm"] * 1e-3      # held brick bottom, start
    z_seat_b = zb + g.H
    tcp_tgt = np.column_stack([off_xy, z_b0 + gdz])
    yaw_err = np.radians(P["yaw_deg"])

    # --- IK to the hover pose, then put the brick in the hand
    pb, rb = zip(*[hand_targets(tcp_tgt[w], yaw_err[w]) for w in range(nw)])
    solver_ik, ik_pos, ik_rot, ik_q = ik_rig(nw, model_ik, np.array(pb), np.array(rb))
    for _ in range(60):
        solver_ik.step(ik_q, ik_q, iterations=24)
    qik = ik_q.numpy()
    jq = model.joint_q.numpy().reshape(nw, -1).copy()
    jqs = model.joint_q_start.numpy()
    ju = int(np.where(model.joint_child.numpy() == up_idx[0])[0][0])
    cu = int(jqs[ju])
    jq[:, :7] = qik[:, :7]
    jq[:, 7:9] = half_w
    model.joint_q.assign(jq.ravel())
    st = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, st)
    bq = st.body_q.numpy()
    tcp_p, tcp_q = bq[ee_idx, :3], bq[ee_idx, 3:7]
    tcp_err = np.linalg.norm(tcp_p - tcp_tgt, axis=1) * 1e3
    bp = tcp_p + qrotv(tcp_q, np.broadcast_to([0.0, 0.0, gdz], tcp_p.shape))
    bqq = qmulv(tcp_q, np.broadcast_to(das.Q_DOWN * [-1.0, -1.0, -1.0, 1.0], tcp_q.shape))   # hand * conj(Q_DOWN)
    jq[:, cu:cu + 7] = np.concatenate([bp, bqq], 1)
    model.joint_q.assign(jq.ravel())
    sim.start()                                             # resets the state from joint_q and captures
    tgt = np.zeros(model.joint_dof_count, np.float32)
    for w in range(nw):
        tgt[w * dpw:w * dpw + 7] = qik[w, :7]
        tgt[w * dpw + 7:w * dpw + 9] = g_shut
    sim.control.joint_target_pos.assign(tgt)
    with wp.ScopedCapture() as c_:
        solver_ik.step(ik_q, ik_q, iterations=24)
    graph_ik = c_.graph

    bq = sim.s0.body_q.numpy()
    pl0 = bq[lo_idx, :3] if lo_idx is not None else np.tile([0.0, 0.0, zb], (nw, 1))
    ql0 = bq[lo_idx, 3:7] if lo_idx is not None else np.tile([0.0, 0.0, 0.0, 1.0], (nw, 1))
    bp0, bq0 = bq[up_idx, :3], bq[up_idx, 3:7]
    off_actual = np.linalg.norm((bp0 - pl0)[:, :2], axis=1) * 1e3
    tcp0_z = bq[ee_idx, 2].copy()

    # --- host state machine, vectorised over worlds
    t_ref = None
    z_cmd = tcp_tgt[:, 2].copy()
    xy_cmd = tcp_tgt[:, :2].copy()
    z_floor = zb + g.H + gdz - a.floor
    trig = np.zeros(nw, bool)
    done = np.zeros(nw, bool)
    kind = np.zeros(nw, int)                                  # 1 force, 2 FK seated, 3 floor, 4 carry
    t_trig = np.full(nw, np.nan)
    dz_trig = np.full(nw, np.nan)                             # dz at the trigger (the last frame's, before the hold)
    dz = np.zeros(nw)
    clutch = Clutch(nw)
    rel_prev = rel_pose(bp0, bq0, pl0, ql0)[0]
    rel_on = np.zeros(nw, bool)
    t_rel = np.full(nw, np.nan)
    fk_rel = np.full(nw, np.nan)                              # FK dz at the end of the hold (release start)
    fire = np.full((nw, 5), np.nan)                           # t, dz, dxy, tilt, yaw at first pass
    end = np.full((nw, 4), np.nan)                            # dz, dxy, tilt, yaw at the last active frame
    min_dz = np.full(nw, np.inf)
    sink_post = np.full(nw, np.inf)                           # min dz after the trigger
    fk_dz = np.zeros(nw)
    peak_f = np.zeros(nw)
    peak_r = np.zeros(nw)
    slip_ref = None
    slip_pk = np.zeros(nw)
    slip_z = np.zeros(nw)
    fpk_prev = np.zeros(nw)
    bb_all = np.ones(nw)
    bb_post = np.ones(nw)
    fpen_all = np.ones(nw)
    ke_f = P["grip_N"] / SQUEEZE
    grip_settle = np.full(nw, np.nan)
    grip_min = np.full(nw, np.inf)
    squeeze_settle = np.full(nw, np.nan)
    nonfinite = np.zeros(nw, bool)
    tmax = T_SETTLE + max(((z_b0 - z_seat_b).max() + a.floor) / max(P["v"][~carry].max(initial=1e-3), 1e-3) + P["hold"].max() + 0.5
                          + REL_TIMEOUT + P["post"].max(),
                          T_CARRY + P["hold"].max() + 0.5)
    tmax = min(tmax, a.tmax)
    nmax = int(tmax * FPS)
    k_end = nmax
    dzh = np.full((nmax, nw), np.nan)                         # dz per frame (rest stability)
    tgt2 = tgt.reshape(nw, dpw)                               # view: fingers open on release
    for k in range(nmax):
        t = k / FPS
        act = ~done
        # commands
        zc_new = np.maximum(tcp_tgt[:, 2] - P["v"] * max(0.0, t - T_SETTLE), z_floor)
        new_force = act & ~trig & ~carry & (t > T_SETTLE + 0.05) & (fpk_prev >= P["F_cap"])
        new_fk = act & ~trig & ~carry & fkstop & (t > T_SETTLE + 0.05) & (fk_dz <= -0.2e-3 * s) & ~new_force
        new_floor = act & ~trig & ~carry & (zc_new <= z_floor + 1e-9) & ~new_force & ~new_fk
        new_carry = act & ~trig & carry & (t >= T_SETTLE + T_CARRY)
        new = new_force | new_fk | new_floor | new_carry
        kind[new_force], kind[new_fk], kind[new_floor], kind[new_carry] = 1, 2, 3, 4
        t_trig[new], dz_trig[new] = t, dz[new]
        trig |= new
        z_cmd = np.where(trig & ~carry, z_cmd, np.where(carry, tcp_tgt[:, 2], zc_new))
        u = np.clip((t - T_SETTLE) / T_CARRY, 0, 1)
        xy_cmd = tcp_tgt[:, :2] + np.where(carry[:, None], 1.0, 0.0) * np.array([0.0, -D_CARRY]) * (u * u * (3 - 2 * u))
        new_rel = act & trig & rel_en & ~rel_on & (t >= t_trig + P["hold"])
        rel_on |= new_rel
        t_rel[new_rel], fk_rel[new_rel] = t, fk_dz[new_rel]
        done = done | (trig & ~rel_en & (t >= t_trig + P["hold"]))
        done = done | (rel_on & ((clutch.opened & (t >= clutch.t_open + P["post"])) | (~clutch.opened & (t >= t_rel + REL_TIMEOUT))))
        if done.all():
            k_end = k
            break
        # IK -> joint targets
        cmd = np.column_stack([xy_cmd, z_cmd])
        ik_pos.assign(np.array([hand_targets(cmd[w], yaw_err[w])[0] for w in range(nw)], np.float32))
        wp.capture_launch(graph_ik)
        q = ik_q.numpy()
        tgt2[:, :7] = q[:, :7]
        tgt2[:, 7:9] = np.where(rel_on, g_open, g_shut)[:, None]
        sim.control.joint_target_pos.assign(tgt)
        sim.fpk.zero_()
        sim.rpk.zero_()
        sim.bbmin.assign(np.ones(nw, np.float32))
        sim.fpen.assign(np.ones(nw, np.float32))
        sim.step()
        sim.check_tri()
        # measure
        bq = sim.s0.body_q.numpy()
        fin = np.isfinite(bq).all(1)
        if not fin.all():
            bad = ~np.isfinite(bq).all(1)
            wbad = np.unique(np.where(bad)[0] // bpw)
            nonfinite[wbad] = True
            done |= nonfinite
            bq = np.nan_to_num(bq)
        act = ~done
        pu, qu = bq[up_idx, :3], bq[up_idx, 3:7]
        pl = bq[lo_idx, :3] if lo_idx is not None else pl0
        ql = bq[lo_idx, 3:7] if lo_idx is not None else ql0
        rel, tilt, yw = rel_pose(pu, qu, pl, ql)
        dxy, dz, tilt, dyaw = gate_of(rel, tilt, yw, g)
        tm = (k + 1) / FPS
        speed = rel_speed(rel[:, :3], rel_prev)                  # mm/s, relative to the partner (frame-to-frame)
        rel_prev = rel[:, :3].copy()
        fpk_prev = sim.fpk.numpy()
        bb, fp = sim.bbmin.numpy(), sim.fpen.numpy()
        fire_now = clutch.update(tm, act, rel_on, fp < 0, speed, gate(dxy, dz, tilt, dyaw, s) & (tm > T_SETTLE), s)
        fire[fire_now] = np.column_stack([np.full(nw, tm), dz, dxy, tilt, dyaw])[fire_now]
        live = act & (tm > T_SETTLE)
        pre = live & ~rel_on                                     # loaded phase: settle .. release
        dzh[k] = np.where(act, dz, np.nan)
        min_dz = np.where(live, np.minimum(min_dz, dz), min_dz)
        sink_post = np.where(pre & trig, np.minimum(sink_post, dz), sink_post)
        end[act] = np.column_stack([dz, dxy, tilt, dyaw])[act]
        fk_dz = (bq[ee_idx, 2] - tcp0_z) - (z_seat_b - z_b0)
        bb_all = np.where(live, np.minimum(bb_all, bb), bb_all)
        bb_post = np.where(pre & trig, np.minimum(bb_post, bb), bb_post)
        fpen_all = np.where(pre, np.minimum(fpen_all, fp), fpen_all)
        qf = sim.s0.joint_q.numpy().reshape(nw, -1)[:, 7:9].mean(1)     # finger drive: N = ke * (target - q)
        n_real = ke_f * (qf - g_shut)       # inward drive force per finger = the contact force it holds
        first = live & np.isnan(grip_settle)
        grip_settle[first], squeeze_settle[first] = n_real[first], (half_w - qf[first]) * 1e3
        grip_min = np.where(pre, np.minimum(grip_min, n_real), grip_min)
        peak_f = np.where(live, np.maximum(peak_f, fpk_prev), peak_f)
        peak_r = np.where(live, np.maximum(peak_r, sim.rpk.numpy()), peak_r)
        hand_rel = qrotv(qconjv(bq[ee_idx, 3:7]), pu - bq[ee_idx, :3])
        if slip_ref is None and tm >= T_SETTLE:
            slip_ref = hand_rel.copy()
        if slip_ref is not None:
            dv = hand_rel - slip_ref
            slip_pk = np.where(act & ~rel_on, np.maximum(slip_pk, np.linalg.norm(dv, axis=1)), slip_pk)
            slip_z = np.where(act & ~rel_on, dv[:, 2], slip_z)
        if a.trace and k % a.trace == 0:     # world 0: t, brick z in hand (mm), brick dz to seated (mm), fk dz, cmd z rel, F sensor
            print("   trace t=%.3f hand_rel=(%+.3f %+.3f %+.3f) dz=%+.3f fk_dz=%+.3f zcmd=%+.3f F=%.3f R=%.3f fpen=%.3f trig=%d rel=%d v=%.2f" % (
                tm, *(hand_rel[0] * 1e3), dz[0], fk_dz[0] * 1e3, (z_cmd[0] - tcp_tgt[0, 2]) * 1e3, fpk_prev[0],
                sim.rpk.numpy()[0], fp[0] * 1e3, trig[0], rel_on[0], speed[0]), "ee=(%+.4f %+.4f %+.4f) brick=(%+.4f %+.4f)" % (*bq[ee_idx[0], :3], *pu[0, :2]))
    rows = []
    tk = (np.arange(nmax) + 1) / FPS
    for w, c in enumerate(cases):
        trg = bool(trig[w])
        fk_end = (fk_rel[w] if rel_on[w] else fk_dz[w]) * 1e3      # FK height at the end of the hold
        fk_seated = bool(fk_end <= 0.5 * s)
        k_os = float(peak_f[w] / c["F_cap"]) if kind[w] == 1 else None
        fired = bool(clutch.fired[w])
        rest = dict(rest_detr_mm=None, rest_raw_mm=None, rest_drift_um_s=None)
        if clutch.opened[w]:                                       # last min(2, post) s of the run after the window opens
            t_end = clutch.t_open[w] + c["post"]
            m = (tk > t_end - min(2.0, c["post"]) - 1e-9) & (tk <= t_end + 1e-9) & np.isfinite(dzh[:, w])
            if m.sum() >= 10:
                rest = dict(zip(("rest_detr_mm", "rest_raw_mm", "rest_drift_um_s"), rest_fit(tk[m], dzh[m, w])))
        r = dict(c)
        r.update(fired=fired, t_fire=_n(fire[w, 0]), dz_fire_mm=_n(fire[w, 1]), lat_fire_mm=_n(fire[w, 2]),
                 tilt_fire_deg=_n(fire[w, 3]), yaw_fire_deg=_n(fire[w, 4]),
                 window_opened=bool(clutch.opened[w]), t_rel=_n(t_rel[w]), t_open=_n(clutch.t_open[w]),
                 gate_transit=bool(clutch.transit[w] > 0), n_gate_transit=int(clutch.transit[w]),
                 dz_end_mm=_n(end[w, 0]), lat_end_mm=_n(end[w, 1]), tilt_end_deg=_n(end[w, 2]), yaw_end_deg=_n(end[w, 3]),
                 min_dz_mm=_n(min_dz[w]), sink_mm=_n(-sink_post[w]) if np.isfinite(sink_post[w]) else None,
                 bb_min_mm=float(bb_all[w] * 1e3) if bb_all[w] < 0.9 else None,
                 bb_min_hold_mm=float(bb_post[w] * 1e3) if bb_post[w] < 0.9 else None,
                 peak_fz_N=float(peak_f[w]), peak_react_N=float(peak_r[w]), triggered=trg,
                 force_reached=bool(peak_f[w] >= c["F_cap"]),
                 trig_kind=["none", "force", "fk", "floor", "carry"][kind[w]] if trg else "none",
                 t_trig=_n(t_trig[w]), dz_trig_mm=_n(dz_trig[w]), k_os=k_os, fk_dz_end_mm=float(fk_end), fk_seated=fk_seated,
                 fk_false_pos=bool(fk_seated and bool(rel_en[w]) and not fired), slip_mm=float(slip_pk[w] * 1e3),
                 slip_z_mm=float(slip_z[w] * 1e3),
                 finger_pen_mm=float(max(0.0, -fpen_all[w]) * 1e3) if fpen_all[w] < 0.9 else 0.0,
                 grip_real_N=_n(grip_settle[w]), grip_real_min_N=_n(grip_min[w]), squeeze_mm=_n(squeeze_settle[w]),
                 tcp_err_mm=float(tcp_err[w]), off_actual_mm=float(off_actual[w]),
                 timeout=bool(not done[w] and not nonfinite[w]), diverged=bool(nonfinite[w]),
                 frames=k_end, **sim.buf_fields(), **rest)
        rows.append(r)
    return rows


def _n(x):
    x = float(x)
    return x if np.isfinite(x) else None


# --- case lists -----------------------------------------------------------------------
def _case(mode, F, grip_mult, s, vp, rng, off_mm=0.0, above_mm=None, level=None, fkstop=True, hold=HOLD_PRESS,
          yaw_free=True, post=WINDOW):
    N = grip_mult * F / MU
    return dict(mode=mode, F_cap=float(F), grip_N=min(N, GRIP_CAP_N), grip_req_N=N, grip_limited=bool(N > GRIP_CAP_N),
                grip_mult=grip_mult, off_mm=float(off_mm), dir_deg=float(rng.uniform(0, 360)),
                yaw_deg=float(rng.uniform(-1, 1)) if yaw_free else 0.0,
                above_mm=float(3 * s if above_mm is None else above_mm), v=vp / 1000.0, fkstop=fkstop, hold=hold,
                level=float(F if level is None else level), post=post)


def cases_d(a, s, vp, rng):
    """(d): press trials (guarded press, release, weld window) and load trials (cap held 1 s, no release). Load trials press from
    the Insert start height at v_press, like the presses: at every level >= F_seat the brick is seated (dz_trig_mm) before the cap
    is reached, so 'seated first' holds where F_pt and D1(iii) read; levels below F_seat only show sink 0. `load_from` labels it;
    d_quant reports how many trials were seated at the trigger."""
    cs = []
    for F in a.fcaps:
        for mode in ("press", "load"):
            for _ in range(a.trials):
                cs.append(dict(_case(mode, F, a.gm, s, vp, rng, fkstop=(mode == "press"),
                                     hold=HOLD_PRESS if mode == "press" else HOLD_LOAD, yaw_free=(mode == "press")),
                               load_from="insert_start" if mode == "load" else None))
    return cs


def cases_b(a, s, vp, rng, fpress):
    cs = []
    for k in a.offsets:
        for _ in range(a.trials):
            cs.append(_case("press", fpress, a.gm, s, vp, rng, off_mm=round(k * 0.1 * s, 6), level=round(k * 0.1 * s, 6)))
    return cs


def cases_c(a, s, vp, rng, fseat):
    """(c): a.trials_c (d)-style presses at F_seat, watched POST_C s after release."""
    return [_case("press", fseat, a.gm, s, vp, rng, post=POST_C) for _ in range(a.trials_c)]


def cases_e(a, s, vp, rng, fpress):
    cs = []
    for mode in ("press", "carry"):
        for m in (1.0, 2.0, 4.0):
            for _ in range(a.trials):
                cs.append(_case(mode, fpress, m, s, vp, rng, above_mm=30.0, level=m))
    return cs


def run_cases(g, lower, cases, a, tag, vp):
    """Chunk by F_cap (slow, high-force trials cluster in the last chunk) and run each chunk as one model."""
    order = sorted(range(len(cases)), key=lambda i: (cases[i]["F_cap"], cases[i]["mode"] != "press"))
    res = [None] * len(cases)
    for j in range(0, len(order), a.chunk):
        idx = order[j:j + a.chunk]
        t0 = time.time()
        rows = run_arm_batch(g, lower, [cases[i] for i in idx], a, vp)
        for i, r in zip(idx, rows):
            res[i] = r
        print("  [%s] chunk %d-%d of %d: %.0f s" % (tag, j, j + len(idx), len(cases), time.time() - t0), flush=True)
    return res


# --- free-brick worlds: (g) step body force ---------------------------
def run_free(g, cases, lower, T, a):
    """press.py's setup: upper brick free, lower fixed or free on the ground; per-case body force F until t_rel."""
    nw, H = len(cases), g.H
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    for c in cases:
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        if lower == "fixed":
            add_brick(b, -1, wp.transform((0, 0, 0), wp.quat_identity()), g)
            z_low = 0.0
        else:
            z_low = 0.0005
            lb = b.add_body(xform=wp.transform((0, 0, z_low), wp.quat_identity()), label="lower")
            add_brick(b, lb, wp.transform_identity(), g)
        ub = b.add_body(xform=wp.transform((c["off_mm"] * 1e-3, 0, z_low + H + c["start_mm"] * 1e-3), wp.quat_identity()),
                        label="upper")
        add_brick(b, ub, wp.transform_identity(), g)
        scene.add_world(b)
    sim = Sim(scene, nw, a.substeps, "sub", cmax=a.cmax, body_force=True)     # press.py collides every substep
    model = sim.model
    bpw = model.body_count // nw
    up = np.array([w * bpw + bpw - 1 for w in range(nw)])
    lo = np.array([w * bpw for w in range(nw)]) if lower != "fixed" else None
    F = np.array([c["F"] for c in cases], np.float32)
    trel = np.array([c.get("t_rel", 1e9) for c in cases])
    fb = np.zeros((model.body_count, 6), np.float32)
    sim.fbuf = wp.array(fb, dtype=wp.spatial_vector)
    sim.start()
    nf = int(round(T * FPS))
    dzs = np.zeros((nf, nw))
    vzs = np.zeros((nf, nw))
    for k in range(nf):
        fb[up, 2] = -np.where(k / FPS < trel, F, 0.0)
        sim.fbuf.assign(fb)
        sim.step()
        sim.check_tri()
        q = sim.s0.body_q.numpy()
        zl = q[lo, 2] if lo is not None else 0.0
        dzs[k] = q[up, 2] - (zl + H)
        vzs[k] = sim.s0.body_qd.numpy()[up, 2]
    q = sim.s0.body_q.numpy()
    out = []
    for w, c in enumerate(cases):
        uq = q[up[w]]
        lq = q[lo[w]] if lo is not None else np.array([0, 0, 0, 0, 0, 0, 1.0])
        x, y, z, _ = uq[3:7]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        r = dict(c)
        r.update(final_dz_mm=dzs[-1, w] * 1e3, min_dz_mm=dzs[:, w].min() * 1e3, lat_x_mm=(uq[0] - lq[0]) * 1e3,
                 lat_y_mm=(uq[1] - lq[1]) * 1e3, tilt_deg=float(tilt), lower_z_mm=lq[2] * 1e3, vz_min=float(vzs[:, w].min()),
                 body_mass_g=float(model.body_mass.numpy()[up[w]] * 1e3),
                 **sim.buf_fields())
        out.append(r)
    return out


# --- (a) contact generation ------------------------------------------------------------------
def run_a(g, a, overlap_um=50.0):
    """Lower brick static, upper free and seated (dz 0); shifted sideways so the LOWER's stud proxy overlaps the UPPER's wall
    proxy by 50 um. One collide() call; count the stud-proxy/wall-proxy contacts (plus a mesh-off / mesh-on comparison)."""
    s = g.s
    clear_y = (g.W * g.pitch / 2 - ex.WALL_THICKNESS * s + ex.COLLIDER_INSET * s) - (g.pitch / 2 + ex.STUD_COLLIDER_RADIUS * s)
    clear_x = (g.L * g.pitch / 2 - ex.WALL_THICKNESS * s + ex.COLLIDER_INSET * s) - ((g.L - 1) * g.pitch / 2 + ex.STUD_COLLIDER_RADIUS * s)
    cases = [dict(axis=ax, sign=sg, mesh=m) for ax in "xy" for sg in (1, -1) for m in (True, False)]
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    meta = []
    for c in cases:
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        lo = add_brick(b, -1, wp.transform((0, 0, 0), wp.quat_identity()), g, mesh_collide=c["mesh"])
        clr = clear_x if c["axis"] == "x" else clear_y
        d = c["sign"] * (clr + overlap_um * 1e-6)
        ub = b.add_body(xform=wp.transform((d if c["axis"] == "x" else 0.0, d if c["axis"] == "y" else 0.0, g.H),
                                           wp.quat_identity()), label="upper")
        up = add_brick(b, ub, wp.transform_identity(), g, mesh_collide=c["mesh"])
        meta.append((set(lo["studs"]), set(up["walls"]), b.shape_count))
        scene.add_world(b)
    scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))
    model = scene.finalize()
    nw = len(cases)
    pipe = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=4096 * nw, broad_phase="sap",
                                    max_triangle_pairs=400000 * nw)
    contacts = pipe.contacts()
    st = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, st)
    pipe.collide(st, contacts)
    tri = int(pipe.narrow_phase.triangle_pairs_count.numpy()[0]), int(pipe.narrow_phase.triangle_pairs.shape[0])
    n = int(contacts.rigid_contact_count.numpy()[0])
    con = n, 4096 * nw
    s0, s1 = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
    p0, p1 = contacts.rigid_contact_point0.numpy()[:n], contacts.rigid_contact_point1.numpy()[:n]
    nrm = contacts.rigid_contact_normal.numpy()[:n]
    sb = model.shape_body.numpy()
    bq = st.body_q.numpy()

    def world(sh, p):
        bd = sb[sh]
        if bd < 0:
            return p
        return qrotv(bq[bd:bd + 1, 3:7], p[None])[0] + bq[bd, :3]

    spw = meta[0][2]
    rows = []
    for w, c in enumerate(cases):
        studs, walls, _ = meta[w]
        nc = ns = 0
        dmin = None
        for i in range(n):
            a_, b_ = int(s0[i]) - w * spw, int(s1[i]) - w * spw
            if not (0 <= a_ < spw and 0 <= b_ < spw):
                continue
            nc += 1
            if (a_ in studs and b_ in walls) or (b_ in studs and a_ in walls):
                ns += 1
                d = float(np.dot(nrm[i], world(int(s1[i]), p1[i]) - world(int(s0[i]), p0[i])))
                dmin = d if dmin is None else min(dmin, d)
        rows.append(dict(axis=c["axis"], sign=c["sign"], mesh_collides=c["mesh"], overlap_um=overlap_um,
                         clearance_um=(clear_x if c["axis"] == "x" else clear_y) * 1e6, n_contacts=nc,
                         n_stud_wall=ns, stud_wall_sep_um=None if dmin is None else dmin * 1e6,
                         tri_peak=tri[0], tri_cap=tri[1], tri_overflow=tri[0] > tri[1],
                         con_peak=con[0], con_cap=con[1], con_overflow=con[0] > con[1]))
    return rows


# --- (f) real-time factor -------------------------------------------------------------------------
def s3_layout(s):
    """S3's 16 bricks at their built poses at scale s, from the planner's geometry (das.make_plan's frame, plate centred at the
    origin): xy and the height above the plate x s. Returns [(type, pos, yaw, [(parent index or None, rel pos, rel yaw)])]."""
    P = das.P
    bricks = P.STRUCTURES["S3"]
    ni = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    nj = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    old, P.VOXEL_ORIGIN = P.VOXEL_ORIGIN, (-ni * P.PITCH / 2, -nj * P.PITCH / 2, das.Z0)
    try:
        plan = P.build_plan("S3", "weakest_joint")
    finally:
        P.VOXEL_ORIGIN = old
    typ = {b["id"]: b for b in plan["bricks"]}
    idx = {st["brick_id"]: n for n, st in enumerate(plan["sequence"])}
    lay = []
    for st in plan["sequence"]:
        b = typ[st["brick_id"]]
        nx, ny = P.footprint(b["type"], b["yaw_index"])
        x, y, z = st["target_pose"][:3]
        lay.append(dict(type="%dx%d" % (min(nx, ny), max(nx, ny)), pos=np.array([x * s, y * s, das.Z0 + (z - das.Z0) * s]),
                        yaw=math.pi / 2 if ny > nx else 0.0, parents=[idx[m[0]] for m in st["mating_studs"]] or [None]))
    return lay, (ni, nj)


def das_contact_max():
    """(f) must match the cell's buffers (dual_arm_sim.Example: contact_max 32768, njmax 2x): MuJoCo Warp launches
    over naconmax/njmax, so a bigger buffer would slow (f) and it would no longer measure the cell."""
    return 32768


def run_f(a, s):
    """A working cell: two arms + plate + 30 bricks + the full weld pool (210 = N 20, 465 = N 30), driven like the executor
    (IK graph, joint targets, sim graph, body_q readback each frame). S3's 16 bricks stand on the plate in their built poses
    with the welds M1 would have enabled (mating pairs, plate welds); 14 spares rest in the placer's tray sector; arm A is
    parked and B's hand sweeps a slow oval (4 s / 2 s periods, 6 cm / 3 cm) so the arm-link contact tests move.
    tri_peak in each row shows the contacts are live. A second scene, `loose`, is the early build: all 30 bricks loose in the
    tray sector (none seated, no welds enabled). Rows carry scene = seated_S3 | loose; the RTF rule reads seated_S3."""
    lay, (ni, nj) = s3_layout(s)
    geo = lambda t: Geo.get(t, s)   # noqa: E731
    tray_types = ["2x4"] * 8 + ["2x2"] * 6
    slots = das.feeder_slots(len(tray_types))
    loose = das.feeder_slots(24) + [(0.10 + 0.07 * i, 0.30) for i in range(6)]      # 30 loose spots (the tray has 24)
    rows = []
    for sc, pool, nb in (("seated_S3", 210, 20), ("seated_S3", 465, 30), ("loose", 210, 20), ("loose", 465, 30)):
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        arm = das.build_arm()
        model_ik = arm.finalize()
        park = das.Example.solve_park(types.SimpleNamespace(model_ik=model_ik))
        arm.joint_q[:9] = park
        arm.joint_target_pos[:9] = park
        fing = [k * arm.shape_count + i for k in (0, 1) for i, bd in enumerate(arm.shape_body) if bd in (12, 13)]
        for pos, yaw in (das.ARM_A, das.ARM_B):
            scene.add_builder(arm, xform=wp.transform(pos, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)))
        add_plate(scene, geo("2x4"), das.Z0, ni=ni + 4, nj=nj + 4, margin=2)
        bodies = []
        for n, b in enumerate(lay):
            pos = b["pos"] if sc == "seated_S3" else np.array([*loose[n], 0.0005])
            bd = scene.add_body(xform=wp.transform(tuple(pos), das.qz(b["yaw"])), label="s3_%d" % n)
            add_brick(scene, bd, wp.transform_identity(), geo(b["type"]))
            bodies.append(bd)
        for n, ((x, y), t) in enumerate(zip(slots if sc == "seated_S3" else loose[len(lay):], tray_types)):
            bd = scene.add_body(xform=wp.transform((x, y, 0.0005), wp.quat_identity()), label="tray%d" % n)
            add_brick(scene, bd, wp.transform_identity(), geo(t))
            bodies.append(bd)
        assert len(bodies) == 30
        mate = {}                              # (parent, child) -> relpose of a built joint; (None, child) = plate
        for n, b in enumerate(lay if sc == "seated_S3" else []):
            for par in b["parents"]:
                p1, y1 = (np.zeros(3), 0.0) if par is None else (lay[par]["pos"], lay[par]["yaw"])
                mate[(par, n)] = wp.transform(das.rotate(das.qz(-y1), b["pos"] - p1), das.qz(b["yaw"] - y1))
        n_on = 0
        for i in range(nb):
            r = mate.get((None, i))
            n_on += r is not None
            scene.add_equality_constraint_weld(body1=-1, body2=bodies[i], relpose=r or wp.transform_identity(), enabled=r is not None,
                                               label="pw%d" % i, custom_attributes=das.WELD_ATTRS)
            for j in range(i + 1, nb):
                r = mate.get((i, j))
                n_on += r is not None
                scene.add_equality_constraint_weld(body1=bodies[i], body2=bodies[j], relpose=r or wp.transform_identity(),
                                                   enabled=r is not None, label="w%d_%d" % (i, j), custom_attributes=das.WELD_ATTRS)
        model = Sim.finalize(scene)
        assert model.equality_constraint_count == pool, model.equality_constraint_count
        for collide in ("frame", "sub"):
            for sub in (16, 8, 4):
                ke, kd = model.shape_material_ke.numpy(), model.shape_material_kd.numpy()
                ke[fing], kd[fing] = FINGER_CONTACT_KE, finger_kd(sub)      # r4 finger contacts, kd_f per n_sub
                model.shape_material_ke.assign(ke)
                model.shape_material_kd.assign(kd)
                sim = Sim(None, 1, sub, collide, cmax=das_contact_max(), model=model, tri=TRI_F)
                ik_pos = wp.array([das.PARK] * 2, dtype=wp.vec3)
                ik_solver = ik.IKSolver(
                    model=model_ik, n_problems=2, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
                    objectives=[ik.IKObjectivePosition(link_index=das.EE, link_offset=wp.vec3(0, 0, 0), target_positions=ik_pos),
                                ik.IKObjectiveRotation(link_index=das.EE, link_offset_rotation=wp.quat_identity(),
                                                       target_rotations=wp.array([das.Q_DOWN] * 2, dtype=wp.vec4)),
                                ik.IKObjectiveJointLimit(joint_limit_lower=model_ik.joint_limit_lower,
                                                         joint_limit_upper=model_ik.joint_limit_upper)])
                ikq = wp.array(np.tile(park, (2, 1)), dtype=wp.float32)
                with wp.ScopedCapture() as c_:
                    ik_solver.step(ikq, ikq, iterations=24)
                sim.start()
                p_s3 = sim.s0.body_q.numpy()[bodies[:len(lay)], :3].copy()
                tgt = sim.control.joint_target_pos.numpy()
                fr = [0]

                def one():
                    t = fr[0] / FPS
                    fr[0] += 1
                    sweep = das.PARK + [0.0, 0.06 * math.sin(2 * math.pi * t / 4.0), 0.03 * math.sin(2 * math.pi * t / 2.0)]
                    ik_pos.assign(np.array([das.PARK, sweep], np.float32))
                    wp.capture_launch(c_.graph)
                    q = ikq.numpy()
                    for k in range(2):
                        tgt[9 * k:9 * k + 7] = q[k, :7]
                    sim.control.joint_target_pos.assign(tgt)
                    sim.step()
                    sim.check_tri()
                    return sim.s0.body_q.numpy()

                for _ in range(30):
                    one()
                nfr = a.rtf_frames
                wp.synchronize()
                t0 = time.time()
                for _ in range(nfr):
                    bq = one()
                wall = time.time() - t0
                rows.append(dict(scale=s, scene=sc, pool=pool, n_bricks=30, n_welds_on=n_on, collide=collide, substeps=sub, ke_f=FINGER_CONTACT_KE,
                                 kd_f=finger_kd(sub), frames=nfr, wall_s=wall, rtf=nfr / FPS / wall, ms_per_frame=1e3 * wall / nfr,
                                 finite=bool(np.isfinite(bq).all()), n_welds=int(model.equality_constraint_count),
                                 s3_drift_mm=float(np.linalg.norm(bq[bodies[:len(lay)], :3] - p_s3, axis=1).max() * 1e3),
                                 **sim.buf_fields()))
                print("  f: %s pool %d collide=%s substeps=%d  RTF %.2f  tri_peak %d/%d  nacon_peak %d/%d  welds on %d  S3 drift %.3f mm" % (
                    sc, pool, collide, sub, rows[-1]["rtf"], sim.tri_peak, sim.tri_cap, sim.nacon_peak, sim.nacon_cap, n_on,
                    rows[-1]["s3_drift_mm"]), flush=True)
    return rows


# --- summary / gate --------------------------------------------------------------------------------
def load_rows(path):
    rows = []
    for p in ([path] if not isinstance(path, (list, tuple)) else path):
        if Path(p).exists():
            rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return rows


def cfg_key(r):
    return (r["scale"], r["brick"], r["lower"], r["vpress"], r["variant"], r["collide"], r["substeps"])


KEYFMT = "s%g/%s/%s/v%g/%s/%s/n%d"
GM = (1.0, 2.0, 4.0)


def _max(v):
    v = [x for x in v if x is not None]
    return max(v) if v else None


def d_quant(press, load, s):
    """(d) at one grip multiple: F_seat, F_pt, sink, k_os, F_press."""
    q = {}
    levels = sorted({r["F_cap"] for r in press})
    seat = {lv: (sum(bool(r["fired"]) for r in press if r["F_cap"] == lv), sum(1 for r in press if r["F_cap"] == lv))
            for lv in levels}
    q["seat_rate"] = {str(lv): "%d/%d" % seat[lv] for lv in levels}
    q["F_seat"] = next((lv for lv in levels if seat[lv][0] >= thr(seat[lv][1])), None)
    q["grip_limited_levels"] = sorted({r["F_cap"] for r in press if r["grip_limited"]})
    sink, pt = {}, None
    for lv in sorted({r["F_cap"] for r in load}):
        rs = [r for r in load if r["F_cap"] == lv]
        sk = [r["sink_mm"] for r in rs if r["sink_mm"] is not None]
        bb = [r["bb_min_hold_mm"] for r in rs if r.get("bb_min_hold_mm") is not None]
        reached = sum(1 for r in rs if r.get("force_reached"))
        seated = sum(1 for r in rs if r.get("dz_trig_mm") is not None and r["dz_trig_mm"] <= 0.3 * s)
        sink[lv] = dict(n_force_reached="%d/%d" % (reached, len(rs)), all_reached=reached == len(rs), n_seated_at_trigger="%d/%d" % (seated, len(rs)),
                        sink_max_mm=max(sk) if sk else None, sink_end_max_mm=_max([-r["dz_end_mm"] for r in rs if r["dz_end_mm"] is not None]),
                        bb_min_hold_mm=min(bb) if bb else None)
        if pt is None and ((sk and max(sk) > 1.0 * s) or (bb and min(bb) < -1.0 * s)):
            pt = lv
    q["sink_by_level"] = {str(k): v for k, v in sink.items()}
    q["F_pt"], q["sink_levels"] = pt, sorted(sink)
    q["load_force_not_reached_levels"] = [lv for lv, v in sink.items() if not v["all_reached"]]   # grip/friction-limited
    kos = [r["k_os"] for r in press if r["k_os"] is not None]
    q["k_os_p95"], q["k_os_n"] = p95(kos), len(kos)
    q["F_press"] = lvl_up(2 * q["F_seat"]) if q["F_seat"] is not None else None
    # D1(iii): sink where the cap is actually reached, at the level of the peak press force k_os * F_press
    lvl = lvl_up(q["k_os_p95"] * q["F_press"], q["sink_levels"]) if q["F_press"] is not None and q["k_os_p95"] is not None and sink else None
    v_ = sink.get(lvl, {}) if lvl is not None else {}
    q["sink_at_peak_press_mm"], q["sink_level"] = (v_.get("sink_end_max_mm") if v_.get("all_reached") else None), lvl
    return q


def b_quant(bs, s):
    """(b) at one grip multiple: contiguous capture window, FK-seated false positives, dz at rest."""
    lv = sorted({r["level"] for r in bs})
    fired = [sum(bool(r["fired"]) for r in bs if r["level"] == x) for x in lv]
    n = max(sum(1 for r in bs if r["level"] == x) for x in lv)
    ch, chm = c_held(lv, fired, n)
    fks = [r for r in bs if r["fk_seated"]]
    return dict(b_F=bs[0]["F_cap"], c_held_mm=ch, c_held_max_mm=chm,
                capture_curve={"%.2f" % x: "%d/%d" % (f, sum(1 for r in bs if r["level"] == x)) for x, f in zip(lv, fired)},
                dz_fire_med_mm=float(np.median([abs(r["dz_fire_mm"]) for r in bs if r["dz_fire_mm"] is not None] or [np.nan])),
                lat_fire_p95_mm=p95([r["lat_fire_mm"] for r in bs if r["fired"]]),
                fk_fp_given_fk_seated=(sum(r["fk_false_pos"] for r in fks) / len(fks)) if fks else None,
                fk_fp_of_all=sum(r["fk_false_pos"] for r in bs) / len(bs), n_gate_transit=sum(bool(r.get("gate_transit")) for r in bs))


def e_stats(rows_e):
    """(e) per grip multiple: final-descent slip, carry slip, realised grip / nominal, pad penetration."""
    out = {}
    for m in sorted({r["grip_mult"] for r in rows_e}):
        rs = [r for r in rows_e if r["grip_mult"] == m]
        ratios = [r["grip_real_N"] / r["grip_N"] for r in rs if r.get("grip_real_N")]
        out[m] = dict(slip_press_p95_mm=p95([r["slip_mm"] for r in rs if r["mode"] == "press"]),
                      slip_carry_max_mm=_max([r["slip_mm"] for r in rs if r["mode"] == "carry"]),
                      grip_ratio_min=min(ratios) if ratios else None, pen_max_mm=_max([r["finger_pen_mm"] for r in rs]),
                      grip_nominal_N=rs[0]["grip_N"], grip_limited=any(r["grip_limited"] for r in rs), n=len(rs))
    return out


def e_verdict(e, s, ch):
    """The (e) gate items at one grip multiple's stats e, given the (b) capture window ch: True/False/None each."""
    f = lambda v, fn: None if v is None else bool(fn(v))   # noqa: E731
    return dict(grip=f(e["grip_ratio_min"], lambda v: v >= 0.8), pen=f(e["pen_max_mm"], lambda v: v <= 0.3 * s),
                carry=f(e["slip_carry_max_mm"], lambda v: v < 0.5),
                slip=None if e["slip_press_p95_mm"] is None or ch is None else bool(ch > 0 and e["slip_press_p95_mm"] <= 0.2 * ch))


def next_m(m):
    return next((x for x in GM if x > m), None)


def rtf_rule(rows, s, collide="frame"):
    """n_sub rule: 16 if >= 0.8x real time at the 465-weld pool, else 8 if that does, else None (U14)."""
    R = {r["substeps"]: r["rtf"] for r in rows if r["sub"] == "f" and r["scale"] == s and r["collide"] == collide and r["pool"] == 465
         and r.get("scene", "seated_S3") == "seated_S3"}
    if not R:
        return None, {}
    return next((n for n in (16, 8) if R.get(n, 0) >= 0.8), None), R


def summarize(rows, chain_p95=None):
    """Gate quantities per (scale, brick, lower, vpress, variant, collide, n_sub) from the JSONL rows; each Gate P1 item is
    True/False, or None where a quantity is missing. chain_p95: mm, or {scale: mm}, adds D1(i)."""
    out = {}
    cfgs = sorted({cfg_key(r) for r in rows if r["sub"] in ("b", "c", "d", "e")}, key=str)
    for key in cfgs:
        s = key[0]
        R = lambda sub, **kw: [r for r in rows if r["sub"] == sub and cfg_key(r) == key  # noqa: E731
                               and all(r.get(k) == v for k, v in kw.items())]
        q = dict(scale=s, brick=key[1], lower=key[2], vpress=key[3], variant=key[4], collide=key[5], nsub=key[6])
        D, B, E, C = R("d"), R("b"), R("e"), R("c")
        gm_of = lambda rs: sorted({r["grip_mult"] for r in rs})   # noqa: E731
        Dp = lambda m: d_quant([r for r in D if r["mode"] == "press" and r["grip_mult"] == m],   # noqa: E731
                               [r for r in D if r["mode"] == "load" and r["grip_mult"] == m], s)
        Bq = {m: b_quant([r for r in B if r["grip_mult"] == m], s) for m in gm_of(B)}
        Eq = e_stats(E)
        ch_of = lambda m: (Bq[m] if m in Bq else Bq[min(Bq)])["c_held_mm"] if Bq else None   # noqa: E731
        # chosen grip multiple: the smallest m meeting the slip gate (final-descent slip and carry slip), then next-m retries
        # while the FK-seated false-positive rate over (b) at m is above 5 %. If (b) exists and no m meets the slip gate, the
        # finger items are a FAIL (evaluated at the largest m, slip False); without (b) the choice is pending (None items).
        m = next((x for x in sorted(Eq) if (lambda v: v["slip"] and v["carry"])(e_verdict(Eq[x], s, ch_of(x)))), None)
        while m is not None and m in Bq and (Bq[m]["fk_fp_given_fk_seated"] or 0) > 0.05 and next_m(m) is not None:
            m = next_m(m)
        no_m = m is None and bool(Bq) and bool(Eq)
        m_used = m if m is not None else (max(Eq) if no_m else None)
        q["chosen_m"], q["m_used"], q["no_m_meets_slip"], q["e_by_m"] = m, m_used, no_m, {str(k): v for k, v in Eq.items()}
        q["e_verdict_by_m"] = {str(k): e_verdict(v, s, ch_of(k)) for k, v in Eq.items()}
        md = m if m in gm_of(D) else (gm_of(D)[0] if D else None)          # (d), (b) at the chosen grip if run, else the lowest
        mb = m if m in Bq else (min(Bq) if Bq else None)
        q["d_grip_mult"], q["b_grip_mult"] = md, mb
        # rows taken at another grip multiple than the chosen one cannot judge the items that depend on it
        mism = dict(d=m is not None and md != m, b=m is not None and mb != m, c=m is not None and m not in gm_of(C))
        q["rerun"] = [x for x in "dbc" if mism[x]]
        if md is not None:
            q.update(Dp(md))
        if mb is not None:
            q.update(Bq[mb])
        F_press, ch, kos = q.get("F_press"), q.get("c_held_mm"), q.get("k_os_p95")
        # band sanity: (b) and (d) presses at rest in [+0.2s, +0.4s] in their weld window
        bd = [r for rs, mm in ((B, mb), (D, md)) for r in rs
              if r["mode"] == "press" and r["grip_mult"] == mm and r["window_opened"] and r["dz_end_mm"] is not None]
        q["band_frac"] = (sum(0.2 * s <= r["dz_end_mm"] <= 0.4 * s for r in bd) / len(bd)) if bd else None
        # (c) rest stability at F_seat, seated trials; stud-top creep of unseated (c)/(d) presses is reported only
        Cs = [r for r in C if r["fired"] and r["rest_detr_mm"] is not None]
        q["rest_detr_max_mm"], q["rest_raw_max_mm"] = _max([r["rest_detr_mm"] for r in Cs]), _max([r["rest_raw_mm"] for r in Cs])
        q["rest_drift_max_um_s"] = _max([abs(r["rest_drift_um_s"]) for r in Cs])
        q["rest_n"] = len(Cs)
        creep = [r["rest_drift_um_s"] for r in C + D if r["mode"] == "press" and not r["fired"] and r["rest_drift_um_s"] is not None]
        q["studtop_creep_med_um_s"], q["studtop_creep_n"] = (float(np.median(creep)) if creep else None), len(creep)
        gr = [r["grip_real_N"] for r in D if r["mode"] == "press" and r.get("grip_real_N") is not None]
        q["grip_real_N_median"] = float(np.median(gr)) if gr else None
        # per-scale (a, f)
        A = [r for r in rows if r["sub"] == "a" and r["scale"] == s]
        q["a_stud_wall_contacts_min"] = min([r["n_stud_wall"] for r in A]) if A else None
        n_rule, rtfs = rtf_rule(rows, s, key[5])
        Fr = [r["rtf"] for r in rows if r["sub"] == "f" and r["scale"] == s and r["collide"] == key[5] and r["substeps"] == key[6] and r["pool"] == 465
              and r.get("scene", "seated_S3") == "seated_S3"]
        q["rtf_465"], q["n_sub_rule"] = (Fr[0] if Fr else None), n_rule
        q["rtf_all"] = {"%s/%s/%d/%d" % (r.get("scene", "seated_S3"), r["collide"], r["substeps"], r["pool"]): round(r["rtf"], 2)
                        for r in rows if r["sub"] == "f" and r["scale"] == s}
        # Gate P1 items
        ev = e_verdict(Eq[m_used], s, ch) if m_used in Eq else None
        if no_m:
            ev["slip"] = False
        chain = chain_p95.get(s) if isinstance(chain_p95, dict) else chain_p95
        G = {}
        G["a_contacts>0"] = None if q["a_stud_wall_contacts_min"] is None else q["a_stud_wall_contacts_min"] > 0
        G["finger_grip>=0.8nominal"] = ev and ev["grip"]
        G["finger_pen<=0.3s"] = ev and ev["pen"]
        G["carry_slip<0.5"] = ev and ev["carry"]
        if kos is None or F_press is None:
            G["D1ii_press_window"] = None
        elif q["F_pt"] is not None:
            G["D1ii_press_window"] = bool(kos * F_press <= 0.5 * q["F_pt"])
        else:            # no punch-through up to the top level: F_pt is only bounded below by it
            G["D1ii_press_window"] = True if q["sink_levels"] and kos * F_press <= 0.5 * q["sink_levels"][-1] else None
        sk = q.get("sink_at_peak_press_mm")
        G["D1iii_sink<=0.5s"] = None if sk is None else bool(sk <= 0.5 * s)
        G["D1iv_grip<=70N"] = None if F_press is None or m_used is None else bool(m_used * F_press / MU <= HAND_N)
        G["rest_stability"] = (None if q["rest_detr_max_mm"] is None else
                               bool(q["rest_detr_max_mm"] < 0.02 * s and q["rest_drift_max_um_s"] <= 100.0 * s))
        G["slip_p95<=0.2c"] = ev and ev["slip"]
        fp = q.get("fk_fp_given_fk_seated")
        G["fk_fp<=5%"] = None if mb is None else (True if not fp or fp <= 0.05 else (False if next_m(mb) is None else None))
        G["band_sanity<=2%"] = None if q["band_frac"] is None else bool(q["band_frac"] <= 0.02)
        G["rtf>=0.8"] = None if q["rtf_465"] is None else bool(q["rtf_465"] >= 0.8)
        G["no_buffer_overflow"] = not any(r.get("tri_overflow") or r.get("con_overflow") for r in rows if r.get("scale") == s)
        if chain is not None and ch is not None:
            G["D1i_chain_p95<=0.5c"] = bool(chain <= 0.5 * ch)
            q["chain_p95_used_mm"] = chain
        for k in ("D1ii_press_window", "D1iii_sink<=0.5s") if mism["d"] else ():
            G[k] = None
        for k in ("slip_p95<=0.2c", "fk_fp<=5%", "D1i_chain_p95<=0.5c") if mism["b"] else ():
            if k in G:
                G[k] = None
        if mism["d"] or mism["b"]:
            G["band_sanity<=2%"] = None
        if mism["c"]:
            G["rest_stability"] = None
        G = {k: (None if v is None else bool(v)) for k, v in G.items()}
        q["gates"] = G
        vals = list(G.values())
        q["gate_P1"] = "PASS" if all(v is True for v in vals) else ("FAIL" if any(v is False for v in vals) else "INCOMPLETE")
        out[KEYFMT % key] = q
    return out


def crosscheck(res):
    """Frame-collision config vs its per-substep twin ((d) at 0 offset / 20 mm/s, (e)): flags per the plan's cross-check."""
    out = {}
    for k, qf in res.items():
        if qf["collide"] != "frame":
            continue
        qs = res.get(k.replace("/frame/", "/sub/"))
        if qs is None:
            continue
        s, ch = qf["scale"], qf.get("c_held_mm")
        fl = {}
        lf, ls = qf.get("F_seat"), qs.get("F_seat")
        kf, ks = qf.get("k_os_p95"), qs.get("k_os_p95")
        sf, ss = qf.get("sink_at_peak_press_mm"), qs.get("sink_at_peak_press_mm")
        fl["F_seat_levels"] = None if None in (lf, ls) else abs(SWEEP.index(int(lf)) - SWEEP.index(int(ls)))
        fl["sink_diff_mm"] = None if None in (sf, ss) else abs(sf - ss)
        fl["k_os_rel_diff"] = None if None in (kf, ks) else abs(kf - ks) / kf
        # a quantity present in one mode and absent in the other is a difference, not a pass
        fl["none_vs_value"] = [n for n, (x, y) in dict(F_seat=(lf, ls), sink=(sf, ss), k_os=(kf, ks)).items() if (x is None) != (y is None)]
        m = qf.get("m_used")
        Ef, Es = qf["e_by_m"].get(str(m)), qs["e_by_m"].get(str(m))
        vf = e_verdict(Ef, s, ch) if Ef else None
        vs = e_verdict(Es, s, ch) if Es else None
        fl["e_verdict_changed"] = None if vf is None or vs is None else [x for x in vf if vf[x] != vs[x]]
        fl["substep_collision_needed"] = bool((fl["F_seat_levels"] or 0) > 1 or (fl["sink_diff_mm"] or 0) > 0.1 * s
                                              or (fl["k_os_rel_diff"] or 0) > 0.25 or fl["e_verdict_changed"] or fl["none_vs_value"])
        out[k] = fl
    return out


REQ_GROUPS = [(b, l) for b in ("2x4", "2x2") for l in ("welded", "proxy")]
CATS = {"finger": ("finger_grip>=0.8nominal", "finger_pen<=0.3s", "carry_slip<0.5"), "rest": ("rest_stability",),
        "band": ("band_sanity<=2%",)}


def verdicts(res, rows, chain_p95=None):
    """Per scale: choose the n_sub (f's rule) and, per (brick, lower), the fastest v_press whose D1(ii) holds (20, 5, then the
    2 mm/s rescue); the scale passes if every group passes. Then the plan's branch precedence. RTF failures are U14, not a
    scale failure."""
    out, h_ok = {}, {}
    for s in sorted({q["scale"] for q in res.values()}):
        n_rule, _ = rtf_rule(rows, s)
        nsub = n_rule or 16
        groups, failed, inc = {}, set(), False
        for q in res.values():
            if q["scale"] == s and q["variant"] == "proto" and q["collide"] == "frame" and q["nsub"] == nsub:
                groups.setdefault((q["brick"], q["lower"]), []).append(q)
        gv = {}
        for g_, qs in groups.items():
            qs.sort(key=lambda q: -q["vpress"])
            pick = next((q for q in qs if q["gates"]["D1ii_press_window"]), qs[-1])
            f = [k for k, v in pick["gates"].items() if v is False and k != "rtf>=0.8"]
            gv["%s/%s v%g" % (*g_, pick["vpress"])] = "FAIL" if f else ("INCOMPLETE" if any(
                v is None for k, v in pick["gates"].items() if k != "rtf>=0.8") else "PASS")
            failed |= set(f)
            inc |= any(v is None for k, v in pick["gates"].items() if k != "rtf>=0.8")
        missing = sorted(set(REQ_GROUPS) - set(groups))         # the plan requires 2x4 and 2x2, proxy and welded
        verdict = "FAIL" if failed else ("INCOMPLETE" if inc or missing else "PASS")
        rt = [q["gates"]["rtf>=0.8"] for q in res.values() if q["scale"] == s and q["nsub"] == nsub and q["variant"] == "proto"]
        out[s] = dict(n_sub=nsub, n_sub_rule=n_rule, groups=gv, missing_groups=["%s/%s" % g_ for g_ in missing], verdict=verdict, failed=sorted(failed),
                      U14=bool(rtf_rule(rows, s)[1]) and n_rule is None, rtf_ok=rt[0] if rt else None)
        hs = [q for q in res.values() if q["scale"] == s and q["variant"] == "clear" and q["collide"] == "frame"]
        if hs:      # (h) has only (d) and (b): judge the items they can compute
            ok = [all(v is True for k, v in q["gates"].items() if v is not None and k != "rtf>=0.8") and "D1i_chain_p95<=0.5c" in q["gates"]
                  for q in hs]
            pend = all("D1i_chain_p95<=0.5c" not in q["gates"] for q in hs)
            h_ok[s] = None if pend else any(ok)
    v = {s: x["verdict"] for s, x in out.items()}
    if len(v) < 2:
        br = "INCOMPLETE: need both scales"
    elif "PASS" in v.values():
        ps = sorted(s for s, x in v.items() if x == "PASS")
        br = "both pass -- D1 decides (U2)" if len(ps) == 2 else "adopt s=%g" % ps[0]
    elif "INCOMPLETE" in v.values():
        br = "INCOMPLETE"
    else:                                       # FAIL at both
        both = [c for c, items in CATS.items() if all(set(items) & set(out[s]["failed"]) for s in out)]
        hh = list(h_ok.values())
        if both:
            br = "PLAN REVISION before D1: r4 item(s) %s fail at both scales (h does not change those mechanisms)" % both
        elif not hh or None in hh and True not in hh:
            br = "FAIL at both scales: adopt (h) if it passes the gate and D1(i) at some s -- (h) results/chain p95 pending"
        elif True in hh:
            br = "adopt (h) at s=%s; record deviation v4_brick_collision_clearance" % [s for s, x in h_ok.items() if x]
        else:
            br = "(h) fails: s=2 with the search only if 1.5*c_held(2) >= chain p95(2), else escalate U1 (PhysX via Isaac Lab)"
    return out, br, h_ok


def print_summary(res, xc=None, vd=None, out=None):
    for k, q in res.items():
        print("== %s   Gate P1: %s" % (k, q["gate_P1"]))
        for f in ("chosen_m", "F_seat", "F_press", "F_pt", "k_os_p95", "sink_at_peak_press_mm", "grip_real_N_median", "c_held_mm",
                  "c_held_max_mm", "fk_fp_given_fk_seated", "fk_fp_of_all", "band_frac", "rest_detr_max_mm", "rest_raw_max_mm",
                  "rest_drift_max_um_s", "studtop_creep_med_um_s", "rtf_465", "n_sub_rule"):
            v = q.get(f)
            print("   %-24s %s" % (f, "n/a" if v is None else ("%.4g" % v if isinstance(v, float) else v)))
        for m, e in q["e_by_m"].items():
            print("   e m=%s: %s" % (m, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in e.items()}))
        print("   gates:", q["gates"])
        if q.get("rerun"):
            print("   INCOMPLETE until re-run at the chosen grip multiple m=%g:" % q["chosen_m"])
            print("     bash scripts/run.sh scripts/10_newton_scale_probe.py --sub %s --scale %g --brick %s --lower %s --vpress %g --nsub %d "
                  "--collide %s --gm %g%s" % (",".join(q["rerun"]), q["scale"], q["brick"], q["lower"], q["vpress"], q["nsub"], q["collide"],
                                            q["chosen_m"], " --out %s" % out if out else ""))
        elif q.get("no_m_meets_slip"):
            print("   no grip multiple in {1,2,4} meets the slip gate (slip p95 <= 0.2 c_held and carry < 0.5 mm): finger items FAIL")
    for k, fl in (xc or {}).items():
        print("== cross-check %s: %s%s" % (k, fl, "   -> cell must collide every substep; re-read (f) for that mode"
                                          if fl["substep_collision_needed"] else ""))
    if vd:
        out, br, h_ok = vd
        for s, x in out.items():
            print("== scale %g: %s  n_sub=%s (f's rule: %s)  groups=%s  failed=%s  rtf_ok=%s" % (
                s, x["verdict"], x["n_sub"], "U14 (neither 16 nor 8 reaches 0.8x)" if x["U14"] else x["n_sub_rule"] or "no (f) rows",
                x["groups"], x["failed"], x["rtf_ok"]) + ("  MISSING groups: %s" % x["missing_groups"] if x["missing_groups"] else ""))
        print("== branch: %s   (h passes: %s)" % (br, h_ok or "not run"))


def parse_chain(txt):
    if txt is None:
        return None
    if "=" not in txt:
        return float(txt)
    return {float(k): float(v) for k, v in (x.split("=") for x in txt.split(","))}


def synth_rows(s, pen=0.05, collide="frame", seat_shift=0, brick="2x4", lower="welded", gm=2.0, slip_scale=1.0, noseat=False):
    """Synthetic rows for one scale that pass Gate P1 (grip multiple 2 chosen; (d), (b), (c) rows at grip multiple gm), with knobs
    to break items."""
    base = dict(scale=s, brick=brick, lower=lower, vpress=20.0, variant="proto", collide=collide, substeps=16, grip_limited=False,
                fk_seated=False, fk_false_pos=False, fired=True, dz_fire_mm=0.1, lat_fire_mm=0.2, dz_end_mm=0.0, bb_min_hold_mm=None,
                sink_mm=None, k_os=None, slip_mm=0.0, finger_pen_mm=pen, grip_mult=gm, grip_N=10.0, grip_real_N=9.5, mode="press",
                window_opened=True, rest_detr_mm=None, rest_raw_mm=None, rest_drift_um_s=None, force_reached=True, gate_transit=False)
    rows = []
    fs = SWEEP[SWEEP.index(5) + seat_shift]                      # F_seat
    for F in SWEEP:
        k = 0 if noseat else 20 if F > fs else 19 if F == fs else 5
        for i in range(20):
            rows.append(dict(base, sub="d", F_cap=float(F), fired=i < k, k_os=1.1 if i % 10 else 1.3))
            rows.append(dict(base, sub="d", F_cap=float(F), mode="load", fired=True, sink_mm=0.2 * s if F < 50 else 1.5 * s,
                             dz_end_mm=-0.2 * s if F < 50 else -1.5 * s, bb_min_hold_mm=-0.1 * s))
    for m, sl in ((1.0, 0.3), (2.0, 0.1), (4.0, 0.05)):
        for i in range(20):
            rows.append(dict(base, sub="e", F_cap=10.0, slip_mm=slip_scale * sl * (0.5 + i / 20), level=m, grip_mult=m, grip_N=m * 14.3, grip_real_N=m * 13.5))
            rows.append(dict(base, sub="e", F_cap=10.0, mode="carry", slip_mm=0.1, level=m, grip_mult=m, grip_N=m * 14.3, grip_real_N=m * 13.5))
    if collide != "frame":                                       # the cross-check twin has (d) and (e) only
        return rows
    for lv in range(0, 21):
        for i in range(20):
            rows.append(dict(base, sub="b", F_cap=10.0, level=lv * 0.1 * s, fired=(i < 20 if lv <= 8 else i < 15 if lv == 9 else i < 20),
                             fk_seated=(i == 0), fk_false_pos=(i == 0 and lv == 9)))
    for i in range(10):
        rows.append(dict(base, sub="c", F_cap=10.0, rest_detr_mm=0.005 * s, rest_raw_mm=0.01 * s, rest_drift_um_s=20.0 * s * (-1) ** i))
    rows.append(dict(sub="a", scale=s, n_stud_wall=3))
    for sub_, r_ in ((16, 1.2), (8, 1.5), (4, 1.8)):
        rows.append(dict(sub="f", scale=s, substeps=sub_, pool=465, collide="frame", rtf=r_))
    return rows


def selfcheck():
    assert thr(20) == 19 and thr(2) == 2 and thr(10) == 10
    assert lvl_up(10) == 10 and lvl_up(11) == 20 and lvl_up(101) is None and lvl_up(0.5) == 1
    assert c_held([0, 1, 2, 3, 4], [20, 20, 19, 18, 20], 20) == (2, 4)
    assert c_held([0, 1], [10, 20], 20) == (-1.0, 1) and c_held([0, 1], [20, 10], 20) == (0.0, 0.0)   # -1: even offset 0 fails
    z = np.zeros(1)
    s = 2.0
    assert gate(z + 2.3, z, z, z, s)[0] and not gate(z + 2.5, z, z, z, s)[0]           # 1.2s = 2.4
    assert gate(z, z - 1.0, z, z, s)[0] and not gate(z, z - 1.01, z, z, s)[0]           # -0.5s
    assert gate(z, z + 0.6, z, z, s)[0] and not gate(z, z + 0.61, z, z, s)[0]           # +0.3s
    assert not gate(z, z, z + 4.0, z, s)[0] and not gate(z, z, z, z + 5.0, s)[0]
    q = np.array([[0, 0, 0, 1.0]])
    rel, tilt, yw = rel_pose(np.array([[0.0005, 0, 0.0096]]), q, np.zeros((1, 3)), q)
    g = types.SimpleNamespace(pitch=0.008, H=0.0096, L=4, W=2)
    dxy, dz, *_ = gate_of(rel, tilt, yw, g)
    assert abs(dxy[0] - 0.5) < 1e-6 and abs(dz[0]) < 1e-6
    # rest fit: 30 um/s ramp + 5 um wiggle -> slope 30, detrended p2p ~ 0.01 mm
    t = np.arange(120) / 60.0
    dp, rw, sl = rest_fit(t, 0.03 * t + 0.005 * (-1.0) ** np.arange(120))
    assert abs(sl - 30) < 1 and abs(dp - 0.01) < 1e-3 and rw > dp
    # clutch window on synthetic traces (s = 1); 5 worlds, 90 frames; frame f has tm = (f + 1) / 60
    #  0 seated at rest under the press, release at f = 20, pads clear at f = 25: fires at the opening frame, the earlier passes are transit
    #  1 moving until f = 40 then at rest, gate passes throughout: fires when 6 rest frames are in, at f = 45
    #  2 at rest but on the stud tops (dz +1.2) throughout: never fires
    #  3 passes only after the 1 s window closed (opened f = 25, last in-window frame f = 85): never fires, later passes are not transit
    #  4 pads never clear: window never opens
    nw, nf = 5, 90
    cl = Clutch(nw)
    fire_f = [None] * nw
    for f in range(nf):
        tm = (f + 1) / 60
        rel_on = np.array([f >= 20] * nw)
        touch = np.array([f < 25, f < 25, f < 25, f < 25, True])
        speed = np.array([0.0, 50.0 if f < 40 else 0.0, 0.0, 0.0, 0.0])
        ok = np.array([True, True, False, f >= 88, True])
        fr = cl.update(tm, np.ones(nw, bool), rel_on, touch, speed, ok, 1.0)
        for w in np.where(fr)[0]:
            fire_f[w] = f
    assert fire_f == [25, 45, None, None, None], fire_f
    assert cl.transit.tolist() == [25, 45, 0, 0, 90], cl.transit
    assert cl.opened.tolist() == [True] * 4 + [False] and abs(cl.t_open[0] - 26 / 60) < 1e-9
    # Gate P1 on synthetic rows: scale 1 passes with grip multiple 2 (m = 1 fails the slip gate), chain p95 0.3 <= 0.5 c_held 0.8
    res = summarize(synth_rows(1.0), chain_p95=0.3)
    q = res["s1/2x4/welded/v20/proto/frame/n16"]
    assert q["F_seat"] == 5 and q["F_press"] == 10 and q["F_pt"] == 50 and q["chosen_m"] == 2.0, q
    assert abs(q["k_os_p95"] - 1.3) < 1e-9 and abs(q["c_held_mm"] - 0.8) < 1e-9 and abs(q["c_held_max_mm"] - 2.0) < 1e-9, q
    assert q["gate_P1"] == "PASS" and q["gates"]["D1iv_grip<=70N"] and q["band_frac"] == 0, q["gates"]
    assert summarize(synth_rows(1.0), chain_p95=0.5)["s1/2x4/welded/v20/proto/frame/n16"]["gates"]["D1i_chain_p95<=0.5c"] is False   # 0.5 > 0.4
    assert summarize(synth_rows(1.0, pen=0.4))["s1/2x4/welded/v20/proto/frame/n16"]["gates"]["finger_pen<=0.3s"] is False
    # band sanity: 3 % of (b)+(d) presses resting at +0.3 fails
    rr = synth_rows(1.0)
    for r in [r for r in rr if r["sub"] == "b"][:13]:
        r["dz_end_mm"] = 0.3
    q = summarize(rr)["s1/2x4/welded/v20/proto/frame/n16"]
    assert q["gates"]["band_sanity<=2%"] is False and 0.02 < q["band_frac"] < 0.04, q["band_frac"]
    # FK false positives above 5 % at the chosen m = 2 (b): the retry m = 4 has no (b) rows -> the b-dependent items are None and
    # the re-run command names b (and d, c: their rows are at m = 2)
    rr = synth_rows(1.0)
    for r in [r for r in rr if r["sub"] == "b" and r["fk_seated"]][:3]:
        r["fk_false_pos"] = True
    q = summarize(rr)["s1/2x4/welded/v20/proto/frame/n16"]
    assert q["chosen_m"] == 4.0 and q["gates"]["fk_fp<=5%"] is None and q["rerun"] == ["d", "b", "c"], (q["chosen_m"], q["gates"], q["rerun"])
    # (d), (b), (c) rows at m = 1 while (e) chooses m = 2: their items are None (INCOMPLETE), not judged at the wrong grip
    q = summarize(synth_rows(1.0, gm=1.0))["s1/2x4/welded/v20/proto/frame/n16"]
    assert q["chosen_m"] == 2.0 and q["rerun"] == ["d", "b", "c"] and q["gate_P1"] == "INCOMPLETE", (q["rerun"], q["gates"])
    assert all(q["gates"][k] is None for k in ("D1ii_press_window", "slip_p95<=0.2c", "fk_fp<=5%", "band_sanity<=2%", "rest_stability")), q["gates"]
    # no m in {1,2,4} meets the slip gate (with (b) present): a FAIL, not INCOMPLETE; c_held 0 or -1 -> slip False, None -> None
    q = summarize(synth_rows(1.0, slip_scale=10.0))["s1/2x4/welded/v20/proto/frame/n16"]
    assert q["chosen_m"] is None and q["no_m_meets_slip"] and q["gates"]["slip_p95<=0.2c"] is False and q["gate_P1"] == "FAIL", q["gates"]
    e_ = dict(slip_press_p95_mm=0.01, slip_carry_max_mm=0.1, grip_ratio_min=1.0, pen_max_mm=0.0)
    assert e_verdict(e_, 1.0, 0.0)["slip"] is False and e_verdict(e_, 1.0, -1.0)["slip"] is False and e_verdict(e_, 1.0, None)["slip"] is None
    # frame-to-frame speed: a static relative pose with 2 um of noise (chatter) is at rest, and the clutch fires; a 5 mm/s drift is not
    rng = np.random.default_rng(0)
    static = rng.normal(0, 1e-6, (60, 1, 3))
    sp = np.array([rel_speed(static[i + 1], static[i])[0] for i in range(59)])
    assert sp.max() < 1.0, sp.max()                                            # < 1s mm/s at s = 1
    assert rel_speed(np.array([[0.0, 0.0, 5e-3 / FPS]]), np.zeros((1, 3)))[0] > 4.9
    cl = Clutch(1)
    got = [cl.update((f + 1) / FPS, np.ones(1, bool), np.array([f >= 5]), np.array([f < 8]), np.array([sp[f % 59]]), np.ones(1, bool), 1.0)[0]
           for f in range(40)]
    assert got.index(True) == 8 and cl.fired[0], got.index(True)

    # cross-check: same F_seat -> no flag; F_seat two sweep levels apart -> flagged
    res = summarize(synth_rows(1.0) + synth_rows(1.0, collide="sub"))
    assert crosscheck(res)["s1/2x4/welded/v20/proto/frame/n16"]["substep_collision_needed"] is False
    res = summarize(synth_rows(1.0) + synth_rows(1.0, collide="sub", seat_shift=2))
    xc = crosscheck(res)["s1/2x4/welded/v20/proto/frame/n16"]
    assert xc["F_seat_levels"] == 2 and xc["substep_collision_needed"] is True, xc
    res = summarize(synth_rows(1.0) + synth_rows(1.0, collide="sub", noseat=True))     # F_seat present in one mode only
    xc = crosscheck(res)["s1/2x4/welded/v20/proto/frame/n16"]
    assert "F_seat" in xc["none_vs_value"] and xc["substep_collision_needed"] is True, xc
    # branch precedence
    allg = lambda s, **kw: [r for b, l in REQ_GROUPS for r in synth_rows(s, brick=b, lower=l, **kw)]   # noqa: E731
    def branch(pen1, pen2, chain=0.3):
        rows = allg(1.0, pen=pen1) + allg(2.0, pen=pen2)
        return verdicts(summarize(rows, chain), rows, chain)
    assert branch(0.05, 0.05)[1] == "both pass -- D1 decides (U2)", branch(0.05, 0.05)[1]
    assert branch(0.4, 0.05)[1] == "adopt s=2"                                 # fails at s=1 only (pen 0.4 > 0.3)
    assert branch(0.05, 0.7)[1] == "adopt s=1"
    assert branch(0.4, 0.7)[1].startswith("PLAN REVISION"), branch(0.4, 0.7)[1]  # finger item at both scales
    rows = allg(1.0) + allg(2.0)                                               # D1(i) fails at both scales, r4 items fine -> (h) branch
    assert verdicts(summarize(rows, 1.0), rows, 1.0)[1].startswith("FAIL at both scales: adopt (h)")
    rows = synth_rows(1.0) + synth_rows(2.0)                                   # only 2x4/welded: the other three groups are missing
    vd = verdicts(summarize(rows, 0.3), rows, 0.3)
    assert vd[0][1.0]["verdict"] == "INCOMPLETE" and vd[0][1.0]["missing_groups"] == ["2x2/proxy", "2x2/welded", "2x4/proxy"], vd[0][1.0]
    print("selfcheck OK")


# --- driver ----------------------------------------------------------------------------------------------
def emit(out, rows, common):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in rows:
            r = dict(common, **r)
            f.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    bad = [r for r in rows if r.get("tri_overflow") or r.get("con_overflow")]
    if bad:     # rows are written (flagged) so the record shows it; the batch stops
        raise SystemExit("BUFFER OVERFLOW in %d row(s): triangle pairs peak %s / %s, contacts peak %s / %s, MuJoCo nacon peak %s / %s. "
                         "Newton/MuJoCo dropped contacts; raise TRI_PER_WORLD/TRI_F or --cmax" % (
                             len(bad), max(r.get("tri_peak", 0) for r in bad), bad[0].get("tri_cap"), max(r.get("con_peak", 0) for r in bad),
                             bad[0].get("con_cap"), max(r.get("nacon_peak", 0) for r in bad), bad[0].get("nacon_cap")))


def auto_gm(rows, key, a, soft=False):
    """--gm auto: the smallest m in {1,2,4} whose (e) rows (proto config of this key) pass the carry gate and slip p95 <= 0.2 c_held,
    with c_held from (b) if it exists, else the provisional --c-prov * s mm. The summary re-derives m from the real c_held."""
    pk = key[:4] + ("proto",) + key[5:]
    Eq = e_stats([r for r in rows if r["sub"] == "e" and cfg_key(r) == pk])
    if not Eq and soft:
        return 1.0                     # (d) before any (e): m = 1
    if not Eq:
        raise SystemExit("--gm auto needs (e) rows for %s: run --sub e first" % (pk,))
    B = [r for r in rows if r["sub"] == "b" and cfg_key(r) == pk]
    ch = b_quant([r for r in B if r["grip_mult"] == min(r2["grip_mult"] for r2 in B)], key[0])["c_held_mm"] if B else a.c_prov * key[0]
    m = next((x for x in sorted(Eq) if (lambda v: v["slip"] and v["carry"])(e_verdict(Eq[x], key[0], ch))), max(Eq))
    print("  gm auto -> m=%g (c_held %s%.3g mm)" % (m, "" if B else "provisional ", ch), flush=True)
    return m


def dq_of(rows, key, gm=None):
    """(d) quantities of one config at grip multiple gm (default: the lowest run), from the rows so far."""
    if gm is None:
        gm = min([r["grip_mult"] for r in rows if r["sub"] == "d" and cfg_key(r) == key] or [1.0])
    D = [r for r in rows if r["sub"] == "d" and cfg_key(r) == key and r["grip_mult"] == gm]
    return d_quant([r for r in D if r["mode"] == "press"], [r for r in D if r["mode"] == "load"], key[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sub", default="all", help="a,b,c,d,e,f,g,h (comma list) or all = a,d,b,c,e,h,g; run f alone and first")
    ap.add_argument("--scale", default="1,2")
    ap.add_argument("--brick", default="2x4,2x2", help="2x4,2x2 (the plan requires both)")
    ap.add_argument("--lower", default="welded,proxy", help="proxy,welded (the plan requires both)")
    ap.add_argument("--vpress", default="20", help="mm/s, comma list: 20,5 (and 2, the (d) rescue for D1(ii))")
    ap.add_argument("--nsub", type=int, choices=(16, 8), default=16, help="substeps per frame; sets kd_f (900 at 16, 450 at 8)")
    ap.add_argument("--collide", choices=("frame", "sub"), default="frame",
                    help="pipe.collide once per frame (default; the cell's) or every substep (the cross-check: (d) at 0 offset / "
                         "20 mm/s and (e)); f runs both")
    ap.add_argument("--gm", default="1", help="grip multiple m for (b), (c), (d), (h); (e) sweeps m = 1, 2, 4. 'auto' = the smallest m "
                    "whose (e) rows meet the slip gate (uses (b)'s c_held if run, else --c-prov); run (d), (e) first")
    ap.add_argument("--c-prov", type=float, default=1.0, help="provisional c_held in units of s mm for --gm auto before (b) exists")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trials-c", type=int, default=10, help="(c) trials at F_seat")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="JSONL, appended (one row per trial or level)")
    ap.add_argument("--summary", action="store_true", help="read --out and print/write Gate P1 per config, the cross-check and the branch")
    ap.add_argument("--chain-p95", default=None, help="mm, or 1=0.4,2=0.7 per scale; adds D1(i) to the summary")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--fcaps", default=",".join(str(x) for x in SWEEP))
    ap.add_argument("--offsets", default=",".join(str(i) for i in range(21)), help="(b) offsets in units of 0.1s mm")
    ap.add_argument("--fpress", type=float, default=None, help="override F_press (N) for b/e")
    ap.add_argument("--fseat", type=float, default=None, help="override F_seat (N) for c")
    ap.add_argument("--chunk", type=int, default=160, help="worlds per model")
    ap.add_argument("--cmax", type=int, default=12000, help="contact buffer per world (a proxy-lower world peaks near 4.5k)")
    ap.add_argument("--zbase", type=float, default=0.05, help="lower brick bottom height [m] (arm posture)")
    ap.add_argument("--floor", type=float, default=0.06, help="TCP target floor below seated [m]")
    ap.add_argument("--tmax", type=float, default=45.0, help="s of sim per chunk, at most (2 mm/s needs ~35)")
    ap.add_argument("--rtf-frames", type=int, default=120)
    ap.add_argument("--forces", default="0,0.5,1,2,3,5,10,20,50,71,100", help="(g) body forces [N]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", type=int, default=0, help="print world 0 every N frames (debug)")
    a = ap.parse_args()
    a.fcaps = [float(x) for x in a.fcaps.split(",")]
    a.offsets = [int(x) for x in a.offsets.split(",")]
    a.substeps, gm_arg = a.nsub, a.gm
    if a.selfcheck:
        return selfcheck()
    if a.summary:
        chain, rows = parse_chain(a.chain_p95), load_rows(a.out)
        res = summarize(rows, chain)
        xc, vd = crosscheck(res), verdicts(res, rows, chain)
        print_summary(res, xc, vd, a.out)
        p = a.out.with_name(a.out.stem + "_summary.json")
        p.write_text(json.dumps(dict(configs=res, crosscheck=xc, scales=vd[0], branch=vd[1], h_ok=vd[2]), indent=1, default=float))
        print("-> %s" % p)
        return
    subs = ["a", "d", "b", "c", "e", "h", "g"] if a.sub == "all" else a.sub.split(",")
    rng = np.random.default_rng(a.seed)
    base = dict(collide=a.collide, substeps=a.nsub, seed=a.seed)
    for s in [float(x) for x in a.scale.split(",")]:
        for sub in subs:
            if sub in ("a", "g", "f"):
                g = Geo.get("2x4", s)
                if sub == "a":
                    rows = run_a(g, a)
                    emit(a.out, rows, dict(base, sub="a", scale=s, brick="2x4", variant="proto"))
                    print("a s=%g: stud-wall contacts %s (mesh on/off, +-x,+-y)" % (s, [r["n_stud_wall"] for r in rows]))
                elif sub == "g":
                    for lower in ("free", "fixed"):
                        cs = [dict(F=float(f), start_mm=2.0, off_mm=0.0) for f in a.forces.split(",")]
                        rows = run_free(g, cs, lower, 1.0, a)
                        emit(a.out, rows, dict(base, collide="sub", sub="g", scale=s, brick="2x4", lower_mode=lower, variant="proto",
                                               note="BODY FORCE reproduction of plan_v4 s0 fact 6 (press.py); not an arm press"))
                        print("# g scale=%g lower=%s substeps=%d  brick mass=%.2f g" % (s, lower, a.substeps, rows[0]["body_mass_g"]))
                        print(" F[N]  final_dz[mm]  min_dz[mm]  lat_x[mm] lat_y[mm] tilt[deg] lower_z[mm] vz_min[m/s]")
                        for r in rows:
                            print("%5.1f  %+10.3f  %+10.3f  %+8.3f %+8.3f %7.2f %+8.3f %+7.3f" % (
                                r["F"], r["final_dz_mm"], r["min_dz_mm"], r["lat_x_mm"], r["lat_y_mm"], r["tilt_deg"],
                                r["lower_z_mm"], r["vz_min"]))
                elif sub == "f":
                    rows = run_f(a, s)
                    emit(a.out, rows, dict(base, sub="f", brick="mixed"))
                continue
            variant = "clear" if sub == "h" else "proto"
            for brick in a.brick.split(","):
                g = Geo.get(brick, s, clear=(variant == "clear"))
                for lower in a.lower.split(","):
                    for vp in [float(x) for x in a.vpress.split(",")]:
                        key = (s, brick, lower, vp, variant, a.collide, a.nsub)
                        tag = "%s s=%g %s %s v=%g%s" % (sub, s, brick, lower, vp, " clear" if variant == "clear" else "")
                        common = dict(base, scale=s, brick=brick, lower=lower, vpress=vp, variant=variant)
                        for sb in (["d", "b"] if sub == "h" else [sub]):
                            a.gm = float(gm_arg) if gm_arg != "auto" else (1.0 if sb == "e" else auto_gm(load_rows(a.out), key, a, soft=(sb == "d")))
                            if sb == "d":
                                cs = cases_d(a, s, vp, rng)
                            else:
                                dq = dq_of(load_rows(a.out), key)
                                fp, fs = a.fpress or dq["F_press"], a.fseat or dq["F_seat"]
                                if (fs if sb == "c" else fp) is None:
                                    raise SystemExit("no F_seat/F_press for %s at --gm %g: run --sub d first (or give --fpress/--fseat)" % (key, a.gm))
                                cs = {"b": lambda: cases_b(a, s, vp, rng, fp), "c": lambda: cases_c(a, s, vp, rng, fs),
                                      "e": lambda: cases_e(a, s, vp, rng, fp)}[sb]()
                            print("%s -> %s: %d worlds" % (tag, sb, len(cs)), flush=True)
                            rows = run_cases(g, lower, cs, a, tag + " " + sb, vp)
                            emit(a.out, rows, dict(common, sub=sb))
                            if sb == "d":
                                v_ = dq_of(load_rows(a.out), key, a.gm)
                                print("  d: F_seat=%s F_press=%s F_pt=%s k_os_p95=%s seat=%s" % (
                                    v_["F_seat"], v_["F_press"], v_["F_pt"], v_["k_os_p95"], v_["seat_rate"]))


if __name__ == "__main__":
    main()
