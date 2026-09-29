"""P2 (plan_v4 §3): cameras -- speed, information content, look pose, V5 prototype.

    bash scripts/run.sh scripts/11_camera_probe.py --part timing            # gate (a)
    bash scripts/run.sh scripts/11_camera_probe.py --part lookpose          # look-pose sweep, picks h_look and mount
    bash scripts/run.sh scripts/11_camera_probe.py --part tray              # 100 tray scenes -> offline set, gate (b)
    bash scripts/run.sh scripts/11_camera_probe.py --part v5                # 50 look renders per scale, gates (c), (d)
    bash scripts/run.sh scripts/11_camera_probe.py --part all --png DIR     # everything, in the plan's order
    bash scripts/run.sh scripts/11_camera_probe.py --summary                # print gates (a)-(d) from the result files
    bash scripts/run.sh scripts/11_camera_probe.py --selfcheck              # V5 lattice fit on a synthetic depth image

Scenes are render-only: dual_arm_sim's arm builder and brick meshes, arms posed by IK, bricks and plate written
straight into body_q (no physics). Ground truth (shape_index, normals, scene poses) only labels and scores here; the
V5 prototype reads depth, the intrinsics, the arm FK (for A's mask) and the plan, as the firewall (plan §2.5.1) allows.
Not an executor: it looks at scenes, it does not move anything.
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as Rot

import newton
import newton.geometry
import newton.ik as ik
from newton.sensors import SensorTiledCamera

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as D  # noqa: E402  (arm builder, IK conventions, park/brace constants; not edited)
import planner as P  # noqa: E402

ex = D.ex
OUT = HERE / "results" / "v4" / "p2"
MM = 1e-3
PLATE_N = 16                      # baseplate studs per side (plan: (NI+4)x(NJ+4), fixed here so one mesh serves all)
POOL = ["2x2"] * 14 + ["2x4"] * 11 + ["2x6"] * 5      # the 30 pre-allocated bricks (S5 needs 13 / 10 / 3)
CAMS = {"top": dict(w=1280, h=960, vfov=60.0), "wrist_B": dict(w=640, h=480, vfov=55.0)}
TOP_POS = np.array([0.10, -0.20, 1.10])
TILT = math.radians(25.0)         # wrist camera tilt toward the tool axis
MOUNT_DZ = 0.060                  # camera above the TCP
DEPTH_C = 2e-3                    # sigma_depth = C r^2  (plan §2.5.2)
EXT_ERR = {"top": (1 * MM, 0.2), "wrist_B": (0.2 * MM, 0.1)}   # hidden extrinsic error, 1 sigma (m, deg)
H_LOOK_MM = (25, 35, 45, 60)      # x s
OFFSETS_MM = (45, 60, 75)         # camera offset along hand x, not scaled
Rz = lambda a: Rot.from_euler("z", a)
RX_PI = Rot.from_euler("x", math.pi)


def sigma_depth(r):
    return DEPTH_C * r * r


# --- rig: arms + plate + 30 bricks, one Newton model per scale, one camera sensor ------------------
class Rig:
    def __init__(self, s, shadows=True):
        self.s = s
        self.p, self.H, self.sh, self.z0 = 0.008 * s, 0.0096 * s, 0.0017 * s, 0.0032 * s
        self.dz_grasp = 0.0089 + 0.0041 * s       # TCP above the brick bottom (13 mm at s=1: D.GRASP_DZ)
        arm = D.build_arm()
        self.model_ik = arm.finalize()
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        for pos, yaw in (D.ARM_A, D.ARM_B):
            scene.add_builder(arm, xform=wp.transform(pos, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)))
        self.nb = arm.body_count
        vis = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        mesh = lambda L, W: (lambda vf: newton.Mesh(vf[0] * s, vf[1].flatten()))(ex._make_brick_mesh(L, W))
        self.plate_body = scene.add_body(xform=wp.transform_identity(), label="plate")
        scene.add_shape_mesh(self.plate_body, mesh=mesh(PLATE_N, PLATE_N), cfg=vis, color=(0.45, 0.47, 0.45))
        meshes, self.brick_body, self.shape2brick = {}, [], {}
        for n, t in enumerate(POOL):
            L, W = sorted(P.BRICKS[t], reverse=True)
            meshes.setdefault(t, mesh(L, W))
            b = scene.add_body(xform=wp.transform_identity(), label="brick%02d" % n)
            self.shape2brick[scene.add_shape_mesh(b, mesh=meshes[t], cfg=vis, color=D.COLORS[n % 6])] = n
            self.brick_body.append(b)
        scene.add_ground_plane(color=(0.62, 0.60, 0.55))
        self.model = scene.finalize()
        self.state = self.model.state()
        self.shape_body = self.model.shape_body.numpy()
        # label codes: brick pool index >= 0, -1 no hit, -2 table, -3 arm A, -4 arm B, -5 plate
        sb = self.shape_body
        self.lut = np.array([-3 if 0 <= b < self.nb else -4 if self.nb <= b < 2 * self.nb else -5 if b == self.plate_body else -2
                             for b in sb] + [-1])
        for sh_, n in self.shape2brick.items():
            self.lut[sh_] = n
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        newton.geometry.build_bvh_shape(self.model, self.state)
        self.sensor = SensorTiledCamera(self.model, config=SensorTiledCamera.RenderConfig(enable_shadows=shadows))
        self.cams = {}
        for name, c in CAMS.items():
            u = self.sensor.utils
            rays = u.compute_pinhole_camera_rays(c["w"], c["h"], math.radians(c["vfov"]))
            self.cams[name] = dict(c, rays=rays, rays_np=rays.numpy()[0, :, :, 1], depth=u.create_depth_image_output(c["w"], c["h"]),
                                   hdr=u.create_hdr_color_image_output(c["w"], c["h"]),
                                   sidx=u.create_shape_index_image_output(c["w"], c["h"]),
                                   nrm=u.create_normal_image_output(c["w"], c["h"]))
        self.ik = {k: self._ik() for k in "AB"}
        park = lambda base: np.array(base[0]) + D.rotate(D.qz(base[1]), D.PARK)
        self.park_q = {k: self.solve(self.ik[k], base, park(base), Rot.from_quat(D.Q_DOWN))[0]
                       for k, base in (("A", D.ARM_A), ("B", D.ARM_B))}
        # A's link spheres in link frames: 'mesh' = bisected visual-mesh vertices (<= 2 cm radius); 'fr3' = P3's
        # motion/fr3.yml, fitted to the collision meshes (arms within 2 mm, fingers 0.5 mm)
        self.spheres = {"mesh": self._sphere_model(range(self.nb)), "fr3": self._fr3_spheres()}
        self.mask_model = "fr3"

    # -- kinematics -------------------------------------------------------------------------------
    def _ik(self):
        q, pos, rot = (wp.array(np.array([D.HOME_Q + [0.01, 0.01]]), dtype=wp.float32),
                       wp.zeros(1, dtype=wp.vec3), wp.zeros(1, dtype=wp.vec4))
        m = self.model_ik
        solver = ik.IKSolver(model=m, n_problems=1, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
                             objectives=[ik.IKObjectivePosition(link_index=D.EE, link_offset=wp.vec3(0, 0, 0), target_positions=pos),
                                         ik.IKObjectiveRotation(link_index=D.EE, link_offset_rotation=wp.quat_identity(), target_rotations=rot),
                                         ik.IKObjectiveJointLimit(joint_limit_lower=m.joint_limit_lower, joint_limit_upper=m.joint_limit_upper)])
        return dict(q=q, pos=pos, rot=rot, solver=solver, st=m.state())

    def solve(self, k, base, p_world, R, q0=None):
        """IK for the hand at world pose (p_world, R). Returns (q[9], err_mm, err_deg); fingers are set by the caller."""
        (bx, by, bz), byaw = base
        p = D.rotate(D.qz(-byaw), np.asarray(p_world) - [bx, by, bz])
        k["pos"].assign(np.array([p], np.float32))
        k["rot"].assign(np.array([D.qmul(D.qz(-byaw), R.as_quat())], np.float32))
        k["q"].assign(np.array([(q0 if q0 is not None else D.HOME_Q + [0.01, 0.01])], np.float32))
        for _ in range(12):
            k["solver"].step(k["q"], k["q"], iterations=24)
            q = k["q"].numpy()[0]
            newton.eval_fk(self.model_ik, wp.array(q, dtype=wp.float32), wp.zeros(9, dtype=wp.float32), k["st"])
            t = k["st"].body_q.numpy()[D.EE]
            e_mm = np.linalg.norm(t[:3] - p) / MM
            e_deg = math.degrees((Rot.from_quat(t[3:]) * Rot.from_quat(D.qmul(D.qz(-byaw), R.as_quat())).inv()).magnitude())
            if e_mm < 0.3 and e_deg < 0.2:
                break
        return q, e_mm, e_deg

    def pose_arms(self, qa, qb, ga, gb):
        jq = self.model.joint_q.numpy().copy()
        jq[0:9], jq[9:18] = list(qa[:7]) + [ga, ga], list(qb[:7]) + [gb, gb]
        self.model.joint_q.assign(jq)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def place(self, plate, bricks):
        """plate (x, y, yaw); bricks {pool index: (pos, Rotation)}. Everything else is parked off-stage. Refits the BVH."""
        bq = self.state.body_q.numpy().copy()
        bq[self.plate_body] = [plate[0], plate[1], self.z0 - self.H, *Rz(plate[2]).as_quat()]
        for n, b in enumerate(self.brick_body):
            pos, R = bricks.get(n, ((3.0 + 0.1 * n, 0.0, 0.0), Rot.identity()))
            bq[b] = [*pos, *R.as_quat()]
        self.state.body_q.assign(bq)
        newton.geometry.refit_bvh_shape(self.model, self.state)

    def hand(self, k):
        t = self.state.body_q.numpy()[k * self.nb + D.EE]
        return t[:3].copy(), Rot.from_quat(t[3:])

    def _sphere_model(self, bodies, rmax=0.02):
        def split(V):
            c = (V.min(0) + V.max(0)) / 2
            r = np.linalg.norm(V - c, axis=1).max()
            if r <= rmax or len(V) < 20:
                return [(c, r)]
            ax = int(np.ptp(V, 0).argmax())
            V = V[np.argsort(V[:, ax])]
            return split(V[:len(V) // 2]) + split(V[len(V) // 2:])
        out, scale, xf = [], self.model.shape_scale.numpy(), self.model.shape_transform.numpy()
        vis = self.model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)
        for b in bodies:
            V = [Rot.from_quat(xf[s, 3:]).apply(self.model.shape_source[s].vertices[::4] * scale[s]) + xf[s, :3]
                 for s in range(self.model.shape_count) if self.shape_body[s] == b and vis[s]
                 and self.model.shape_source[s] is not None]
            out += [(b, c, r) for c, r in split(np.concatenate(V))] if V else []
        return out

    def _fr3_spheres(self):
        import yaml
        cs = yaml.safe_load((HERE / "motion" / "fr3.yml").read_text())["robot_cfg"]["kinematics"]["collision_spheres"]
        names = [str(l).split("/")[-1] for l in self.model.body_label[:self.nb]]
        return [(names.index(l), np.array(sp["center"]), sp["radius"]) for l, v in cs.items() if l in names for sp in v]

    def fk_world(self, k, base, q, g):
        """Link poses of arm k in the world from its joint angles alone (proprioception): (positions, rotations)."""
        st = self.ik[k]["st"]
        newton.eval_fk(self.model_ik, wp.array(np.r_[q[:7], g, g], dtype=wp.float32), wp.zeros(9, dtype=wp.float32), st)
        t = st.body_q.numpy()
        (bx, by, bz), yaw = base
        return Rz(yaw).apply(t[:, :3]) + [bx, by, bz], Rz(yaw) * Rot.from_quat(t[:, 3:])

    def mask_A(self, cam, pos, R, depth, links, pad=0.005):
        """A's padded link spheres, moved by A's FK (`links` from fk_world) and projected into the camera at its
        nominal pose: bool image. Uses nothing but proprioception, the calibrated camera and the depth image."""
        lp, lR = links
        rays = self.cams[cam]["rays_np"] @ R.as_matrix().T
        hit = np.zeros(rays.shape[:2], bool)
        for b, c, r in self.spheres[self.mask_model]:
            oc = lR[b].apply(c) + lp[b] - pos
            if np.linalg.norm(oc) - r > 0.45:         # farther than any surface in a look image: cannot occlude
                continue
            bb = rays @ oc
            disc = bb * bb - oc @ oc + (r + pad) ** 2
            with np.errstate(invalid="ignore"):
                hit |= (disc > 0) & (bb > 0) & (bb - np.sqrt(np.maximum(disc, 0)) < depth + 0.01)
        return hit

    # -- rendering ----------------------------------------------------------------------------------
    def set_lights(self, key_dir, ring=None):
        """Key: a directional light (random per seed). Ring: co-located spot at the camera, along its axis (the
        renderer's only point-like light: fixed 18-32 deg cone, no intensity parameter)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = self.sensor.render_context
        L = [(1, key_dir, (0, 0, 0))] + ([(0, ring[1], ring[0])] if ring else [])
        d = self.model.device
        rc.lights_active = wp.array([True] * len(L), dtype=wp.bool, device=d)
        rc.lights_type = wp.array([x[0] for x in L], dtype=wp.int32, device=d)
        rc.lights_cast_shadow = wp.array([True] * len(L), dtype=wp.bool, device=d)
        rc.lights_position = wp.array([x[2] for x in L], dtype=wp.vec3f, device=d)
        rc.lights_orientation = wp.array([x[1] for x in L], dtype=wp.vec3f, device=d)

    def render(self, cam, pos, R, labels=False, download=True):
        """One capture at the (true) camera pose. Depth is ray distance [m], 0 = no hit; labels adds shape_index/normals."""
        c = self.cams[cam]
        newton.geometry.refit_bvh_shape(self.model, self.state)
        xf = wp.array([[wp.transformf(wp.vec3f(*pos), wp.quatf(*R.as_quat()))]], dtype=wp.transformf)
        self.sensor.update(self.state, xf, c["rays"], depth_image=c["depth"], hdr_color_image=c["hdr"],
                           shape_index_image=c["sidx"] if labels else None, normal_image=c["nrm"] if labels else None)
        if not download:
            wp.synchronize()
            return None
        out = dict(depth=c["depth"].numpy()[0, 0], hdr=c["hdr"].numpy()[0, 0])
        if labels:
            si = c["sidx"].numpy()[0, 0]
            out["brick"] = self.lut[np.where(si == 0xFFFFFFFF, self.model.shape_count, si)]   # see the label codes in __init__
            out["normal"] = c["nrm"].numpy()[0, 0]
        return out


def cam_pose(name, hand=None, dx=0.045, rng=None, hidden=True):
    """(true_pos, true_R, nominal_pos, nominal_R). Hidden extrinsic error per seed (plan §2.5.2)."""
    if name == "top":
        pos, R = TOP_POS, Rot.identity()                    # looks along -z, image up = +y
    else:
        hp, hR = hand
        f = np.array([-math.sin(TILT), 0, math.cos(TILT)])   # hand frame: z along the tool, tilted toward the axis
        up = np.array([math.cos(TILT), 0, math.sin(TILT)])
        Rm = np.stack([[0, 1, 0], up, -f], 1)                # columns: right, up, back (OpenGL camera)
        pos, R = hp + hR.apply([dx, 0, -MOUNT_DZ]), hR * Rot.from_matrix(Rm)
    st, sr = EXT_ERR[name]
    if hidden and rng is not None:
        return pos + rng.normal(0, st, 3), R * Rot.from_rotvec(np.radians(rng.normal(0, sr, 3))), pos, R
    return pos, R, pos, R


def project(cam, pos, R, P_w):
    """World points -> (px, py, ray distance) in a camera's image (px, py float pixel coordinates)."""
    c = CAMS[cam]
    q = R.inv().apply(np.atleast_2d(P_w) - pos)
    h = math.tan(math.radians(c["vfov"]) / 2)
    z = np.maximum(-q[:, 2], 1e-9)
    return ((q[:, 0] / z / (2 * h * c["w"] / c["h"]) + 0.5) * c["w"] - 0.5,
            (-q[:, 1] / z / (2 * h) + 0.5) * c["h"] - 0.5, np.linalg.norm(q, axis=1))


def sense(out, rng, gain=1.0, dropout=0.01, s=1.0):
    """Sensor noise (plan §2.5.2): depth sigma 2e-3 r^2 on the ray distance, dropout at depth edges (a jump of more
    than 3 s mm in the clean depth, i.e. about a stud height at either scale; read literally: 1 % of edge pixels),
    RGB sigma 2/255 after `gain` (the per-seed light
    intensity, applied to the linear HDR image because the renderer has no intensity parameter) and a soft clip."""
    d = out["depth"].astype(np.float64)
    edge = np.zeros(d.shape, bool)
    jv, jh = np.abs(np.diff(d, axis=0)) > 3 * MM * s, np.abs(np.diff(d, axis=1)) > 3 * MM * s
    edge[:-1] |= jv
    edge[1:] |= jv
    edge[:, :-1] |= jh
    edge[:, 1:] |= jh
    noisy = np.where(d > 0, d + rng.normal(0, 1, d.shape) * sigma_depth(d), 0.0)
    noisy[edge & (rng.random(d.shape) < dropout)] = 0.0
    rgb = np.clip(1 - np.exp(-out["hdr"] * gain) + rng.normal(0, 2 / 255, out["hdr"].shape), 0, 1)   # soft-clip exposure
    return noisy.astype(np.float32), (rgb * 255 + 0.5).astype(np.uint8)


def to_points(depth, rays, pos, R):
    """Depth (ray distance) -> world points through the per-pixel camera rays; depth 0 -> nan."""
    return np.where((depth > 0)[..., None], pos + (depth[..., None] * rays) @ R.as_matrix().T, np.nan)




# --- scenes: a structure on a randomly posed plate, the step's target, a tray, a held brick -----------------
STRUCTS = ["S1", "S2", "S3", "S4", "S5", "S3C", "S3L", "S2A"]


def rand_plate(rng):
    return (rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), math.radians(rng.uniform(-15, 15)))


class Scene:
    """Planner brick tuples (id, type, i, j, k, yaw_index) on the plate, the step's `target`, and world geometry."""

    def __init__(self, rig, plate, placed, target=None):
        self.rig, self.plate, self.placed, self.target = rig, plate, placed, target
        allb = placed + ([target] if target else [])
        NI = max([b[2] + P.footprint(b[1], b[5])[0] for b in allb], default=2)
        NJ = max([b[3] + P.footprint(b[1], b[5])[1] for b in allb], default=2)
        self.oi, self.oj = (PLATE_N - NI) // 2, (PLATE_N - NJ) // 2
        self.R2 = Rz(plate[2]).as_matrix()[:2, :2]

    def world_xy(self, i, j):
        """World xy of the (fractional, array-able) stud coordinate (i, j): plate-frame lattice, then the plate pose."""
        loc = np.stack(np.broadcast_arrays(i + self.oi - PLATE_N / 2, j + self.oj - PLATE_N / 2), -1) * self.rig.p
        return np.array(self.plate[:2]) + loc @ self.R2.T

    def brick_pose(self, b):
        nx, ny = P.footprint(b[1], b[5])
        return (*self.world_xy(b[2] + nx / 2, b[3] + ny / 2), self.rig.z0 + b[4] * self.rig.H), \
            Rz(self.plate[2] + (math.pi / 2 if ny > nx else 0.0))

    def z_face(self, k):
        return self.rig.z0 + k * self.rig.H                    # top of course k-1 (the plate top for k = 0)

    def exposed(self):
        """(cells, single, below): unobstructed studs of the course the target sits on, within 3 pitches of its
        footprint; whether they are only its own footprint (single footprint: tower / pier); the course below."""
        i0, j0, k = self.target[2:5]
        nx, ny = P.footprint(self.target[1], self.target[5])
        cells = lambda bs: {c for b in bs for c in P.cells(b)[0]}
        cover = cells([b for b in self.placed if b[4] == k])
        below = ({(i - self.oi, j - self.oj) for i in range(PLATE_N) for j in range(PLATE_N)} if k == 0
                 else cells([b for b in self.placed if b[4] == k - 1]))
        cs = sorted(c for c in below - cover if i0 - 3 <= c[0] < i0 + nx + 3 and j0 - 3 <= c[1] < j0 + ny + 3)
        foot = {(i0 + a, j0 + b) for a in range(nx) for b in range(ny)}
        return cs, bool(k > 0 and cs and set(cs) <= foot), below

    def brace(self):
        """A's spot (world xyz, stud top) and outward direction: the farthest free stud of the target's support course
        (plan `nearest`); else any free stud; for a fully covered single-footprint support a stand-in on that support."""
        i0, j0, k = self.target[2:5]
        nx, ny = P.footprint(self.target[1], self.target[5])
        ctr = self.world_xy(i0 + nx / 2, j0 + ny / 2)
        top = {}
        for b in self.placed:
            for c in P.cells(b)[0]:
                top[c] = max(top.get(c, -1), b[4])
        foot = {(i0 + a, j0 + b) for a in range(nx) for b in range(ny)}
        pool = ([c for c in top if c not in foot and top[c] == k - 1] or [c for c in top if c not in foot]
                or [c for c in top if top[c] == k - 1])
        xy = lambda c: self.world_xy(c[0] + .5, c[1] + .5)
        c = max(pool, key=lambda c: np.linalg.norm(xy(c) - ctr))
        v = xy(c) - ctr
        return np.array([*xy(c), self.z_face(top[c] + 1) + self.rig.sh]), v / (np.linalg.norm(v) or 1.0)

    def bricks(self):
        """({pool index: (pos, R)} for the placed bricks, free pool slots per type)."""
        free = {t: [n for n, tt in enumerate(POOL) if tt == t] for t in set(POOL)}
        return {free[b[1]].pop(0): self.brick_pose(b) for b in self.placed}, free


def A_brace_pose(rig, sc):
    """A's hand pose pressing closed fingers on the brace spot, leaning away from B (dual_arm_sim.queue_brace)."""
    spot, v = sc.brace()
    tool = np.array([*(-v * math.sin(D.BRACE_TILT)), -math.cos(D.BRACE_TILT)])
    tilt = np.array([-v[1], v[0], 0.0]) * D.BRACE_TILT
    press = spot - tool * (D.TIP_BELOW_TCP - D.BRACE_PRESS)
    return press, Rot.from_quat(D.qmul(D.qrot(tilt), D.qmul(D.qz(math.atan2(v[1], v[0])), D.Q_DOWN)))


def look_hand(rig, sc, h_look, grasp, flip, rng):
    """B's TCP pose over the target at look height h_look (brick bottom above the target's stud tops), the brick's
    actual and nominal pose in the hand, and the grasp width. 'short' closes across the 2-stud axis (14.8 mm at s=1),
    the long axis then points at the camera; 'long' closes across the long axis."""
    nx, ny = P.footprint(sc.target[1], sc.target[5])
    bp, bR = sc.brick_pose(sc.target)
    yl = 0.0 if grasp == "short" else math.pi / 2
    psi = bR.as_euler("xyz")[2] - yl + (math.pi if flip else 0.0)
    across = (min(nx, ny) if grasp == "short" else max(nx, ny)) * rig.p
    tcp = np.array(bp) + [0, 0, rig.sh + h_look + rig.dz_grasp]
    tcp[:2] += rng.normal(0, 0.5 * MM, 2)                 # V4-fine / hover error: the look pose is not exactly on target
    psi += math.radians(rng.normal(0, 0.3))
    err = np.array([rng.uniform(-1.5 * MM, 1.5 * MM), rng.normal(0, 0.1 * MM), rng.uniform(-0.5 * MM, 0.5 * MM)])
    tilt = np.radians([rng.normal(0, 0.3), rng.uniform(-1.5, 1.5), rng.normal(0, 0.5)])   # in-hand pose error
    act = (np.array([err[0], err[1], rig.dz_grasp + err[2]]), RX_PI * Rot.from_rotvec(tilt) * Rz(yl))
    nom = (np.array([0, 0, rig.dz_grasp]), RX_PI * Rz(yl))
    return tcp, Rz(psi) * RX_PI, act, nom, across


def brick_studs(rig, btype, pos, R):
    nx, ny = sorted(P.BRICKS[btype], reverse=True)
    loc = np.array([[(i - (nx - 1) / 2) * rig.p, (j - (ny - 1) / 2) * rig.p, rig.H + rig.sh]
                    for i in range(nx) for j in range(ny)])
    return R.apply(loc) + pos                             # stud tops, world xyz


def face_pts(rig, btype, pose, f, n=8):
    """Grid on vertical face f (0 +x, 1 -x, 2 +y, 3 -y of the mesh) of a brick at `pose`, world."""
    L, W = sorted(P.BRICKS[btype], reverse=True)
    u = np.linspace(-0.9, 0.9, n)
    a, z = np.meshgrid(u, np.linspace(0.1, 0.9, n))
    a, z = a.ravel(), z.ravel() * rig.H
    sg = 1 if f % 2 == 0 else -1
    loc = np.c_[sg * L * rig.p / 2 * np.ones_like(a), a * W * rig.p / 2, z] if f < 2 else \
        np.c_[a * L * rig.p / 2, sg * W * rig.p / 2 * np.ones_like(a), z]
    return pose[1].apply(loc) + pose[0]


# --- one look render: the full V5 input, plus ground-truth scoring --------------------------------------------
def look_render(rig, sc, h_look, dx, grasp, braced, flip, rng):
    """B at the look pose over sc.target with the brick in hand; A braced on the brace spot or parked. None-like dict
    (ik_ok False) if IK fails. Ground truth (the label render) is only used to score visibility."""
    qa, ea, da = rig.park_q["A"], 0.0, 0.0
    if braced:
        pa, Ra = A_brace_pose(rig, sc)
        qa, ea, da = rig.solve(rig.ik["A"], D.ARM_A, pa, Ra, rig.park_q["A"])
    tcp, Rh, act, nom, across = look_hand(rig, sc, h_look, grasp, flip, rng)
    qb, eb, db = rig.solve(rig.ik["B"], D.ARM_B, tcp, Rh)
    if max(eb, ea, db, da) > 1.0:
        return dict(ik_ok=False, ik_err=float(max(eb, ea, db, da)))
    rig.pose_arms(qa, qb, 0.0, (across - 1.2 * MM * rig.s) / 2)
    hp, hR = rig.hand(1)
    bricks, free = sc.bricks()
    held = free[sc.target[1]].pop(0)
    bricks[held] = (hp + hR.apply(act[0]), hR * act[1])
    rig.place(sc.plate, bricks)
    pos, R, npos, nR = cam_pose("wrist_B", (hp, hR), dx, rng)
    rig.set_lights(tuple(-np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)])),
                   (tuple(pos), tuple(R.apply([0, 0, -1]))))
    lab = rig.render("wrist_B", pos, R, labels=True)
    depth, rgb = sense(lab, rng, gain=rng.uniform(0.5, 2.0), s=rig.s)
    links = rig.fk_world("A", D.ARM_A, qa, 0.0) if braced else None
    mask = rig.mask_A("wrist_B", npos, nR, depth, links) if braced else np.zeros(depth.shape, bool)
    row = dict(ik_ok=True, ik_err=float(max(eb, ea)), pos=pos, R=R, npos=npos, nR=nR, hand=(hp, hR), held=held, lab=lab,
               held_nom=(hp + hR.apply(nom[0]), hR * nom[1]), held_act=bricks[held], depth=depth, rgb=rgb, mask=mask, links=links,
               across=across)
    row.update(gt_visibility(rig, sc, row))
    return row


def gt_visibility(rig, sc, row):
    """Gate (c)'s information counts from the label render: visible, unmasked target-course studs; the support's
    top-face outline sides; the held brick's camera-side face pixels."""
    pos, R, lab, mask, d = row["pos"], row["R"], row["lab"], row["mask"], row["lab"]["depth"]
    Hh, Ww = d.shape
    tol = 0.3 * MM * rig.s

    def visible(Pw):
        px, py, r = project("wrist_B", pos, R, Pw)
        ix, iy = np.round(px).astype(int), np.round(py).astype(int)
        ok = (ix >= 0) & (ix < Ww) & (iy >= 0) & (iy < Hh)
        ix, iy = np.clip(ix, 0, Ww - 1), np.clip(iy, 0, Hh - 1)
        return ok & (np.abs(d[iy, ix] - r) < tol) & ~mask[iy, ix]
    cells, single, below = sc.exposed()
    k = sc.target[4]
    xy = sc.world_xy(np.array([c[0] + .5 for c in cells]), np.array([c[1] + .5 for c in cells])) if cells else np.zeros((0, 2))
    studs = np.c_[xy, np.full(len(cells), sc.z_face(k) + rig.sh)]
    sides = 0
    if single:
        i_lo, i_hi = min(c[0] for c in cells), max(c[0] for c in cells) + 1
        j_lo, j_hi = min(c[1] for c in cells), max(c[1] for c in cells) + 1
        t = (np.arange(24) + 0.5) / 24
        e = 0.5 * MM * rig.s / rig.p                          # samples sit just inside the edge, on the top face
        for side, (i, j, out) in enumerate([
                (i_lo + e, j_lo + t * (j_hi - j_lo), (i_lo - 1, j_lo)), (i_hi - e, j_lo + t * (j_hi - j_lo), (i_hi, j_lo)),
                (i_lo + t * (i_hi - i_lo), j_lo + e, (i_lo, j_lo - 1)), (i_lo + t * (i_hi - i_lo), j_hi - e, (i_lo, j_hi))]):
            if out in below:                                  # a support-course brick alongside: no drop, no outline
                continue
            w = np.c_[sc.world_xy(i + 0 * t, j + 0 * t), np.full(len(t), sc.z_face(k))]
            sides += int(visible(w).mean() >= 0.5)
    hp, hR = row["hand"]
    face = (lab["brick"] == row["held"]) & (lab["normal"] @ hR.apply([1, 0, 0]) > 0.9)
    r_t = np.linalg.norm(studs.mean(0) - pos) if len(studs) else 1.0
    hs = brick_studs(rig, sc.target[1], *row["held_act"])           # the held brick's studs: in frame and unoccluded
    a_px = lab["brick"] == -3                                      # A's own pixels: how much of them the mask covers
    return dict(held_studs_visible=int(visible(hs).sum()), held_studs=len(hs), mask_px=int(mask.sum()), a_px=int(a_px.sum()), a_px_unmasked=int((a_px & ~mask).sum()),
                n_studs_visible=int(visible(studs).sum()) if len(studs) else 0, n_studs_expected=len(cells),
                single=bool(single), outline_sides=sides, face_px=int(face.sum()), bump_sigma=float(rig.sh / sigma_depth(r_t)))


def gate_c(r):
    """Gate (c), per render (information, not pipeline quality). The held-stud clause is new: the plan's list lacks it,
    but V5(b) needs the held brick's studs in frame (a 2x2 at offset 75 mm loses its near row)."""
    ok = (r["n_studs_visible"] >= 2 and r["outline_sides"] >= 2) if r["single"] else r["n_studs_visible"] >= 4
    return bool(ok and r["bump_sigma"] >= 10 and r["face_px"] >= 500 and r["held_studs_visible"] >= min(4, r["held_studs"]))


# --- V5 prototype (the seed of vision.py): depth stud blobs, lattice fit, held-brick face fit -----------------
def top_disc(q, sh, pos, k=3.5):
    """Centre of a stud's top disc from its points q. A plane is fitted to the points near the top (it follows a tilted
    brick), points farther than k sigma_depth from it (the visible side wall) are dropped, and the rest are averaged
    with surface-area weights r^2/cos(theta): a plain pixel mean is pulled toward the camera (near side is denser)."""
    sig = sigma_depth(np.linalg.norm(q.mean(0) - pos))
    sel = q[q[:, 2] >= np.percentile(q[:, 2], 95) - 0.3 * sh]
    c0 = sel.mean(0)
    for _ in range(3):
        if len(sel) < 10:
            return q.mean(0)
        co = np.linalg.lstsq(np.c_[np.ones(len(sel)), sel[:, :2] - c0[:2]], sel[:, 2], rcond=None)[0]
        sel = q[np.abs(q[:, 2] - np.c_[np.ones(len(q)), q[:, :2] - c0[:2]] @ co) < k * sig]
    d = sel - pos
    w = np.linalg.norm(d, axis=1) ** 3 / np.maximum(np.abs(d[:, 2]), 1e-9)
    return (sel * w[:, None]).sum(0) / w.sum()


def stud_blobs(P3, h, mask, lo, hi, sh, pos, min_px=40):
    """Connected components of stud pixels (height h above the top face in [lo, hi]) -> the top-disc centre
    (`top_disc`) of each. Dropped:
    blobs touching A's mask or the image border (a partial stud biases the centroid), blobs that are not compact
    (wall strips at stud height), and blobs under 0.7 x the median area (partly hidden). `pos`: the camera
    (nominal). Isolated dropout pixels are tolerated."""
    bad = ~np.isfinite(P3[..., 2])
    cap = ~bad & (h > lo) & (h < hi) & ~mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(cap.astype(np.uint8), connectivity=8)
    near_bad = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    L = lab[cap]
    order = np.argsort(L, kind="stable")
    pts, edges = P3[cap][order], np.searchsorted(L[order], np.arange(n + 1))
    nb = np.bincount(lab[near_bad & cap], minlength=n)
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if (a >= min_px and x > 0 and y > 0 and x + w < lab.shape[1] and y + h < lab.shape[0] and nb[i] == 0
                and max(w, h) < 2.5 * min(w, h) and a > 0.5 * w * h):      # compact, disc-like: not a wall strip
            keep.append((top_disc(pts[edges[i]:edges[i + 1]], sh, pos), a))
    if not keep:
        return np.zeros((0, 3))
    med = np.median([a for _, a in keep])
    return np.array([c for c, a in keep if a >= 0.7 * med]).reshape(-1, 3)


def held_height(P3, pose, btype, rig):
    """Height of each pixel above the held brick's top face, for pixels over its nominal footprint (nan elsewhere):
    the face is the dominant level there (a plane fitted from the height histogram's mode, so it follows the in-hand
    tilt); the studs stand out of it. Uses only the nominal in-hand pose and the depth."""
    L, W = sorted(P.BRICKS[btype], reverse=True)
    lb = pose[1].inv().apply(np.nan_to_num(P3.reshape(-1, 3), nan=9.0) - pose[0])      # brick frame, z up from its bottom
    m = 1.5 * MM
    reg = (np.abs(lb[:, 0]) < L * rig.p / 2 + m) & (np.abs(lb[:, 1]) < W * rig.p / 2 + m) & (np.abs(lb[:, 2] - rig.H) < 2 * MM * rig.s + rig.sh)
    q = lb[reg]
    out = np.full(len(lb), np.nan)
    if len(q) < 100:
        return out.reshape(P3.shape[:2])
    hst, e = np.histogram(q[:, 2], bins=np.arange(q[:, 2].min(), q[:, 2].max() + 2e-4, 2e-4))
    f = q[np.abs(q[:, 2] - e[hst.argmax()] - 1e-4) < 0.35 * MM * rig.s]
    for _ in range(3):
        co = np.linalg.lstsq(np.c_[np.ones(len(f)), f[:, :2]], f[:, 2], rcond=None)[0]
        res = q[:, 2] - np.c_[np.ones(len(q)), q[:, :2]] @ co
        f = q[np.abs(res) < max(3 * res[np.abs(res) < 0.3 * rig.sh].std(), 0.06 * MM)]
    out[reg] = q[:, 2] - np.c_[np.ones(len(q)), q[:, :2]] @ co
    return out.reshape(P3.shape[:2])


def fit_rigid(nodes, obs, gate, iters=3):
    """obs ~ R(dpsi)(node - c0) + c0 + t, obs matched to the nearest prior node within `gate` (the prior fixes the
    lattice indices, plan §2.5.3 V5(a)). Needs >= 2 matches. dict(dpsi, t, rms, n, apply) or None."""
    c0, R, t = nodes.mean(0), np.eye(2), np.zeros(2)
    for _ in range(iters):
        pred = (nodes - c0) @ R.T + c0 + t
        d = np.linalg.norm(obs[:, None] - pred[None], axis=2)
        j = d.argmin(1)
        ok = d[np.arange(len(obs)), j] < (gate if _ == 0 else gate / 3)     # after the first pass the prior error is gone
        if ok.sum() < 2 or len(set(j[ok])) < 2:
            return None
        A, B = nodes[j[ok]], obs[ok]
        U, _, Vt = np.linalg.svd((A - A.mean(0)).T @ (B - B.mean(0)))
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = B.mean(0) - c0 - R @ (A.mean(0) - c0)
    res = np.linalg.norm(B - ((A - c0) @ R.T + c0 + t), axis=1)
    return dict(dpsi=math.atan2(R[1, 0], R[0, 0]), t=t, rms=float(np.sqrt((res ** 2).mean())), n=int(ok.sum()),
                apply=lambda x: (np.atleast_2d(x) - c0) @ R.T + c0 + t)


def plane_fit(Q, tol0=None):
    """x = a + b y + c z through hand-frame points Q. With tol0, the fit starts from the points within tol0 of the
    densest 0.2 mm slice of x (contamination from the brick's top face or the fingers cannot capture it); then 3
    rounds of 3-sigma trimming. -> (a, b, c) or None."""
    if tol0 is not None and len(Q):
        h, e = np.histogram(Q[:, 0], bins=np.arange(Q[:, 0].min(), Q[:, 0].max() + 2e-4, 2e-4))
        Q = Q[np.abs(Q[:, 0] - e[h.argmax()] - 1e-4) < tol0]
    for _ in range(3):
        if len(Q) < 30:
            return None
        A = np.c_[np.ones(len(Q)), Q[:, 1], Q[:, 2]]
        co = np.linalg.lstsq(A, Q[:, 0], rcond=None)[0]
        res = Q[:, 0] - A @ co
        Q = Q[np.abs(res) < max(3 * res.std(), 0.05 * MM)]
    return co


def v5(rig, sc, row, seed, prior_sig=(0.4 * MM, 0.25), true_cam=False):
    """One look image -> target lattice pose, held-brick pose, held face plane; each scored against ground truth in
    the hand frame (the frame the TCP correction is applied in). Inputs: the depth image, the nominal camera pose
    (the hidden extrinsic error stays in, so absolute errors carry the hand-eye term; the target-minus-held error
    cancels its translation; true_cam=True gives the pixel-only terms, without the hand-eye error), the hand FK, A's mask, the nominal in-hand pose and the plan. The plate prior is V4-fine's
    pose (the plan's lattice moved by prior_sig). Ground truth (`held_act`, the true lattice) only scores."""
    rng = np.random.default_rng(seed)                               # the same prior error for the nominal and true-camera runs
    pos, R = (row["pos"], row["R"]) if true_cam else (row["npos"], row["nR"])
    hp, hR, mask = *row["hand"], row["mask"]
    P3 = to_points(row["depth"], rig.cams["wrist_B"]["rays_np"], pos, R)
    cells, single, _ = sc.exposed()
    k = sc.target[4]
    ax = hR.as_matrix()[:2, :2]                                     # columns: hand x, y in world xy
    hand_mm = lambda e: [float(e @ ax[:, 0]) / MM, float(e @ ax[:, 1]) / MM]
    out = {}
    # (a) target course: lattice fit to the stud blobs, prior = the plan's lattice under V4-fine's pose error
    nodes = sc.world_xy(np.array([c[0] + .5 for c in cells]), np.array([c[1] + .5 for c in cells])) if cells else np.zeros((0, 2))
    dpsi_p, t_p = math.radians(rng.normal(0, prior_sig[1])), rng.normal(0, prior_sig[0], 2)
    c_t = nodes.mean(0) if len(nodes) else np.zeros(2)
    prior = (nodes - c_t) @ Rz(dpsi_p).as_matrix()[:2, :2].T + c_t + t_p
    z0 = sc.z_face(k)
    b = stud_blobs(P3, P3[..., 2] - z0, mask, 0.5 * rig.sh, 1.4 * rig.sh, rig.sh, pos)
    out["n_blobs"] = len(b)
    ft = fit_rigid(prior, b[:, :2], 0.45 * rig.p) if len(b) >= 2 and len(nodes) >= 2 else None
    if ft:
        out.update(t_n=ft["n"], t_rms_um=ft["rms"] * 1e6, t_err=hand_mm(ft["apply"](prior).mean(0) - nodes.mean(0)),
                   t_yaw_deg=math.degrees(ft["dpsi"] + dpsi_p))
    # (b) held brick: its top studs (nominal in-hand pose as prior) and its camera-side face
    bt = sc.target[1]
    nom, act = brick_studs(rig, bt, *row["held_nom"]), brick_studs(rig, bt, *row["held_act"])
    bh = stud_blobs(P3, held_height(P3, row["held_nom"], bt, rig), mask, 0.5 * rig.sh, 1.4 * rig.sh, rig.sh, pos, min_px=25)
    fh = fit_rigid(nom[:, :2], bh[:, :2], 0.45 * rig.p) if len(bh) >= 2 else None
    gt = fit_rigid(nom[:, :2], act[:, :2], 0.45 * rig.p)            # exact studs: ground-truth in-hand displacement
    out["h_n"] = fh["n"] if fh else 0
    if fh:
        out.update(h_err=hand_mm(fh["apply"](nom[:, :2]).mean(0) - gt["apply"](nom[:, :2]).mean(0)),
                   h_yaw_deg=math.degrees(fh["dpsi"] - gt["dpsi"]), h_rms_um=fh["rms"] * 1e6)
        if ft:                                                      # the budget's combined term: target relative to held
            out.update(rel_err=[x - y for x, y in zip(out["t_err"], out["h_err"])], rel_yaw_deg=out["t_yaw_deg"] - out["h_yaw_deg"])
    Rn = row["held_nom"][1]
    f = max(range(4), key=lambda f: (hR.inv() * Rn).apply([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)][f])[0])
    Qn = hR.inv().apply(face_pts(rig, bt, row["held_nom"], f) - hp)
    Qa = hR.inv().apply(face_pts(rig, bt, row["held_act"], f) - hp)
    n_h = (hR.inv() * Rn).apply([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)][f])
    Q = hR.inv().apply(P3.reshape(-1, 3)[np.isfinite(P3[..., 2]).ravel()] - hp)
    L, W = sorted(P.BRICKS[bt], reverse=True)
    half = (W if f < 2 else L) * rig.p / 2 - 2 * MM * rig.s
    sel = ((np.abs((Q - Qn.mean(0)) @ n_h) < 2.5 * MM) & (np.abs(Q[:, 1] - Qn[:, 1].mean()) < half)
           & (Q[:, 2] > Qn[:, 2].min() + 0.1 * rig.H) & (Q[:, 2] < Qn[:, 2].max() + 0.3 * MM))   # not the top face
    ce, cg = plane_fit(Q[sel], 0.6 * MM * rig.s), plane_fit(Qa)
    out["face_n"] = int(sel.sum())
    if ce is not None and cg is not None:
        zm = Qa[:, 2].mean()
        out.update(face_dx_um=((ce[0] + ce[2] * zm) - (cg[0] + cg[2] * zm)) * 1e6,
                   face_tilt_deg=math.degrees(math.atan(ce[2]) - math.atan(cg[2])),
                   face_yaw_deg=math.degrees(math.atan(ce[1]) - math.atan(cg[1])))
    out["_blobs"], out["_hblobs"] = b, bh
    return out


# --- self-check: the lattice fit on a synthetic depth image -----------------------------------------------
def selfcheck():
    """A straight-down depth camera over a stud lattice (unknown pose within the prior's capture range): the blob
    detector plus fit_rigid must recover the lattice to < 0.05 mm and 0.1 deg. Pure numpy, no GPU."""
    rng = np.random.default_rng(0)
    p, sh, r_st, h_cam = 0.008, 0.0017, 0.0024, 0.12
    W = Hh = 320
    f = W / (2 * math.tan(math.radians(40) / 2))
    u = (np.arange(W)[None] + 0.5 - W / 2) / f * np.ones((Hh, 1))
    v = -(np.arange(Hh)[:, None] + 0.5 - Hh / 2) / f * np.ones((1, W))
    rays = np.stack([u, v, -np.ones_like(u)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    psi, off = math.radians(7.0), np.array([0.0013, -0.0021])
    Rl = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
    nodes = np.array([(i - 3, j - 3) for i in range(7) for j in range(7)]) * p
    true = nodes @ Rl.T + off
    depth = np.full((Hh, W), h_cam) / -rays[..., 2]                         # table plane z = 0
    xy = rays[..., :2] * (h_cam - sh) / -rays[..., 2:3]                    # where each ray meets the stud-top plane
    d2 = np.linalg.norm(xy[:, :, None] - true[None, None], axis=-1).min(-1)
    depth = np.where(d2 < r_st, (h_cam - sh) / -rays[..., 2], depth)
    depth += rng.normal(0, sigma_depth(h_cam), depth.shape)
    P3 = depth[..., None] * rays
    P3[..., 2] += h_cam                                                     # world: camera at z = h_cam looking down
    blobs = stud_blobs(P3, P3[..., 2], np.zeros(depth.shape, bool), 0.5 * sh, 1.4 * sh, sh, np.array([0, 0, h_cam]))
    assert len(blobs) >= 30, len(blobs)
    prior = nodes @ np.array([[1, 0], [0, 1]]) + 0.0
    c = math.radians(0.3)
    pr = nodes @ np.array([[math.cos(psi + c), -math.sin(psi + c)], [math.sin(psi + c), math.cos(psi + c)]]).T + off + [3e-4, -3e-4]
    ft = fit_rigid(pr, blobs[:, :2], 0.45 * p)
    e = ft["apply"](pr) - true
    assert ft["n"] >= 40 and np.abs(e).max() < 5e-5, (ft["n"], np.abs(e).max())
    assert abs(ft["dpsi"] + c) < math.radians(0.1), ft["dpsi"]
    ft2 = fit_rigid(pr, blobs[:, :2][blobs[:, 0] < -0.002], 0.45 * p)        # half the studs occluded: still recovered
    assert ft2 is not None and np.abs(ft2["apply"](pr) - true).mean() < 1e-4
    print("selfcheck ok: %d blobs, fit max err %.1f um, yaw err %.3f deg" % (
        len(blobs), np.abs(e).max() * 1e6, math.degrees(abs(ft["dpsi"] + c))))




# --- parts -------------------------------------------------------------------------------------------------
dump = lambda o: json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
scalars = lambda r: {k: v for k, v in r.items() if isinstance(v, (int, float, bool, str, list)) and not k.startswith("_")}


def intrinsics(name):
    c = CAMS[name]
    f = c["h"] / (2 * math.tan(math.radians(c["vfov"]) / 2))
    return dict(width=c["w"], height=c["h"], vfov_deg=c["vfov"], fx=f, fy=f, cx=c["w"] / 2, cy=c["h"] / 2,
                convention="OpenGL camera (x right, y up, looks -z); pixel (px,py) ray = normalize(((px+.5)/W-.5)*2h*W/H, "
                           "-((py+.5)/H-.5)*2h, -1), h = tan(vfov/2); depth = distance along that ray [m], 0 = invalid")


def png_depth(d, lo, hi):
    v = np.clip((d - lo) / (hi - lo), 0, 1)
    return np.where((d > 0)[..., None], cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO), 0).astype(np.uint8)


def save_png(png, name, rgb, depth, lo, hi):
    if png:
        png.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png / (name + "_rgb.png")), rgb[..., ::-1])
        cv2.imwrite(str(png / (name + "_depth.png")), png_depth(depth, lo, hi))


def tray_scene(rig, rng):
    """Random partial structure on a randomly posed plate + 6 tray bricks (annular sector of B's pick ring, bearings
    50-85 deg toward -y, >= 25 mm apart by bounding circles, studs up, uniform yaw); a tray that does not fit (big
    bricks at s=2) is redrawn. -> (Scene, {pool index: pose}, tray slots)."""
    base = np.array(D.ARM_B[0][:2])
    while True:
        seq = P.sequence(P.STRUCTURES[STRUCTS[rng.integers(len(STRUCTS))]])
        sc = Scene(rig, rand_plate(rng), seq[:int(rng.integers(0, min(len(seq), 24) + 1))])
        bricks, free = sc.bricks()
        slots = [int(n) for n in rng.permutation([n for v in free.values() for n in v])[:6]]
        placed = []
        for n in slots:
            L, W = sorted(P.BRICKS[POOL[n]], reverse=True)
            rc = math.hypot(L, W) * rig.p / 2
            for _ in range(300):
                r, th = math.sqrt(rng.uniform(0.42 ** 2, 0.62 ** 2)), math.radians(rng.uniform(50, 85))
                xy = base + r * np.array([-math.cos(th), -math.sin(th)])
                if all(np.linalg.norm(xy - o[0]) >= rc + o[1] + 0.025 for o in placed):
                    placed.append((xy, rc, rng.uniform(0, 2 * math.pi)))
                    break
        if len(slots) == 6 and len(placed) == 6:
            for n, (xy, _, yaw) in zip(slots, placed):
                bricks[n] = ((*xy, 0.0), Rz(yaw))
            return sc, bricks, slots


def capture_top(rig, sc, bricks, rng):
    rig.pose_arms(rig.park_q["A"], rig.park_q["B"], 0.0, 0.0)
    rig.place(sc.plate, bricks)
    pos, R, npos, nR = cam_pose("top", rng=rng)
    kd = tuple(-np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)]))
    rig.set_lights(kd)
    lab = rig.render("top", pos, R, labels=True)
    gain = rng.uniform(0.5, 2.0)
    depth, rgb = sense(lab, rng, gain, s=rig.s)
    return dict(pos=pos, R=R, npos=npos, nR=nR, lab=lab, depth=depth, rgb=rgb, gain=gain, key_dir=kd)


def bench(fn, n):
    for _ in range(3):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), n=n)


def course_scenes(rig, rng):
    """The sweep's two representative targets: a 2x4 on a course with neighbours (S2-like), and a 2x4 on a 2x2 pier
    with a second pier 8 pitches away (S3: single-footprint support; A braces on the other pier)."""
    b = lambda t, i, j, k: ("b", t, i, j, k, 0)
    plate = rand_plate(rng)
    return {"course": Scene(rig, plate, [b("2x4", 0, 0, 0), b("2x4", 0, 4, 0), b("2x4", 0, 8, 0), b("2x4", 0, 6, 1)], b("2x4", 0, 2, 1)),
            "small": Scene(rig, plate, [b("2x4", 0, 0, 0), b("2x4", 0, 4, 0), b("2x4", 0, 8, 0), b("2x4", 0, 6, 1)], b("2x2", 0, 2, 1)),
            "single": Scene(rig, plate, [b("2x2", 0, 0, 0), b("2x2", 0, 0, 1), b("2x2", 0, 0, 2),
                                         b("2x2", 0, 8, 0), b("2x2", 0, 8, 1), b("2x2", 0, 8, 2), b("2x2", 0, 8, 3)], b("2x4", 0, 0, 3))}


def part_timing(scales, n, out, seed, shadows):
    res = []
    for s in scales:
        rig = Rig(s, shadows)
        rng = np.random.default_rng(seed)
        sc, bricks, _ = tray_scene(rig, rng)
        cap = capture_top(rig, sc, bricks, rng)
        row = {"scale": s, "n_bricks": len(POOL)}
        for sh in (True, False):
            rig.sensor.render_config.enable_shadows = sh
            top = {"gpu": bench(lambda: rig.render("top", cap["pos"], cap["R"], download=False), n),
                   "capture": bench(lambda: rig.render("top", cap["pos"], cap["R"]), n),
                   "capture_labels": bench(lambda: rig.render("top", cap["pos"], cap["R"], labels=True), n)}
            look = None
            for h, dx in ((35, 45), (25, 60), (60, 45), (45, 75)):
                sc2 = course_scenes(rig, rng)["course"]
                look = look_render(rig, sc2, h * MM * s, dx * MM, "short", False, 0, rng)
                if look["ik_ok"]:
                    break
            rig.sensor.render_config.enable_shadows = sh
            wr = {"gpu": bench(lambda: rig.render("wrist_B", look["pos"], look["R"], download=False), n),
                  "capture": bench(lambda: rig.render("wrist_B", look["pos"], look["R"]), n),
                  "capture_labels": bench(lambda: rig.render("wrist_B", look["pos"], look["R"], labels=True), n)}
            row["shadows_on" if sh else "shadows_off"] = {"top": top, "wrist_B": wr}
            print("s=%g shadows=%s  top capture %.1f ms (gpu %.1f)  wrist capture %.1f ms (gpu %.1f)" % (
                s, sh, top["capture"]["median_ms"], top["gpu"]["median_ms"], wr["capture"]["median_ms"], wr["gpu"]["median_ms"]))
        res.append(row)
    (out / "timing.json").write_text(dump(res))


def part_lookpose(scales, n, out, seed, shadows):
    grid = [(h, dx) for h in H_LOOK_MM for dx in OFFSETS_MM][:n]
    f = (out / "lookpose.jsonl").open("w")
    rows = []
    for s in scales:
        rig = Rig(s, shadows)
        for (h, dx) in grid:
            for si, name in enumerate(("course", "single", "small")):
                for gi, grasp in enumerate(("short", "long")):
                    for bi, braced in enumerate((False, True)):
                        for fi in (0, 1):
                            rng = np.random.default_rng(seed + 1000 * si + 100 * gi + 10 * bi + fi)
                            sc = course_scenes(rig, rng)[name]
                            r = look_render(rig, sc, h * MM * s, dx * MM, grasp, braced, fi, rng)
                            row = dict(scale=s, h_mm=h, offset_mm=dx, target=name, grasp=grasp, braced=braced, flip=fi,
                                       **scalars(r))
                            row["gate_c"] = bool(r["ik_ok"] and gate_c(r))
                            f.write(dump(row) + "\n")
                            rows.append(row)
            print("s=%g h_look=%d s mm offset=%d mm: %d/%d combos pass (best of both yaw flips), %d/%d single looks" % (
                s, h, dx, sum(combos(rows, s, h, dx).values()), len(combos(rows, s, h, dx)),
                sum(r["gate_c"] for r in rows if (r["scale"], r["h_mm"], r["offset_mm"]) == (s, h, dx)),
                sum(1 for r in rows if (r["scale"], r["h_mm"], r["offset_mm"]) == (s, h, dx))), flush=True)
    f.close()
    pick_lookpose(rows, out)


def combos(rows, s, h, dx):
    """{(target, grasp, braced): passes with either yaw flip (the plan's 180 deg second look)}"""
    c = {}
    for r in rows:
        if (r["scale"], r["h_mm"], r["offset_mm"]) == (s, h, dx):
            k = (r["target"], r["grasp"], r["braced"])
            c[k] = c.get(k, False) or r["gate_c"]
    return c


def pick_lookpose(rows, out):
    grid = sorted({(r["h_mm"], r["offset_mm"]) for r in rows})
    scales = sorted({r["scale"] for r in rows})
    bof = {g: sum(sum(combos(rows, s, *g).values()) for s in scales) for g in grid}      # best of the two yaw flips
    one = {g: sum(r["gate_c"] for r in rows if (r["h_mm"], r["offset_mm"]) == g) for g in grid}   # every single look
    best = min(grid, key=lambda g: (-bof[g], -one[g], g[0], abs(g[1] - 45)))
    nc = len(combos(rows, scales[0], *grid[0]))                                           # combos per scale
    nr = sum(1 for r in rows if (r["scale"], r["h_mm"], r["offset_mm"]) == (scales[0], *grid[0]))
    res = dict(chosen=dict(h_look_mm_per_s=best[0], offset_mm=best[1]),
               best_of_two_flips_passes={"%d/%d" % g: "%d/%d" % (v, nc * len(scales)) for g, v in bof.items()},
               single_look_passes={"%d/%d" % g: "%d/%d" % (v, nr * len(scales)) for g, v in one.items()},
               rot_offset_term_mm={"s=%g" % s: 1.75e-3 * best[0] * s for s in scales},
               rule="max passes over (scale x target x grasp x braced) with the better of two yaw flips; then max single-look "
                    "passes; then smaller h_look, offset nearest 45")
    (out / "lookpose.json").write_text(json.dumps(res, indent=1))
    print("look pose chosen: h_look = %d s mm, offset %d mm  (rotation x offset term %s mm)" % (
        best[0], best[1], res["rot_offset_term_mm"]))


def chosen_pose(out, args):
    if args.h_look and args.offset:
        return args.h_look, args.offset
    f = out / "lookpose.json"
    if f.exists():
        c = json.loads(f.read_text())["chosen"]
        return c["h_look_mm_per_s"], c["offset_mm"]
    print("no lookpose.json in %s: using h_look = 35 s mm, offset 45 mm" % out)
    return 35, 45


def part_tray(scales, n, out, seed, shadows, dx, png):
    off = out / "offline"
    off.mkdir(parents=True, exist_ok=True)
    f = (out / "tray.jsonl").open("w")
    meta, labels = dict(intrinsics={c: intrinsics(c) for c in CAMS}, noise=dict(depth_sigma="2e-3*r^2 m on ray distance",
                        depth_edge_dropout=0.01, rgb_sigma=2 / 255, hand_eye_offset_mm=dx), scenes=[]), dict(scenes=[])
    for s in scales[:1]:                   # the offline set is at one scale (the first given)
        rig = Rig(s, shadows)
        for i in range(n):
            rng = np.random.default_rng(seed + i)
            sc, bricks, slots = tray_scene(rig, rng)
            cap = capture_top(rig, sc, bricks, rng)
            P3 = to_points(cap["lab"]["depth"], rig.cams["top"]["rays_np"], cap["pos"], cap["R"])
            per = []
            for n_ in slots:
                m = (cap["lab"]["brick"] == n_) & (P3[..., 2] >= 0.97 * rig.H)          # top face + studs, label render
                px = int(m.sum())
                r = float(cap["lab"]["depth"][m].mean()) if px else 0.0
                per.append(dict(slot=n_, type=POOL[n_], top_px=px, range_m=r, sigma_depth_mm=sigma_depth(r) / MM,
                                ok=bool(px >= 50 and rig.H >= 3 * sigma_depth(r))))
            # wrist hover over one tray brick: B 100 mm above a V2-like estimate (5 mm error), ring light on
            j = int(rng.integers(6))
            hov = None
            for _ in range(4):
                pos_b = np.array(bricks[slots[j]][0][:2]) + rng.normal(0, 5 * MM, 2)
                psi = bricks[slots[j]][1].as_euler("xyz")[2] + rng.integers(4) * math.pi / 2
                q, e, d = rig.solve(rig.ik["B"], D.ARM_B, [*pos_b, rig.H + 0.100], Rz(psi) * RX_PI)
                if e < 1.0 and d < 1.0:
                    rig.pose_arms(rig.park_q["A"], q, 0.03, 0.03)
                    rig.place(sc.plate, bricks)
                    hp, hR = rig.hand(1)
                    wpos, wR, wnp, wnR = cam_pose("wrist_B", (hp, hR), dx * MM, rng)
                    rig.set_lights(cap["key_dir"], (tuple(wpos), tuple(wR.apply([0, 0, -1]))))
                    wl = rig.render("wrist_B", wpos, wR, labels=True)
                    wd, wr = sense(wl, rng, cap["gain"], s=rig.s)
                    hov = dict(brick_slot=slots[j], pos=wpos, R=wR, npos=wnp, nR=wnR)
                    break
            u16 = lambda d: np.clip(d * 1e4, 0, 65535).astype(np.uint16)
            arrs = dict(top_depth=u16(cap["depth"]), top_rgb=cap["rgb"])
            larrs = dict(top_brick=cap["lab"]["brick"].astype(np.int8), top_depth_clean=u16(cap["lab"]["depth"]))
            if hov:
                arrs.update(wrist_depth=u16(wd), wrist_rgb=wr)
                larrs.update(wrist_brick=wl["brick"].astype(np.int8), wrist_depth_clean=u16(wl["depth"]))
            np.savez_compressed(off / ("scene_%03d.npz" % i), **arrs)
            np.savez_compressed(off / ("labels_%03d.npz" % i), **larrs)
            ext = lambda p, R: dict(pos=p, quat_xyzw=R.as_quat())
            meta["scenes"].append(dict(scene=i, scale=s, top=dict(nominal=ext(cap["npos"], cap["nR"])),
                                       wrist_B=hov and dict(nominal=ext(hov["npos"], hov["nR"]), hover_over_tray_slot=hov["brick_slot"])))
            labels["scenes"].append(dict(
                scene=i, plate=sc.plate, key_light_dir=cap["key_dir"], light_gain=cap["gain"],
                bricks=[dict(slot=n_, type=POOL[n_], pos=p, quat_xyzw=R.as_quat(), role="tray" if n_ in slots else "placed")
                        for n_, (p, R) in bricks.items()], tray_slots=slots, tray=per,
                extrinsics_true=dict(top=ext(cap["pos"], cap["R"]), wrist_B=hov and ext(hov["pos"], hov["R"])),
                joint_q=dict(A=rig.park_q["A"], B=rig.park_q["B"])))
            f.write(dump(dict(scene=i, scale=s, bricks=per)) + "\n")
            f.flush()
            if i == 0:
                save_png(png, "top", cap["rgb"], cap["depth"], 0.9, 1.3)
                if hov:
                    save_png(png, "wrist", wr, wd, 0.05, 0.25)
            print("scene %d/%d: tray top-face px %s" % (i + 1, n, [p["top_px"] for p in per]), flush=True)
    (off / "scenes.json").write_text(dump(meta))
    (off / "labels.json").write_text(dump(labels))
    size = sum(p.stat().st_size for p in off.iterdir())
    print("offline set: %d scenes, %.1f MB (%.1f MB per scene) in %s" % (n, size / 1e6, size / 1e6 / max(n, 1), off))


def part_v5(scales, n, out, seed, shadows, choice, png):
    h, dx = choice
    f = (out / "v5.jsonl").open("w")
    for s in scales:
        rig = Rig(s, shadows)
        rng = np.random.default_rng(seed + int(s * 7))
        cands = []
        for name in STRUCTS:
            seq = P.sequence(P.STRUCTURES[name])
            cands += [(name, m, Scene(rig, (0, 0, 0), seq[:m], seq[m]).exposed()[1]) for m in range(len(seq))]
        n_braced, done, tries, skipped = round(0.4 * n), 0, 0, 0
        while done < n and tries < 5 * n:
            tries += 1
            braced, want_single = done < n_braced, done % 5 == 0
            pool = [c for c in cands if c[2] == want_single and (c[1] >= 1 or not braced)]
            name, m, single = pool[rng.integers(len(pool))]
            seq = P.sequence(P.STRUCTURES[name])
            sc = Scene(rig, rand_plate(rng), seq[:m], seq[m])
            g = P.grasp_for(seq[m], seq[:m])
            nx, ny = P.footprint(seq[m][1], seq[m][5])
            n_across = round((g["grip_width_m"] + 0.0012) / 0.008)
            grasp = "short" if n_across == min(nx, ny) or n_across * 0.008 * s > 0.07 else "long"
            fl = int(rng.integers(2))
            r = look_render(rig, sc, h * MM * s, dx * MM, grasp, braced, fl, rng)
            if not r["ik_ok"]:
                r = look_render(rig, sc, h * MM * s, dx * MM, grasp, braced, 1 - fl, rng)
            if not r["ik_ok"]:
                skipped += 1                      # kept in the file: the summary counts it as a gate (c) failure
                f.write(dump(dict(scale=s, i=-1, ik_skipped=True, structure=name, step=m, type=seq[m][1], grasp=grasp, braced=braced,
                                  single=bool(single), gate_c=False, h_mm=h, offset_mm=dx)) + "\n")
                continue
            sd = int(rng.integers(1 << 31))
            res = v5(rig, sc, r, sd)
            px = v5(rig, sc, r, sd, true_cam=True)                       # pixel-only terms: the same image, true camera pose
            res.update({"px_" + k: px[k] for k in ("t_err", "t_yaw_deg", "h_err", "h_yaw_deg", "rel_err", "rel_yaw_deg", "face_dx_um", "face_tilt_deg") if k in px})
            row = dict(scale=s, i=done, structure=name, step=m, type=seq[m][1], grasp=grasp, braced=braced, h_mm=h, offset_mm=dx,
                       gate_c=gate_c(r), **scalars(r), **scalars(res))
            f.write(dump(row) + "\n")
            f.flush()
            if done == 0 and png:
                save_png(png, "wrist_s%g" % s, r["rgb"], r["depth"], 0.05, 0.25)
                ov = r["rgb"][..., ::-1].copy()
                ov[r["mask"]] = (0.5 * ov[r["mask"]] + [127, 0, 0]).astype(np.uint8)
                for blobs, col in ((res["_blobs"], (0, 255, 0)), (res["_hblobs"], (0, 255, 255))):
                    for px, py, _ in zip(*project("wrist_B", r["npos"], r["nR"], blobs)) if len(blobs) else []:
                        cv2.circle(ov, (int(px), int(py)), 6, col, 1)
                cv2.imwrite(str(png / ("wrist_s%g_v5.png" % s)), ov)
            print("s=%g render %d/%d %s step %d %s%s: studs %d/%d vis, outline %d, face %d px, blobs %d, gate c %s" % (
                s, done + 1, n, name, m, "single " if r["single"] else "", "braced" if braced else "",
                r["n_studs_visible"], r["n_studs_expected"], r["outline_sides"], r["face_px"], res["n_blobs"], gate_c(r)), flush=True)
            done += 1
        print("s=%g: %d renders, %d skipped (IK)" % (s, done, skipped))
    f.close()


# --- summary: gates (a)-(d) from the result files ------------------------------------------------------------
def summary(out):
    rd = lambda n: [json.loads(x) for x in (out / n).read_text().splitlines()] if (out / n).exists() else []
    print("== P2 gates from %s" % out)
    t = out / "timing.json"
    if t.exists():
        for r in json.loads(t.read_text()):
            k = r["shadows_on"]
            top, wr = k["top"]["capture"]["median_ms"], k["wrist_B"]["capture"]["median_ms"]
            print("(a) s=%g: top %.1f ms (p95 %.1f) [<=50] %s | wrist %.1f ms (p95 %.1f) [<=10] %s | shadows off: %.1f / %.1f ms" % (
                r["scale"], top, k["top"]["capture"]["p95_ms"], "PASS" if top <= 50 else "FAIL", wr, k["wrist_B"]["capture"]["p95_ms"],
                "PASS" if wr <= 10 else "FAIL", r["shadows_off"]["top"]["capture"]["median_ms"], r["shadows_off"]["wrist_B"]["capture"]["median_ms"]))
    else:
        print("(a) no timing.json")
    if (out / "lookpose.json").exists():
        print("look pose:", json.dumps(json.loads((out / "lookpose.json").read_text())["chosen"]))
    tray = rd("tray.jsonl")
    if tray:
        b = [x for r in tray for x in r["bricks"]]
        ok = np.mean([x["ok"] for x in b])
        print("(b) %d scenes, %d tray bricks: %.2f%% have >=50 top-face px and height >= 3 sigma_depth [>=99%%] %s; min px %d" % (
            len(tray), len(b), 100 * ok, "PASS" if ok >= 0.99 else "FAIL", min(x["top_px"] for x in b)))
    else:
        print("(b) no tray.jsonl")
    v5r = rd("v5.jsonl")
    for s in sorted({r["scale"] for r in v5r}):
        allr = [r for r in v5r if r["scale"] == s]
        rs = [r for r in allr if not r.get("ik_skipped")]
        ns = len(allr) - len(rs)
        for nm, key in (("all", lambda r: True), ("course", lambda r: not r["single"]), ("single-footprint", lambda r: r["single"])):
            sel, sel_all = [r for r in rs if key(r)], [r for r in allr if key(r)]
            if not sel:
                continue
            fr = lambda k: 100 * np.mean([k(r) for r in sel])
            print("(c) s=%g %-17s n=%2d (+%d IK-skipped): gate c %.0f%% of rendered, %.0f%% counting skips as failures [>=95%%] | studs %.0f%% | "
                  "bump/sigma>=10 %.0f%% | face>=500px %.0f%% | held studs %.0f%% | braced %d" % (
                      s, nm, len(sel), len(sel_all) - len(sel), fr(lambda r: r["gate_c"]), 100 * np.mean([r["gate_c"] for r in sel_all]),
                      fr(lambda r: (r["n_studs_visible"] >= 2 and r["outline_sides"] >= 2) if r["single"] else r["n_studs_visible"] >= 4),
                      fr(lambda r: r["bump_sigma"] >= 10), fr(lambda r: r["face_px"] >= 500),
                      fr(lambda r: r["held_studs_visible"] >= min(4, r["held_studs"])), sum(r["braced"] for r in sel)))
        # the budget's target term is "with >= 4 studs"; fits on 2-3 studs (single footprint) are left out here
        sd = lambda key, ax=None, rows=rs: [(r[key] if ax is None else r[key][ax]) for r in rows if key in r and (
            not key.startswith(("t_", "rel_")) or r.get("t_n", 0) >= 4)]
        # a face-plane fit that is off by > 20 um or 0.5 deg with the true camera is contamination, not noise: counted,
        # and left out of the face statistics
        bad = [r for r in rs if "px_face_dx_um" in r and (abs(r["px_face_dx_um"]) > 20 or abs(r["px_face_tilt_deg"]) > 0.5)]
        good = [r for r in rs if r not in bad]
        for pre, tag in (("px_", "pixel-only (true camera)"), ("", "with hand-eye error (nominal camera)")):
            for lab, key, ax, unit in (("target lattice x", "t_err", 0, "mm"), ("target lattice y", "t_err", 1, "mm"),
                                       ("target lattice yaw", "t_yaw_deg", None, "deg"), ("held pose x", "h_err", 0, "mm"),
                                       ("held pose y", "h_err", 1, "mm"), ("held yaw", "h_yaw_deg", None, "deg"),
                                       ("target-held x", "rel_err", 0, "mm"), ("target-held y", "rel_err", 1, "mm"),
                                       ("target-held yaw", "rel_yaw_deg", None, "deg"), ("face offset", "face_dx_um", None, "um"),
                                       ("face tilt", "face_tilt_deg", None, "deg")):
                v = np.array(sd(pre + key, ax, good if key.startswith("face") else rs))
                if len(v):
                    print("(d) s=%g %-18s n=%2d: rms %.3f %s (bias %+.3f, std %.3f, p95 |e| %.3f)  [%s]" % (
                        s, lab, len(v), np.sqrt((v ** 2).mean()), unit, v.mean(), v.std(), np.percentile(np.abs(v), 95), tag))
        print("(d) s=%g face-plane outliers (|dx| > 20 um or |tilt| > 0.5 deg, true camera): %d of %d; held-brick fits found: %d of %d" % (
            s, len(bad), len(rs), sum(r.get("h_n", 0) >= 2 for r in rs), len(rs)))
        h = rs[0]["h_mm"] * s
        print("(d) s=%g analytic hand-eye rotation x look offset (0.1 deg x %g mm) = %.3f mm" % (s, h, 1.75e-3 * h))
    if not v5r:
        print("(c)/(d) no v5.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["timing", "lookpose", "tray", "v5", "all"], default="all")
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0, 2.0], choices=[1.0, 2.0])
    ap.add_argument("--n", type=int, help="timing: repeats (30); lookpose: grid points (12); tray: scenes (100); v5: renders per scale (50)")
    ap.add_argument("--out", type=Path, default=OUT, help="result directory (default results/v4/p2)")
    ap.add_argument("--png", type=Path, help="save one top and one wrist image (rgb, depth, V5 overlay) here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h-look", type=int, help="override the chosen look height, mm per s")
    ap.add_argument("--offset", type=int, help="override the chosen camera offset along hand x, mm")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.selfcheck or a.part in ("v5", "all") and not a.summary:
        selfcheck()
    if a.summary:
        summary(a.out)
    elif not a.selfcheck:
        sh, parts = not a.no_shadows, ("timing", "lookpose", "tray", "v5") if a.part == "all" else (a.part,)
        n = lambda d: a.n or d
        for part in parts:
            t0 = time.time()
            if part == "timing":
                part_timing(a.scale, n(30), a.out, a.seed, sh)
            elif part == "lookpose":
                part_lookpose(a.scale, n(12), a.out, a.seed, sh)
            elif part == "tray":
                h, dx = chosen_pose(a.out, a)
                part_tray(a.scale, n(100), a.out, a.seed, sh, dx, a.png)
            else:
                part_v5(a.scale, n(50), a.out, a.seed, sh, chosen_pose(a.out, a), a.png)
            print("[%s] %.0f s" % (part, time.time() - t0))
        summary(a.out)
