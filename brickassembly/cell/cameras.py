"""Simulated cameras (plan_v4_perception S1.1): the camera models of P2's probe (scripts/11_camera_probe.py), and
`Cameras`, which attaches them to the cell's own Newton model.

    CAMS, cam_pose, project, intrinsics, sense, to_points   the camera definitions, the sensor noise (plan_v4 §2.5.2)
    set_lights, mask_A, sphere_model, fr3_spheres            lights; A's own-arm mask from FK and the FR3 spheres
    Cameras(model, seed).capture(cam, ee_pose, state=...)    one noised capture of the current state

Ground truth (shape_index labels, the clean depth, the true camera pose) leaves only through `ground_truth`, for scoring. What `capture` returns by default is what an executor may read: noised rgb and depth, the
intrinsics and the NOMINAL camera pose (the mount and the hand FK, without the hidden extrinsic error).
"""

import math
import time
import warnings
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as Rot

import newton
import newton.geometry
from newton.sensors import SensorTiledCamera

HERE = Path(__file__).resolve().parent.parent
MM = 1e-3
CAMS = {"top": dict(w=1280, h=960, vfov=60.0), "wrist_B": dict(w=1280, h=960, vfov=55.0)}      # ACC remedy: wrist_B 1280x960 (same VFOV and mount; P2 ran 640x480)
TOP_POS = np.array([0.10, -0.20, 1.10])
TILT = math.radians(25.0)         # wrist camera tilt toward the tool axis
MOUNT_DZ = 0.060                  # camera above the TCP
WRIST_DX = 0.060                  # camera offset along hand x (P2's chosen look pose; the probe passes its own)
DEPTH_C = 2e-3                    # sigma_depth = C r^2  (plan §2.5.2)
EXT_ERR = {"top": (1 * MM, 0.2), "wrist_B": (0.2 * MM, 0.1)}   # hidden extrinsic error, 1 sigma (m, deg)


def sigma_depth(r):
    return DEPTH_C * r * r


def hidden_error(name, rng):
    """One draw of the hidden extrinsic error of a camera: (translation [m], rotation vector [rad]), 1 sigma per axis."""
    st, sr = EXT_ERR[name]
    return rng.normal(0, st, 3), np.radians(rng.normal(0, sr, 3))


def nominal_pose(name, hand=None, dx=0.045):
    """(pos, R): the camera's nominal pose in the world. Top: fixed. Wrist: the mount on the hand (pos, Rotation)."""
    if name == "top":
        return TOP_POS, Rot.identity()                      # looks along -z, image up = +y
    hp, hR = hand
    f = np.array([-math.sin(TILT), 0, math.cos(TILT)])      # hand frame: z along the tool, tilted toward the axis
    up = np.array([math.cos(TILT), 0, math.sin(TILT)])
    Rm = np.stack([[0, 1, 0], up, -f], 1)                   # columns: right, up, back (OpenGL camera)
    return hp + hR.apply([dx, 0, -MOUNT_DZ]), hR * Rot.from_matrix(Rm)


def cam_pose(name, hand=None, dx=0.045, rng=None, hidden=True):
    """(true_pos, true_R, nominal_pos, nominal_R). Hidden extrinsic error per seed (plan §2.5.2)."""
    pos, R = nominal_pose(name, hand, dx)
    if hidden and rng is not None:
        dt, dr = hidden_error(name, rng)
        return pos + dt, R * Rot.from_rotvec(dr), pos, R
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


def intrinsics(name):
    c = CAMS[name]
    f = c["h"] / (2 * math.tan(math.radians(c["vfov"]) / 2))
    return dict(width=c["w"], height=c["h"], vfov_deg=c["vfov"], fx=f, fy=f, cx=c["w"] / 2, cy=c["h"] / 2,
                convention="OpenGL camera (x right, y up, looks -z); pixel (px,py) ray = normalize(((px+.5)/W-.5)*2h*W/H, "
                           "-((py+.5)/H-.5)*2h, -1), h = tan(vfov/2); depth = distance along that ray [m], 0 = invalid")


# --- renderer: one tiled-camera sensor over a model -------------------------------------------------------------
def make_cams(sensor):
    """Per camera in CAMS: its pinhole rays and the sensor's output images."""
    u, out = sensor.utils, {}
    for name, c in CAMS.items():
        rays = u.compute_pinhole_camera_rays(c["w"], c["h"], math.radians(c["vfov"]))
        out[name] = dict(c, rays=rays, rays_np=rays.numpy()[0, :, :, 1], depth=u.create_depth_image_output(c["w"], c["h"]),
                         hdr=u.create_hdr_color_image_output(c["w"], c["h"]),
                         sidx=u.create_shape_index_image_output(c["w"], c["h"]),
                         nrm=u.create_normal_image_output(c["w"], c["h"]))
    return out


def set_lights(sensor, device, key_dir, ring=None):
    """Key: a directional light (random per seed). Ring: co-located spot at the camera, along its axis (the
    renderer's only point-like light: fixed 18-32 deg cone, no intensity parameter)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = sensor.render_context
    L = [(1, key_dir, (0, 0, 0))] + ([(0, ring[1], ring[0])] if ring else [])
    rc.lights_active = wp.array([True] * len(L), dtype=wp.bool, device=device)
    rc.lights_type = wp.array([x[0] for x in L], dtype=wp.int32, device=device)
    rc.lights_cast_shadow = wp.array([True] * len(L), dtype=wp.bool, device=device)
    rc.lights_position = wp.array([x[2] for x in L], dtype=wp.vec3f, device=device)
    rc.lights_orientation = wp.array([x[1] for x in L], dtype=wp.vec3f, device=device)


def render(sensor, model, state, c, pos, R, labels=False, download=True):
    """One render of `state` at the (true) camera pose: dict(depth [m, ray distance, 0 = no hit], hdr[, sidx, normal]).
    The shape BVH is refitted first. download=False renders only (and synchronises)."""
    newton.geometry.refit_bvh_shape(model, state)
    xf = wp.array([[wp.transformf(wp.vec3f(*pos), wp.quatf(*R.as_quat()))]], dtype=wp.transformf)
    sensor.update(state, xf, c["rays"], depth_image=c["depth"], hdr_color_image=c["hdr"],
                  shape_index_image=c["sidx"] if labels else None, normal_image=c["nrm"] if labels else None)
    if not download:
        wp.synchronize()
        return None
    out = dict(depth=c["depth"].numpy()[0, 0], hdr=c["hdr"].numpy()[0, 0])
    if labels:
        out["sidx"] = c["sidx"].numpy()[0, 0]
        out["normal"] = c["nrm"].numpy()[0, 0]
    return out


# --- A's own-arm mask: link spheres + FK, projected into a camera ---------------------------------------------
def sphere_model(model, nb, bodies, rmax=0.02):
    """Link spheres (body, centre in the link frame, radius) from the visual meshes of `bodies`: bisected vertices,
    radius <= rmax. `nb` is unused here (the bodies are given)."""
    shape_body = model.shape_body.numpy()

    def split(V):
        c = (V.min(0) + V.max(0)) / 2
        r = np.linalg.norm(V - c, axis=1).max()
        if r <= rmax or len(V) < 20:
            return [(c, r)]
        ax = int(np.ptp(V, 0).argmax())
        V = V[np.argsort(V[:, ax])]
        return split(V[:len(V) // 2]) + split(V[len(V) // 2:])
    out, scale, xf = [], model.shape_scale.numpy(), model.shape_transform.numpy()
    vis = model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)
    for b in bodies:
        V = [Rot.from_quat(xf[s, 3:]).apply(model.shape_source[s].vertices[::4] * scale[s]) + xf[s, :3]
             for s in range(model.shape_count) if shape_body[s] == b and vis[s]
             and model.shape_source[s] is not None]
        out += [(b, c, r) for c, r in split(np.concatenate(V))] if V else []
    return out


def fr3_spheres(model, nb):
    """P3's motion/fr3.yml link spheres (arms within 2 mm of the collision meshes, fingers 0.5 mm), by body index of
    the first arm (`nb` bodies)."""
    import yaml
    cs = yaml.safe_load((HERE / "motion" / "fr3.yml").read_text())["robot_cfg"]["kinematics"]["collision_spheres"]
    names = [str(l).split("/")[-1] for l in model.body_label[:nb]]
    return [(names.index(l), np.array(sp["center"]), sp["radius"]) for l, v in cs.items() if l in names for sp in v]


def mask_A(spheres, rays_np, pos, R, depth, links, pad=0.005):
    """A's padded link spheres, moved by A's FK (`links`: (positions, rotations) per link) and projected into the camera
    at its nominal pose: bool image. Uses nothing but proprioception, the calibrated camera and the depth image."""
    lp, lR = links
    rays = rays_np @ R.as_matrix().T
    hit = np.zeros(rays.shape[:2], bool)
    for b, c, r in spheres:
        oc = lR[b].apply(c) + lp[b] - pos
        if np.linalg.norm(oc) - r > 0.45:         # farther than any surface in a look image: cannot occlude
            continue
        bb = rays @ oc
        disc = bb * bb - oc @ oc + (r + pad) ** 2
        with np.errstate(invalid="ignore"):
            hit |= (disc > 0) & (bb > 0) & (bb - np.sqrt(np.maximum(disc, 0)) < depth + 0.01)
    return hit


# --- the cell's cameras ---------------------------------------------------------------------------------------
class Cameras:
    """The top camera and wrist_B, attached to the cell's own model (the cell's brick, plate and arm shapes are what
    they see). Per seed: the hidden extrinsic error (the top camera's in the world; wrist_B's as a fixed hand-eye
    error in the hand frame), the key light and its gain, and the noise stream.

    Rendering is an ordinary kernel launch on the model's current state: call it outside a captured CUDA graph (between
    graph launches is fine, the stream orders it)."""

    def __init__(self, model, seed, shadows=True):
        self.model, self.seed = model, seed
        newton.geometry.build_bvh_shape(model, model.state())      # sizes the BVH; every render refits it
        self.sensor = SensorTiledCamera(model, config=SensorTiledCamera.RenderConfig(enable_shadows=shadows))
        self.cams = make_cams(self.sensor)
        rng = np.random.default_rng(seed)
        self.hidden = {name: hidden_error(name, rng) for name in CAMS}
        self.key_dir = tuple(-np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)]))
        self.gain = float(rng.uniform(0.5, 2.0))
        self.noise = np.random.default_rng([seed, 1])
        self.shape_body = model.shape_body.numpy()
        self.timing = {name: [] for name in CAMS}                  # per capture: dict(render_ms, sense_ms)

    def poses(self, cam, ee_pose=None):
        """(true_pos, true_R, nominal_pos, nominal_R). wrist_B needs `ee_pose` = [x y z qx qy qz qw], B's TCP from FK."""
        hand = None
        if cam == "wrist_B":
            hand = (np.asarray(ee_pose[:3], float), Rot.from_quat(ee_pose[3:7]))
        npos, nR = nominal_pose(cam, hand, WRIST_DX)
        dt, dr = self.hidden[cam]
        if cam == "wrist_B":
            return npos + hand[1].apply(dt), nR * Rot.from_rotvec(dr), npos, nR
        return npos + dt, nR * Rot.from_rotvec(dr), npos, nR

    def _render(self, cam, ee_pose, state, labels):
        pos, R, npos, nR = self.poses(cam, ee_pose)
        if cam == "wrist_B":                                        # co-located ring light at the true camera
            set_lights(self.sensor, self.model.device, self.key_dir, (tuple(pos), tuple(R.apply([0, 0, -1]))))
        else:
            set_lights(self.sensor, self.model.device, self.key_dir)
        return render(self.sensor, self.model, state, self.cams[cam], pos, R, labels=labels), (pos, R, npos, nR)

    def capture(self, cam, ee_pose=None, *, state, frame=None, sim_time=None):
        """Render `state` from `cam` and add the sensor noise. -> dict(rgb uint8 (H, W, 3), depth float32 (H, W) [m, ray
        distance, 0 = invalid], intrinsics, pos / quat_xyzw (the nominal camera pose), frame, sim_time, render_ms,
        sense_ms)."""
        t0 = time.perf_counter()
        out, (_, _, npos, nR) = self._render(cam, ee_pose, state, False)
        t1 = time.perf_counter()
        depth, rgb = sense(out, self.noise, self.gain)
        t2 = time.perf_counter()
        res = dict(rgb=rgb, depth=depth, intrinsics=intrinsics(cam), pos=npos, quat_xyzw=nR.as_quat(), frame=frame,
                   sim_time=sim_time, render_ms=(t1 - t0) * 1e3, sense_ms=(t2 - t1) * 1e3)
        self.timing[cam].append(dict(render_ms=res["render_ms"], sense_ms=res["sense_ms"]))
        return res

    def ground_truth(self, cam, ee_pose=None, *, state):
        """Scoring only, never for acting: the same view with labels. -> dict(shape_index int32 (-1 = no hit), clean depth,
        normal, the TRUE camera pos / quat_xyzw, light_gain)."""
        out, (pos, R, _, _) = self._render(cam, ee_pose, state, True)
        si = out["sidx"]
        return dict(shape_index=np.where(si == 0xFFFFFFFF, -1, si.astype(np.int64)).astype(np.int32), depth=out["depth"],
                    normal=out["normal"], pos=pos, quat_xyzw=R.as_quat(), light_gain=self.gain)
