"""V5 relative alignment, estimation half (plan_v4_perception §2.2a, §2.4; plan_v4 §2.5.3 V5).

One wrist look -> the target's lattice pose T̂ (xy, yaw) and the held brick's pose B̂ (xyz, yaw), both in the world,
plus the acceptance decision. Reads the depth image, the camera rays (intrinsics), the nominal camera pose, the hand
FK, A's mask and the plan; nothing else. Pure numpy / scipy / cv2: no newton, warp, torch, dual_arm_sim or cell.*
(the firewall runs the executor side with those blocked). Ground truth never enters here.

Units: metres and radians; world frame, z up; cameras OpenGL (x right, y up, looks -z); depth is ray distance, 0 = invalid.
Scale s = 1 is the only one V1 runs (D2); geometry and the length thresholds scale with it.

    res = v5_estimate(depth, rays_or_intrinsics, (cam_pos, cam_R), (hand_pos, hand_R), step_ctx, prior, mask)
    res.accepted, res.T_hat, res.B_hat, res.dpsi, res.estimator
    d_h = hand_offset(ee, R_hand, res.B_hat[:3]);  B_est(t) = brick_from_hand(ee(t), R_hand(t), d_h)
    fz = fuse_looks([...accepted looks...])        the two looks of a step averaged (hand frame d_h, in-hand yaw, T): derivation in its docstring

ACC re-freeze: wrist_B 1280 x 960, look height 35 mm, and the edge fallback: on a single-footprint target (step_ctx.outline from the plan) whose stud
rule fails (< 4 matched or collinear studs), the support's top-face outline (wall points and face-to-ground transitions, a line per side) joins the
>= 2 matched studs in one small-angle fit; estimator "edge". Never tried when the stud rule passes. Two common-mode corrections for the camera's
hand-fixed extrinsic error: B_hat's z loses the camera's z offset read off the target's bare face (face_z_offset, PARAMS z_common_mode), and
fuse_looks removes the xy offset seen as the disagreement of T between the two looks (PARAMS fuse_common_mode).
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

MM = 1e-3
PARAMS = dict(                       # every threshold the estimator uses; frozen before any V1 run (PARAMS_HASH goes in the rows)
    depth_c=2e-3,                    # sigma_depth = depth_c r^2 (plan §2.5.2), sets the wall-point cut in top_disc
    pitch=0.008, brick_h=0.0096, stud_h=0.0017,                   # at s = 1
    disc_k=3.5, disc_top_pct=95, disc_top_band=0.3, disc_min_pts=40, disc_iters=3,   # top_disc (band x stud_h)
    blob_lo=0.5, blob_hi=1.4,        # stud pixels: height above the face in [lo, hi] x stud_h
    wrist_w=1280, wrist_h=960, h_look=0.035,          # ACC re-freeze: wrist_B resolution (cell/cameras.py, tests check it) and the look height (dual_arm_sim.H_LOOK)
    blob_min_px_target=160, blob_min_px_held=100, blob_mask_dilate=13, blob_aspect=2.5, blob_fill=0.5, blob_area_frac=0.7,   # pixel counts at 1280x960 (4 x the 640x480 values, mask dilation 2 x)
    held_margin=1.5 * MM, held_z_band=2 * MM, held_min_pts=400, held_hist_bin=2e-4, held_face_tol=0.35 * MM,
    held_res_band=0.3, held_sig_floor=0.06 * MM, held_iters=3,    # held_height (band: x stud_h, tol / z_band: x s)
    fit_gate=0.45, fit_iters=3, fit_refine_div=3,                 # fit_rigid: match gate x pitch; after pass 1 gate / div
    plane_tol0=0.6 * MM, plane_min_pts=120, plane_floor=0.05 * MM, plane_iters=3, plane_bin=2e-4,
    face_slab=2.5 * MM, face_end=2 * MM, face_bottom=0.1, face_top=0.3 * MM,   # held side-face selection (end / slab: x s)
    accept_min_target=4, accept_min_held=4, accept_span_pitch=1, accept_dpsi_deg=3.0,   # plan §2.4; fit RMS is not a criterion
    z_common_mode=0, zc_band=0.8 * MM, zc_r0=3 * MM, zc_r1=5.6 * MM, zc_min_pts=200,   # the camera's z offset, from the target's face (see face_z_offset); OFF: see results/v4/vision/acc_dev, hollow_box step 7
    fuse_common_mode=1, fuse_cm_min_sv=1.0,              # fuse_looks: remove the hand-fixed camera offset seen in T across the two looks (1 = on)
    # edge fallback (single-footprint targets whose stud rule fails): the support's top-face outline + the visible studs
    edge_face_tol=0.6 * MM,          # |z - z_face| below this: a top-face pixel; below -this: a drop (wall or lower ground)
    edge_wall_lo=1.0 * MM, edge_wall_hi=6.5 * MM, edge_wall_dist=1.0 * MM,   # wall points: h in (-hi, -lo); a drop pixel farther than this from its ray's face crossing saw the ground
    edge_gate=1.5 * MM, edge_along_margin=2 * MM,       # an edge sample joins the nearest predicted outline line within gate, away from its ends
    edge_min_samples=40, edge_min_cover=0.5, edge_min_span=8 * MM,   # per side: samples, and the span they cover along it (>= min_span, or cover x the side's usable length if shorter)
    edge_min_sides=2, edge_min_studs=2,                 # sides with a line fit; matched studs (distinct nodes)
    edge_sigma_stud=0.08 * MM, edge_sigma_wall=0.04 * MM, edge_sigma_pair=0.08 * MM, edge_max_sd_xy=0.15 * MM, edge_max_sd_yaw_deg=0.4, edge_max_rms=0.5 * MM,   # row noise (stud centre, wall-seen side, transition-seen side); the accepted covariance and residual
)
PARAMS_HASH = hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:16]
_P = PARAMS


@dataclass
class StepCtx:
    """What the plan knows for this step."""
    btype: str                       # held brick type, "2x4"
    z_face: float                    # world z of the target's support course top face (stud base)
    cells: np.ndarray                # (N, 2) integer lattice cells (i, j) of the exposed support studs, same order as Prior.nodes_xy
    held_in_hand: tuple              # nominal grasp: (offset (3,) in the hand frame, Rotation brick<-hand); brick origin = bottom centre
    s: float = 1.0
    outline: tuple = ()              # single-footprint targets only (the edge fallback): the support's drop sides, (cell index into `cells`, direction)
                                     # with direction 0 +u, 1 -u, 2 +v, 3 -v of the lattice; a side of an exposed cell whose neighbour is lower ground


@dataclass
class Prior:
    """V4-fine's pose applied to the plan: world xy of the exposed support studs, the target brick's footprint centre and yaw."""
    nodes_xy: np.ndarray             # (N, 2)
    target_xy: np.ndarray            # (2,)
    target_yaw: float
    sigma_xy: float = 0.4 * MM       # P2's V4-fine prior error, 1 sigma; documentation, the match gate is 0.45 pitch
    sigma_yaw_deg: float = 0.25


@dataclass
class V5Result:
    T_hat: np.ndarray                # (x, y, yaw) target brick in the world; nan if the target fit failed
    B_hat: np.ndarray                # (x, y, z, yaw) held brick origin (bottom centre, nominal-pose convention) in the world
    n_target_studs: int
    target_noncollinear: bool        # matched studs span >= 1 pitch on both lattice axes
    n_held_studs: int
    dpsi: float                      # wrap(T_hat yaw - B_hat yaw), modulo 180 deg (a footprint is 180-symmetric)
    t_rms: float                     # fit RMS (m), logged only
    h_rms: float
    accepted: bool
    reject_reason: str               # "" if accepted, else the failed clauses joined by ","
    diag: dict = field(default_factory=dict)   # n_blobs, held blobs, face plane vs nominal (logged only)
    params_hash: str = PARAMS_HASH
    estimator: str = "studs"         # "edge" when the target came from the outline fallback (the stud rule failed)

    def as_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


# --- the estimate pair -> the hand offset (plan §2.2a) ----------------------------------------------------------
def _mat(R):
    return R.as_matrix() if hasattr(R, "as_matrix") else np.asarray(R)


def hand_offset(ee_pos, R_hand, B_hat_pos):
    """d_h = R_hand^T (ee - B_hat): the full 3-D vector in the hand frame. Keeps the 13 mm lever, so tilt and yaw are modelled."""
    return _mat(R_hand).T @ (np.asarray(ee_pos) - np.asarray(B_hat_pos))


def brick_from_hand(ee_pos, R_hand, d_h):
    """B_est = ee - R_hand d_h."""
    return np.asarray(ee_pos) - _mat(R_hand) @ np.asarray(d_h)


def _mean_near(a, period=math.pi):
    """Mean of angles that agree modulo `period`: a[0] + the mean of the wrapped deviations from it (stays near a[0], no branch cut)."""
    a = np.asarray(a, float)
    return float(a[0] + np.mean([(x - a[0] + period / 2) % period - period / 2 for x in a]))


def fuse_looks(looks, ref=None, common_mode=None):
    """The estimate of a step from its accepted looks (ACC: two looks per step, the second with the hand turned 180 deg). Each look k is
    dict(T=(x, y, yaw) target, B=(x, y, z, yaw) held brick, ee=hand TCP (3,), R=hand rotation, psi_cmd=the hand's commanded yaw), all world.

    The grasp is rigid, so what is common to the looks lives in the hand frame, not the world: the hand-frame offset
    d_h,k = R_k^T (ee_k - B_k) and the in-hand yaw phi_k = B_yaw,k - psi_FK,k (psi_FK,k = the hand's FK yaw: the brick follows the hand
    as it IS, not as commanded; this is the 77b551d hand-tracking correction). The target is fixed in the world. So
        d_h = mean_k d_h,k          (hand frame; the 180 deg turn drops out)
        T   = mean_k T_k            (x, y; the yaw as an angle mean mod pi, a footprint is 180-symmetric)
        phi = mean_k phi_k  (mod pi),  psi_cmd = mean_k psi_cmd,k  (mod pi: the turned look's command is psi + pi - 1e-6)
        dpsi = wrap(T_yaw - phi - psi_cmd)        the yaw added to every remaining waypoint (those carry psi_cmd, so the hand ends at
                                                    psi_cmd + dpsi and the brick at psi_cmd + dpsi + phi = T_yaw once it tracks)
    One look gives the pre-ACC formula, wrap(T - B_yaw + (psi_FK - psi_cmd)) = wrap(T - phi - psi_cmd); with a perfect rigid grasp, any number
    of looks at any hand yaws and tracking offsets give the same d_h, dpsi and T as one perfect look. B (for logs and scoring) is the brick
    at the hand pose `ref` = (ee, R) (default: the last look's): ee_ref - R_ref d_h, yaw psi_FK,ref + phi.

    Common mode (PARAMS fuse_common_mode, `common_mode` overrides): the camera rides on the hand, so its hidden extrinsic error is a fixed
    hand-frame xy offset e that shifts T_k and B_k alike: T_k = T + R_k e, B_k = B_k,true + R_k e (xy, plus noise). In the pair (T, d_h) the
    two do not cancel across a 180 deg turn the way they do within one look: T_k flips with the hand (the mean cancels it) but d_h,k = ... - e does
    not (the mean keeps it), and a hand that ends at look 0's yaw then misses by R e, twice a single look's residual. The target is fixed in the
    world, so e is observable: T_k - R_k e must agree across the looks, (R_k - R_0)_xy e = T_k - T_0 (least squares over k >= 1, used when the
    hand yaws differ by >= ~60 deg: smallest singular value >= fuse_cm_min_sv). Then T_k <- T_k - R_k e and d_h,k <- d_h,k + [e, 0] before the
    averages. A yaw error of the camera shifts T_yaw and B_yaw alike whatever the hand yaw, so it cancels in dpsi as it is. One look, or looks
    less than ~60 deg apart: no correction. e is returned (mm in the row).
    -> dict(T (3,), d_h (3,), phi, psi_cmd, dpsi, B (4,), n, e_h (2,) or None)"""
    cm = bool(_P["fuse_common_mode"]) if common_mode is None else common_mode
    R = [Rot.from_matrix(_mat(l["R"])) for l in looks]
    Rm = [r.as_matrix() for r in R]
    psi = [_yaw(r) for r in R]
    Ts = [np.asarray(l["T"], float) for l in looks]
    d_k = [hand_offset(l["ee"], r, np.asarray(l["B"])[:3]) for l, r in zip(looks, R)]
    e_h = None
    if cm and len(looks) >= 2:
        A = np.vstack([Rm[k][:2, :2] - Rm[0][:2, :2] for k in range(1, len(looks))])
        if np.linalg.svd(A, compute_uv=False).min() >= _P["fuse_cm_min_sv"]:
            e_h = np.linalg.lstsq(A, np.concatenate([Ts[k][:2] - Ts[0][:2] for k in range(1, len(looks))]), rcond=None)[0]
            Ts = [np.array([*(t[:2] - m[:2, :2] @ e_h), t[2]]) for t, m in zip(Ts, Rm)]
            d_k = [d + np.array([*e_h, 0.0]) for d in d_k]
    d_h = np.mean(d_k, axis=0)
    phi = _mean_near([float(np.asarray(l["B"])[3]) - ps for l, ps in zip(looks, psi)])
    pc = _mean_near([l["psi_cmd"] for l in looks])
    Ty = _mean_near([t[2] for t in Ts])
    T = np.array([*np.mean([t[:2] for t in Ts], axis=0), Ty])
    e_ref, R_ref = (looks[-1]["ee"], R[-1]) if ref is None else (ref[0], Rot.from_matrix(_mat(ref[1])))
    B = np.array([*brick_from_hand(e_ref, R_ref, d_h), _yaw(R_ref) + phi])
    return dict(T=T, d_h=d_h, phi=phi, psi_cmd=pc, dpsi=_wrap(Ty - phi - pc), B=B, n=len(looks), e_h=e_h)


# --- geometry, rays, points -------------------------------------------------------------------------------------
def _geom(s):
    return SimpleNamespace(s=s, p=_P["pitch"] * s, H=_P["brick_h"] * s, sh=_P["stud_h"] * s)


def rays_from_intrinsics(w, h, vfov_deg):
    """Per-pixel unit rays, OpenGL camera: ray(px, py) = normalize(((px+.5)/W-.5) 2t W/H, -((py+.5)/H-.5) 2t, -1), t = tan(vfov/2)."""
    t = math.tan(math.radians(vfov_deg) / 2)
    u = ((np.arange(w) + 0.5) / w - 0.5) * 2 * t * w / h * np.ones((h, 1))
    v = -((np.arange(h)[:, None] + 0.5) / h - 0.5) * 2 * t * np.ones((1, w))
    r = np.stack([u, v, -np.ones_like(u)], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def to_points(depth, rays, pos, R):
    """Depth (ray distance) -> world points through the per-pixel camera rays; depth 0 -> nan."""
    return np.where((depth > 0)[..., None], np.asarray(pos) + (depth[..., None] * rays) @ _mat(R).T, np.nan)


def sigma_depth(r):
    return _P["depth_c"] * r * r


def _yaw(R):
    M = _mat(R)
    return math.atan2(M[1, 0], M[0, 0])


def _wrap(a):
    """To (-pi/2, pi/2]: yaw differences of a 180-symmetric footprint."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


def _dims(btype):
    return tuple(sorted(map(int, btype.split("x")), reverse=True))


def brick_studs(g, dims, pos, R):
    nx, ny = dims
    loc = np.array([[(i - (nx - 1) / 2) * g.p, (j - (ny - 1) / 2) * g.p, g.H + g.sh] for i in range(nx) for j in range(ny)])
    return Rot.from_matrix(_mat(R)).apply(loc) + pos          # stud tops, world xyz


def face_pts(g, dims, pose, f, n=8):
    """Grid on vertical face f (0 +x, 1 -x, 2 +y, 3 -y of the mesh) of a brick at `pose`, world."""
    L, W = dims
    u = np.linspace(-0.9, 0.9, n)
    a, z = np.meshgrid(u, np.linspace(0.1, 0.9, n))
    a, z = a.ravel(), z.ravel() * g.H
    sg = 1 if f % 2 == 0 else -1
    loc = np.c_[sg * L * g.p / 2 * np.ones_like(a), a * W * g.p / 2, z] if f < 2 else \
        np.c_[a * L * g.p / 2, sg * W * g.p / 2 * np.ones_like(a), z]
    return Rot.from_matrix(_mat(pose[1])).apply(loc) + pose[0]


# --- stud blobs, lattice fit, held-brick face -------------------------------------------------------------------
def top_disc(q, sh, pos, k=None):
    """Centre of a stud's top disc from its points q. A plane is fitted to the points near the top (it follows a tilted
    brick), points farther than k sigma_depth from it (the visible side wall) are dropped, and the rest are averaged
    with surface-area weights r^2/cos(theta): a plain pixel mean is pulled toward the camera (near side is denser)."""
    k = k or _P["disc_k"]
    sig = sigma_depth(np.linalg.norm(q.mean(0) - pos))
    sel = q[q[:, 2] >= np.percentile(q[:, 2], _P["disc_top_pct"]) - _P["disc_top_band"] * sh]
    c0 = sel.mean(0)
    for _ in range(_P["disc_iters"]):
        if len(sel) < _P["disc_min_pts"]:
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
    kd = _P["blob_mask_dilate"]
    near_bad = cv2.dilate(mask.astype(np.uint8), np.ones((kd, kd), np.uint8)) > 0
    L = lab[cap]
    order = np.argsort(L, kind="stable")
    pts, edges = P3[cap][order], np.searchsorted(L[order], np.arange(n + 1))
    nb = np.bincount(lab[near_bad & cap], minlength=n)
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if (a >= min_px and x > 0 and y > 0 and x + w < lab.shape[1] and y + h < lab.shape[0] and nb[i] == 0
                and max(w, h) < _P["blob_aspect"] * min(w, h) and a > _P["blob_fill"] * w * h):   # compact, disc-like: not a wall strip
            keep.append((top_disc(pts[edges[i]:edges[i + 1]], sh, pos), a))
    if not keep:
        return np.zeros((0, 3))
    med = np.median([a for _, a in keep])
    return np.array([c for c, a in keep if a >= _P["blob_area_frac"] * med]).reshape(-1, 3)


def held_height(P3, pose, dims, g):
    """Height of each pixel above the held brick's top face, for pixels over its nominal footprint (nan elsewhere):
    the face is the dominant level there (a plane fitted from the height histogram's mode, so it follows the in-hand
    tilt); the studs stand out of it. Uses only the nominal in-hand pose and the depth.
    -> (height image, face plane z = c0 + c1 x + c2 y in the nominal brick frame, or None)."""
    L, W = dims
    lb = Rot.from_matrix(_mat(pose[1])).inv().apply(np.nan_to_num(P3.reshape(-1, 3), nan=9.0) - pose[0])   # brick frame, z up from its bottom
    m = _P["held_margin"]
    reg = (np.abs(lb[:, 0]) < L * g.p / 2 + m) & (np.abs(lb[:, 1]) < W * g.p / 2 + m) & (np.abs(lb[:, 2] - g.H) < _P["held_z_band"] * g.s + g.sh)
    q = lb[reg]
    out = np.full(len(lb), np.nan)
    if len(q) < _P["held_min_pts"]:
        return out.reshape(P3.shape[:2]), None
    b = _P["held_hist_bin"]
    hst, e = np.histogram(q[:, 2], bins=np.arange(q[:, 2].min(), q[:, 2].max() + b, b))
    f = q[np.abs(q[:, 2] - e[hst.argmax()] - b / 2) < _P["held_face_tol"] * g.s]
    for _ in range(_P["held_iters"]):
        co = np.linalg.lstsq(np.c_[np.ones(len(f)), f[:, :2]], f[:, 2], rcond=None)[0]
        res = q[:, 2] - np.c_[np.ones(len(q)), q[:, :2]] @ co
        f = q[np.abs(res) < max(3 * res[np.abs(res) < _P["held_res_band"] * g.sh].std(), _P["held_sig_floor"])]
    out[reg] = q[:, 2] - np.c_[np.ones(len(q)), q[:, :2]] @ co
    return out.reshape(P3.shape[:2]), co


def face_z_offset(P3, h, nodes, mask, g):
    """The camera's z offset at the target (m): the median height h of the exposed cells' bare face (|h| < zc_band, between zc_r0 and zc_r1 of the
    nearest stud: not the stud, not the lower ground) over its plan height ctx.z_face. The camera rides on the hand, so a hand-frame z error moves T and B
    alike (it does not flip with the hand yaw): subtracting it from B_hat's z removes the bias. None if too few face points (no correction)."""
    ok = np.isfinite(h) & (np.abs(h) < _P["zc_band"]) & ~mask
    if not len(nodes) or ok.sum() < _P["zc_min_pts"]:
        return None
    q = P3[ok][:, :2]
    dmin = np.linalg.norm(q[:, None] - np.asarray(nodes)[None, :, :2], axis=2).min(1)
    sel = (dmin > _P["zc_r0"] * g.s) & (dmin < _P["zc_r1"] * g.s)
    return float(np.median(h[ok][sel])) if sel.sum() >= _P["zc_min_pts"] else None


def fit_rigid(nodes, obs, gate, iters=None):
    """obs ~ R(dpsi)(node - c0) + c0 + t, obs matched to the nearest prior node within `gate` (the prior fixes the
    lattice indices, plan §2.5.3 V5(a)). Needs >= 2 matches on distinct nodes. dict(dpsi, t, rms, n, idx, apply) or None;
    idx = the distinct prior nodes matched in the last pass."""
    iters = iters or _P["fit_iters"]
    c0, R, t = nodes.mean(0), np.eye(2), np.zeros(2)
    for _ in range(iters):
        pred = (nodes - c0) @ R.T + c0 + t
        d = np.linalg.norm(obs[:, None] - pred[None], axis=2)
        j = d.argmin(1)
        ok = d[np.arange(len(obs)), j] < (gate if _ == 0 else gate / _P["fit_refine_div"])   # after the first pass the prior error is gone
        if ok.sum() < 2 or len(set(j[ok])) < 2:
            return None
        A, B = nodes[j[ok]], obs[ok]
        U, _, Vt = np.linalg.svd((A - A.mean(0)).T @ (B - B.mean(0)))
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = B.mean(0) - c0 - R @ (A.mean(0) - c0)
    res = np.linalg.norm(B - ((A - c0) @ R.T + c0 + t), axis=1)
    return dict(dpsi=math.atan2(R[1, 0], R[0, 0]), t=t, rms=float(np.sqrt((res ** 2).mean())), n=int(ok.sum()),
                idx=np.array(sorted(set(j[ok]))), apply=lambda x: (np.atleast_2d(x) - c0) @ R.T + c0 + t)


def plane_fit(Q, tol0=None):
    """x = a + b y + c z through hand-frame points Q. With tol0, the fit starts from the points within tol0 of the
    densest 0.2 mm slice of x (contamination from the brick's top face or the fingers cannot capture it); then 3
    rounds of 3-sigma trimming. -> (a, b, c) or None."""
    bn = _P["plane_bin"]
    if tol0 is not None and len(Q):
        h, e = np.histogram(Q[:, 0], bins=np.arange(Q[:, 0].min(), Q[:, 0].max() + bn, bn))
        Q = Q[np.abs(Q[:, 0] - e[h.argmax()] - bn / 2) < tol0]
    for _ in range(_P["plane_iters"]):
        if len(Q) < _P["plane_min_pts"]:
            return None
        A = np.c_[np.ones(len(Q)), Q[:, 1], Q[:, 2]]
        co = np.linalg.lstsq(A, Q[:, 0], rcond=None)[0]
        res = Q[:, 0] - A @ co
        Q = Q[np.abs(res) < max(3 * res.std(), _P["plane_floor"])]
    return co


def _noncollinear(cells):
    c = np.asarray(cells, float)
    return bool((np.ptp(c, 0) >= _P["accept_span_pitch"]).all() and np.linalg.svd(c - c.mean(0), compute_uv=False)[1] > 1e-6)


# --- edge fallback (plan_v4_perception §2.4): the support's top-face outline + the visible studs ---------------------------------
def _J(x):
    """Rotate 2-D vectors (rows) by +90 deg."""
    x = np.atleast_2d(x)
    return np.stack([-x[:, 1], x[:, 0]], 1)


def outline_lines(nodes, outline, lam, p):
    """The plan's drop sides merged into lines (one per direction and line index; the lattice is rotated by `lam`, the prior's lattice yaw).
    -> [dict(n outward unit normal, t along unit, s a point on the line, lo, hi along range of the sides)], all in the prior's frame."""
    u = np.array([math.cos(lam), math.sin(lam)])
    e = [u, -u, _J(u)[0], -_J(u)[0]]
    groups = {}
    for idx, d in outline:
        s = nodes[idx] + e[d] * p / 2
        groups.setdefault((d, round(float(e[d] @ s), 4)), []).append(s)
    out = []
    for (d, _), ss in groups.items():
        t = _J(e[d])[0]
        a = [float(t @ (x - ss[0])) for x in ss]
        out.append(dict(n=e[d], t=t, s=ss[0], lo=min(a) - p / 2, hi=max(a) + p / 2))
    return out


def edge_samples(P3, h, rays, pos, R, z_face, mask):
    """(wall, pairs): world xy of points ON the top face's edge against the drop, from two kinds of evidence:
    - wall points: the support's visible side wall, h in (-wall_hi, -wall_lo): a vertical wall's xy is the edge itself (no pixel quantisation);
    - ground-seen-past-the-edge pairs: a face pixel and a neighbouring drop pixel whose ray reaches the lower ground (its measured point is more
      than edge_wall_dist from where the ray crosses the face plane, so it is not a wall hit): the edge lies between the face pixel's point and that
      crossing, so the midpoint is unbiased to a fraction of a pixel.
    Face: |h| < tol; drop: h < -tol (a stud, the held brick or a taller wall is neither). Pixels near A's mask are skipped; invalid pixels are
    neither face nor drop."""
    tol, kd = _P["edge_face_tol"], _P["blob_mask_dilate"]
    bad = cv2.dilate(mask.astype(np.uint8), np.ones((kd, kd), np.uint8)) > 0
    fin = np.isfinite(P3[..., 2])
    face, drop = fin & (np.abs(h) < tol) & ~bad, fin & (h < -tol) & ~bad
    rw = rays @ _mat(R).T
    wall, out = P3[fin & (h < -_P["edge_wall_lo"]) & (h > -_P["edge_wall_hi"]) & ~bad][:, :2], []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:, 1:], np.s_[:, :-1]), (np.s_[:-1], np.s_[1:]), (np.s_[1:], np.s_[:-1])):
        m = face[a] & drop[b]
        ray, q = rw[b][m], P3[b][m][:, :2]
        Q = np.asarray(pos) + ((z_face - pos[2]) / ray[:, 2])[:, None] * ray
        g = np.linalg.norm(q - Q[:, :2], axis=1) > _P["edge_wall_dist"]
        out.append((P3[a][m][g][:, :2] + Q[g, :2]) / 2)
    return wall, np.concatenate(out)


def edge_fit(P3, h, rays, cam, mask, ctx, prior, blobs, g):
    """The target from the outline sides and the matched studs, jointly: unknowns u = (t_x, t_y, theta) of the prior lattice (small angle about
    the nodes' centre c0, X(x) = x + t + theta J (x - c0)). A matched stud o_k of node n_k gives o_k - n_k = t + theta J (n_k - c0); an outline
    side with a line fit across = a + b along gives, at the two ends A of its sample span, n . (t + theta J (A - c0)) = a + b along_A. Each
    row is weighted by 1 / sigma (edge_sigma_stud / _wall / _pair: a visible wall is exact, a side seen only as a face-to-ground transition
    is quantised to a pixel). A side uses the wall points if it has edge_min_samples of them, else the transition pairs. Needs >=
    edge_min_sides sides (each >= edge_min_samples samples over >= edge_min_cover of its length), >= edge_min_studs matched studs, and a
    covariance ((A^T W A)^-1) and residual under the PARAMS limits.
    -> dict(ok, sides, n_studs, idx, rms, sd_xy, sd_yaw_deg[, t, theta, apply]) or None if it cannot be set up."""
    nodes, p = np.asarray(prior.nodes_xy, float), g.p
    c0 = nodes.mean(0)
    lines = outline_lines(nodes, ctx.outline, (prior.target_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4, p) if len(ctx.outline) else []
    if not lines or not len(blobs):
        return None
    d = np.linalg.norm(blobs[:, None, :2] - nodes[None], axis=2)
    j = d.argmin(1)
    ok = d[np.arange(len(blobs)), j] < _P["fit_gate"] * p
    pair = {}                                                            # node -> its nearest blob
    for b_, n_ in zip(np.nonzero(ok)[0], j[ok]):
        if n_ not in pair or d[b_, n_] < d[pair[n_], n_]:
            pair[n_] = b_
    rows, rhs, sig = [], [], []
    for n_, b_ in pair.items():
        r = nodes[n_] - c0
        rows += [[1, 0, -r[1]], [0, 1, r[0]]]
        rhs += list(blobs[b_, :2] - nodes[n_])
        sig += [_P["edge_sigma_stud"]] * 2
    (wall, trans), sides, gate, mg = edge_samples(P3, h, rays, cam[0], cam[1], ctx.z_face, mask), 0, _P["edge_gate"], _P["edge_along_margin"]
    for L in lines:
        for xy, sg in ((wall, _P["edge_sigma_wall"]), (trans, _P["edge_sigma_pair"])):
            r = xy - L["s"]
            ac, al = r @ L["n"], r @ L["t"]
            m = (np.abs(ac) < gate) & (al > L["lo"] + mg) & (al < L["hi"] - mg)
            if m.sum() < _P["edge_min_samples"]:
                continue
            ac, al, keep = ac[m], al[m], np.ones(m.sum(), bool)
            for _ in range(3):
                co = np.polyfit(al[keep], ac[keep], 1)
                res = ac - np.polyval(co, al)
                keep = np.abs(res) < max(3 * res[keep].std(), 0.05 * MM)
            if keep.sum() < _P["edge_min_samples"] or np.ptp(al[keep]) < min(_P["edge_min_span"], _P["edge_min_cover"] * (L["hi"] - L["lo"] - 2 * mg)):
                continue
            for a in (al[keep].min(), al[keep].max()):
                rows.append([*L["n"], float(L["n"] @ _J(L["s"] + L["t"] * a - c0)[0])])
                rhs.append(float(np.polyval(co, a)))
                sig.append(sg)
            sides += 1
            break
    out = dict(ok=False, sides=sides, n_studs=len(pair), idx=np.array(sorted(pair)), rms=math.nan, sd_xy=math.nan, sd_yaw_deg=math.nan)
    if sides < _P["edge_min_sides"] or len(pair) < _P["edge_min_studs"]:
        return out
    w = 1 / np.array(sig)
    A, y = np.array(rows, float) * w[:, None], np.array(rhs, float) * w
    u, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    if rank < 3:
        return out
    sd = np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    rms = float(np.sqrt((((A @ u - y) / w) ** 2).mean()))
    out.update(rms=rms, sd_xy=float(sd[:2].max()), sd_yaw_deg=math.degrees(sd[2]), t=u[:2], theta=float(u[2]),
               apply=lambda x: np.atleast_2d(x) + u[:2] + u[2] * _J(np.atleast_2d(x) - c0))
    out["ok"] = bool(out["sd_xy"] <= _P["edge_max_sd_xy"] and out["sd_yaw_deg"] <= _P["edge_max_sd_yaw_deg"] and rms <= _P["edge_max_rms"])
    return out


# --- V5 ----------------------------------------------------------------------------------------------------------
def v5_estimate(depth, rays_or_intrinsics, cam_pose_nominal, hand_pose_fk, step_ctx, prior, mask=None):
    """One look image -> V5Result. `rays_or_intrinsics`: (H, W, 3) unit camera rays, or dict(w, h, vfov) (degrees).
    Poses are (pos (3,), R) with R a scipy Rotation or a 3x3 matrix; the camera pose is the *nominal* one."""
    g, ctx = _geom(step_ctx.s), step_ctx
    rays = rays_or_intrinsics if isinstance(rays_or_intrinsics, np.ndarray) else rays_from_intrinsics(
        *(rays_or_intrinsics[k] for k in ("w", "h", "vfov")))
    mask = np.zeros(depth.shape, bool) if mask is None else mask
    pos, hp, hR = np.asarray(cam_pose_nominal[0]), np.asarray(hand_pose_fk[0]), Rot.from_matrix(_mat(hand_pose_fk[1]))
    P3 = to_points(depth, rays, pos, cam_pose_nominal[1])
    gate, lo, hi = _P["fit_gate"] * g.p, _P["blob_lo"] * g.sh, _P["blob_hi"] * g.sh
    reasons, diag = [], {}
    nodes, cells = np.asarray(prior.nodes_xy, float), np.asarray(ctx.cells)
    # (a) target course: lattice fit to the stud blobs; the prior fixes the lattice indices
    b = stud_blobs(P3, P3[..., 2] - ctx.z_face, mask, lo, hi, g.sh, pos, _P["blob_min_px_target"])
    ft = fit_rigid(nodes, b[:, :2], gate) if len(b) >= 2 and len(nodes) >= 2 else None
    n_t, noncol, T_hat, t_rms, est = 0, False, np.full(3, np.nan), math.nan, "studs"
    if ft:
        n_t, noncol, t_rms = len(ft["idx"]), _noncollinear(cells[ft["idx"]]), ft["rms"]
        T_hat = np.array([*ft["apply"](np.asarray(prior.target_xy))[0], prior.target_yaw + ft["dpsi"]])
    if (n_t < _P["accept_min_target"] or not noncol) and len(ctx.outline):      # the stud rule fails on a single-footprint target: the edge fallback
        fe = edge_fit(P3, P3[..., 2] - ctx.z_face, rays, (pos, cam_pose_nominal[1]), mask, ctx, prior, b, g)
        if fe is not None:
            diag["edge"] = {k: fe[k] for k in ("ok", "sides", "n_studs", "rms", "sd_xy", "sd_yaw_deg")}
        if fe is not None and fe["ok"]:
            est, n_t, noncol, t_rms = "edge", fe["n_studs"], _noncollinear(cells[fe["idx"]]), fe["rms"]
            T_hat = np.array([*fe["apply"](np.asarray(prior.target_xy))[0], prior.target_yaw + fe["theta"]])
    if est == "edge":
        pass
    elif n_t < _P["accept_min_target"]:
        reasons.append("target_studs<%d" % _P["accept_min_target"])
    elif not noncol:
        reasons.append("target_collinear")
    # (b) held brick: its top studs (nominal in-hand pose as prior) and its camera-side face
    dims, (off, Rin) = _dims(ctx.btype), ctx.held_in_hand
    nom = (hp + hR.apply(off), hR * Rot.from_matrix(_mat(Rin)))
    h_img, co = held_height(P3, nom, dims, g)
    bh = stud_blobs(P3, h_img, mask, lo, hi, g.sh, pos, _P["blob_min_px_held"])
    fh = fit_rigid(brick_studs(g, dims, *nom)[:, :2], bh[:, :2], gate) if len(bh) >= 2 else None
    n_h, h_rms, B_hat = 0, math.nan, np.full(4, np.nan)
    if fh:
        n_h, h_rms = len(fh["idx"]), fh["rms"]
        dz = (co[0] - g.H) * _mat(nom[1])[2, 2] if co is not None else 0.0
        zc = face_z_offset(P3, P3[..., 2] - ctx.z_face, nodes, mask, g) if _P["z_common_mode"] else None
        if _P["z_common_mode"]:
            diag["face_dz_um"] = None if zc is None else zc * 1e6                # the camera's z offset at the target, removed from B_hat's z
        B_hat = np.array([*fh["apply"](nom[0][:2])[0], nom[0][2] + dz - (zc or 0.0), _yaw(nom[1]) + fh["dpsi"]])
    if n_h < _P["accept_min_held"]:
        reasons.append("held_studs<%d" % _P["accept_min_held"])
    # held side face vs the nominal face (logged only: tilt > 3 deg means regrasp, plan §2.5.3 V5(c))
    Rn = hR.inv() * nom[1]
    nrm = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    f = max(range(4), key=lambda f: Rn.apply(nrm[f])[0])
    Qn, n_h_ax = hR.inv().apply(face_pts(g, dims, nom, f) - hp), Rn.apply(nrm[f])
    ok = np.isfinite(P3[..., 2]).ravel()
    Q = hR.inv().apply(P3.reshape(-1, 3)[ok] - hp)
    half = (dims[1] if f < 2 else dims[0]) * g.p / 2 - _P["face_end"] * g.s
    sel = ((np.abs((Q - Qn.mean(0)) @ n_h_ax) < _P["face_slab"] * g.s) & (np.abs(Q[:, 1] - Qn[:, 1].mean()) < half)
           & (Q[:, 2] > Qn[:, 2].min() + _P["face_bottom"] * g.H) & (Q[:, 2] < Qn[:, 2].max() + _P["face_top"] * g.s))   # not the top face
    ce, cn = plane_fit(Q[sel], _P["plane_tol0"] * g.s), plane_fit(Qn)
    diag.update(n_blobs=len(b), n_held_blobs=len(bh), face_n=int(sel.sum()))
    if ce is not None and cn is not None:
        zm = Qn[:, 2].mean()
        diag.update(face_dx_um=((ce[0] + ce[2] * zm) - (cn[0] + cn[2] * zm)) * 1e6,
                    face_tilt_deg=math.degrees(math.atan(ce[2]) - math.atan(cn[2])),
                    face_yaw_deg=math.degrees(math.atan(ce[1]) - math.atan(cn[1])))
    # acceptance (plan §2.4): >= 4 non-collinear target studs, >= 4 held studs, |dpsi| <= 3 deg; fit RMS is not a criterion
    have_T = bool(ft) or est == "edge"
    dpsi = _wrap(T_hat[2] - B_hat[3]) if have_T and fh else math.nan
    if have_T and fh and abs(math.degrees(dpsi)) > _P["accept_dpsi_deg"]:
        reasons.append("dpsi>%g" % _P["accept_dpsi_deg"])
    return V5Result(T_hat=T_hat, B_hat=B_hat, n_target_studs=n_t, target_noncollinear=noncol, n_held_studs=n_h, dpsi=dpsi,
                    t_rms=t_rms, h_rms=h_rms, accepted=not reasons, reject_reason=",".join(reasons), diag=diag, estimator=est)
