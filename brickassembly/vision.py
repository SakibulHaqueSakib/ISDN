"""V5 relative alignment, estimation half (plan_v4_perception §2.2a, §2.4; plan_v4 §2.5.3 V5).

One wrist look -> the target's lattice pose T̂ (xy, yaw) and the held brick's pose B̂ (xyz, yaw), both in the world,
plus the acceptance decision. Reads the depth image, the camera rays (intrinsics), the nominal camera pose, the hand
FK, A's mask and the plan; nothing else. Pure numpy / scipy / cv2: no newton, warp, torch, dual_arm_sim or cell.*
(the firewall runs the executor side with those blocked). Ground truth never enters here.

Units: metres and radians; world frame, z up; cameras OpenGL (x right, y up, looks -z); depth is ray distance, 0 = invalid.
Scale s = 1 is the only one V1 runs (D2); geometry and the length thresholds scale with it.

    res = v5_estimate(depth, rays_or_intrinsics, (cam_pos, cam_R), (hand_pos, hand_R), step_ctx, prior, mask)
    res.accepted, res.T_hat, res.B_hat, res.dpsi
    d_h = hand_offset(ee, R_hand, res.B_hat[:3]);  B_est(t) = brick_from_hand(ee(t), R_hand(t), d_h)
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
    disc_k=3.5, disc_top_pct=95, disc_top_band=0.3, disc_min_pts=10, disc_iters=3,   # top_disc (band x stud_h)
    blob_lo=0.5, blob_hi=1.4,        # stud pixels: height above the face in [lo, hi] x stud_h
    blob_min_px_target=40, blob_min_px_held=25, blob_mask_dilate=7, blob_aspect=2.5, blob_fill=0.5, blob_area_frac=0.7,
    held_margin=1.5 * MM, held_z_band=2 * MM, held_min_pts=100, held_hist_bin=2e-4, held_face_tol=0.35 * MM,
    held_res_band=0.3, held_sig_floor=0.06 * MM, held_iters=3,    # held_height (band: x stud_h, tol / z_band: x s)
    fit_gate=0.45, fit_iters=3, fit_refine_div=3,                 # fit_rigid: match gate x pitch; after pass 1 gate / div
    plane_tol0=0.6 * MM, plane_min_pts=30, plane_floor=0.05 * MM, plane_iters=3, plane_bin=2e-4,
    face_slab=2.5 * MM, face_end=2 * MM, face_bottom=0.1, face_top=0.3 * MM,   # held side-face selection (end / slab: x s)
    accept_min_target=4, accept_min_held=4, accept_span_pitch=1, accept_dpsi_deg=3.0,   # plan §2.4; fit RMS is not a criterion
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
    n_t, noncol, T_hat, t_rms = 0, False, np.full(3, np.nan), math.nan
    if ft:
        n_t, noncol, t_rms = len(ft["idx"]), _noncollinear(cells[ft["idx"]]), ft["rms"]
        T_hat = np.array([*ft["apply"](np.asarray(prior.target_xy))[0], prior.target_yaw + ft["dpsi"]])
    if n_t < _P["accept_min_target"]:
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
        B_hat = np.array([*fh["apply"](nom[0][:2])[0], nom[0][2] + dz, _yaw(nom[1]) + fh["dpsi"]])
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
    dpsi = _wrap(T_hat[2] - B_hat[3]) if ft and fh else math.nan
    if ft and fh and abs(math.degrees(dpsi)) > _P["accept_dpsi_deg"]:
        reasons.append("dpsi>%g" % _P["accept_dpsi_deg"])
    return V5Result(T_hat=T_hat, B_hat=B_hat, n_target_studs=n_t, target_noncollinear=noncol, n_held_studs=n_h, dpsi=dpsi,
                    t_rms=t_rms, h_rms=h_rms, accepted=not reasons, reject_reason=",".join(reasons), diag=diag)
