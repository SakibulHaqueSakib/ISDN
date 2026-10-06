"""vision.py V5 (plan_v4_perception §2.2a, §2.4): no GPU for everything except test_regression_vs_probe.

  - the probe's synthetic lattice self-check, against vision.py;
  - the acceptance rule end to end on a synthetic wrist look (target lattice + held brick, straight-down camera);
  - hand_offset / brick_from_hand (the full 3-D lever);
  - vision.py imports with newton and warp blocked;
  - regression: render P2-style look scenes through the probe's rig and compare vision.v5_estimate with the probe's own
    v5() on the same images (needs the Isaac interpreter and a GPU; skipped when the probe cannot be imported).

    bash scripts/run.sh tests/test_vision.py          # or: python -m pytest tests/test_vision.py
"""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import vision as V  # noqa: E402

MM = 1e-3
GRASP_DZ = 0.013
P_, H_, SH_ = V.PARAMS["pitch"], V.PARAMS["brick_h"], V.PARAMS["stud_h"]
RX_PI = Rot.from_euler("x", math.pi)
Rz = lambda a: Rot.from_euler("z", a)


# --- the probe's self-check, re-expressed ----------------------------------------------------------------------
def test_selfcheck_lattice_fit():
    """A straight-down depth camera over a stud lattice (unknown pose within the prior's capture range): the blob
    detector plus fit_rigid recover the lattice to < 0.05 mm and 0.1 deg."""
    rng = np.random.default_rng(0)
    p, sh, r_st, h_cam = 0.008, 0.0017, 0.0024, 0.12
    rays = V.rays_from_intrinsics(320, 320, 40.0)
    psi, off = math.radians(7.0), np.array([0.0013, -0.0021])
    Rl = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
    nodes = np.array([(i - 3, j - 3) for i in range(7) for j in range(7)]) * p
    true = nodes @ Rl.T + off
    depth = h_cam / -rays[..., 2]                                           # table plane z = 0
    xy = rays[..., :2] * (h_cam - sh) / -rays[..., 2:3]                    # where each ray meets the stud-top plane
    d2 = np.linalg.norm(xy[:, :, None] - true[None, None], axis=-1).min(-1)
    depth = np.where(d2 < r_st, (h_cam - sh) / -rays[..., 2], depth)
    depth += rng.normal(0, V.sigma_depth(h_cam), depth.shape)
    cam = np.array([0, 0, h_cam])
    P3 = V.to_points(depth, rays, cam, np.eye(3))
    blobs = V.stud_blobs(P3, P3[..., 2], np.zeros(depth.shape, bool), 0.5 * sh, 1.4 * sh, sh, cam)
    assert len(blobs) >= 30, len(blobs)
    c = math.radians(0.3)
    Rp = np.array([[math.cos(psi + c), -math.sin(psi + c)], [math.sin(psi + c), math.cos(psi + c)]])
    pr = nodes @ Rp.T + off + [3e-4, -3e-4]
    ft = V.fit_rigid(pr, blobs[:, :2], 0.45 * p)
    e = ft["apply"](pr) - true
    assert ft["n"] >= 40 and np.abs(e).max() < 5e-5, (ft["n"], np.abs(e).max())
    assert abs(ft["dpsi"] + c) < math.radians(0.1), ft["dpsi"]
    ft2 = V.fit_rigid(pr, blobs[:, :2][blobs[:, 0] < -0.002], 0.45 * p)      # half the studs occluded: still recovered
    assert ft2 is not None and np.abs(ft2["apply"](pr) - true).mean() < 1e-4
    print("selfcheck: %d blobs, fit max err %.1f um, yaw err %.3f deg" % (len(blobs), np.abs(e).max() * 1e6, math.degrees(abs(ft["dpsi"] + c))))


# --- acceptance rule, end to end on a synthetic look -----------------------------------------------------------
W, Hh, Z0, HC = 640, 480, 0.0032, 0.14          # image, plate top, camera height (straight down, nominal = true)
CAM = (np.array([0.0, 0.0, HC]), np.eye(3))
TARGET_O, HELD_O = np.array([-0.035, 0.0]), np.array([0.030, 0.0])      # world xy of the lattice origin / the held brick


def look(cells, held_dyaw_deg=1.0, mask_x=None, seed=0):
    """-> (depth, ctx, prior, mask, truth). Target: studs on `cells` (i, j) of a lattice with yaw 0.3 deg, top faces at
    Z0 + stud height; held: a 2x4 whose actual pose is the nominal + (1.2, -0.6) mm, 0.3 mm up, held_dyaw_deg."""
    rng = np.random.default_rng(seed)
    rays = V.rays_from_intrinsics(W, Hh, 55.0)
    cells = np.asarray(cells)
    psi_t = math.radians(0.3)
    nodes = (cells + 0.5) * P_ @ Rz(psi_t).as_matrix()[:2, :2].T + TARGET_O
    ctr_loc = np.array([3.0, 2.0]) * P_                                     # target footprint centre: 2x4 on the patch
    T_ctr = ctr_loc @ Rz(psi_t).as_matrix()[:2, :2].T + TARGET_O
    zb = Z0 + SH_ + 0.045                                                   # held brick bottom: 45 mm over the stud tops
    hand = (np.array([*HELD_O, zb + GRASP_DZ]), Rz(0.0) * RX_PI)
    nom = (hand[0] + hand[1].apply([0, 0, GRASP_DZ]), Rz(0.0))
    act = (nom[0] + [1.2 * MM, -0.6 * MM, 0.3 * MM], Rz(math.radians(held_dyaw_deg)))
    hstuds = V.brick_studs(V._geom(1.0), (4, 2), *act)
    # nearest of: table, target stud tops, held top face, held stud tops (all horizontal planes, straight-down camera)
    rz, best = -rays[..., 2], np.full((Hh, W), np.inf)
    for zp, test in ((Z0, lambda xy: np.ones(len(xy), bool)),
                     (Z0 + SH_, lambda xy: np.linalg.norm(xy[:, None] - nodes[None], axis=2).min(1) < 0.0024),
                     (act[0][2] + H_, lambda xy: (np.abs(act[1].inv().apply(np.c_[xy, np.zeros(len(xy))] - [*act[0][:2], 0])[:, 0]) < 2 * P_)
                      & (np.abs(act[1].inv().apply(np.c_[xy, np.zeros(len(xy))] - [*act[0][:2], 0])[:, 1]) < 1 * P_)),
                     (act[0][2] + H_ + SH_, lambda xy: np.linalg.norm(xy[:, None] - hstuds[None, :, :2], axis=2).min(1) < 0.0024)):
        d = (HC - zp) / rz
        xy = (CAM[0] + d[..., None] * rays)[..., :2].reshape(-1, 2)
        best = np.minimum(best, np.where(test(xy).reshape(d.shape), d, np.inf))
    best += rng.normal(0, V.sigma_depth(best), best.shape)
    mask = np.zeros((Hh, W), bool)
    if mask_x is not None:
        mask = V.to_points(best, rays, *CAM)[..., 0] > mask_x
    pr_yaw, pr_t = math.radians(-0.2), np.array([0.3 * MM, -0.25 * MM])    # V4-fine's error on the lattice
    f = lambda x: (x - nodes.mean(0)) @ Rz(pr_yaw).as_matrix()[:2, :2].T + nodes.mean(0) + pr_t
    prior = V.Prior(f(nodes), f(T_ctr), psi_t + pr_yaw)
    ctx = V.StepCtx("2x4", Z0, cells, (np.array([0, 0, GRASP_DZ]), RX_PI))
    return dict(depth=best, rays=rays, hand=hand, ctx=ctx, prior=prior, mask=mask,
                truth=dict(T=np.r_[T_ctr, psi_t], B=np.r_[act[0], math.radians(held_dyaw_deg)]))


def run(**kw):
    s = look(**kw)
    res = V.v5_estimate(s["depth"], s["rays"], CAM, s["hand"], s["ctx"], s["prior"], s["mask"])
    return res, s["truth"]


PATCH = [(i, j) for i in range(6) for j in range(4)]


def test_accepted_and_accurate():
    res, t = run(cells=PATCH)
    assert res.accepted, res.reject_reason
    assert res.n_target_studs == 24 and res.n_held_studs == 8 and res.target_noncollinear
    assert np.abs(res.T_hat[:2] - t["T"][:2]).max() < 5e-5 and abs(res.T_hat[2] - t["T"][2]) < math.radians(0.1)
    assert np.abs(res.B_hat[:3] - t["B"][:3]).max() < 1e-4 and abs(res.B_hat[3] - t["B"][3]) < math.radians(0.15)
    assert abs(res.dpsi - (t["T"][2] - t["B"][3])) < math.radians(0.15)
    assert res.params_hash == V.PARAMS_HASH and isinstance(res.as_dict()["T_hat"], list)
    print("accepted look: T err %.1f um, B err %.1f um, dpsi %.3f deg" % (
        1e6 * np.abs(res.T_hat[:2] - t["T"][:2]).max(), 1e6 * np.abs(res.B_hat[:3] - t["B"][:3]).max(), math.degrees(res.dpsi)))


def test_four_collinear_target_studs_rejected():
    res, _ = run(cells=[(0, 0), (1, 0), (2, 0), (3, 0)])
    assert res.n_target_studs == 4 and not res.target_noncollinear
    assert not res.accepted and "target_collinear" in res.reject_reason


def test_three_target_studs_rejected():
    res, _ = run(cells=[(0, 0), (1, 0), (0, 1)])
    assert res.n_target_studs == 3 and not res.accepted and "target_studs<4" in res.reject_reason


def test_four_noncollinear_studs_accepted():
    res, _ = run(cells=[(0, 0), (1, 0), (0, 1), (1, 1)])
    assert res.n_target_studs == 4 and res.target_noncollinear and res.accepted, res.reject_reason


def test_held_studs_below_four_rejected():
    res, _ = run(cells=PATCH, mask_x=0.020)                  # A's mask over the held brick's far studs
    assert res.n_held_studs < 4 and not res.accepted and "held_studs<4" in res.reject_reason


def test_dpsi_threshold():
    ok, _ = run(cells=PATCH, held_dyaw_deg=0.3 - 2.0)        # dpsi = T - B = +2 deg
    assert ok.accepted and abs(math.degrees(ok.dpsi) - 2.0) < 0.15, (ok.reject_reason, math.degrees(ok.dpsi))
    bad, _ = run(cells=PATCH, held_dyaw_deg=0.3 - 4.0)       # +4 deg
    assert not bad.accepted and bad.reject_reason == "dpsi>3" and abs(math.degrees(bad.dpsi) - 4.0) < 0.15
    assert bad.n_target_studs == 24 and bad.n_held_studs == 8        # rejected on yaw alone; fit RMS is not a criterion


# --- hand offset ------------------------------------------------------------------------------------------------
def test_hand_offset_round_trip():
    rng = np.random.default_rng(1)
    R = Rz(math.radians(30)) * RX_PI * Rot.from_euler("y", math.radians(3))      # yawed 30 deg, tilted 3 deg
    ee = rng.normal(0, 0.1, 3)
    B = ee + R.apply([0.4 * MM, -0.7 * MM, -GRASP_DZ + 0.2 * MM])                  # brick, 13 mm below the TCP in the hand frame
    d = V.hand_offset(ee, R, B)
    assert np.abs(V.brick_from_hand(ee, R, d) - B).max() < 1e-9
    R2 = Rz(math.radians(41)) * RX_PI * Rot.from_euler("x", math.radians(-2))     # the grasp is rigid: d_h carries to a new pose
    ee2 = ee + [0.05, -0.02, 0.03]
    assert np.abs(V.brick_from_hand(ee2, R2, d) - (ee2 - R2.apply(R.inv().apply(ee - B)))).max() < 1e-9
    assert np.abs(V.hand_offset(ee, R.as_matrix(), B) - d).max() < 1e-12            # matrix input


def test_dropping_the_z_lever_costs_0p227_mm_at_1_deg():
    R = Rot.from_euler("y", math.radians(1.0)) * RX_PI                              # hand tilted 1 deg
    ee = np.array([0.3, -0.1, 0.2])
    B = ee - R.apply([0, 0, GRASP_DZ])                                              # brick exactly GRASP_DZ along the tool axis
    old = ee - np.array([0, 0, GRASP_DZ])                                           # the old world-frame [0, 0, GRASP_DZ] subtraction
    lat = np.linalg.norm((old - B)[:2])
    assert abs(lat - GRASP_DZ * math.sin(math.radians(1.0))) < 1e-9 and 0.226e-3 < lat < 0.228e-3, lat
    assert np.abs(V.brick_from_hand(ee, R, V.hand_offset(ee, R, B)) - B).max() < 1e-9   # the full vector keeps it
    print("z-lever dropped at 1 deg: %.3f mm lateral" % (lat * 1e3))


# --- import guard -----------------------------------------------------------------------------------------------
def test_imports_without_newton_or_warp():
    code = ("import sys; [sys.modules.__setitem__(m, None) for m in ('newton', 'warp', 'torch')]; sys.path.insert(0, %r); "
            "import vision; bad = [m for m in ('dual_arm_sim', 'cell') if m in sys.modules]; assert not bad, bad; print(vision.PARAMS_HASH)"
            % str(HERE))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == V.PARAMS_HASH, r.stderr[-500:]


# --- regression against the probe's own V5 on rendered P2 look scenes -----------------------------------------
def _probe():
    spec = importlib.util.spec_from_file_location("camera_probe", HERE / "scripts" / "11_camera_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_regression_vs_probe():
    try:
        pr = _probe()
        rig = pr.Rig(1.0, shadows=True)
    except Exception as e:                                   # no newton / GPU / probe mid-edit: not a vision.py failure
        pytest.skip("probe rig unavailable: %r" % (e,))
    rng = np.random.default_rng(5)
    diffs, errs = [], []
    for _ in range(60):
        if len(errs) >= 10:
            break
        name = pr.STRUCTS[rng.integers(len(pr.STRUCTS))]
        seq = pr.P.sequence(pr.P.STRUCTURES[name])
        m = int(rng.integers(len(seq)))
        sc = pr.Scene(rig, pr.rand_plate(rng), seq[:m], seq[m])
        cells, single, _ = sc.exposed()
        if single or len(cells) < 4:
            continue
        r = pr.look_render(rig, sc, 45 * pr.MM, 60 * pr.MM, "short", m >= 1 and rng.random() < 0.5, int(rng.integers(2)), rng)
        if not r["ik_ok"]:
            continue
        sd = int(rng.integers(1 << 31))
        old = pr.v5(rig, sc, r, sd)
        if old.get("t_n", 0) < 4 or "rel_err" not in old:
            continue
        # the same V4 prior as the probe's v5 draws it
        g = np.random.default_rng(sd)
        nodes = sc.world_xy(np.array([c[0] + .5 for c in cells]), np.array([c[1] + .5 for c in cells]))
        dpsi_p, t_p = math.radians(g.normal(0, 0.25)), g.normal(0, 0.4 * pr.MM, 2)
        c_t = nodes.mean(0)
        tp, tyaw = sc.brick_pose(sc.target)
        f = lambda x: (x - c_t) @ pr.Rz(dpsi_p).as_matrix()[:2, :2].T + c_t + t_p
        prior = V.Prior(f(nodes), f(np.array(tp[:2])), tyaw.as_euler("xyz")[2] + dpsi_p)
        bt = sc.target[1]
        ctx = V.StepCtx(bt, sc.z_face(sc.target[4]), np.array(cells), (np.array([0, 0, rig.dz_grasp]), pr.RX_PI * pr.Rz(0.0)))
        hp, hR = r["hand"]
        # held_nom is hand * (offset, RX_PI Rz(yl)); the probe's grasp 'short' has yl = 0
        ctx.held_in_hand = (np.array([0, 0, rig.dz_grasp]), pr.RX_PI)
        res = V.v5_estimate(r["depth"], rig.cams["wrist_B"]["rays_np"], (r["npos"], r["nR"]), (hp, hR), ctx, prior, r["mask"])
        assert res.accepted or res.n_held_studs < 4, res.reject_reason
        if not res.accepted:
            continue
        # error of (target - held) in the hand frame against the truth; the held reference is the actual brick's stud centroid
        # shifted like the nominal origin is from the nominal stud centroid (B_hat's convention)
        ax = hR.as_matrix()[:2, :2]
        bs = lambda pose: V.brick_studs(V._geom(rig.s), V._dims(bt), *pose).mean(0)[:2]
        act_c = bs(r["held_act"]) - (bs(r["held_nom"]) - r["held_nom"][0][:2])
        e = ((res.T_hat[:2] - res.B_hat[:2]) - (np.array(tp[:2]) - act_c)) @ ax / pr.MM
        errs.append(e)
        diffs.append(np.abs(e - np.array(old["rel_err"])).max())
    assert len(errs) >= 3, len(errs)
    rad = np.linalg.norm(errs, axis=1)
    print("regression: %d scenes, rel radial max %.3f mm; max |new - probe v5 rel_err| %.3f mm" % (len(errs), rad.max(), max(diffs)))
    assert rad.max() < 0.5, rad                              # P2: p95 0.234 mm, max 0.378 mm over 45 rows
    assert max(diffs) < 0.1, diffs                           # the same algorithm: only the reference point differs
