"""FK aim controller and look schedule of dual_arm_sim (plan_v4_perception §2.2a, S1.3a): no simulator, no GPU.

  - rigid grasp, hand yawed and tilted: B_est = ee_FK - R_hand d_h equals the brick's position to < 0.05 mm as the hand moves;
  - the bias formula puts B_est on T_hat + [i_xy, 0] (xy) and on the target's z, for any hand yaw and tilt (the 13 mm lever cancels);
  - the look schedule: look waypoints between transport and pre-insert, a second one with the wrist turned 180 deg on EVERY step (ACC),
    the wrist turned back; the schedule does not depend on the aim mode; the look is H_LOOK = 35 mm up, below the travel height;
  - averaging (V.fuse_looks, the derivation is its docstring): two looks at different hand yaws and tracking offsets give the same d_h, delta-psi
    and T as one perfect look; errors average; one accepted look is used alone; none is perception_v5; look_done records v["used"] and
    v["aim"]["used_errors"];
  - look_done (fk_oracle): delta-psi is added to the yaw of every remaining waypoint of the step (pre-insert, insert, release,
    retract), once; the wrist's turn-back is left exactly pi from the turned look (Arm.update wraps yaw differences at 2 pi);
  - the pre-weld screen thresholds are the r4 ones.

    ~/Codes/CAIRSS/Issac/bin/python -m pytest tests/test_aim.py     (or run it as a script)
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import dual_arm_sim as D  # noqa: E402
import vision as V  # noqa: E402
from test_arm_protocol import load, stub  # noqa: E402

MM = 1e-3
DT = 1.0 / 60


def rand_hand(rng):
    """A hand pose: pointing down, yawed anywhere, tilted a few degrees."""
    R = Rot.from_euler("z", rng.uniform(-math.pi, math.pi)) * D.RX_PI * Rot.from_rotvec(np.radians(rng.normal(0, 3, 3)))
    return np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.05, 0.12)]), R


def test_b_est_equals_brick_rigid_grasp():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        p_b = np.array([rng.normal(0, 1.5 * MM), rng.normal(0, 1.5 * MM), -D.GRASP_DZ + rng.normal(0, 0.5 * MM)])   # brick origin in the hand frame
        ee_E, R_E = rand_hand(rng)
        B_E = ee_E + R_E.apply(p_b)                                  # t_E: the estimate pair's brick (here exact)
        d_h = V.hand_offset(ee_E, R_E, B_E)
        for _ in range(5):                                           # the hand moves, yaws (a 180 deg turn included) and tilts
            ee, R = rand_hand(rng)
            B = ee + R.apply(p_b)                                    # the rigid brick
            worst = max(worst, np.abs(V.brick_from_hand(ee, R, d_h) - B).max())
    assert worst < 0.05 * MM, worst
    print("B_est vs brick: worst %.2e mm" % (worst / MM))


def test_bias_puts_b_est_on_target_setpoint():
    rng = np.random.default_rng(2)
    for _ in range(50):
        tgt = np.array([*rng.uniform(-0.05, 0.05, 2), 0.0032])
        T_xy, i_xy = tgt[:2] + rng.normal(0, 0.5 * MM, 2), rng.normal(0, 1 * MM, 2)
        d_h = np.array([*rng.normal(0, 1.5 * MM, 2), D.GRASP_DZ + rng.normal(0, 0.5 * MM)])
        _, R = rand_hand(rng)
        ee = tgt + [0, 0, D.GRASP_DZ] + D.fk_bias(R, d_h, T_xy, tgt[:2], i_xy)      # the waypoint plus the bias, tracked perfectly
        B_est = V.brick_from_hand(ee, R, d_h)
        assert np.abs(B_est[:2] - (T_xy + i_xy)).max() < 1e-12 and abs(B_est[2] - tgt[2]) < 1e-12


def test_integral_gain_and_clip():
    i = D.fk_integral(np.zeros(2), np.array([0.001, 0.0]), np.zeros(2), 1 / 60)
    assert np.allclose(i, [D.ALIGN_GAIN / 60 * 0.001, 0])
    assert np.allclose(D.fk_integral(np.zeros(2), np.array([1.0, -1.0]), np.zeros(2), 1.0), [0.004, -0.004])


# --- the look schedule and delta-psi on the stub Example --------------------------------------------------------
def look_stub(name="S3", aim="gt", shadow=False):
    plan = load(name)
    e = stub(plan)
    e.args = SimpleNamespace(legacy_brace=False, continue_=False, look=True)
    e.name, e.aim, e.look, e.yaw_shift, e.grasp_yaw = name, aim, True, None, 0.0
    e.sees = aim == "fk_vision" or shadow
    e.NIJ = (max(b["grid_pos"][0] + D.P.footprint(b["type"], b["yaw_index"])[0] for b in plan["bricks"]),
             max(b["grid_pos"][1] + D.P.footprint(b["type"], b["yaw_index"])[1] for b in plan["bricks"]))
    e.vis, e.cand, e.fk, e.inject, e.b_phase_t, e.frame = {}, {}, None, None, {}, 0
    e.look_psi, e.look_frame, e.look_turn, e.hand_off = {}, {}, 0.0, None
    return e, plan


def queue(e, plan, n):
    e.B.moves.clear()
    e.queue_place(plan["sequence"][n])
    return list(e.B.moves)


def test_look_schedule():
    e, plan = look_stub()
    order = D.P.sequence(D.P.STRUCTURES["S3"])
    singles = [n for n in range(len(order)) if D.exposed_cells(order, n, *e.NIJ)[1]]
    assert singles and len(singles) < len(order), singles               # S3 has piers: some single-footprint targets, not all
    assert D.H_LOOK == 0.035 and D.H_LOOK + D.ex.STUD_HEIGHT < D.TRAVEL
    for n in range(len(order)):
        moves = queue(e, plan, n)
        ph = [m["phase"] for m in moves]
        i, j = ph.index("transport"), ph.index("pre-insert")
        assert set(ph[i + 1:j]) == {"look"} and ph[j:] == ["pre-insert", "insert", "release", "retract"], ph
        looks = moves[i + 1:j]
        assert len(looks) == 5, (n, len(looks))                             # every step: look, turn, look, turn back (and the first move)
        assert all(np.allclose(m["pos"], looks[0]["pos"]) for m in looks)    # the hand stays over the target
        h = looks[0]["pos"][2] - e.target[plan["sequence"][n]["brick_id"]][0][2] - D.GRASP_DZ
        assert abs(h - (D.ex.STUD_HEIGHT + D.H_LOOK)) < 1e-9, h             # brick bottom H_LOOK above the stud tops
        y = [m["yaw"] for m in looks]
        assert abs(abs(y[2] - y[0]) - math.pi) < 1e-5 and y[3] == y[2] and y[4] == y[0], y      # turned 180 deg, then back
        assert looks[1]["then"] is not None and looks[3]["then"] is not None
        # the nearer-wrist choice is taken once, at the grasp: raw yaw + yaw_shift is where the hand points
        assert abs(e.yaw_shift / math.pi - round(e.yaw_shift / math.pi)) < 1e-9
        assert abs(y[0] + e.yaw_shift - (D.ARM_B[1] + D.wrap(y[0] - D.ARM_B[1], math.pi))) < 1e-9
    print("single-footprint steps of S3:", singles)


def test_dpsi_on_every_remaining_waypoint():
    e, plan = look_stub(aim="fk_oracle")
    order = D.P.sequence(D.P.STRUCTURES["S3"])
    for n in range(len(order)):
        moves = queue(e, plan, n)
        single = D.exposed_cells(order, n, *e.NIJ)[1]
        first = [m["phase"] for m in moves].index("transport") + 1
        e.B.moves[:] = moves[first + 4:]                               # the final look's dwell has ended: what is left of the step
        before = [m["yaw"] for m in e.B.moves]
        bid = plan["sequence"][n]["brick_id"]
        tgt, yaw_t, _ = e.target[bid]
        q = np.array([*tgt, *D.qz(yaw_t + math.radians(2.5))])            # the brick sits 2.5 deg off the target yaw
        ee = np.r_[tgt + [0, 0, D.GRASP_DZ + 0.047], (Rot.from_euler("z", 1.0) * D.RX_PI).as_quat()]
        e.B.cmd = (e.B.cmd[0], 1.0 - e.yaw_shift, *e.B.cmd[2:])         # the hand is at its commanded yaw (1.0 rad)
        e.look_done(plan["sequence"][n], dict(looks=[]), ee, q)
        # the pre-insert, insert, release and retract waypoints get delta-psi; the wrist's turn-back (phase look) keeps its yaw
        assert e.fk is not None and len(e.B.moves) == len(before) and all(
            abs((m["yaw"] - b) - (0.0 if m["phase"] == "look" else math.radians(-2.5))) < 1e-9 for m, b in zip(e.B.moves, before)), (n, [m["yaw"] - b for m, b in zip(e.B.moves, before)])
        assert sum(m["phase"] == "look" for m in e.B.moves) == 1
        assert [m["phase"] for m in e.B.moves][-4:] == ["pre-insert", "insert", "release", "retract"]
        assert np.allclose(e.fk["T"], tgt[:2])
    print("delta-psi on all remaining waypoints: ok")


def ee_at_cmd(e, tgt):
    """The hand pose (xyzw) at its commanded yaw."""
    return np.r_[tgt + [0, 0, D.GRASP_DZ + 0.047], (Rot.from_euler("z", e.B.cmd[1] + e.yaw_shift) * D.RX_PI).as_quat()]


def test_hand_tracking_offset_at_t_E_is_removed():
    """The estimate pair is taken with the hand at its FK yaw (cmd + 0.37 deg here, as in the turned look); the brick's yaw is then
    hand yaw + const. dpsi = T - B + (FK - cmd): applied to the commanded waypoints, the brick ends at T once the hand reaches cmd."""
    for off_deg in (0.37, -0.37, 0.0):
        e, plan = look_stub(aim="fk_oracle")
        n = 0
        queue(e, plan, n)
        bid = plan["sequence"][n]["brick_id"]
        tgt, yaw_t, _ = e.target[bid]
        psi_cmd, rel = 1.0, math.radians(2.5)                           # the hand's commanded yaw at t_E; the brick sits 2.5 deg off it
        e.B.cmd = (e.B.cmd[0], psi_cmd - e.yaw_shift, *e.B.cmd[2:])
        psi_fk = psi_cmd + math.radians(off_deg)
        ee = np.r_[tgt + [0, 0, D.GRASP_DZ + 0.047], (Rot.from_euler("z", psi_fk) * D.RX_PI).as_quat()]
        q = np.array([*tgt, *D.qz(psi_fk + rel)])                       # the brick is rigid in the hand
        v = dict(looks=[])
        e.look_done(plan["sequence"][n], v, ee, q)
        dpsi = math.radians(v["aim"]["dpsi_deg"])
        assert abs(v["aim"]["hand_yaw_track_deg"] - off_deg) < 1e-9
        # the hand later at its commanded yaw + dpsi: the brick is at psi_cmd + dpsi + rel, which must be T's yaw (mod pi)
        assert abs(D.wrap(psi_cmd + dpsi + rel - yaw_t, math.pi)) < 1e-9, (off_deg, dpsi)


def test_turn_back_goes_the_way_it_came():
    """Run a single-footprint step's queue through Arm.update with delta-psi of either sign: the commanded wrist yaw never leaves
    [psi - pi, psi] (a 2 pi wrap the other way would send the wrist past its joint limit); it ends at psi + delta-psi."""
    for sign in (+1, -1):
        e, plan = look_stub(aim="fk_oracle")
        order = D.P.sequence(D.P.STRUCTURES["S3"])
        n = next(n for n in range(len(order)) if D.exposed_cells(order, n, *e.NIJ)[1])
        moves = queue(e, plan, n)
        bid = plan["sequence"][n]["brick_id"]
        tgt, yaw_t, _ = e.target[bid]
        q = np.array([*tgt, *D.qz(yaw_t - sign * math.radians(2.5))])      # brick yaw error -sign * 2.5 deg: delta-psi = +sign * 2.5 deg
        ee = np.r_[tgt + [0, 0, D.GRASP_DZ + 0.047], (Rot.from_euler("z", 1.0) * D.RX_PI).as_quat()]
        e.on_look = lambda s, k, final: e.look_done(s, dict(looks=[]), ee_at_cmd(e, tgt), q) if final else None   # the hand tracks perfectly
        e.on_pre_insert = lambda s: None
        e.B.moves[:] = moves[[m["phase"] for m in moves].index("transport"):]   # from the transport on
        psi = moves[0]["yaw"]
        lo, hi, ee_B = psi, psi, e.B.cmd[0].copy()
        for _ in range(4000):
            if e.B.idle():
                break
            ee_B = e.B.update(DT, ee_B)[0]
            lo, hi = min(lo, e.B.cmd[1]), max(hi, e.B.cmd[1])
        assert e.B.idle() and abs(e.B.cmd[1] - (psi + sign * math.radians(2.5))) < 2e-6, (e.B.cmd[1], psi)
        assert lo > psi - math.pi - 1e-5 and hi < psi + math.radians(3), (lo - psi, hi - psi, sign)


def test_fk_vision_without_an_accepted_look_fails():
    e, plan = look_stub(aim="fk_vision")
    e.welds = {}
    n = 0
    queue(e, plan, n)
    e.look_done(plan["sequence"][n], dict(looks=[]), np.r_[0.0, 0.0, 0.1, D.RX_PI.as_quat()], np.array([0, 0, 0.1, 0, 0, 0, 1.0]))
    assert e.failure == "perception_v5" and e.fk is None and e.failed_at_step == e.b_step


# --- averaging the two looks (V.fuse_looks) ----------------------------------------------------------------------
def rigid_looks(rng, psi_cmd=(0.4, 0.4 + math.pi - 1e-6), track_deg=(0.37, -0.52), phi0=0.9, p_b=(0.4 * MM, -0.7 * MM, -D.GRASP_DZ + 0.2 * MM),
                T=(0.012, -0.034, 1.3), noise=None, e_cm=None):
    """Two looks of a rigid grasp (brick pose in the hand fixed): the hand at a commanded yaw plus a tracking offset, tilted a little, over the
    target. Estimates exact, or with `noise` = (sigma xy m, sigma yaw rad) added to T and B in the world. -> list of fuse_looks inputs, the true brick pose
    in the hand (p_b, phi0)."""
    out = []
    for pc, tr in zip(psi_cmd, track_deg):
        psi = pc + math.radians(tr)
        R = Rot.from_euler("z", psi) * D.RX_PI * Rot.from_rotvec(np.radians(rng.normal(0, 1.5, 3)) * [1, 1, 0])
        ee = np.array([T[0] + rng.normal(0, 1 * MM), T[1] + rng.normal(0, 1 * MM), 0.06])
        B = np.r_[ee + R.apply(p_b), V._yaw(R) + phi0]
        Tk = np.array(T, float)
        if noise:
            B = B + np.r_[rng.normal(0, noise[0], 3) * [1, 1, 0], rng.normal(0, noise[1])]
            Tk = Tk + np.r_[rng.normal(0, noise[0], 2), rng.normal(0, noise[1])]
        if e_cm is not None:                                                   # the camera's hand-fixed offset: shifts T and B alike, R_k e in the world
            B[:3], Tk[:2] = B[:3] + R.apply([*np.asarray(e_cm, float), 0.0]), Tk[:2] + R.as_matrix()[:2, :2] @ np.asarray(e_cm, float)
        out.append(dict(T=Tk, B=B, ee=ee, R=R, psi_cmd=pc))
    return out


def test_two_looks_equal_one_perfect_look():
    """Two looks, hand yawed 180 deg apart with different tracking offsets and tilts: d_h, delta-psi and T equal what ONE perfect look at
    any hand yaw gives (and each look alone gives the same, the estimates being exact)."""
    rng = np.random.default_rng(3)
    p_b, phi0, T = np.array([0.4 * MM, -0.7 * MM, -D.GRASP_DZ + 0.2 * MM]), 0.9, (0.012, -0.034, 1.3)
    looks = rigid_looks(rng)
    fz = V.fuse_looks(looks)
    # the true hand-frame offset d_h = R^T (ee - B) = -p_b for any hand pose; the one-perfect-look reference at the commanded yaw
    assert np.abs(fz["d_h"] - (-p_b)).max() < 1e-12, fz["d_h"]
    want = D.wrap(T[2] - phi0 - 0.4, math.pi)                                  # T - phi - psi_cmd, the pre-ACC single-look formula
    assert abs(D.wrap(fz["dpsi"] - want, math.pi)) < 2e-6 and np.allclose(fz["T"][:2], T[:2]) and abs(fz["T"][2] - T[2]) < 1e-12   # 2e-6: the turned look's command is pi - 1e-6
    for l in looks:
        one = V.fuse_looks([l])
        assert np.abs(one["d_h"] - fz["d_h"]).max() < 1e-12 and abs(D.wrap(one["dpsi"] - fz["dpsi"], math.pi)) < 2e-6
        # the pre-ACC formula of look_done: wrap(T - B_yaw + (psi_FK - psi_cmd))
        old = D.wrap(T[2] - l["B"][3] + D.wrap(V._yaw(l["R"]) - l["psi_cmd"], math.pi), math.pi)
        assert abs(D.wrap(one["dpsi"] - old, math.pi)) < 1e-12
    # the brick at an arbitrary reference hand pose is the rigid brick there
    ee2, R2 = rand_hand(rng)
    fz2 = V.fuse_looks(looks, ref=(ee2, R2))
    assert np.abs(fz2["B"][:3] - (ee2 + R2.apply(p_b))).max() < 1e-12 and abs(D.wrap(fz2["B"][3] - (V._yaw(R2) + phi0), math.pi)) < 1e-12
    # applied: the brick ends on T_yaw once the hand tracks its commanded yaw (+ dpsi): psi_cmd + dpsi + phi = T_yaw (mod pi)
    assert abs(D.wrap(0.4 + fz["dpsi"] + phi0 - T[2], math.pi)) < 2e-6


def test_common_mode_camera_offset_is_removed_across_the_turn():
    """The camera's hidden extrinsic error is a hand-frame xy offset e (0.2, -0.3 mm), the same in T_k and B_k. Plain averaging (T in the world, d_h
    in the hand) cancels it in T but keeps it in d_h: the brick misses by R e. fuse_looks' common-mode step finds e from the T_k disagreement across the
    turned looks (the target is fixed in the world) and removes it: exact. One look, or looks 30 deg apart: no correction."""
    rng = np.random.default_rng(6)
    p_b, e = np.array([0.4 * MM, -0.7 * MM, -D.GRASP_DZ + 0.2 * MM]), np.array([0.2 * MM, -0.3 * MM])
    looks = rigid_looks(rng, e_cm=e)
    plain, cm = V.fuse_looks(looks, common_mode=False), V.fuse_looks(looks, common_mode=True)
    assert np.abs(cm["d_h"] - (-p_b)).max() < 1e-12 and np.allclose(cm["T"][:2], (0.012, -0.034), atol=1e-12) and np.allclose(cm["e_h"], e, atol=1e-12)
    assert np.allclose(plain["d_h"][:2] - (-p_b)[:2], -e) and plain["e_h"] is None           # plain: d_h keeps -e (hand frame), T has lost it
    assert np.allclose(plain["T"][:2], (0.012, -0.034), atol=1e-5)            # (the hands' tilts leave a few um)
    assert abs(D.wrap(cm["dpsi"] - plain["dpsi"], math.pi)) < 1e-12                          # yaw is unaffected (a camera yaw error shifts T and B alike)
    ee2, R2 = rand_hand(rng)
    assert np.abs(V.fuse_looks(looks, ref=(ee2, R2))["B"][:3] - (ee2 + R2.apply(p_b))).max() < 1e-12
    assert np.abs(np.hypot(*(V.fuse_looks(looks, ref=(ee2, R2), common_mode=False)["B"][:2] - (ee2 + R2.apply(p_b))[:2])) - np.hypot(*e)) < 1e-5   # plain: |e| off (to the hand tilt)
    one = V.fuse_looks(looks[:1], common_mode=True)
    assert one["e_h"] is None and np.allclose(one["d_h"][:2], (-p_b)[:2] - e)                  # one look: the camera offset stays (and cancels against T in rel)
    near = rigid_looks(rng, psi_cmd=(0.4, 0.4 + math.radians(30)), track_deg=(0.0, 0.0), e_cm=e)
    assert V.fuse_looks(near, common_mode=True)["e_h"] is None                                 # 30 deg apart: e is not observable well enough
    three = rigid_looks(rng, e_cm=e)
    three.append(dict(three[0], R=Rot.from_euler("z", 0.4 + math.pi / 2) * D.RX_PI))         # a third look at 90 deg: lstsq over both differences
    assert V.fuse_looks(rigid_looks(rng, e_cm=e), common_mode=True)["e_h"] is not None


def test_averaging_reduces_error_and_handles_the_circular_wrap():
    rng = np.random.default_rng(4)
    e1, e2 = [], []
    for _ in range(200):
        looks = rigid_looks(rng, noise=(0.1 * MM, math.radians(0.15)))
        fz, f1 = V.fuse_looks(looks), V.fuse_looks(looks[:1])
        e1.append(D.wrap(f1["dpsi"] - D.wrap(1.3 - 0.9 - 0.4, math.pi), math.pi))
        e2.append(D.wrap(fz["dpsi"] - D.wrap(1.3 - 0.9 - 0.4, math.pi), math.pi))
    assert np.std(e2) < 0.8 * np.std(e1), (np.std(e1), np.std(e2))              # about 1/sqrt(2) of one look
    # the yaws sit across the +-pi/2 cut of a mod-pi angle: still the mean of two close angles
    looks = rigid_looks(rng, phi0=math.pi / 2 - 0.001, T=(0.0, 0.0, math.pi / 2 - 0.002))
    looks[1]["T"] = looks[1]["T"] + [0, 0, 0.004]                               # T yaws pi/2 -0.002 and +0.002, across the cut
    assert abs(D.wrap(V.fuse_looks(looks)["T"][2] - (math.pi / 2), math.pi)) < 1e-9


def test_drop_sides_of_a_one_stud_wall():
    """hollow_box step 7 (a 1x4 on a one-stud-wide wall, exposed cells (5, 1..4)): the two long sides drop to the ground (8 sides); the ends meet
    the wall's course-1 neighbours at (5, 0) and (5, 5), already placed: a rise, not a drop."""
    bricks = D.P.sequence(list(__import__("blueprint").load("hollow_box")))
    cells, single = D.exposed_cells(bricks, 7, 8, 8)
    assert single and cells == [(5, 1), (5, 2), (5, 3), (5, 4)]
    sd = D.drop_sides(bricks, 7, 8, 8, cells)
    assert sorted(sd) == [(a, d) for a in range(4) for d in (0, 1)], sd


def fake_res(look, est="studs"):
    return SimpleNamespace(T_hat=look["T"], B_hat=look["B"], estimator=est)


def fused_step(accepted, aim="fk_vision", n=0, est=("studs", "studs"), e_cm=None):
    """look_done on the stub with `accepted` of the two rigid looks accepted. -> (example, v, truth q)."""
    e, plan = look_stub(aim=aim, shadow=aim != "fk_vision")
    e.welds = {}
    moves = queue(e, plan, n)
    first = [m["phase"] for m in moves].index("transport") + 1
    e.B.moves[:] = moves[first + 4:]                                            # the final look's dwell has ended: what is left of the step
    bid = plan["sequence"][n]["brick_id"]
    tgt, yaw_t, _ = e.target[bid]
    rng = np.random.default_rng(5)
    psi0 = 1.0 - e.yaw_shift
    looks = rigid_looks(rng, psi_cmd=(1.0, 1.0 + math.pi - 1e-6), T=(*tgt[:2], yaw_t), phi0=0.3, e_cm=e_cm)
    e.B.cmd = (e.B.cmd[0], psi0, *e.B.cmd[2:])
    e.cand = {n: [(fake_res(looks[k], est[k]), np.r_[looks[k]["ee"], looks[k]["R"].as_quat()], 100 + 10 * k, looks[k]["psi_cmd"], k) for k in accepted]}
    e.look_psi[n], e.look_frame[n] = {k: looks[k]["psi_cmd"] for k in (0, 1)}, {0: 100, 1: 110}
    ee, R = looks[1]["ee"], looks[1]["R"]                                       # the final look's hand pose; the brick is rigid in it
    q = np.r_[ee + R.apply([0.4 * MM, -0.7 * MM, -D.GRASP_DZ + 0.2 * MM]), Rot.from_euler("z", V._yaw(R) + 0.3).as_quat()]
    v = dict(looks=[dict(k=k, accepted=k in accepted, estimator=est[k]) for k in (0, 1)])
    e.look_done(plan["sequence"][n], v, np.r_[ee, R.as_quat()], q)
    return e, v, (tgt, yaw_t)


def test_look_done_averages_two_looks_and_uses_one_alone():
    for accepted, nu in (((0, 1), 2), ((0,), 1), ((1,), 1)):
        e, v, (tgt, yaw_t) = fused_step(accepted)
        assert e.failure is None and e.fk is not None
        a, u = v["aim"], v["used"]
        assert a["src"] == "v5" and a["n_looks_used"] == nu == u["n_used"] and a["used_errors"] == u["errors"]
        assert np.allclose(e.fk["d_h"], [-0.4 * MM, 0.7 * MM, D.GRASP_DZ - 0.2 * MM], atol=1e-12) and np.allclose(e.fk["T"], tgt[:2])
        er = u["errors"]                                                          # exact looks: zero error against the truth, whichever were used
        assert er["rel_radial_mm"] < 1e-6 and abs(er["rel_yaw_deg"]) < 1e-6 and abs(er["held_z_mm"]) < 1e-6, er
        assert abs(D.wrap(math.radians(a["dpsi_deg"]) - D.wrap(yaw_t - 0.3 - 1.0, math.pi), math.pi)) < 2e-6
        assert abs(a["hand_yaw_track_deg"] - (0.37 if accepted == (0,) else -0.52)) < 0.05          # the last accepted look's tracking offset (the hand tilt moves its atan2 yaw a hair)
    e, v, _ = fused_step(())                                                      # none accepted: perception_v5
    assert e.failure == "perception_v5" and e.fk is None and v["failure"] == "perception_v5" and v["used"]["n_used"] == 0


def test_edge_is_a_fallback_at_step_level():
    """Plan 2.4: edge looks are used only when the stud rule fails for the whole step. A stud-accepted look present: the edge looks are dropped from
    the average (recorded in v["used"]["dropped"] with the look and why); no stud look: the edge looks are used (alone or together)."""
    e, v, _ = fused_step((0, 1), est=("studs", "edge"))                                # look 0 stud, look 1 edge: edge dropped
    u = v["used"]
    assert u["estimators"] == ["studs"] and u["n_used"] == 1 and v["aim"]["n_looks_used"] == 1 and u["errors"]["rel_radial_mm"] < 1e-6
    assert len(u["dropped"]) == 1 and u["dropped"][0]["k"] == 1 and u["dropped"][0]["estimator"] == "edge" and "fallback" in u["dropped"][0]["why"]
    assert abs(v["aim"]["hand_yaw_track_deg"] - (0.37)) < 0.05                          # the stud look 0 is the last used one (its tracking offset)
    _, v, _ = fused_step((0, 1), est=("edge", "studs"))                                # the other way round
    assert v["used"]["estimators"] == ["studs"] and [d["k"] for d in v["used"]["dropped"]] == [0]
    _, v, _ = fused_step((0, 1), est=("edge", "edge"))                                 # both edge: both used, nothing dropped
    assert v["used"]["estimators"] == ["edge", "edge"] and v["used"]["dropped"] == [] and v["aim"]["n_looks_used"] == 2
    _, v, _ = fused_step((1,), est=("studs", "edge"))                                  # only the edge look accepted: it is used
    assert v["used"]["estimators"] == ["edge"] and v["used"]["dropped"] == []
    _, v, _ = fused_step((0, 1), est=("studs", "studs"))
    assert v["used"]["estimators"] == ["studs", "studs"] and v["used"]["dropped"] == []
    # the averaged estimate with the edge look dropped equals the stud look alone (not a mix)
    _, v2, _ = fused_step((0,), est=("studs", "studs"))
    assert v["used"]["d_h_mm"] is not None and np.allclose(v["used"]["d_h_mm"], v2["used"]["d_h_mm"])


def test_single_look_inserted_at_its_own_yaw_cancels_the_camera_offset():
    """Plan 2.2a: target and brick come from one image, and the insertion uses that image's hand frame. With the camera's hand-fixed offset e
    (0.2, -0.3 mm) in T and B, a single used look (look 0, or the turned look 1) inserted at its own hand yaw, and the fused pair (common mode solved,
    inserted at look 0's yaw), all end within 0.03 mm of the truth at the hand pose they are inserted with."""
    e = (0.2 * MM, -0.3 * MM)
    for acc, turned in (((0,), False), ((1,), True), ((0, 1), False)):
        ex, v, _ = fused_step(acc, e_cm=e)
        er = v["used"]["errors"]
        assert er["rel_radial_mm"] < 0.03 and abs(er["held_z_mm"]) < 0.01, (acc, er)
        assert v["insert"]["turned"] is turned and v["insert"]["look"] == (1 if turned else 0), v["insert"]
        assert v["used"]["scored_at"] == "the insertion hand pose"


def test_insertion_waypoints_follow_the_chosen_look():
    """Single turned look: no turn-back before the pre-insert; pre-insert .. retract carry the turned look's yaw + dpsi; an unturn move after the retract
    goes back exactly the turn (pi - 1e-6) the way it came. Look 0 alone or the fused pair: the turn-back stays, the yaw is the base yaw + dpsi."""
    for acc, turned in (((1,), True), ((0,), False), ((0, 1), False)):
        e0, plan = look_stub(aim="fk_vision")
        moves = queue(e0, plan, 0)
        first = [m["phase"] for m in moves].index("transport") + 1
        base = moves[first + 4]["yaw"]                                          # the turn-back's yaw = the step's base yaw
        ex, v, _ = fused_step(acc, e_cm=(0.2 * MM, -0.3 * MM))
        ph, dpsi, turn = [m["phase"] for m in ex.B.moves], math.radians(v["aim"]["dpsi_deg"]), ex.look_turn
        if turned:
            assert ph == ["pre-insert", "insert", "release", "retract", "look"], ph
            assert all(abs(m["yaw"] - (base + turn + dpsi)) < 1e-9 for m in ex.B.moves[:4])
            assert abs((ex.B.moves[3]["yaw"] - ex.B.moves[4]["yaw"]) - turn) < 1e-9 and abs(ex.B.moves[4]["yaw"] - (base + dpsi)) < 1e-9
            assert np.allclose(ex.B.moves[4]["pos"], ex.B.moves[3]["pos"])
        else:
            assert ph == ["look", "pre-insert", "insert", "release", "retract"], ph
            assert abs(ex.B.moves[0]["yaw"] - base) < 1e-9 and all(abs(m["yaw"] - (base + dpsi)) < 1e-9 for m in ex.B.moves[1:])
        assert abs(v["insert"]["yaw_deg"] - math.degrees(ex.look_psi[0][1 if turned else 0] + dpsi)) < 1e-9 and "look" in v["insert"]["reason"]


def test_every_arm_picks_the_same_insertion_yaw():
    """gt and fk_oracle use the shadow V5's acceptance to choose, as fk_vision would: single look k at its own yaw, both looks look 0's, none look 0's;
    without V5 in the arm, look 0's. Same waypoint phases in every arm."""
    for acc, est, want in (((1,), ("studs", "studs"), True), ((0,), ("studs", "studs"), False), ((0, 1), ("studs", "studs"), False),
                           ((0, 1), ("edge", "studs"), True)):               # edge look 0 dropped: look 1 alone
        res = {}
        for aim in ("fk_vision", "fk_oracle", "gt"):
            ex, v, _ = fused_step(acc, aim=aim, est=est)
            res[aim] = (v["insert"]["turned"], [m["phase"] for m in ex.B.moves])
        assert all(r[0] is want for r in res.values()) and len({tuple(r[1]) for r in res.values()}) == 1, (acc, est, res)
    for aim in ("fk_oracle", "gt"):                                              # the shadow V5 rejects both looks: look 0's yaw
        ex, v, _ = fused_step((), aim=aim)
        assert v["insert"]["turned"] is False and "no accepted look" in v["insert"]["reason"]
    ex2, plan = look_stub(aim="fk_oracle")                                      # no V5 in this arm at all: look 0's yaw, with the reason
    assert ex2.sees is False
    moves = queue(ex2, plan, 0)
    first = [m["phase"] for m in moves].index("transport") + 1
    ex2.B.moves[:] = moves[first + 4:]
    v = dict(looks=[])
    ex2.look_done(plan["sequence"][0], v, np.r_[0.0, 0.0, 0.1, D.RX_PI.as_quat()], np.array([0, 0, 0.1, 0, 0, 0, 1.0]))
    assert v["insert"]["turned"] is False and "no V5" in v["insert"]["reason"] and [m["phase"] for m in ex2.B.moves][0] == "look"
    for acc in ((1,), (0,)):                                                     # the oracle's B is the ground truth at the chosen look's frame
        ex, v, _ = fused_step(acc, aim="fk_oracle")
        assert v["aim"]["src"] == "oracle" and v["insert"]["turned"] is (acc == (1,)) and v["aim"]["t_E_frame"] == (110 if acc == (1,) else 100)


def test_turned_insertion_stays_inside_the_wrist_range_both_ways():
    """Run a step through Arm.update with the turned look as the only used look, for delta-psi of either sign: the commanded yaw never leaves
    [psi - pi, psi] (+ a few deg), the unturn goes back the way it came, and it ends at psi + delta-psi."""
    for sign in (+1, -1):
        e, plan = look_stub(aim="fk_vision")
        order = D.P.sequence(D.P.STRUCTURES["S3"])
        n = next(n for n in range(len(order)) if D.exposed_cells(order, n, *e.NIJ)[1])
        moves = queue(e, plan, n)
        bid = plan["sequence"][n]["brick_id"]
        tgt, yaw_t, _ = e.target[bid]
        psi = moves[0]["yaw"]
        psi_wp = psi + e.yaw_shift
        rng = np.random.default_rng(7)
        looks = rigid_looks(rng, psi_cmd=(psi_wp, psi_wp + e.look_turn), track_deg=(0.0, 0.0), T=(*tgt[:2], yaw_t),
                            phi0=D.wrap(yaw_t - psi_wp - sign * math.radians(2.5), math.pi))
        e.cand = {n: [(fake_res(looks[1]), np.r_[looks[1]["ee"], looks[1]["R"].as_quat()], 110, looks[1]["psi_cmd"], 1)]}
        e.sees = True
        ee, R = looks[1]["ee"], looks[1]["R"]
        qb = np.r_[ee, Rot.from_euler("z", V._yaw(R) + 0.2).as_quat()]
        vv = dict(looks=[dict(k=0, accepted=False, estimator="studs"), dict(k=1, accepted=True, estimator="studs")])
        e.on_look = lambda s, kk, final: e.look_done(s, vv, np.r_[ee, R.as_quat()], qb) if final else None
        e.on_pre_insert = lambda s: None
        e.B.moves[:] = moves[[m["phase"] for m in moves].index("transport"):]
        lo, hi, ee_B = psi, psi, e.B.cmd[0].copy()
        for _ in range(6000):
            if e.B.idle():
                break
            ee_B = e.B.update(DT, ee_B)[0]
            lo, hi = min(lo, e.B.cmd[1]), max(hi, e.B.cmd[1])
        lim = sorted((psi, psi + e.look_turn))
        assert vv["insert"]["turned"] is True
        assert e.B.idle() and abs(e.B.cmd[1] - (psi + sign * math.radians(2.5))) < 2e-6, (e.B.cmd[1], psi)
        assert lo > lim[0] - math.radians(3) and hi < lim[1] + math.radians(3), (lo - psi, hi - psi, sign)


def test_used_estimators_are_recorded_and_shadow_arms_log_but_do_not_steer():
    _, v, _ = fused_step((0, 1), est=("edge", "edge"))
    assert v["used"]["estimators"] == ["edge", "edge"]
    e, v, _ = fused_step((0, 1), aim="gt")                                        # a shadow arm: v["used"] is logged, the aim is the GT servo
    assert v["used"]["n_used"] == 2 and "aim" not in v and e.fk is None
    e, v, _ = fused_step((0, 1), aim="gt", est=("studs", "edge"))                 # the shadow arm applies the same fallback rule
    assert v["used"]["estimators"] == ["studs"] and len(v["used"]["dropped"]) == 1


def test_inject_for():
    e, _ = look_stub(aim="fk_oracle")
    assert e.inject_for(3) == (0.0, 0.0, 0.0)
    e.inject = {"dx_mm": 0.4, "dyaw_deg": -1.0}
    assert e.inject_for(3) == (0.4e-3, 0.0, -1.0)                    # one vector: every step
    e.inject = {"3": {"dx_mm": 0.4, "dy_mm": 0.2, "dyaw_deg": 0.5}}
    assert e.inject_for(3) == (0.4e-3, 0.2e-3, 0.5) and e.inject_for(4) == (0.0, 0.0, 0.0)   # {step: vector}


def test_screen_thresholds_are_r4():
    assert D.SCREEN == {"lateral_mm": 1.2, "dz_mm": (-0.5, 0.3), "tilt_deg": 4.0, "yaw_deg": 5.0}


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
