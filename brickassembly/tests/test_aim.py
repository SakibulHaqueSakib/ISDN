"""FK aim controller and look schedule of dual_arm_sim (plan_v4_perception §2.2a, S1.3a): no simulator, no GPU.

  - rigid grasp, hand yawed and tilted: B_est = ee_FK - R_hand d_h equals the brick's position to < 0.05 mm as the hand moves;
  - the bias formula puts B_est on T_hat + [i_xy, 0] (xy) and on the target's z, for any hand yaw and tilt (the 13 mm lever cancels);
  - the look schedule: look waypoints between transport and pre-insert, a second one with the wrist turned 180 deg iff the target
    is single-footprint, the wrist turned back; the schedule does not depend on the aim mode;
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
def look_stub(name="S3", aim="gt"):
    plan = load(name)
    e = stub(plan)
    e.args = SimpleNamespace(legacy_brace=False, continue_=False, look=True)
    e.name, e.aim, e.look, e.yaw_shift, e.grasp_yaw = name, aim, True, None, 0.0
    e.NIJ = (max(b["grid_pos"][0] + D.P.footprint(b["type"], b["yaw_index"])[0] for b in plan["bricks"]),
             max(b["grid_pos"][1] + D.P.footprint(b["type"], b["yaw_index"])[1] for b in plan["bricks"]))
    e.vis, e.cand, e.fk, e.inject, e.b_phase_t, e.frame = {}, {}, None, None, {}, 0
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
    for n in range(len(order)):
        moves = queue(e, plan, n)
        ph = [m["phase"] for m in moves]
        i, j = ph.index("transport"), ph.index("pre-insert")
        assert set(ph[i + 1:j]) == {"look"} and ph[j:] == ["pre-insert", "insert", "release", "retract"], ph
        looks = moves[i + 1:j]
        assert len(looks) == (5 if n in singles else 2), (n, len(looks))
        assert all(np.allclose(m["pos"], looks[0]["pos"]) for m in looks)    # the hand stays over the target
        h = looks[0]["pos"][2] - e.target[plan["sequence"][n]["brick_id"]][0][2] - D.GRASP_DZ
        assert abs(h - (D.ex.STUD_HEIGHT + D.H_LOOK)) < 1e-9, h             # brick bottom H_LOOK above the stud tops
        y = [m["yaw"] for m in looks]
        if n in singles:                                                  # turned 180 deg, then back
            assert abs(abs(y[2] - y[0]) - math.pi) < 1e-5 and y[3] == y[2] and y[4] == y[0], y
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
        e.B.moves[:] = moves[first + (4 if single else 2):]            # the final look's dwell has ended: what is left of the step
        before = [m["yaw"] for m in e.B.moves]
        bid = plan["sequence"][n]["brick_id"]
        tgt, yaw_t, _ = e.target[bid]
        q = np.array([*tgt, *D.qz(yaw_t + math.radians(2.5))])            # the brick sits 2.5 deg off the target yaw
        ee = np.r_[tgt + [0, 0, D.GRASP_DZ + 0.047], (Rot.from_euler("z", 1.0) * D.RX_PI).as_quat()]
        e.look_done(plan["sequence"][n], dict(looks=[]), ee, q)
        # the pre-insert, insert, release and retract waypoints get delta-psi; the wrist's turn-back (phase look) keeps its yaw
        assert e.fk is not None and len(e.B.moves) == len(before) and all(
            abs((m["yaw"] - b) - (0.0 if m["phase"] == "look" else math.radians(-2.5))) < 1e-9 for m, b in zip(e.B.moves, before)), (n, [m["yaw"] - b for m, b in zip(e.B.moves, before)])
        assert sum(m["phase"] == "look" for m in e.B.moves) == (1 if single else 0)
        assert [m["phase"] for m in e.B.moves][-4:] == ["pre-insert", "insert", "release", "retract"]
        assert np.allclose(e.fk["T"], tgt[:2])
    print("delta-psi on all remaining waypoints: ok")


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
        e.on_look = lambda s, k, final: e.look_done(s, dict(looks=[]), ee, q) if final else None
        e.on_pre_insert = lambda s: None
        e.B.moves[:] = moves[[m["phase"] for m in moves].index("transport"):]   # from the transport on
        psi = moves[0]["yaw"]
        lo, hi, ee_B = psi, psi, e.B.cmd[0].copy()
        for _ in range(4000):
            if e.B.idle():
                break
            ee_B = e.B.update(DT, ee_B)[0]
            lo, hi = min(lo, e.B.cmd[1]), max(hi, e.B.cmd[1])
        assert e.B.idle() and abs(e.B.cmd[1] - (psi + sign * math.radians(2.5))) < 1e-6, (e.B.cmd[1], psi)
        assert lo > psi - math.pi - 1e-5 and hi < psi + math.radians(3), (lo - psi, hi - psi, sign)


def test_fk_vision_without_an_accepted_look_fails():
    e, plan = look_stub(aim="fk_vision")
    e.welds = {}
    n = 0
    queue(e, plan, n)
    e.look_done(plan["sequence"][n], dict(looks=[]), np.r_[0.0, 0.0, 0.1, D.RX_PI.as_quat()], np.array([0, 0, 0.1, 0, 0, 0, 1.0]))
    assert e.failure == "perception_v5" and e.fk is None and e.failed_at_step == e.b_step


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
