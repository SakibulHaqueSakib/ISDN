"""Gate G3's controller checks -- master_report §WP4 acceptance, in the twin.

    cd brickassembly && ../cpuenv/bin/python -m pytest tests/test_control.py -q

  * impedance step response < 5% overshoot          test_step_overshoot
  * null-space posture does not perturb the task     test_nullspace_isolation
    (< 0.1 mm)
  * F/T sensor within 5% of a known static load      test_ft_static_load
  * the gripper squeezes what it is asked, closing   test_grip_force
    no faster than the Franka hand

Arm B, empty cell (S1's plan, bricks in the feeder), free space above the
build area.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

import planner as P
from sim.mj import control as C
from sim.mj import runtime as RT
from sim.mj import scene as S

OUT = Path(__file__).with_suffix(".json")
RESULTS = {}


@pytest.fixture()
def sim():
    S.centre_plan_origin(P.STRUCTURES["S1"])
    cell = S.build_cell(P.build_plan("S1", "none"))
    s = RT.Sim(cell)
    s.goto("B", [0.15, 0.0, 0.25], C.down_rot(0.0), duration=1.0, settle=0.5)
    return s


@pytest.fixture(scope="module", autouse=True)
def dump():
    yield
    OUT.write_text(json.dumps(RESULTS, indent=1, default=float))


def _track(sim, name, seconds, axis=None):
    arm = sim.arms[name]
    out = []
    sim.hooks.append(lambda s_: out.append(arm.tcp(s_.d)[0].copy()))
    sim.run(seconds)
    sim.hooks.pop()
    return np.array(out)


@pytest.mark.parametrize("axis,step", [(0, 0.01), (1, 0.01), (2, 0.01), (2, -0.02)])
def test_step_overshoot(sim, axis, step):
    """A raw set-point step (no trajectory, no feed-forward) settles with
    < 5% overshoot: D = 2 zeta sqrt(K Lambda) with zeta = 1 per axis."""
    arm = sim.arms["B"]
    x0 = arm.tcp(sim.d)[0]
    arm.x_d = x0.copy()
    arm.x_d[axis] += step
    xs = _track(sim, "B", 1.5)
    travel = (xs[:, axis] - x0[axis]) * np.sign(step)
    final = travel[-1]
    over = max(0.0, travel.max() - abs(step)) / abs(step)
    RESULTS.setdefault("step_overshoot", {})["%s%+.0fmm" % ("xyz"[axis], step * 1000)] = {
        "overshoot_pct": 100 * over, "final_err_mm": 1000 * (abs(step) - final),
        "rise_90_s": float(np.argmax(travel > 0.9 * abs(step)) * sim.m.opt.timestep)}
    assert over < 0.05
    assert abs(abs(step) - final) < 0.0005          # within 0.5 mm after 1.5 s (gravity comp.)


def test_nullspace_isolation(sim):
    """Swinging the posture target (elbow and wrist, 0.4 rad) moves the TCP
    < 0.1 mm: the posture torque is projected through the dynamically
    consistent null space."""
    arm = sim.arms["B"]
    arm.hold(sim.d)
    sim.run(0.5)
    x0 = arm.tcp(sim.d)[0]
    q0 = arm.q(sim.d)
    arm.q_null = q0 + np.array([0.0, 0.0, 0.4, 0.0, 0.4, 0.0, 0.0])
    xs = _track(sim, "B", 2.0)
    dev = float(np.max(np.linalg.norm(xs - x0, axis=1)))
    moved = float(np.linalg.norm(arm.q(sim.d) - q0))
    RESULTS["nullspace"] = {"max_tcp_dev_mm": 1000 * dev, "joint_motion_rad": moved}
    assert moved > 0.05                     # the posture actually moved
    assert dev < 1e-4


@pytest.mark.parametrize("load", [(0, 0, -20.0), (10.0, 0, 0), (0, -5.0, -10.0)])
def test_ft_static_load(sim, load):
    """A known static force on the hand (at the TCP) reads back within 5%
    once the arm has settled against it (wrist sensor, tared, filtered)."""
    arm = sim.arms["B"]
    arm.hold(sim.d)
    sim.run(0.3)
    f = np.array(load)
    x, _ = arm.tcp(sim.d)
    hand = arm.hand

    def apply(s_):
        # a Cartesian force at the TCP, as a wrench at the hand's centre of mass
        r = x - s_.d.xipos[hand]
        s_.d.xfrc_applied[hand, :3] = f
        s_.d.xfrc_applied[hand, 3:] = np.cross(r, f)

    sim.hooks.append(apply)
    apply(sim)
    sim.run(1.5)
    read = sim.ft("B")[:3]
    sim.hooks.pop()
    sim.d.xfrc_applied[hand, :] = 0
    err = float(np.linalg.norm(read - f) / np.linalg.norm(f))
    RESULTS.setdefault("ft_static", {})[str(load)] = {"read_N": read.tolist(), "rel_err": err}
    assert err < 0.05


def test_grip_force(sim):
    """Closing on a 16 mm brick: fingers close at <= 0.1 m/s and then squeeze
    the commanded force per finger (pad contact normal force)."""
    from motion import skills as K
    arm = sim.arms["B"]
    bid = "b_000"
    p, _ = K.brick_pose(sim, bid)
    grasp = p + [0, 0, K.GRASP_TCP_ABOVE_BOTTOM]
    R = C.down_rot(K.grip_yaw(sim, "B", 0.0, grasp))
    sim.goto_joint("B", grasp + [0, 0, K.APPROACH], R)
    sim.goto("B", grasp, R, duration=0.8, settle=0.5)
    arm.grip_force = 30.0
    arm.grip_open = 0.0
    vmax = []
    sim.hooks.append(lambda s_: vmax.append(abs(float(s_.d.qvel[arm.finger_v].mean()))))
    K.close(sim, "B")
    sim.hooks.pop()
    b = sim.m.body(bid).id
    normal = []
    for i in range(sim.d.ncon):
        c = sim.d.contact[i]
        bodies = {sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2]}
        if b in bodies and (bodies - {b}) and sim.m.body((bodies - {b}).pop()).name.startswith("B/"):
            fr = np.zeros(6)
            mujoco.mj_contactForce(sim.m, sim.d, i, fr)
            normal.append(abs(fr[0]))
    per_finger = sum(normal) / 2
    RESULTS["grip"] = {"commanded_N": 30.0, "per_finger_N": per_finger,
                       "max_close_speed_mps": max(vmax)}
    assert max(vmax) <= 0.1 + 1e-3
    assert per_finger == pytest.approx(30.0, rel=0.1)
