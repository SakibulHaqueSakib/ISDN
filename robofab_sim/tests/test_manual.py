"""Manual control of dual_arm_sim --manual (cell/liveview.py): no simulator, no GPU.

  - clamp_target: height, mid-plane crossing, reach and base-column limits, yaw; an allowed target is left alone; clamping is idempotent;
  - ManualCtl.advance: the command is slew-limited (V_MAX), stays within LEASH of the measured hand, and reaches a reachable target;
  - nudge / home: targets stay inside the workspace for any sequence of nudges.

    bash scripts/run.sh tests/test_manual.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cell import liveview as LV  # noqa: E402

A, B = (-0.45, 0.0, 0.0), (0.45, 0.0, 0.0)
DT = 1.0 / 60


def inside(base, p, yaw):
    b, p = np.array(base), np.array(p)
    s = b + [0, 0, LV.SHOULDER_Z]
    return (LV.Z_MIN - 1e-9 <= p[2] <= LV.Z_MAX + 1e-9 and np.linalg.norm(p - s) <= LV.REACH + 1e-9
            and np.linalg.norm(p[:2] - b[:2]) >= LV.R_MIN - 1e-9 and abs(yaw) <= LV.YAW_MAX + 1e-9
            and (p[0] >= -LV.CROSS - 1e-9 if b[0] > 0 else p[0] <= LV.CROSS + 1e-9))


def test_clamp_limits():
    p, y = LV.clamp_target(B, [0.2, -0.3, 0.05], 0.3)
    assert np.allclose(p, [0.2, -0.3, 0.05]) and y == 0.3                          # an allowed target is left alone
    assert LV.clamp_target(B, [0.3, 0, -0.1], 0)[0][2] == LV.Z_MIN                 # not through the table
    assert LV.clamp_target(B, [0.3, 0, 5.0], 0)[0][2] == LV.Z_MAX
    assert LV.clamp_target(B, [-0.5, 0, 0.1], 0)[0][0] == -LV.CROSS                # B stays on its side of the mid-plane (+ CROSS)
    assert LV.clamp_target(A, [0.5, 0, 0.1], 0)[0][0] == LV.CROSS
    far = LV.clamp_target(B, [0.45, -2.0, 0.3], 0)[0]
    assert np.isclose(np.linalg.norm(far - [0.45, 0, LV.SHOULDER_Z]), LV.REACH)    # out of reach: pulled to the reach sphere
    near = LV.clamp_target(B, [0.45, 0.01, 0.3], 0)[0]
    assert np.isclose(np.linalg.norm(near[:2] - [0.45, 0]), LV.R_MIN)              # inside the base column: pushed out
    assert LV.clamp_target(B, [0.3, 0, 0.1], 9.0)[1] == LV.YAW_MAX
    rng = np.random.default_rng(0)
    for _ in range(500):
        base = (A, B)[rng.integers(2)]
        p, y = LV.clamp_target(base, rng.uniform(-1, 1, 3), rng.uniform(-4, 4))
        assert inside(base, p, y)
        p2, y2 = LV.clamp_target(base, p, y)
        assert np.allclose(p, p2, atol=1e-9) and y == y2                            # idempotent
    # the farthest feeder slot of the pick (0.65 m from B's base, TCP 13.5 mm up) is reachable
    slot = [0.45 - 0.0, -0.65, 0.0135]
    assert np.allclose(LV.clamp_target(B, slot, 0)[0], slot, atol=1e-6)


def test_advance_slew_and_leash():
    m = LV.ManualCtl(((A, 0.0), (B, np.pi)), [np.array([-0.15, 0, 0.35]), np.array([0.15, 0, 0.35])])
    ee = np.array([0.15, 0, 0.35])
    cmd = (ee.copy(), np.pi, 0.01, np.zeros(3))                                     # B's base yaw: its hand parked
    m.nudge(1, (0.1, 0, 0), 0.5, 0.03)
    p, yaw, g, _ = m.advance(1, cmd, ee, DT)
    assert np.linalg.norm(p - cmd[0]) <= LV.V_MAX * DT + 1e-12 and 0 < yaw - np.pi <= LV.W_MAX * DT and 0.01 < g <= 0.01 + LV.G_MAX * DT
    for _ in range(600):                                                            # a hand that follows exactly: the command arrives
        cmd = m.advance(1, cmd, cmd[0], DT)
    assert np.allclose(cmd[0], m.tgt[1]["p"]) and np.isclose(cmd[1], np.pi + 0.5) and np.isclose(cmd[2], 0.03)
    stuck = np.array([0.0, -0.3, 0.1])                                              # a hand that does not follow: no wind-up past the leash
    cmd = (stuck.copy(), 0.0, 0.01, np.zeros(3))
    m.nudge(1, (0.0, -0.1, 0.0))
    for _ in range(600):
        cmd = m.advance(1, cmd, stuck, DT)
    assert np.linalg.norm(cmd[0] - stuck) <= LV.LEASH + 1e-9


def test_nudge_home_stay_inside():
    m = LV.ManualCtl(((A, 0.0), (B, np.pi)), [np.array([-0.15, 0, 0.35]), np.array([0.15, 0, 0.35])])
    rng = np.random.default_rng(1)
    for _ in range(300):
        k = int(rng.integers(2))
        m.nudge(k, rng.normal(0, 0.1, 3), rng.normal(0, 0.5), rng.uniform(-0.1, 0.1))
        assert inside((A, B)[k], m.tgt[k]["p"], m.tgt[k]["yaw"]) and 0.0 <= m.tgt[k]["grip"] <= 0.04
    m.home(1)
    assert np.allclose(m.tgt[1]["p"], [0.15, 0, 0.35]) and m.tgt[1]["yaw"] == 0.0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
