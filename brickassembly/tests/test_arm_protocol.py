"""Hand-off protocol of dual_arm_sim (plan_v4 r5-5, S3.3): no GPU physics.

A stub Example (Example.__new__ plus the plan-derived attributes) drives the real Arm, queue_place, queue_brace and
schedule with perfect tracking (the hand is where it was last commanded). Checked on the committed arch and S3 plans,
an all-braced arch, and an arch whose braced step has no feasible brace:
  - every step completes within MAX_UPDATES updates and no wait times out;
  - at no frame of a braced step do both arms make a moving move (a move without a wait; "stage" and "hold" wait);
  - A's protocol events come in order (staged < braced < placed < retracted per step);
  - the infeasible step runs unbraced (A never leaves park) and is flagged brace_infeasible.

    bash scripts/run.sh tests/test_arm_protocol.py
"""

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as D  # noqa: E402

MAX_UPDATES = 20000
DT = 1.0 / 60


def load(name):
    return json.loads((HERE / "plans" / ("sim_%s_weakest_joint.json" % name)).read_text())


def all_braced(plan):
    """Every step needs a brace: the plan's first brace, reused (the protocol does not care where it grips)."""
    plan = copy.deepcopy(plan)
    ref = next(s["brace"] for s in plan["sequence"] if s["requires_brace"])
    for s in plan["sequence"]:
        s["requires_brace"], s["brace"] = True, copy.deepcopy(ref)
    return plan


def one_infeasible(plan):
    plan = copy.deepcopy(plan)
    s = next(s for s in plan["sequence"] if s["requires_brace"])
    s["brace"] = dict(s["brace"], feasible=False)
    return plan, s["step"]


def stub(plan, legacy=False):
    """An Example without its scene: the attributes queue_place, queue_brace and schedule read."""
    e = D.Example.__new__(D.Example)
    steps = plan["sequence"]
    bricks = {b["id"]: b for b in plan["bricks"]}
    e.args, e.plan, e.legacy = SimpleNamespace(legacy_brace=legacy, continue_=False), plan, legacy
    e.failure = e.failed_at_step = e.built_at_failure = None
    e.stopped = e.done = False
    e.start_step, e.end_step, e.sham_s = None, len(steps), None      # no nominal start, no sham
    e.z_travel = max(s["target_pose"][2] for s in steps) + D.P.BRICK_H + D.TRAVEL
    slots = D.feeder_slots(len(steps))
    e.feeder = {s["brick_id"]: np.array([*slots[n], 0.0]) for n, s in enumerate(steps)}
    e.target = {}
    for s in steps:
        nx, ny = D.P.footprint(bricks[s["brick_id"]]["type"], bricks[s["brick_id"]]["yaw_index"])
        e.target[s["brick_id"]] = (np.array(s["target_pose"][:3]), math.pi / 2 if ny > nx else 0.0, nx == ny)
    park = lambda base: np.array(base[0]) + D.rotate(D.qz(base[1]), D.PARK)
    e.A, e.B = D.Arm("A stabilizer", D.ARM_A, park(D.ARM_A)), D.Arm("B placer", D.ARM_B, park(D.ARM_B))
    e.parks = {e.A: park(D.ARM_A), e.B: park(D.ARM_B)}
    e.next_step, e.b_step, e.sim_time = 0, None, 0.0
    e.staged, e.braced, e.retracted, e.placed, e.queued = set(), set(), set(), set(), set()
    e.brace_by_step, e.a_step, e.close_t, e.guard_n = {}, None, None, 0
    e.episodes, e.place_t = [], {}
    e.log_path = Path("/dev/null")
    e.timeouts = []
    for arm in (e.A, e.B):
        arm.on_timeout = lambda a, ph: e.timeouts.append((a.name[0], ph))
    e.snap = lambda s: e.episodes.append({"success": True})
    return e


def moving(arm):
    m = arm.cur
    return m is not None and m["wait"] is None and m["phase"] != "park"


def run(plan, legacy=False):
    """Step the stub until done; returns (example, updates, both-moving frames inside a braced step, A moves during
    an unbraced-infeasible step)."""
    e = stub(plan, legacy)
    steps = plan["sequence"]
    ee = {e.A: e.A.cmd[0].copy(), e.B: e.B.cmd[0].copy()}
    e.A.push("park", e.parks[e.A], 0.0, 0.0, 1.5)      # what Example.__init__ queues
    e.B.push("park", e.parks[e.B], 0.0, 0.01, 1.5)
    both, a_moves = [], []
    for u in range(MAX_UPDATES):
        e.schedule()
        for arm in (e.A, e.B):
            ee[arm] = arm.update(DT, ee[arm])[0]
        e.sim_time += DT
        n = e.b_step
        if n is not None and steps[n]["requires_brace"] and steps[n]["brace"].get("feasible", True) \
                and moving(e.A) and moving(e.B):
            both.append((u, n, e.A.phase, e.B.phase))
        if n is not None and steps[n]["requires_brace"] and not steps[n]["brace"].get("feasible", True) \
                and e.A.cur is not None:
            a_moves.append((u, n, e.A.phase))
        if e.done and e.A.idle() and e.B.idle():
            return e, u + 1, both, a_moves
    return e, MAX_UPDATES, both, a_moves


def check(plan, infeasible=None):
    e, n_upd, both, a_moves = run(plan)
    steps = plan["sequence"]
    assert e.done and n_upd < MAX_UPDATES, "not done after %d updates (step %s, A %s, B %s)" % (
        n_upd, e.b_step, e.A.phase, e.B.phase)
    assert e.failure is None and not e.timeouts, (e.failure, e.timeouts)
    assert len(e.placed) == len(steps), sorted(e.placed)
    assert not both, "both arms moving inside a braced step: %s" % both[:5]
    for s in steps:
        n = s["step"]
        if s["requires_brace"] and n != infeasible:
            assert {n} <= e.staged & e.braced & e.retracted & e.queued, n
            assert e.brace_by_step[n]["executed"] and not e.brace_by_step[n]["infeasible"]
    if infeasible is not None:
        assert infeasible not in e.braced | e.staged | e.queued, "the infeasible step was braced"
        assert e.brace_by_step[infeasible]["infeasible"] and not e.brace_by_step[infeasible]["planned_feasible"]
        assert not a_moves, "A moved during the infeasible step: %s" % a_moves[:3]
    return n_upd


def test_arch():
    print("arch updates:", check(load("arch")))


def test_S3():
    print("S3 updates:", check(load("S3")))


def test_all_braced():
    plan = all_braced(load("arch"))
    print("all-braced arch updates:", check(plan))
    plan = all_braced(load("S3"))
    print("all-braced S3 updates:", check(plan))


def test_infeasible_step():
    plan, n = one_infeasible(load("arch"))
    print("arch, step %d infeasible, updates:" % n, check(plan, infeasible=n))


def test_legacy_protocol_unchanged():
    """--legacy-brace keeps the old rules: A braces at step start, B's pre-insert waits."""
    e, n_upd, _, _ = run(load("arch"), legacy=True)
    assert e.done and not e.timeouts and e.failure is None and not e.staged, (n_upd, e.timeouts)


def test_timeout():
    """A wait nobody satisfies ends in a timeout callback after WAIT_TIMEOUT, and the arm goes on."""
    e = stub(load("arch"))
    seen = []
    e.B.on_timeout = lambda a, ph: seen.append(ph)
    e.B.push("stage", e.parks[e.B], 0.0, 0.0, 0.1, wait=lambda: False)
    ee = e.B.cmd[0].copy()
    for u in range(int((D.WAIT_TIMEOUT + 3) / DT)):
        ee = e.B.update(DT, ee)[0]
        if e.B.idle():
            break
    assert seen == ["stage"] and e.B.idle(), (seen, u * DT)
    assert D.WAIT_TIMEOUT < u * DT < D.WAIT_TIMEOUT + 2


def test_hold_at_measured_pose():
    """close completes -> queued hold/open/retract are re-aimed at the measured hand, not at brace_pose."""
    e = stub(load("arch"))
    n = 10
    e.b_step = n
    brace = e.plan["sequence"][n]["brace"]
    e.queue_brace(n, brace)
    A, ee = e.A, e.A.cmd[0].copy()
    off = np.array([0.0, 0.0, 0.0013])                  # the hand sits 1.3 mm off where it was told
    for _ in range(4000):
        p = A.update(DT, ee)[0]
        ee = p + (off if A.phase in ("close", "hold") else 0)
        if n in e.braced:
            break
    assert n in e.braced
    pose = np.array(brace["brace_pose"][:3])
    hold = next(m for m in A.moves if m["phase"] == "hold")
    assert np.allclose(hold["pos"], A.ee, atol=1e-9) and not np.allclose(hold["pos"], pose, atol=1e-6)
    assert np.allclose(hold["pos"], pose + off) and np.allclose(A.cmd[0], A.ee)
    # and a guard stop rebases the same way
    e2 = stub(load("arch"))
    e2.b_step = n
    e2.queue_brace(n, brace)
    A2, ee2 = e2.A, e2.A.cmd[0].copy()
    for _ in range(4000):
        ee2 = A2.update(DT, ee2)[0]
        if A2.phase == "brace-guard":
            break
    stop = ee2 + [0.0, 0.001, 0.0]
    A2.stop_here(stop)
    close = next(m for m in A2.moves if m["phase"] == "close")
    assert np.allclose(close["pos"], stop) and A2.cur is None and np.allclose(A2.cmd[0], stop)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
    print("test_arm_protocol OK")
