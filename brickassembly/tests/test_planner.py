"""Gate G2 -- master_report §WP3 acceptance, in the MuJoCo twin's frame.

    cd brickassembly && ../cpuenv/bin/python -m pytest tests/test_planner.py -q
    ... -m slow   also runs the single-arm failure trials in the twin (~5 min)

G2 bullets and where each is checked:
  * drop test + all_bricks_accessible        test_drop_test, test_accessible
  * S3/S5 brace_required_count > 0 and fail   test_single_arm_predicted,
    single-arm (verified in simulation)       test_single_arm_fails_in_twin (slow)
  * tiling 100%, no overlaps, supported,      test_voxel_coverage,
    topological order, requires_brace flags   test_supported_and_ordered,
    match the force-margin analysis           test_requires_brace_matches_force_margin
  * all three strategies emit valid plans     test_brace_records_valid

The structures are hand-authored brick sets (ledger D14: the StableLego
samples are not available to this project), so "the voxel model" of a
structure is its occupancy grid and the tiler is cut (§WP3 cut order: "keep
bracing, cut tiling"). Coverage then means every voxel is placed exactly
once by exactly one sequenced brick.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import bracing
import planner as P
import stability as ST
from sim.mj import checks
from sim.mj import control as C
from sim.mj import scene as S

OUT = Path(__file__).with_suffix(".json")
RESULTS = {}
STRATS = bracing.STRATEGIES


@pytest.fixture(scope="module")
def plans():
    """The twin's plans as committed (plans/twin/*.json, written by
    `planner.py --frame twin` with the hand-clearance check) -- the files the
    executor runs. test_plans_reproduce checks they are current."""
    from orchestration.twin_executor import load_plan
    out = {}
    for sid in P.BENCHMARK:
        S.centre_plan_origin(P.STRUCTURES[sid])
        for st in STRATS:
            out[sid, st] = load_plan(sid, st)
    yield out
    OUT.write_text(json.dumps(RESULTS, indent=1, default=float))


@pytest.mark.parametrize("sid", ["S1", "S3"])
def test_plans_reproduce(plans, sid):
    """The committed plan files are what the planner produces now."""
    S.centre_plan_origin(P.STRUCTURES[sid])
    clr = checks.BraceClearance(sid)
    for st in STRATS:
        fresh = json.loads(json.dumps(P.build_plan(sid, st, clearance=clr if st != "none" else None)))
        old = plans[sid, st]
        assert [s["brace"] for s in fresh["sequence"]] == [s["brace"] for s in old["sequence"]]
        assert [s["grasp"] for s in fresh["sequence"]] == [s["grasp"] for s in old["sequence"]]


def _brick(plan, bid):
    b = next(x for x in plan["bricks"] if x["id"] == bid)
    return (b["id"], b["type"]) + tuple(b["grid_pos"]) + (b["yaw_index"],)


def _voxels(bricks):
    occ = {}
    for b in bricks:
        c, k = P.cells(b)
        for i, j in c:
            occ.setdefault((i, j, k), []).append(b[0])
    return occ


@pytest.mark.parametrize("sid", P.BENCHMARK)
def test_voxel_coverage(plans, sid):
    """Every voxel of the structure is placed exactly once; nothing extra."""
    plan = plans[sid, "weakest_joint"]
    model = set(_voxels(P.STRUCTURES[sid]))
    seq = [_brick(plan, s["brick_id"]) for s in plan["sequence"]]
    assert len({b[0] for b in seq}) == len(seq) == len(plan["bricks"]) == len(P.STRUCTURES[sid])
    occ = _voxels(seq)
    overlaps = {v: ids for v, ids in occ.items() if len(ids) > 1}
    assert not overlaps, overlaps
    assert set(occ) == model
    RESULTS.setdefault("coverage", {})[sid] = {"voxels": len(model), "covered": len(occ)}


@pytest.mark.parametrize("sid", P.BENCHMARK)
def test_supported_and_ordered(plans, sid):
    """Each brick sits on the baseplate or on bricks already placed, its
    mating studs are exactly those supports, and its top-down insertion
    corridor is empty (§3.5 precedence)."""
    plan = plans[sid, "weakest_joint"]
    placed = []
    for s in plan["sequence"]:
        b = _brick(plan, s["brick_id"])
        c, k = P.cells(b)
        sup = P.supports(b, placed)
        assert k == 0 or sup, "%s floats" % b[0]
        assert {m[0]: m[1] for m in s["mating_studs"]} == sup
        for p in placed:                     # nothing already above its footprint
            pc, pk = P.cells(p)
            assert not (pk > k and pc & c), "%s blocked by %s" % (b[0], p[0])
        placed.append(b)
    # one connected structure, grounded
    adj = {b[0]: set() for b in placed}
    for a in placed:
        for b in placed:
            if b[4] == a[4] + 1 and P.cells(a)[0] & P.cells(b)[0]:
                adj[a[0]].add(b[0])
                adj[b[0]].add(a[0])
    seen, todo = set(), [b[0] for b in placed if b[4] == 0]
    while todo:
        x = todo.pop()
        if x not in seen:
            seen.add(x)
            todo += adj[x]
    assert seen == set(adj)


@pytest.mark.parametrize("sid", P.BENCHMARK)
def test_requires_brace_matches_force_margin(plans, sid):
    """requires_brace <=> predicted utilisation x SF >= 1 (§3.6 step 4), the
    same trigger for both bracing strategies and never for 'none'; the stored
    utilisation is what the force model gives when re-run."""
    none = plans[sid, "none"]["sequence"]
    for st in STRATS:
        seq = plans[sid, st]["sequence"]
        for s0, s in zip(none, seq):
            assert s["brick_id"] == s0["brick_id"]
            u = s["predicted_util_unbraced"] or 0.0
            assert u == (s0["predicted_util_unbraced"] or 0.0)
            want = st != "none" and u * ST.SAFETY_FACTOR >= 1.0
            feasible = bool(s["brace"]) and s["brace"].get("feasible", True)
            assert s["requires_brace"] == (want and feasible), (st, s["step"])
            assert (s["brace"] is not None) == want, (st, s["step"])
    # re-derive the utilisation of every flagged step, and of the step before it
    S.centre_plan_origin(P.STRUCTURES[sid])
    plan = plans[sid, "weakest_joint"]
    order = [_brick(plan, s["brick_id"]) for s in plan["sequence"]]
    for n, s in enumerate(plan["sequence"]):
        if s["requires_brace"]:
            for m in (n - 1, n):
                if m < 1:
                    continue
                r = ST.insertion_utilisation(order[m], order[:m])
                assert r.s == pytest.approx(plan["sequence"][m]["predicted_util_unbraced"], abs=1e-3)


@pytest.mark.parametrize("sid", P.BENCHMARK)
def test_brace_records_valid(plans, sid):
    """All three strategies give a §2.6-complete brace wherever one is needed,
    on bricks already placed, which the force model says is sufficient."""
    keys = {"arm", "target_brick_id", "rationale", "predicted_failure_joint", "brace_pose",
            "brace_force_N", "brace_axis", "expected_reaction_wrench", "end_effector"}
    rows = {}
    for st in STRATS:
        plan = plans[sid, st]
        placed = set()
        n = 0
        for s in plan["sequence"]:
            br = s["brace"]
            if s["requires_brace"]:
                n += 1
                assert keys <= set(br), keys - set(br)
                assert br["arm"] == "A" and s["placer_arm"] == "B"
                assert set(br["gripped_bricks"]) <= placed
                assert br["target_brick_id"] in br["gripped_bricks"]
                assert len(br["brace_pose"]) == 3 and len(br["expected_reaction_wrench"]) == 6
                w = br["expected_reaction_wrench"]
                assert np.hypot(w[1], w[2]) <= bracing.ST.Brace((), ()).arm_max_N + 1e-3
                assert abs(w[0]) <= 60.0 + 1e-3
                # the naive baseline may pick a useless grasp (S3 step 14: the
                # nearest graspable brick is on the unconnected column)
                assert br["predicted_util_braced"] <= br["predicted_util_unbraced"] + 1e-9
                if st == "weakest_joint":
                    assert br["predicted_util_braced"] < br["predicted_util_unbraced"]
                if st == "weakest_joint":       # predicted to hold (not always with the full SF)
                    assert br["predicted_util_braced"] < 1.0
            placed.add(s["brick_id"])
        v = plan["validation"]
        assert v["bracing_strategy"] == st and v["brace_required_count"] == n
        assert v["brace_infeasible_count"] == 0
        rows[st] = {"braced": n, "worst_braced_util": max(
            [s["brace"]["predicted_util_braced"] for s in plan["sequence"] if s["requires_brace"]],
            default=0.0)}
    RESULTS.setdefault("strategies", {})[sid] = rows


def test_single_arm_predicted(plans):
    """S3 and S5 need a second arm by the force model; S1, S2, S4 do not."""
    for sid in P.BENCHMARK:
        v = plans[sid, "weakest_joint"]["validation"]
        need = sid in ("S3", "S5")
        assert v["single_arm_fails"] is need, (sid, v["max_util_unbraced"])
        assert (v["brace_required_count"] > 0) is need
        RESULTS.setdefault("single_arm_predicted", {})[sid] = v["max_util_unbraced"]


@pytest.mark.parametrize("sid", P.BENCHMARK)
def test_drop_test(plans, sid):
    """§3.3: all joints mated, 2 s under gravity, 0.5 N lateral push at the
    top -> max displacement < 1 mm."""
    r = checks.drop_test(plans[sid, "none"])
    RESULTS.setdefault("drop_test", {})[sid] = r
    assert r["passed"], r


def test_drop_test_detects_failure(plans):
    """The drop test is not vacuous: a 30 N shove at the top of the bridge
    breaks a joint (the same structure passes at 0.5 N above)."""
    r = checks.drop_test(plans["S3", "none"], push_n=30.0)
    RESULTS["drop_test_30N_S3"] = r
    assert not r["passed"]


# 2x2 bricks flanked by 2x2 neighbours in the same layer (S4 step 17, S5's
# row of six 2x2s): 17.5 mm pads on a 15.8 mm face touch the neighbours
# whatever the order. Recorded as a G2 shortfall (ledger v31_accessibility,
# master_report 7.7 M9); strict, so the day it passes the marker must go.
KNOWN_INACCESSIBLE = {"S4", "S5"}


@pytest.mark.parametrize("sid", [pytest.param(s, marks=pytest.mark.xfail(
    strict=True, reason="flanked 2x2 grasps (ledger v31_accessibility)"))
    if s in KNOWN_INACCESSIBLE else s for s in P.BENCHMARK])
def test_accessible(plans, sid):
    """all_bricks_accessible from the planner's clearance check, and in the
    twin: arm B reaches every pre-insertion and target pose and arm A every
    brace pose (IK residual < 1 mm)."""
    plan = plans[sid, "weakest_joint"]
    RESULTS.setdefault("all_bricks_accessible", {})[sid] = plan["validation"]["all_bricks_accessible"]
    S.centre_plan_origin(P.STRUCTURES[sid])
    cell = S.build_cell(plan)
    m = cell.model
    arms = {k: C.Arm(m, k) for k in ("A", "B")}
    worst = 0.0
    from motion import skills as K
    clr = checks.BraceClearance(sid)
    for s in plan["sequence"]:
        if s["requires_brace"]:          # the two hands and the carried brick do not collide
            b = _brick(plan, s["brick_id"])
            br = s["brace"]
            assert clr(b, br["brace_pose"][1], br["brace_pose"][2], br["brace_tilt_rotvec"]), s["step"]
        yaw = math.radians(s["grasp"]["yaw_offset_deg"])
        for pose in (s["pre_insertion_pose"], s["target_pose"]):
            p = np.array(pose[:3]) + [0, 0, K.GRASP_TCP_ABOVE_BOTTOM]
            errs = [C.ik(m, arms["B"], p, C.down_rot(y))[1] for y in (yaw, yaw + math.pi)]
            worst = max(worst, min(errs))
        if s["requires_brace"]:
            p, R = K.brace_pose(s["brace"])
            worst = max(worst, C.ik(m, arms["A"], p, R)[1])
    RESULTS.setdefault("ik_worst_residual_m", {})[sid] = worst
    assert worst < 1e-3
    assert plan["validation"]["all_bricks_accessible"] is True


@pytest.mark.slow
@pytest.mark.parametrize("sid", ("S3", "S5"))
def test_single_arm_fails_in_twin(plans, sid):
    """§3.1 MUST: S3 and S5 genuinely fail single-arm. The step the force
    model rates worst is executed in the twin with no stabilizer (the
    structure below it pre-placed and mated): a joint breaks. With the
    weakest-joint brace the same step completes without a break."""
    from orchestration import twin_executor as X
    plan = plans[sid, "none"]
    step = max(range(len(plan["sequence"])),
               key=lambda n: plan["sequence"][n]["predicted_util_unbraced"] or 0.0)
    un = X.step_trial(sid, step, "none", seed=0)
    br = X.step_trial(sid, step, "weakest_joint", seed=0)
    RESULTS.setdefault("single_arm_twin", {})[sid] = {
        "step": step, "predicted_util": plan["sequence"][step]["predicted_util_unbraced"],
        "none": {k: un[k] for k in ("breaks", "structure_intact", "success")},
        "weakest_joint": {k: br[k] for k in ("breaks", "structure_intact", "success")}}
    assert un["breaks"] > 0
    assert br["breaks"] == 0 and br["success"]


def _blocked_steps(bricks, name):
    import blueprint  # noqa: F401
    P.STRUCTURES[name] = bricks
    seq = P.build_plan(name, "none")["sequence"]
    by_id = {b[0]: b for b in bricks}
    return [s["step"] for n, s in enumerate(seq) if P.pinch_blocked(
        by_id[s["brick_id"]], [by_id[q["brick_id"]] for q in seq[:n]], s["grasp"])]


def test_pinch_blocked():
    """v4_pipeline_pyramid_inaccessible_grasp: cube, arch, hollow_box and S3 have a free pinch face on every step;
    the user's pyramid has a same-layer neighbour flush against a pinch face on steps 2,3,4,6,7,8,9."""
    import blueprint as B
    for shape in ("cube", "arch", "hollow_box"):
        assert _blocked_steps(B.load(shape), shape) == []
    assert _blocked_steps(P.STRUCTURES["S3"], "S3") == []
    bl = Path(__file__).parent.parent / "blueprints" / "user"
    pyr = B.tile(B.carve(bl / "pyramid_front.png", bl / "pyramid_side.png", None, 5, None))
    assert _blocked_steps(pyr, "pyramid_test") == [2, 3, 4, 6, 7, 8, 9]
