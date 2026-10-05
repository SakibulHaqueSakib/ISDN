"""RL bracing task library (plan_v4_rl_brace S0.1), CPU only:

    bash scripts/run.sh tests/test_brace_bandit.py

generator validity and determinism; mirror-invariant de-duplication and split purity; features are a
pure function of the plan (no hidden draw, no result file, no newton / warp); hidden() is random-access,
stratified and shared across arms; assign(strategy="learned") dispatches before the LP threshold.
"""

import builtins
import importlib
import inspect
import sys

import numpy as np
import pytest

import blueprint as BP
import bracing
import planner as P
import stability as ST
from tasks import brace_bandit as BB

E0 = BB.EXE["E0"]


@pytest.fixture(scope="module")
def structs():
    return BB.unique_structures(12)


def test_generator_valid_and_deterministic():
    for seed in range(40):
        b = BB.corbel_family(seed)
        assert 5 <= len(b) <= 24, seed
        assert BP.components(b) == 1, seed
        assert all(P.footprint(x[1], x[5])[0] == 2 for x in b), seed          # depth 2 in x
        assert BP.tile(BB._voxels(BB._cells(b))) == b                         # the tiler accepts its own voxels
        assert b == BB.corbel_family(seed)
        assert len(BB.feeder_slots(len(b))) == len(b)                         # the cell has a slot per brick


def test_mirror_shares_key_and_split_is_disjoint(structs):
    for seed in range(20):
        b = BB.corbel_family(seed)
        m = BP.tile(BB._voxels(BB._mirror(BB._cells(b))[1]))
        assert BB.canonical_key(m) == BB.canonical_key(b)
    keys = [e["key"] for e in structs]
    assert len(set(keys)) == len(keys)                                         # de-duplicated
    for e in structs:                                                          # canonical orientation is a function of the key
        assert e["bricks"] == BB._oriented(e["key"]) and BB.canonical_key(e["bricks"]) == e["key"]
    train, val, test = BB.split("train"), BB.split("val"), BB.split("test", 5)
    assert (len(train), len(val), len(test)) == (72, 18, 5)
    ks = [{e["key"] for e in s} for s in (train, val, test)]
    assert not (ks[0] & ks[1] or ks[0] & ks[2] or ks[1] & ks[2])
    assert [e["seed"] for e in train] == sorted(e["seed"] for e in train)
    assert BB.split("test", 3) == test[:3]                                     # on demand, prefix-stable


@pytest.fixture(scope="module")
def critical():
    for e in BB.unique_structures(12):
        for c in BB.contexts(e["bricks"]):
            if c["pool"] == "critical":
                return c
    pytest.skip("no critical context among 12 structures")


def test_features_pure(critical):
    c = critical
    acts = BB.actions(c)
    assert acts[0] is None and all(len(a) == 3 and a[2] in BB.LEANS for a in acts[1:])
    assert list(inspect.signature(BB.features).parameters) == ["bricks", "step", "action", "exe"]
    sample = [acts[0]] + acts[1:60:7]
    ref = [BB.features(c["bricks"], c["step"], a, E0) for a in sample]
    assert all(v.shape == (BB.NF,) and np.isfinite(v).all() for v in ref)
    key = [list(k) for k in c["key"]]
    for stage in ("a", "b"):                                                   # hidden draws in between change nothing
        for r in range(4):
            BB.hidden(stage, c["key"], c["step"], r, 4)
    BB._context_features.cache_clear(); BB._candidate_features.cache_clear(); BB._util.cache_clear()
    real_open = builtins.open

    def no_results(path, *a, **k):
        assert "results" not in str(path), "features read %s" % path
        return real_open(path, *a, **k)

    builtins.open = no_results                                                 # results/ absent as far as features can tell
    try:
        again = [BB.features(c["bricks"], c["step"], a, E0) for a in sample]
    finally:
        builtins.open = real_open
    assert all(np.array_equal(x, y) for x, y in zip(ref, again))
    # the frame is internal: a caller's origin (the planner's, or another structure's) changes nothing
    old = P.VOXEL_ORIGIN
    P.VOXEL_ORIGIN = (0.1, 0.2, 0.3)
    try:
        assert np.array_equal(BB.features(c["bricks"], c["step"], sample[-1], E0), ref[-1])
        assert P.VOXEL_ORIGIN == (0.1, 0.2, 0.3)
    finally:
        P.VOXEL_ORIGIN = old


def test_features_without_newton_or_warp(critical, monkeypatch):
    for name in ("newton", "warp", "torch"):
        monkeypatch.setitem(sys.modules, name, None)                           # import raises ImportError
    with pytest.raises(ImportError):
        import newton  # noqa: F401
    monkeypatch.delitem(sys.modules, "tasks.brace_bandit")
    fresh = importlib.import_module("tasks.brace_bandit")
    c = critical
    a = fresh.actions(c)[3]
    assert np.array_equal(fresh.features(c["bricks"], c["step"], a, E0), BB.features(c["bricks"], c["step"], a, E0))
    monkeypatch.setitem(sys.modules, "tasks.brace_bandit", BB)


def test_hidden():
    key = BB.unique_structures(1)[0]["key"]
    R = 8
    fwd = [BB.hidden("s", key, 5, r, R) for r in range(R)]
    assert fwd == [BB.hidden("s", key, 5, r, R) for r in reversed(range(R))][::-1]      # random access
    assert fwd[3] == BB.hidden("s", key, 5, 3, R)
    lam = sorted(int((d["lam"] - 0.5) / 0.8 * R) for d in fwd)
    assert lam == list(range(R))                                               # one draw per equal-width stratum
    assert 0.5 <= min(d["lam"] for d in fwd) and max(d["lam"] for d in fwd) <= 1.3
    assert all(0 <= d["lat_mag"] <= 3 for d in fwd)
    assert len({int(d["lat_angle"] / (2 * np.pi) * R) for d in fwd}) >= 4      # sectors spread the circle (phase-shifted)
    assert BB.hidden("s", key, 5, 2, R) != BB.hidden("t", key, 5, 2, R)        # stages differ
    assert BB.hidden("s", key, 5, 2, R) != BB.hidden("s", key, 6, 2, R)        # contexts differ
    assert BB.hidden("s", key, 5, 2, R, stratified=False) == BB.hidden("s", key, 5, 2, R, stratified=False)
    assert BB.hidden("s", key, 5, 2, R, stratified=False) != fwd[2]
    band = (0.9, 1.3)
    assert all(0.9 <= BB.hidden("s", key, 5, r, R, band=band)["lam"] <= 1.3 for r in range(R))
    # arms share draws within a stage: the draw takes no arm argument and is a pure function of its tuple
    assert "arm" not in inspect.signature(BB.hidden).parameters


def test_hidden_job_order_and_workers():
    """A6: draws do not depend on job order or on the worker count."""
    from concurrent.futures import ProcessPoolExecutor
    key = BB.unique_structures(1)[0]["key"]
    jobs = [(st, n, r) for st in ("a", "b") for n in (3, 7) for r in range(6)]
    ref = {j: BB.hidden(j[0], key, j[1], j[2], 6) for j in jobs}
    shuffled = list(jobs)
    np.random.default_rng(1).shuffle(shuffled)
    assert {j: BB.hidden(j[0], key, j[1], j[2], 6) for j in shuffled} == ref
    with ProcessPoolExecutor(3) as ex:
        futs = {j: ex.submit(BB.hidden, j[0], key, j[1], j[2], 6) for j in shuffled}
        assert {j: f.result() for j, f in futs.items()} == ref


def test_pickers_and_screen(critical):
    c = critical
    assert BB.pick_none(c) is None
    acts = set(BB.actions(c))
    sh, co = BB.pick_lp_shared(c, E0), BB.pick_lp_committed(c, E0)
    assert sh in acts and (co is None or co in acts)
    assert BB.pick_lp_shared(dict(c, u0={1.0: 0.0}), E0) is None
    assert BB.pick_random(c, np.random.default_rng(0)) in acts
    scr = BB.screen_candidates(c, E0)
    assert len(scr) == len(set(scr)) == 12 and set(scr) <= acts and scr == BB.screen_candidates(c, E0)
    row = lambda i, s, u, aa=False, lp=1.0, dy=0.0: dict(index=i, success=s, u2_peak=u, arm_arm=aa, lp_u=lp, dy=dy)
    assert BB.screen_pick([row(0, True, 0.5), row(1, True, 0.3), row(2, False, 0.1)])["index"] == 1
    assert BB.screen_pick([row(0, True, 0.30, lp=0.9), row(1, True, 0.305, lp=0.5)])["index"] == 1     # tie: lower LP u
    assert BB.screen_pick([row(0, True, 0.30, lp=0.5, dy=.02), row(1, True, 0.30, lp=0.5, dy=.01)])["index"] == 1
    assert BB.screen_pick([row(0, False, 0.9, aa=True), row(1, False, 0.95)])["index"] == 1            # no success: skip arm_arm
    assert BB.screen_pick([row(0, False, 0.9, aa=True), row(1, False, 0.95, aa=True)])["index"] == 0


def test_learned_dispatch_precedes_threshold():
    s3 = tuple(P.sequence(P.STRUCTURES["S3"]))
    n = 9                                                                      # u0 x SF < 1 here
    brick, placed = s3[n], list(s3[:n])
    assert bracing.predicted(brick, placed).s * ST.SAFETY_FACTOR < 1.0
    assert bracing.assign(brick, placed, "weakest_joint") is None              # the heuristics stand down
    y, z = bracing.candidates(brick, placed, clearance=lambda *a: True)[0]
    b = bracing.assign(brick, placed, "learned", selector=lambda br, pl: (y, z, 0), grip_N=14.3, mu=0.7)
    assert b["feasible"] and b["strategy"] == "learned" and b["brace_pose"][1:] == [round(y, 5), round(z, 5)]
    assert b["brace_tilt_rotvec"] == [0.0, 0.0, 0.0] and b["brace_force_N"] == 14.3 and b["brace_mu"] == 0.7
    b45 = bracing.brace_at(brick, placed, y, z, 45, 14.3, 0.7)
    assert b45["brace_tilt_rotvec"] == [round(np.pi / 4, 4), 0.0, 0.0]
    assert bracing.assign(brick, placed, "learned", selector=lambda br, pl: None) is None
    with pytest.raises(ValueError):
        bracing.assign(brick, placed, "learned")
    assert bracing.assign(brick, [], "learned", selector=lambda br, pl: (y, z, 0)) is None   # still nothing to brace
    # defaults unchanged: the scripted lean and the committed strategies' records are what they were
    r = bracing.assign(s3[13], list(s3[:13]), "weakest_joint")
    assert r["brace_tilt_rotvec"] == [round(v, 4) for v in bracing.lean_for(s3[13], r["brace_pose"][1])]


def test_contexts_need_sequence_order():
    s3 = tuple(P.STRUCTURES["S3"])                                             # raw tuples are not in sequence order
    assert s3 != tuple(P.sequence(list(s3)))
    with pytest.raises(AssertionError, match="sequence order"):
        BB.contexts(s3)


# --- the R0 runner (experiments/v4_brace.py): job ids, de-duplication, the pool; no newton, no cell ---------------

def test_job_ids_stable_and_order_independent():
    from experiments import v4_brace as V
    sid, act = next(iter(V.registry())), (0.0125, 0.0192, 45)
    a = V.job("signal", sid, 7, "none", None, 0.8123456789, [1.0, -2.0], "E0", 0, "signal", 2, {"pool": "critical"})
    b = V.job("harm", sid, 7, "lp_shared", None, 0.8123456789, [1.0, -2.0], "E0", 0, "other", 9, {})     # other labels, same run
    assert a["id"] == b["id"] and len(a["id"]) == 16
    assert a["id"] == V.core_id(dict(reversed(list(a["core"].items()))))                            # key order is irrelevant
    assert V.job("x", sid, 7, "a", act, exe="E0")["id"] != V.job("x", sid, 7, "a", act, exe="E1")["id"]
    assert V.job("x", sid, 7, "a", act, rep=0)["id"] != V.job("x", sid, 7, "a", act, rep=1)["id"]
    assert V.job("x", sid, 7, "a", act, 1.0, V.NO_LATERAL)["id"] != V.job("x", sid, 7, "a", act)["id"]   # fixture on / off
    assert V.job("x", sid, 7, "a", act + (), 1.0)["id"] == V.job("x", sid, 7, "a", list(act), 1.0)["id"]
    assert V.job("x", sid, 7, "a", ("sham", None))["id"] is None                                   # staging time not known yet
    assert V.job("x", sid, 7, "a", ("sham", 5.6))["id"] != V.job("x", sid, 7, "a", ("sham", 5.7))["id"]
    ids = [V.job("x", sid, n, "a", None, 1.0, V.NO_LATERAL)["id"] for n in range(30)]
    assert len(set(ids)) == 30
    spec = V.cell_spec(a, V.ROOT)                                                                  # keys the cell's spec parser accepts
    assert {"bricks", "name", "plan", "start_step", "only_step", "brace_json", "press_scale", "press_lateral", "exe", "out", "tag", "repeat"} == set(spec)
    assert V.cell_spec(V.job("x", "shape:cube", 7, "a", None), V.ROOT)["shape"] == "cube"


def test_run_jobs_dedupes_and_builds_each_plan_once(tmp_path, monkeypatch):
    import json
    import threading
    import time
    from experiments import v4_brace as V
    started, lock, running = [], threading.Lock(), {}

    def fake(j, root, timeout):
        sid = j["core"]["s"]
        with lock:
            started.append((sid, V.plan_path(root, sid).exists()))
            running[sid] = running.get(sid, 0) + 1
            assert running[sid] == 1 or V.plan_path(root, sid).exists(), "two jobs built one plan at once"
        time.sleep(0.1)
        V.plan_path(root, sid).parent.mkdir(parents=True, exist_ok=True)
        V.plan_path(root, sid).write_text("{}")
        V.rows_dir(root).mkdir(parents=True, exist_ok=True)
        rec = {"id": j["id"], "core": j["core"], "meta": {}, "wall_s": 0.1, "row": {"episode": {"success": True}}}
        (V.rows_dir(root) / (j["id"] + ".json")).write_text(json.dumps(rec))
        with lock:
            running[sid] -= 1
        return rec

    monkeypatch.setattr(V, "run_one", fake)
    jobs = [V.job("e1", s, n, "none", None) for s in ("a", "b") for n in range(1, 4)]
    jobs += [V.job("e2", "a", 1, "other_label", None)]                                             # a duplicate of e1's first job
    st = V.run_jobs(jobs, tmp_path, 4)
    assert st["unique"] == 6 and st["ran"] == 6 and not st["errors"]
    first = {}
    for sid, had in started:
        first.setdefault(sid, had)
    assert first == {"a": False, "b": False}                                                       # each plan built by exactly one job
    assert V.run_jobs(jobs, tmp_path, 4)["ran"] == 0                                               # resume: rows exist, nothing runs
    assert V.merge("e2", jobs[-1:], tmp_path)["with_record"] == 1
