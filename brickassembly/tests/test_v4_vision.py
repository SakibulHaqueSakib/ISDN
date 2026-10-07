"""experiments/v4_vision.py: the G-V1 table logic on synthetic rows, the job-id invariance, the draws. No simulator, no GPU.

  - existing v4_brace job ids are unchanged (the vision field is part of the core only when given);
  - the E2 manipulation check (amended): the realised offset minus the mean realised offset of the context's controls vs the injected xy;
  - native contexts (a control that fails the screen) are excluded from the pass rates;
  - G-V1: INCONCLUSIVE takes precedence over ARCH / ACC; unavailable is never a pass; marginal = within 2 trials of the 95 % threshold;
  - the E2 draws follow the SHA-256 seed contract, one from the top decile; scale 2 = 2 e_xy + drift e_xy / |e_xy|, 2 dpsi;
  - E2b runs with --continue (spec key continue_, a new job id): every step with a snap is scored; steps after the first gate miss are
    after_failure, counted separately; agreement is over all scored steps and over the pre-failure ones; no snap = unobserved with a reason;
  - the ACC axis sweep: 31 contexts x 12 = 372 jobs (natives left out), the table per signed axis, largest passing level, non-monotone contexts;
  - the edge fallback's qualification (>= 20 cases and E2 scale-2 pass >= 95 % on the single-footprint contexts): true / false / unavailable;
    edge-steered steps count in the yield only if qualified; an fk_vision build an unqualified edge estimate steered fails (perception_v5);
  - E3 / the pool: one vector per step, the used estimate's error (aim.used_errors), typed by the plan; per-look statistics stay as an extra table;
  - exposed_single agrees with dual_arm_sim.exposed_cells on every E2 / E1b context.

    ~/Codes/CAIRSS/Issac/bin/python -m pytest tests/test_v4_vision.py     (or run it as a script)
"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from experiments import v4_brace as VB  # noqa: E402
from experiments import v4_vision as X  # noqa: E402


def trial(ctx, kind, screen=True, realised=None, scale=None, inj=None, k=0, typ="course"):
    return dict(ctx=ctx, type=typ, kind=kind, k=k, scale=scale, inj=inj, realised=realised, screen=screen)


def ctx_trials(ctx, controls=(True, True), real0=(1.0, 0.2), injected=(), typ="course"):
    """Two controls (screen outcomes `controls`, realised offset real0 each) and injected trials [(scale, k, inj, realised, screen)]."""
    ctx = (ctx, 0)
    out = [trial(ctx, "control", s, list(real0), typ=typ, k=i) for i, s in enumerate(controls)]
    return out + [trial(ctx, "inject", s, r, sc, list(i), k, typ) for sc, k, i, r, s in injected]


def test_existing_job_ids_unchanged():
    core = {"s": "28583c6d06dd", "n": 18, "a": None, "lam": 0.562643, "lat": [1.424469, -1.767465], "exe": "E0", "rep": 0}
    assert VB.core_id(VB.core_of("28583c6d06dd", 18, None, 0.562643, [1.424469, -1.767465], "E0", 0)) == "02f24af2ac947670"
    assert VB.core_of("28583c6d06dd", 18, None, 0.562643, [1.424469, -1.767465], "E0", 0) == core
    assert "v" not in VB.core_of("s", 1, None, None, None, "E0", 0, None) and "v" not in VB.core_of("s", 1, None, None, None, "E0", 0, {})
    j = VB.job("x", "shape:cube", None, "gt", None, v=X.vflags("gt", 0, shadow=True, record=True))
    assert j["core"]["n"] is None and j["core"]["v"]["_record"] is True and j["id"]
    spec = VB.cell_spec(j, "/r")                                  # a full build: no step-episode keys; runner directives are not spec keys
    assert "start_step" not in spec and "brace_json" not in spec and spec["look"] and spec["shadow_vision"] and spec["record_all"].startswith("/r/recordings/")
    assert "_record" not in spec


def test_manipulation_check_amended_rule():
    """Realised offsets are 1.0-1.2 mm with no injection: the check subtracts the controls' mean (here (1.0, 0.2))."""
    inj = (1.0, 0.0, 0.1)
    good = ctx_trials("a", injected=[(1, 0, inj, [2.0, 0.2], True)])                    # diff = (1.0, 0.0) = injected
    assert X.e2_analysis(good)["manipulation"]["frac"] == 1.0
    raw = ctx_trials("a", injected=[(1, 0, inj, [1.0, 0.2], True)])                      # diff = 0, injected 1.0 mm: the control offset is not the injection
    assert X.e2_analysis(raw)["manipulation"]["frac"] == 0.0
    edge = ctx_trials("a", injected=[(1, 0, inj, [2.0 + 0.099, 0.2], True)])             # off by 0.099 mm <= max(0.1, 10 % of 1.0): inside
    assert X.e2_analysis(edge)["manipulation"]["frac"] == 1.0
    out = ctx_trials("a", injected=[(1, 0, inj, [2.0 + 0.11, 0.2], True)])
    assert X.e2_analysis(out)["manipulation"]["frac"] == 0.0
    big = (4.0, 0.0, 0.0)                                                                # 10 % of 4 mm = 0.4 mm
    assert X.e2_analysis(ctx_trials("a", injected=[(2, 0, big, [1.0 + 4.39, 0.2], True)]))["manipulation"]["frac"] == 1.0
    assert X.e2_analysis(ctx_trials("a", injected=[(2, 0, big, [1.0 + 4.41, 0.2], True)]))["manipulation"]["frac"] == 0.0
    none = ctx_trials("a", injected=[(1, 0, inj, None, False)])                           # no realised offset: fails
    assert X.e2_analysis(none)["manipulation"]["frac"] == 0.0
    mix = ctx_trials("a", injected=[(1, k, inj, [2.0, 0.2] if k < 19 else [9.0, 0.0], True) for k in range(20)])
    assert X.e2_analysis(mix)["manipulation"]["frac"] == 0.95                            # 19 of 20 passes: >= 95 %


def test_native_exclusion():
    inj = [(1, 0, (1.0, 0.0, 0.0), [2.0, 0.2], False), (2, 0, (2.0, 0.0, 0.0), [3.0, 0.2], False)]
    ok = ctx_trials("ok", injected=[(1, 0, (1.0, 0.0, 0.0), [2.0, 0.2], True), (2, 0, (2.0, 0.0, 0.0), [3.0, 0.2], True)])
    nat1 = ctx_trials("n1", controls=(True, False), injected=inj)                         # one control fails the screen: native
    nat2 = ctx_trials("n2", controls=(False, False), injected=inj, typ="single")
    a = X.e2_analysis(ok + nat1 + nat2)
    assert sorted(a["native"]) == [["n1", 0], ["n2", 0]]
    assert a["n_native"] == 2 and a["included"] == 1
    assert a["pass"]["scale2"]["all"] == {"pass": 1, "n": 1}                              # the native contexts' failures do not count
    assert a["pass"]["scale1"]["course"] == {"pass": 1, "n": 1} and a["pass"]["scale1"]["single"] == {"pass": 0, "n": 0}
    assert X.e2_analysis(ctx_trials("only_one_control", controls=(True,)))["n_native"] == 1    # both controls are needed


def test_margin_marginal():
    n = 100                                                                               # threshold 95 passes
    assert X.margin_status(98, n) == "pass" and X.margin_status(97, n) == "marginal" and X.margin_status(93, n) == "marginal"
    assert X.margin_status(92, n) == "fail" and X.margin_status(100, n) == "pass"
    assert X.margin_status(123, 132) == "fail" and X.margin_status(124, 132) == "marginal"    # 0.95 x 132 = 125.4: two trials below is still marginal


def gate(v=(True,) * 4, c=(True,) * 4, marginal=False):
    return X.gate_v1({k: (x, "") for k, x in zip("abcd", v)}, {i + 1: (x, "") for i, x in enumerate(c)}, marginal)


def test_gate_order():
    assert gate() == "GO" and gate(marginal=True).startswith("GO (marginal")
    assert gate(v=(True, True, False, True), c=(False, True, True, True)) == "INCONCLUSIVE"       # validity before ARCH
    assert gate(v=(True, False, True, True), c=(True, False, True, True)) == "INCONCLUSIVE"       # before ACC
    assert gate(v=(True, True, True, False), c=(True, True, False, True)) == "INCONCLUSIVE"
    assert gate(v=(False, True, True, True)) == "INCONCLUSIVE"
    assert gate(c=(False, True, True, True)) == "ARCH"
    assert gate(c=(True, False, True, True)) == "ACC" and gate(c=(True, True, False, True)) == "ACC"
    assert gate(c=(True, False, True, False)) == "ACC"
    assert gate(c=(True, True, True, False)) == "LOOP"
    assert gate(v=(True, True, True, None)) == "unavailable"                                       # never a pass
    assert gate(c=(None, True, True, True)) == "unavailable" and gate(c=(True, None, True, True)) == "unavailable"
    assert gate(c=(True, True, True, None)) == "unavailable"
    assert gate(v=(None, True, True, False)) == "INCONCLUSIVE"                                     # a violated item decides even if another is unavailable


def test_e2b_agreement():
    steps = [("cube", 0, 0, True, True, False, None), ("cube", 0, 1, True, False, False, None), ("cube", 0, 2, False, False, True, "no_snap: x"),
             ("cube", 0, 3, True, True, True, None)]
    a = X.e2b_agreement(steps, {("cube", 0, 0): True, ("cube", 2, 0): True, ("cube", 3, 0): False})        # keys (shape, step, k): step 1 has no E2 match
    assert a["observed"] == 3 and a["unobserved"] == 1 and a["agree"] == 1 and a["no_e2_match"] == 1 and a["frac"] == 1 / 3
    assert a["pre_failure"] == {"observed": 2, "pass": 1, "agree": 1, "frac": 0.5} and a["after_failure_steps"] == 1
    assert a["after_failure"]["observed"] == 1 and a["after_failure"]["agree"] == 0 and a["unobserved_reasons"] == {"no_snap: x": 1}


def snap(screen, gate):
    return {"snap": {"screen_pass": screen, "gate_ok": gate}}


def test_e2b_continue_scoring():
    """A build with a gate miss at step 2 and no snap at step 5: steps 3, 4 are scored after_failure, step 5 is unobserved with the failure."""
    by = {"0": snap(True, True), "1": snap(False, True), "2": snap(False, False), "3": snap(True, True), "4": snap(False, False)}
    row = {"total": 6, "failure": "gate_miss", "failed_at_step": 2, "vision": {"by_step": by}}
    st = X.e2b_steps_of([({"meta": {"shape": "cube", "k": 0}}, {"row": row})])
    assert [s[3] for s in st] == [True] * 5 + [False] and [s[5] for s in st] == [False, False, False, True, True, True]
    assert st[5][6] == "no_snap: gate_miss at step 2" and st[0][6] is None
    e2 = {("cube", n, 0): v for n, v in enumerate([True, False, True, True, False])}
    a = X.e2b_agreement(st, e2)
    assert a["observed"] == 5 and a["unobserved"] == 1 and a["after_failure_steps"] == 2 and a["no_e2_match"] == 0
    assert a["agree"] == 4 and a["frac"] == 0.8 and a["pre_failure"]["observed"] == 3 and a["pre_failure"]["agree"] == 2
    assert a["after_failure"]["agree"] == 2
    row2 = dict(row, failure="joint_break", failed_at_step=3, vision={"by_step": {"0": snap(True, True), "1": snap(True, True), "3": snap(True, True)}})
    st2 = X.e2b_steps_of([({"meta": {"shape": "cube", "k": 1}}, {"row": row2})])
    assert [s[5] for s in st2] == [False, False, False, True, True, True]         # a non-gate failure at step 3: step 3 on is after_failure
    assert X.e2b_steps_of([({"meta": {"shape": "cube", "k": 1}}, {"row": None})]) == []


def test_e2b_spec_continue():
    pool = {"course": [[0.1 * i, -0.05 * i, 0.01 * i] for i in range(40)], "single": [[0.2 * i, 0.1 * i, 0.02 * i] for i in range(10)], "drift_max_mm": 0.3}
    pool["hash"] = X.pool_hash(pool)
    jobs = X.build_e2b(pool)
    for j in jobs:
        spec = VB.cell_spec(j, "/r")
        assert spec["continue_"] is True and spec["look"] and spec["aim"] == "fk_oracle" and "start_step" not in spec
    v = {k: x for k, x in jobs[0]["core"]["v"].items() if k != "continue_"}
    assert VB.job("e2b", jobs[0]["core"]["s"], None, "fk_oracle", None, rep=0, v=v)["id"] != jobs[0]["id"]      # the pre-continue E2b rows are not reused
    assert all("continue_" not in j["core"]["v"] for j in X.build_e1() + X.build_e1b() + X.build_e2(pool))


NATIVES = [["S5p13", 12], ["arch", 10]]


def test_sweep_jobs_and_table():
    jobs = X.build_sweep(NATIVES)
    assert len(jobs) == 372 and len({j["id"] for j in jobs}) == 372 and len(X.sweep_contexts(NATIVES)) == 31 and len(X.sweep_contexts([])) == 33
    assert not any(j["meta"]["ctx"] in NATIVES for j in jobs)
    assert len({(j["meta"]["axis"], j["meta"]["sign"], j["meta"]["level"]) for j in jobs}) == 12
    j = next(j for j in jobs if (j["meta"]["axis"], j["meta"]["sign"], j["meta"]["level"]) == ("short", -1, 0.6))
    assert j["core"]["v"]["aim_inject"] == {"dx_mm": 0.0, "dy_mm": -0.6, "dyaw_deg": 0.0} and j["core"]["n"] is not None
    assert "continue_" not in j["core"]["v"] and j["arm"] == "fk_oracle"
    # ctx a (course): long+ passes 0.3, 0.6, fails 1.0; ctx b (single): long+ fails 0.3, passes 0.6 (non-monotone), fails 1.0; long- all fail
    tr = lambda ctx, typ, axis, sign, level, ok: dict(ctx=ctx, type=typ, axis=axis, sign=sign, level=level, screen=ok)
    ts = [tr(("a", 0), "course", "long", 1, l, ok) for l, ok in ((0.3, True), (0.6, True), (1.0, False))]
    ts += [tr(("b", 1), "single", "long", 1, l, ok) for l, ok in ((0.3, False), (0.6, True), (1.0, False))]
    ts += [tr(("b", 1), "single", "long", -1, l, False) for l in X.SWEEP_LEVELS] + [tr(("a", 0), "course", "short", 1, l, True) for l in X.SWEEP_LEVELS]
    a = X.sweep_analysis(ts)
    assert set(a["axes"]) == {"long+", "long-", "short+", "short-"} and a["episodes"] == 12 and a["contexts"] == 2
    L = a["axes"]["long+"]
    assert L["pass_by_level"]["0.3"]["all"] == {"pass": 1, "n": 2} and L["pass_by_level"]["0.6"]["all"] == {"pass": 2, "n": 2}
    assert L["pass_by_level"]["0.6"]["single"] == {"pass": 1, "n": 1} and L["pass_by_level"]["1.0"]["course"] == {"pass": 0, "n": 1}
    assert L["largest_passing_by_context"] == {"a:0": 0.6, "b:1": 0.6} and L["non_monotone"] == ["b:1"]
    assert L["largest_passing_counts"]["single"] == {"0.6": 1}
    assert a["axes"]["long-"]["largest_passing_by_context"] == {"b:1": None} and a["axes"]["short+"]["largest_passing_by_context"] == {"a:0": 1.0}
    assert a["axes"]["long-"]["largest_passing_counts"]["single"] == {"None": 1} and a["axes"]["short-"]["pass_by_level"]["0.3"]["all"] == {"pass": 0, "n": 0}


def edge_looks(n, estimator="edge", typ="single"):
    return [dict(src="e1b", arm="fk_vision", shape="S1", step=i, seed=0, type=typ, yielded=True, yielded_stud=False, accepted=True, estimator=estimator,
                 vec=[0.1 * i, 0.05, 0.1], radial=0.1 + 0.01 * i) for i in range(n)]


def e2_edge_trials(looks, drift, screens):
    """E2 scale-2 single-footprint trials whose vectors are the edge ones (screens cycled), plus one trial with an unrelated vector."""
    out = [dict(ctx=("S1", i), type="single", kind="inject", k=0, scale=2, inj=X.scale2([round(x, 4) for x in l["vec"]], drift), realised=None,
                screen=screens[i % len(screens)]) for i, l in enumerate(looks)]
    return out + [dict(ctx=("S1", 99), type="single", kind="inject", k=0, scale=2, inj=[9.0, 9.0, 0.0], realised=None, screen=False)]


def test_edge_fallback_qualification():
    drift = 0.3
    looks = edge_looks(20) + edge_looks(5, estimator=None)
    pool = {"single": sorted([round(x, 4) for x in l["vec"]] for l in looks), "course": [], "drift_max_mm": drift}
    q = X.edge_qualification(looks, pool, e2_edge_trials(looks[:20], drift, [True]), [])
    assert q["qualified"] is True and q["n_cases"] == 20 and q["in_pool"] == 20 and q["e2_scale2_single"] == {"pass": 20, "n": 20}   # the unrelated trial is not counted
    assert q["rel_radial_mm_p95"][0] > 0 and abs(q["rel_yaw_deg_p95"][0] - 0.1) < 1e-9
    bad = X.edge_qualification(looks, pool, e2_edge_trials(looks[:20], drift, [True] * 18 + [False, False]), [])
    assert bad["qualified"] is False and bad["e2_scale2_single"] == {"pass": 18, "n": 20}                                  # 90 % < 95 %
    nat = X.edge_qualification(looks, pool, e2_edge_trials(looks[:20], drift, [True]), [["S1", i] for i in range(20)])      # native contexts are excluded
    assert nat["e2_scale2_single"]["n"] == 0 and nat["qualified"] is False
    few = X.edge_qualification(edge_looks(19), pool, e2_edge_trials(looks[:19], drift, [True]), [])
    assert few["qualified"] is False and few["n_cases"] == 19
    assert X.edge_qualification(looks, pool, [], [])["qualified"] is None and X.edge_qualification(looks, None, [], [])["qualified"] is None   # no E2: unavailable
    for none in ([], edge_looks(30, estimator=None), edge_looks(30, typ="course")):                                          # no edge single-footprint case
        assert "unavailable" in X.edge_qualification(none, pool, [], []) and X.edge_qualification(none, pool, [], [])["qualified"] is None


def used_ent(arm, shape, n, used, looks=None, snap=True, aim=None, failure=None, built=None, total=None):
    """One (episodes entry, record) with a step n: its looks (default two), the used estimate (`used`: None = not seen, or dict(estimators, errors))."""
    v = {"n_looks": 2, "looks": looks if looks is not None else [{"accepted": False}, {"accepted": False}]}
    if used is not None:
        v["used"] = used
    if snap:
        v["snap"] = dict(screen_pass=True, gate_ok=True, lateral_mm=0.1, dz_mm=0.0)
    row = {"vision": {"aim": aim or arm, "by_step": {str(n): v}}, "failure": failure, "built": built, "total": total}
    return {"arm": arm, "meta": {"shape": shape}}, {"row": row}


def err(dx=0.1, dy=0.0, dyaw=0.2):
    return {"rel_xy_mm": [dx, dy], "rel_yaw_deg": dyaw, "rel_radial_mm": math.hypot(dx, dy)}


def test_pool_is_one_vector_per_step_from_the_used_estimate():
    """E3 / the pool: ONE vector per step, the error of the estimate the controller steers with (the average of the accepted looks), typed from the
    plan (every step has two looks now); the per-look records stay as an extra table."""
    lk = lambda dx: {"accepted": True, "estimator": "studs", "errors": err(dx)}
    cube3 = ("cube", 3)                                              # a single-footprint step of cube (exposed_single)
    assert X.look_type("cube", 3) == "single" and X.look_type("cube", 0) == "course"
    ents = [used_ent("fk_vision", "cube", 3, dict(n_used=2, estimators=["studs", "studs"], errors=err(0.05, 0.0, 0.1)), [lk(0.3), lk(-0.2)]),   # two looks, one averaged vector
            used_ent("fk_vision", "cube", 0, dict(n_used=1, estimators=["studs"], errors=err(0.2, 0.1, -0.3)), [lk(0.2), {"accepted": False}]),
            used_ent("fk_oracle", "cube", 1, dict(n_used=0, estimators=[]), [{"accepted": False}, {"accepted": False}])]   # nothing accepted: no vector
    steps, looks = X.collect_v5([(ents, "e1")])
    assert len(steps) == 3 and len(looks) == 6                                           # one record per step; two per step in the per-look table
    assert [(s["type"], s["yielded"], s["n_used"]) for s in steps] == [("single", True, 2), ("course", True, 1), ("course", False, 0)]
    assert steps[0]["vec"] == [0.05, 0.0, 0.1] and steps[1]["vec"] == [0.2, 0.1, -0.3] and steps[2]["vec"] is None
    assert [l["vec"] and l["vec"][0] for l in looks[:2]] == [0.3, -0.2]                  # per-look vectors are still there
    sm = X.v5_summary(steps, looks)
    assert sm["all"]["yielded"] == 2 and abs(sm["all"]["yield"] - 2 / 3) < 1e-12 and sm["all"]["n_averaged"] == 1
    assert sm["single"]["radial_mm_max"] == 0.05 and sm["all"]["per_look"]["accepted_looks"] == 3 and sm["all"]["per_look"]["radial_mm_max"] == 0.3
    assert sm["course"]["yield"] == 0.5


def test_yield_counts_edge_only_if_qualified():
    u = lambda *est: dict(n_used=len(est), estimators=list(est), errors=err())
    ents = [used_ent("fk_vision", "S1", 1, u("studs")), used_ent("fk_vision", "S1", 2, u("edge")), used_ent("fk_vision", "S1", 3, u("studs", "edge")),
            used_ent("fk_vision", "S1", 4, dict(n_used=0, estimators=[]))]          # S1's steps 1-4: single footprints (a pier)
    steps, looks = X.collect_v5([(ents, "e1b")])
    assert [(s["yielded"], s["yielded_stud"], s["estimator"]) for s in steps] == [(True, True, "studs"), (True, False, "edge"), (True, False, "edge"), (False, False, None)]
    on, off = X.v5_summary(steps, looks, edge=True), X.v5_summary(steps, looks, edge=False)
    assert on["all"]["yielded"] == 3 and off["all"]["yielded"] == 1 and off["single"]["steps"] == 4 and off["single"]["yield"] == 0.25    # the edge steps do not count toward yield
    assert off["all"]["radial_mm_max"] is not None and on["all"]["yield"] == 0.75
    pool_like = [s["vec"] for s in steps if s["accepted"]]
    assert len(pool_like) == 3                                                           # the pool keeps the edge steps' vectors (E2 tests them at 2x for the qualification)


def test_unqualified_edge_fails_its_fk_vision_build():
    """Edge not qualified: an fk_vision build in which an edge estimate steered a step is a perception_v5 failure at that step (T-V1a, criterion 4);
    a stud-only build, an fk_oracle build with a shadow edge estimate, and a qualified edge are untouched."""
    u = lambda *est: dict(n_used=len(est), estimators=list(est), errors=err())
    ok_build = lambda arm, used, **kw: used_ent(arm, "cube", 0, used, built=1, total=1, **kw)
    studs, edge, mixed = ok_build("fk_vision", u("studs", "studs")), ok_build("fk_vision", u("studs", "edge")), ok_build("fk_vision", u("edge"))
    assert X.build_stats(studs[1], edge_ok=False)["ok"] is True and X.build_stats(studs[1])["edge_steps"] == []
    for e in (edge, mixed):
        bad, good = X.build_stats(e[1], edge_ok=False), X.build_stats(e[1], edge_ok=True)
        assert bad["ok"] is False and bad["failure"].startswith("perception_v5") and bad["edge_steps"] == [0]
        assert good["ok"] is True and good["failure"] is None and good["edge_steps"] == [0]          # qualified: it steers like any estimate
    shadow = ok_build("fk_oracle", u("edge"), aim="fk_oracle")                          # a shadow arm logs the estimate; it steers nothing
    assert X.build_stats(shadow[1], edge_ok=False)["ok"] is True and X.build_stats(shadow[1])["edge_steps"] == []
    real = used_ent("fk_vision", "cube", 0, u("edge"), built=0, total=1, failure="gate_miss")
    assert X.build_stats(real[1], edge_ok=False)["failure"] == "gate_miss"                # an existing failure is not overwritten
    # criterion 4 / T-V1a over three gated shapes x three seeds: one edge-steered build flips fk_vision's result, not fk_oracle's
    def e1(arm, edge_build):
        ents = []
        for shape in X.GATED:
            for seed in X.SEEDS:
                n = 1 if (shape, seed) == edge_build and arm == "fk_vision" else 0
                ent, rec = ok_build(arm, u("edge" if n else "studs"))
                ent["meta"] = {"shape": shape, "seed": seed, "rr": 0}
                ents.append((ent, rec))
        return ents
    ents = e1("fk_vision", ("arch", 1))
    assert X.e1_criterion(ents, "fk_vision", edge_ok=True)[0] is True
    assert X.e1_criterion(ents, "fk_vision", edge_ok=False)[0] is False
    t = X.t_v1a(ents, edge_ok=False)
    assert t["fk_vision/arch"]["built_all"] == 2 and t["fk_vision/arch"]["edge_steered_steps"] == [[1, [0]]] and t["fk_vision/cube"]["built_all"] == 3


def test_draws_and_scale():
    pool = {"course": [[0.1 * i, -0.05 * i, 0.01 * i] for i in range(40)], "single": [[0.2 * i, 0.1 * i, 0.02 * i] for i in range(10)], "drift_max_mm": 0.3}
    d = X.draw_vectors(pool, "cube", 0)
    assert d == X.draw_vectors(pool, "cube", 0) and len(d) == X.K_DRAWS and len({tuple(v) for v in d}) == X.K_DRAWS
    top = sorted(pool["course"], key=lambda v: math.hypot(v[0], v[1]))[-4:]
    assert d[0] in top                                                                          # one of the 4 from the top decile
    assert X.scale2([3.0, 4.0, 0.5], 0.5) == [6.3, 8.4, 1.0]                                    # 2 e_xy + 0.5 (0.6, 0.8); 2 dpsi
    assert X.scale2([0.0, 0.0, 0.5], 0.5) == [0.5, 0.0, 1.0]
    assert X.pool_hash(pool) == X.pool_hash(dict(pool, extra=1)) != X.pool_hash(dict(pool, drift_max_mm=0.4))


def test_exposed_single_agrees_with_cell():
    import dual_arm_sim as D
    for name, step in X.e2_contexts() + [(n, s) for n, ss in X.E1B for s in ss]:
        bricks = X.bricks_of(name)
        order = X.P.sequence(bricks)
        assert X.exposed_single(order, step, *X.nij(bricks)) == D.exposed_cells(order, step, *X.nij(bricks))[1], (name, step)
    assert len(X.e2_contexts()) == 33 and sum(len(s) for _, s in X.E1B) == 9
    X.check_s5_prefix()


def test_jobs_counts():
    assert len(X.build_e1()) == 36 and len({j["id"] for j in X.build_e1()}) == 36 and len(X.build_e1({"cube": 1})) == 45
    assert len(X.build_e1b()) == 54 and len({j["id"] for j in X.build_e1b()}) == 54
    pool = {"course": [[0.1 * i, -0.05 * i, 0.01 * i] for i in range(40)], "single": [[0.2 * i, 0.1 * i, 0.02 * i] for i in range(10)], "drift_max_mm": 0.3}
    pool["hash"] = X.pool_hash(pool)
    e2 = X.build_e2(pool)
    assert len(e2) == 330 and len({j["id"] for j in e2}) == 330
    assert len(X.build_e2b(pool)) == 6


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
