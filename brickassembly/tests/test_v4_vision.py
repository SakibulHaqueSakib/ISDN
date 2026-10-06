"""experiments/v4_vision.py: the G-V1 table logic on synthetic rows, the job-id invariance, the draws. No simulator, no GPU.

  - existing v4_brace job ids are unchanged (the vision field is part of the core only when given);
  - the E2 manipulation check (amended): the realised offset minus the mean realised offset of the context's controls vs the injected xy;
  - native contexts (a control that fails the screen) are excluded from the pass rates;
  - G-V1: INCONCLUSIVE takes precedence over ARCH / ACC; unavailable is never a pass; marginal = within 2 trials of the 95 % threshold;
  - the E2 draws follow the SHA-256 seed contract, one from the top decile; scale 2 = 2 e_xy + drift e_xy / |e_xy|, 2 dpsi;
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
    steps = [("cube", 0, 0, True, True), ("cube", 0, 1, True, False), ("cube", 0, 2, False, False)]
    a = X.e2b_agreement(steps, {("cube", 0, 0): True, ("cube", 2, 0): True})                  # keys (shape, step, k): step 1 has no E2 match
    assert a["observed"] == 2 and a["unobserved"] == 1 and a["agree"] == 1 and a["no_e2_match"] == 1 and a["frac"] == 0.5


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
