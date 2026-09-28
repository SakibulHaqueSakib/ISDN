"""Statistics and figures -- master_report §4.2 / WP8.

    python -m experiments.analyze            # -> results/analysis.json, results/figures/

§4.2: mean +- 95% CI over seeds; for A6 a PAIRED test across matched seeds
with an effect size. Success is binary, so:
  * rates with Wilson 95% intervals,
  * paired comparisons by the exact McNemar test on the discordant seeds,
    effect size = difference in success rate (paired) and its 95% CI,
  * peak placer force: paired Wilcoxon on seeds where both conditions ran,
    reported as % change (the [DualArmSnapFit] benchmark is -30%).
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
STRATS = ("none", "nearest", "weakest_joint")


def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a, b):
    """a, b: {seed: bool}. Exact two-sided McNemar on matched seeds, and the
    paired difference in success rate (b - a) with a normal 95% CI."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    x = np.array([a[s] for s in seeds], float)
    y = np.array([b[s] for s in seeds], float)
    n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    d = y - x
    diff = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"n_pairs": len(seeds), "b_only": n01, "a_only": n10, "p_exact": float(p),
            "diff": diff, "diff_ci95": [diff - 1.96 * se, diff + 1.96 * se]}


def a6():
    trials = [r for r in rows(RES / "a6" / "trials.jsonl") if r.get("kind") == "step"]
    out = {"cells": {}, "paired": {}, "peak_force": {}, "brace_prediction": {}}
    by = defaultdict(dict)
    force = defaultdict(dict)
    for r in trials:
        key = (r["structure_id"], r["step"])
        by[key + (r["strategy"],)][r["seed"]] = bool(r.get("success"))
        if r.get("peak_force_N") is not None:
            force[key + (r["strategy"],)][r["seed"]] = float(r["peak_force_N"])
    steps = sorted({k[:2] for k in by})
    for sid, step in steps:
        cell = {}
        for st in STRATS:
            d = by.get((sid, step, st), {})
            k, n = sum(d.values()), len(d)
            p, lo, hi = wilson(k, n)
            f = list(force.get((sid, step, st), {}).values())
            cell[st] = {"success": k, "n": n, "rate": p, "ci95": [lo, hi],
                        "peak_force_mean_N": float(np.mean(f)) if f else None,
                        "peak_force_ci95": (1.96 * float(np.std(f, ddof=1)) / math.sqrt(len(f))
                                            if len(f) > 1 else None)}
        out["cells"]["%s/%d" % (sid, step)] = cell
        for a, b in (("none", "weakest_joint"), ("nearest", "weakest_joint"), ("none", "nearest")):
            m = mcnemar(by.get((sid, step, a), {}), by.get((sid, step, b), {}))
            if m:
                out["paired"]["%s/%d %s->%s" % (sid, step, a, b)] = m
            fa, fb = force.get((sid, step, a), {}), force.get((sid, step, b), {})
            seeds = sorted(set(fa) & set(fb))
            if len(seeds) >= 5:
                x = np.array([fa[s] for s in seeds])
                y = np.array([fb[s] for s in seeds])
                try:
                    pw = float(stats.wilcoxon(x, y).pvalue)
                except ValueError:
                    pw = 1.0
                out["peak_force"]["%s/%d %s->%s" % (sid, step, a, b)] = {
                    "n": len(seeds), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                    "change_pct": float(100 * (y.mean() - x.mean()) / x.mean()),
                    "p_wilcoxon": pw}
    # pooled over each structure's critical steps (seed-matched per step)
    for sid in sorted({s for s, _ in steps}):
        pooled = {}
        for st in STRATS:
            k = n = 0
            for (s2, step) in steps:
                if s2 == sid:
                    d = by.get((sid, step, st), {})
                    k += sum(d.values())
                    n += len(d)
            p, lo, hi = wilson(k, n)
            pooled[st] = {"success": k, "n": n, "rate": p, "ci95": [lo, hi]}
        out["cells"]["%s/pooled" % sid] = pooled
        for a, b in (("none", "weakest_joint"), ("nearest", "weakest_joint")):
            A, B = {}, {}
            for (s2, step) in steps:
                if s2 == sid:
                    A.update({(step, s): v for s, v in by.get((sid, step, a), {}).items()})
                    B.update({(step, s): v for s, v in by.get((sid, step, b), {}).items()})
            m = mcnemar(A, B)
            if m:
                out["paired"]["%s/pooled %s->%s" % (sid, a, b)] = m
    # predicted vs measured brace force (forces only: the measured torque is about
    # the wrist sensor, not the brace point)
    for st in ("nearest", "weakest_joint"):
        err, pred, meas = [], [], []
        for r in trials:
            if r["strategy"] == st and r.get("brace_measured_wrench") and r.get("brace_expected_wrench"):
                pm = np.array(r["brace_measured_wrench"][:3])
                pe = np.array(r["brace_expected_wrench"][:3])
                err.append(float(np.linalg.norm(pm - pe)))
                pred.append(float(np.linalg.norm(pe)))
                meas.append(float(np.linalg.norm(pm)))
        if err:
            out["brace_prediction"][st] = {
                "n": len(err), "abs_err_mean_N": float(np.mean(err)),
                "abs_err_median_N": float(np.median(err)),
                "predicted_mean_N": float(np.mean(pred)), "measured_mean_N": float(np.mean(meas)),
                "corr": float(np.corrcoef(pred, meas)[0, 1]) if len(err) > 2 else None}
    return out, by


def g3():
    trials = [r for r in rows(RES / "g3" / "trials.jsonl") if r.get("kind") == "assembly"]
    out = {}
    for sid in sorted({r["structure_id"] for r in trials}):
        rs = [r for r in trials if r["structure_id"] == sid]
        k = sum(bool(r.get("success")) for r in rs)
        p, lo, hi = wilson(k, len(rs))
        placed = [r.get("placed", 0) / max(1, r.get("of", 1)) for r in rs]
        out[sid] = {"success": k, "n": len(rs), "rate": p, "ci95": [lo, hi],
                    "mean_fraction_placed": float(np.mean(placed)),
                    "fatal": sorted({str(r.get("fatal"))[:60] for r in rs if r.get("fatal")}),
                    "mean_attempts_per_brick": float(np.mean([r.get("attempts", 0) / max(1, r.get("of", 1))
                                                              for r in rs])),
                    "sim_s_per_brick": float(np.mean([r.get("sim_s", 0) / max(1, r.get("of", 1))
                                                      for r in rs]))}
    return out


def g6():
    """WP7 full loop: every structure x strategy with the RL-first inserter."""
    trials = [r for r in rows(RES / "g6" / "trials.jsonl") if r.get("kind") == "assembly"]
    eps = rows(RES / "g6" / "episodes.jsonl")
    out = {"runs": {}, "inserter": {}, "brace": {}}
    by_run = defaultdict(list)
    for e in eps:
        by_run[e.get("run_id")].append(e)
    for r in trials:
        run_id = "%s_%s_seed%02d" % (r["structure_id"], r["strategy"], r["seed"])
        last = by_run.get(run_id, [])[-1:] or [{}]
        # where the assembly ended: a break during an insertion (whose), or
        # after the brick had seated (verify / release)
        if r.get("success"):
            ended = None
        elif last[0].get("joint_breaks"):
            ended = "during %s insertion" % last[0].get("policy")
        elif str(r.get("fatal", "")).startswith("joint_break"):
            ended = "after a %s seat" % last[0].get("policy")
        else:
            ended = r.get("fatal")
        out["runs"]["%s/%s/seed%d" % (r["structure_id"], r["strategy"], r["seed"])] = {
            "success": bool(r.get("success")), "placed": r.get("placed"), "of": r.get("of"),
            "fatal": r.get("fatal"), "attempts": r.get("attempts"), "ended": ended,
            "planned": r["seed"] == 0}
    by_pol = defaultdict(list)
    for e in eps:
        by_pol[e.get("policy")].append(e)
    for pol, es in by_pol.items():
        ok = [e for e in es if e.get("success")]
        probs = [(e["predicted_success_prob"], float(e["success"])) for e in es
                 if e.get("predicted_success_prob") is not None]
        auc = None
        if probs and 0 < sum(y for _, y in probs) < len(probs):
            pp, yy = np.array([p for p, _ in probs]), np.array([y for _, y in probs])
            auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pp[yy == 1] for b in pp[yy == 0]]))
        out["inserter"][pol] = {"attempts": len(es), "success": len(ok),
                                "rate": len(ok) / len(es) if es else None,
                                "ci95": wilson(len(ok), len(es)) if es else None,
                                "peak_force_mean_N": float(np.mean([e["peak_force_N"] for e in es
                                                                    if e.get("peak_force_N") is not None]))
                                if es else None,
                                "breaks_during": sum(1 for e in es if e.get("joint_breaks")),
                                "success_head_auc": auc,
                                "success_prob_mean": float(np.mean([p for p, _ in probs])) if probs else None,
                                "n_with_success_prob": len(probs)}
    # the full system (weakest-joint bracing) with the RL-first inserter
    # against G3's scripted-only pipeline, per structure (Fisher exact)
    g3t = [r for r in rows(RES / "g3" / "trials.jsonl") if r.get("kind", "assembly") == "assembly"]
    out["ended"] = dict(Counter(v["ended"] for v in out["runs"].values() if not v["success"]))
    out["vs_g3"] = {}
    for sid in sorted({r["structure_id"] for r in trials}):
        a = [r for r in trials if r["structure_id"] == sid and r["strategy"] == "weakest_joint"]
        b = [r for r in g3t if r["structure_id"] == sid]
        if a and b:
            ka, kb = sum(bool(r.get("success")) for r in a), sum(bool(r.get("success")) for r in b)
            out["vs_g3"][sid] = {"g6_success": ka, "g6_n": len(a), "g3_success": kb, "g3_n": len(b),
                                 "g6_placed_frac": float(np.mean([r["placed"] / r["of"] for r in a])),
                                 "g3_placed_frac": float(np.mean([r["placed"] / r["of"] for r in b])),
                                 "p_fisher": float(stats.fisher_exact([[ka, len(a) - ka],
                                                                       [kb, len(b) - kb]])[1])}
    br = [e for e in eps if e.get("brace_measured_wrench") and e.get("brace_expected_wrench")]
    if br:
        err = [float(np.linalg.norm(np.array(e["brace_measured_wrench"][:3])
                                    - np.array(e["brace_expected_wrench"][:3]))) for e in br]
        out["brace"] = {"placements": len(br), "abs_err_mean_N": float(np.mean(err)),
                        "abs_err_median_N": float(np.median(err))}
    return out


def wp5():
    ev = {}
    for r in rows(RES / "wp5" / "eval.jsonl"):
        ev[(r["name"], r["stage"], r.get("joint", "calibrated"), r.get("vision"))] = r
    curves = {}
    for run in ("R1", "R2", "R3", "R4", "R1ht", "R1hc"):
        rs = rows(RES / "wp5" / run / "train.jsonl")
        if rs:
            curves[run] = [(r["samples"], r["success_100"], r["stage"]) for r in rs]
    out = {"eval": {"%s|stage%d|%s|vision=%s" % k: v for k, v in ev.items()}, "comparisons": {}}

    def comp(name, a, b):
        ra, rb = ev.get(a), ev.get(b)
        if ra and rb:
            ka, na = round(ra["success_rate"] * ra["episodes"]), ra["episodes"]
            kb, nb = round(rb["success_rate"] * rb["episodes"]), rb["episodes"]
            p = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
            out["comparisons"][name] = {"a": a[0], "b": b[0], "rate_a": ra["success_rate"],
                                        "rate_b": rb["success_rate"],
                                        "diff": rb["success_rate"] - ra["success_rate"],
                                        "p_fisher": float(p),
                                        "peak_a": ra["peak_force_mean_N"],
                                        "peak_b": rb["peak_force_mean_N"]}
    for S in (0, 3):      # stage 0: in the training distribution; stage 3: G4's criterion
        tag = " @stage%d" % S
        comp("A1 scripted->R1" + tag, ("scripted", S, "calibrated", True), ("R1", S, "calibrated", True))
        comp("A4 R3(e2e)->R2(residual)" + tag, ("R3", S, "calibrated", True),
             ("R2", S, "calibrated", True))
        comp("A-dense R1(sparse)->R2(dense)" + tag, ("R1", S, "calibrated", True),
             ("R2", S, "calibrated", True))
        comp("A3 R1(vision)->R4(no vision)" + tag, ("R1", S, "calibrated", True),
             ("R4", S, "calibrated", False))
        comp("A10 R1(calibrated)->R1ht(hand-tuned)" + tag, ("R1", S, "calibrated", True),
             ("R1ht", S, "calibrated", True))
        comp("A10 R1(calibrated)->R1hc(hand-tuned, calibrated budget)" + tag,
             ("R1", S, "calibrated", True), ("R1hc", S, "calibrated", True))
    return out, curves


def figures(a6out, by, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"none": "#999999", "nearest": "#e69f00", "weakest_joint": "#0072b2"}
    cells = [(k, v) for k, v in a6out["cells"].items() if not k.endswith("pooled")]
    if cells:
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(cells)), 3.4))
        w = 0.26
        for i, st in enumerate(STRATS):
            xs = np.arange(len(cells)) + (i - 1) * w
            ys = [v[st]["rate"] if v[st]["n"] else np.nan for _, v in cells]
            lo = [max(0.0, v[st]["rate"] - v[st]["ci95"][0]) if v[st]["n"] else 0 for _, v in cells]
            hi = [max(0.0, v[st]["ci95"][1] - v[st]["rate"]) if v[st]["n"] else 0 for _, v in cells]
            ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, color=colors[st], label=st)
        ax.set_xticks(np.arange(len(cells)))
        ax.set_xticklabels([k.replace("/", " step ") for k, _ in cells], fontsize=8)
        ax.set_ylabel("placement success (no joint breaks)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("A6: bracing strategy at the critical steps (95% Wilson CI)", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "a6_success.png", dpi=150)
        plt.close(fig)
    if curves:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for run, c in curves.items():
            c = np.array(c)
            ax.plot(c[:, 0] / 1000, c[:, 1], label=run)
        ax.set_xlabel("environment steps (k)")
        ax.set_ylabel("success, last 100 episodes")
        ax.set_title("WP5 training (curriculum stage resets the window)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "wp5_training.png", dpi=150)
        plt.close(fig)


def main():
    a6out, by = a6()
    out = {"A6": a6out, "G3": g3(), "G6": g6()}
    w, curves = wp5()
    out["WP5"] = w
    figures(a6out, by, curves)
    (RES / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"A6_cells": {k: {s: "%d/%d" % (v[s]["success"], v[s]["n"]) for s in STRATS}
                                   for k, v in a6out["cells"].items()},
                      "A6_paired": {k: (round(v["diff"], 2), round(v["p_exact"], 4))
                                    for k, v in a6out["paired"].items()},
                      "G3": {k: "%d/%d" % (v["success"], v["n"]) for k, v in out["G3"].items()},
                      "WP5": w["comparisons"]}, indent=1))


if __name__ == "__main__":
    main()
