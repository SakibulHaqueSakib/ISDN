"""Markdown tables for Docs/results_report.md, from results/analysis.json.

    python -m experiments.tables > /tmp/tables.md

Every number in the report's tables comes from here, so none is transcribed
by hand.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATS = ("none", "nearest", "weakest_joint")
MIN_PAIRS_SHOWN = 10


def pct(x):
    return "—" if x is None else "%.0f%%" % (100 * x)


def ci(c):
    return "[%.0f, %.0f]" % (100 * c[0], 100 * c[1])


def main():
    a = json.loads((ROOT / "results" / "analysis.json").read_text())
    out = []
    g3 = a.get("G3", {})
    if g3:
        out += ["### G3 benchmark (scripted inserter, ground-truth poses, weakest-joint bracing)", "",
                "| structure | complete | rate [95% CI] | mean fraction placed | attempts / brick | failure modes |",
                "|---|---|---|---|---|---|"]
        for sid, v in g3.items():
            out.append("| %s | %d/%d | %s %s | %.2f | %.2f | %s |" % (
                sid, v["success"], v["n"], pct(v["rate"]), ci(v["ci95"]), v["mean_fraction_placed"],
                v["mean_attempts_per_brick"], "; ".join(v["fatal"][:3]) or "—"))
        out.append("")
    a6 = a.get("A6", {})
    if a6.get("cells"):
        out += ["### A6 — placement success at the critical steps (no joint may break)", "",
                "| structure / step | none | nearest | weakest_joint |", "|---|---|---|---|"]
        for key, cell in a6["cells"].items():
            out.append("| %s | " % key + " | ".join(
                "%d/%d = %s %s" % (cell[s]["success"], cell[s]["n"], pct(cell[s]["rate"]), ci(cell[s]["ci95"]))
                if cell.get(s, {}).get("n") else "—" for s in STRATS) + " |")
        out += ["", "Paired comparisons on matched seeds (exact McNemar; difference in success rate, "
                "b − a, with 95%% CI). Comparisons with fewer than %d pairs (S5's single steps) are in "
                "`results/analysis.json` only:" % MIN_PAIRS_SHOWN, "",
                "| comparison | pairs | only b succeeds | only a succeeds | difference [95% CI] | p |",
                "|---|---|---|---|---|---|"]
        for key, m in a6["paired"].items():
            if m["n_pairs"] < MIN_PAIRS_SHOWN:
                continue
            out.append("| %s | %d | %d | %d | %+.2f [%+.2f, %+.2f] | %.3g |" % (
                key, m["n_pairs"], m["b_only"], m["a_only"], m["diff"], m["diff_ci95"][0],
                m["diff_ci95"][1], m["p_exact"]))
        if a6.get("peak_force"):
            out += ["", "Placer peak force (matched seeds, Wilcoxon; cells with at least %d pairs):"
                    % MIN_PAIRS_SHOWN, "",
                    "| comparison | n | mean a (N) | mean b (N) | change | p |", "|---|---|---|---|---|---|"]
            for key, m in a6["peak_force"].items():
                if m["n"] < MIN_PAIRS_SHOWN:
                    continue
                out.append("| %s | %d | %.1f | %.1f | %+.1f%% | %.3g |" % (
                    key, m["n"], m["mean_a"], m["mean_b"], m["change_pct"], m["p_wilcoxon"]))
        if a6.get("brace_prediction"):
            out += ["", "Brace force, predicted (`expected_reaction_wrench`) vs measured at the placer's "
                    "peak (forces only):", "",
                    "| strategy | placements | predicted mean (N) | measured mean (N) | |error| mean (N) | median (N) |",
                    "|---|---|---|---|---|---|"]
            for st, v in a6["brace_prediction"].items():
                out.append("| %s | %d | %.1f | %.1f | %.1f | %.1f |" % (
                    st, v["n"], v["predicted_mean_N"], v["measured_mean_N"], v["abs_err_mean_N"],
                    v["abs_err_median_N"]))
        out.append("")
    wp5 = a.get("WP5", {})
    if wp5.get("eval"):
        out += ["### WP5 — policies on held-out episodes (100 each; the calibrated joint unless stated)", "",
                "| policy | evaluated on | stage | success | peak force mean (N) | peak / seating force | success-head AUC | outcomes |",
                "|---|---|---|---|---|---|---|---|"]
        order = ["scripted", "R1", "R2", "R3", "R4", "R1ht", "R1hc"]
        rank = lambda kv: (kv[1].get("joint", "calibrated") != "calibrated",
                           order.index(kv[1]["name"]) if kv[1].get("name") in order else 99,
                           kv[1]["stage"])
        for key, v in sorted(wp5["eval"].items(), key=rank):
            out.append("| %s | %s | %d | %s ± %.0f | %.1f | %.2f | %s | %s |" % (
                v.get("name"), v.get("joint", "calibrated"), v["stage"], pct(v["success_rate"]),
                100 * v["success_ci95"],
                v["peak_force_mean_N"], v["peak_over_nominal_press"],
                "%.2f" % v["success_head_auc"] if v.get("success_head_auc") is not None else "—",
                ", ".join("%s %d" % kv for kv in sorted(v["reasons"].items()))))
        out += ["", "| ablation | a | b | success a → b | p (Fisher) | peak force a → b (N) |",
                "|---|---|---|---|---|---|"]
        for key, v in wp5["comparisons"].items():
            out.append("| %s | %s | %s | %s → %s | %.3g | %.0f → %.0f |" % (
                key, v["a"], v["b"], pct(v["rate_a"]), pct(v["rate_b"]), v["p_fisher"],
                v["peak_a"], v["peak_b"]))
        out.append("")
    g6 = a.get("G6", {})
    if g6.get("runs"):
        out += ["### G6 — full loop, RL-first inserter with scripted fallback", "",
                "| run | complete | placed | attempts | ended | joints broken |",
                "|---|---|---|---|---|---|"]
        runs = sorted(g6["runs"].items(), key=lambda kv: (not kv[1].get("planned", True), kv[0]))
        for key, v in runs:
            out.append("| %s%s | %s | %s/%s | %s | %s | %s |" % (
                key, "" if v.get("planned", True) else " (extra)", "yes" if v["success"] else "no",
                v["placed"], v["of"], v["attempts"], v.get("ended") or "—",
                (v["fatal"] or "—").replace("joint_break: ", "")[:60]))
        if g6.get("inserter"):
            out += ["", "| inserter | attempts | seated [95% CI] | breaks during its insertions | "
                    "peak force mean (N) | success head: mean p, AUC |", "|---|---|---|---|---|---|"]
            for pol, v in g6["inserter"].items():
                out.append("| %s | %d | %d = %s %s | %d | %s | %s |" % (
                    pol, v["attempts"], v["success"], pct(v["rate"]),
                    ci(v["ci95"][1:]) if v.get("ci95") else "", v.get("breaks_during", 0),
                    "%.1f" % v["peak_force_mean_N"] if v["peak_force_mean_N"] is not None else "—",
                    "%.2f, %.2f (n = %d)" % (v["success_prob_mean"], v["success_head_auc"],
                                             v["n_with_success_prob"])
                    if v.get("success_head_auc") is not None else "—"))
        if g6.get("vs_g3"):
            out += ["", "Full system (weakest-joint bracing) with the RL-first inserter vs G3's "
                    "scripted-only pipeline:", "",
                    "| structure | G6 complete | G3 complete | p (Fisher) | mean fraction placed, G6 → G3 |",
                    "|---|---|---|---|---|"]
            for sid, v in g6["vs_g3"].items():
                out.append("| %s | %d/%d | %d/%d | %.3g | %.2f → %.2f |" % (
                    sid, v["g6_success"], v["g6_n"], v["g3_success"], v["g3_n"], v["p_fisher"],
                    v["g6_placed_frac"], v["g3_placed_frac"]))
        if g6.get("ended"):
            out += ["", "How the failed runs ended: " + "; ".join(
                "%s %d" % kv for kv in sorted(g6["ended"].items(), key=lambda kv: -kv[1])) + "."]
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
