"""Project status at a glance -- reads the ledger and the artifacts on disk.

    python scripts/status.py            # summary
    python scripts/status.py --full     # every ledger entry, newest last
    python scripts/status.py --open     # only what is blocking or escalated

Nothing here recomputes anything: it reports what has actually been measured
and written down, so it cannot claim progress that did not happen.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger.jsonl"

# Work packages in the report's order, with what "done" means for each.
PACKAGES = [
    ("WP0", "environment & platform"),
    ("WP1", "joint mechanics"),
    ("WP2", "joint calibration (static)"),
    ("WP2b", "joint calibration (live sim)"),
    ("WP3", "planner + bracing"),
    ("WP4", "motion stack + scripted baseline"),
    ("WP5", "RL insertion"),
    ("WP6", "perception (conditional)"),
    ("WP7", "integration"),
    ("WP8", "experiments & analysis"),
]


def entries():
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def latest_gate(rows, gate_id):
    """The most recent verdict for a gate -- the ledger is append-only, so a
    later entry supersedes an earlier one."""
    hits = [r for r in rows if r.get("type") == "gate" and r.get("id") == gate_id]
    return hits[-1] if hits else None


def artifact(name):
    p = ROOT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {"_raw": True}


def summary(rows):
    print("=" * 66)
    print("  brickassembly -- status from %d ledger entries" % len(rows))
    print("=" * 66)

    print("\nGATES")
    for gate in ("G0", "G1", "WP2", "G2", "G3", "G4", "G5", "G6", "G7"):
        g = latest_gate(rows, gate)
        if not g:
            print("  %-4s  not reached" % gate)
            continue
        print("  %-4s  %-22s %s" % (gate, g.get("result", "?"),
                                    g.get("status", "")))

    print("\nWORK PACKAGES  (activity recorded in the ledger)")
    for wp, label in PACKAGES:
        touched = [r for r in rows if r.get("wp") == wp]
        if not touched:
            print("  %-5s %-32s -" % (wp, label))
            continue
        fails = sum(1 for r in touched if r.get("type") == "failure")
        print("  %-5s %-32s %d entries%s"
              % (wp, label, len(touched),
                 ", %d recorded failures" % fails if fails else ""))

    cal = artifact("joint_calibration.json")
    if cal:
        print("\nCALIBRATION")
        st = cal.get("bricksim_settings", {})
        for path in ("static_solve", "live_sim"):
            if path in st:
                print("  %-12s preloaded_force %-8s -> %s N/stud"
                      % (path, st[path]["preloaded_force"],
                         st[path]["gives_n_per_stud"]))
        dyn = cal.get("dynamic", {})
        if dyn.get("target_unreachable"):
            u = dyn["target_unreachable"]
            print("  target %s N/stud not reachable; best measured %s "
                  "(%.2f%% short)" % (u["target_n_per_stud"],
                                      u["best_measured_n_per_stud"],
                                      u["shortfall_pct"]))

    # A criterion the ledger has CUT must not keep showing as a red failure:
    # the run that produced it is still on disk, but it no longer counts.
    cut_text = " ".join(str(r.get("target", "")) for r in rows
                        if r.get("type") == "cut")
    guarded = {"double_mate_rejected"}
    # test keys are shorter than the criterion names the ledger cuts by
    aliases = {"scripted_insertion_2mm": "scripted_insertion_snaps_from_2mm",
               "gate_boundary_lateral": "gate_boundary_6_cases_per_condition",
               "gate_boundary_yaw": "gate_boundary_6_cases_per_condition"}

    def verdict_for(name, ok):
        if name in cut_text or aliases.get(name, "\0") in cut_text:
            return "CUT"
        if name in guarded:
            return "GUARD"
        return "?" if ok is None else ("PASS" if ok else "FAIL")

    joint = artifact("tests/test_joint.json")
    gate = artifact("tests/test_gate.json")
    if joint or gate:
        print("\nLATEST TEST RUNS   (CUT / GUARD annotated from the ledger)")
    if joint:
        for name, r in joint.items():
            print("  [%-5s] %s" % (verdict_for(name, r.get("pass")), name))
    if gate and "verdict" in gate:
        for name, ok in gate["verdict"].items():
            print("  [%-5s] %s" % (verdict_for(name, ok), name))

    # v3.1 twin test suites and experiment summaries (artifacts written by the
    # tests themselves; nothing is recomputed here)
    twin = [("tests/test_clutch.json", "G1 twin joint"), ("tests/test_planner.json", "G2 planner"),
            ("tests/test_control.json", "G3 controller"), ("tests/test_env.json", "G4 environment")]
    shown = [(f, lab) for f, lab in twin if artifact(f)]
    if shown:
        print("\nTWIN TEST ARTIFACTS  (re-run the suite to refresh)")
        for f, lab in shown:
            print("  %-16s %s" % (lab, f))
    ana = artifact("results/analysis.json")
    if ana and not ana.get("_raw"):
        print("\nEXPERIMENTS  (results/analysis.json)")
        for k, v in ana.get("G3", {}).items():
            print("  G3 %-4s %d/%d assemblies" % (k, v["success"], v["n"]))
        for k, v in ana.get("A6", {}).get("cells", {}).items():
            print("  A6 %-10s " % k + "  ".join("%s %d/%d" % (st, v[st]["success"], v[st]["n"])
                                                 for st in ("none", "nearest", "weakest_joint")
                                                 if st in v))
        for k, v in ana.get("WP5", {}).get("comparisons", {}).items():
            print("  %-48s %.2f -> %.2f (p=%.3g)" % (k, v["rate_a"], v["rate_b"], v["p_fisher"]))

    # An escalation answered by a later decision, or retracted by a later
    # entry, is not still open.
    settled = {r.get("id") for r in rows if r.get("type") == "decision"}
    settled |= {r["retracts"].split()[-1].strip("()")
                for r in rows if r.get("retracts")}
    settled |= {r.get("retracts", "").split("entry ")[-1].split(" ")[0]
                for r in rows if r.get("retracts")}
    key = lambda r: r.get("id") or r.get("item") or r.get("probe")
    blockers = [r for r in rows
                if r.get("escalate") and key(r) not in settled]
    if blockers:
        print("\nESCALATED / NEEDS A DECISION  (%d)" % len(blockers))
        for r in blockers:
            print("  %-6s %-28s %s" % (r.get("wp", "-"), r.get("id", r.get("type")),
                                       (r.get("implication") or r.get("finding")
                                        or "")[:60]))

    print("\nNEXT: python scripts/status.py --open     for the open items")
    print("      python scripts/status.py --full     for the whole ledger")


def show(rows, only_open):
    for r in rows:
        if only_open and not (r.get("escalate")
                              or r.get("type") in ("failure", "deviation")
                              or (r.get("type") == "gate"
                                  and "pass" not in str(r.get("result")))):
            continue
        head = "%s  %-10s %-9s %s" % (r.get("ts", "")[:16], r.get("wp", "-"),
                                      r.get("type", "-"),
                                      r.get("id", r.get("item", "")))
        print(head)
        for key in ("decision", "outcome", "finding", "implication",
                    "correction", "result", "status", "recommendation"):
            if r.get(key):
                text = str(r[key])
                print("      %s: %s" % (key, text[:300]
                                        + ("..." if len(text) > 300 else "")))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    rows = entries()
    if args.full or args.open:
        show(rows, only_open=args.open)
    else:
        summary(rows)
