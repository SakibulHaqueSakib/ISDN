"""Render WORKLOG.md from ledger.jsonl.

    python scripts/worklog.py           # rewrite WORKLOG.md
    python scripts/worklog.py --print   # to stdout

The log is GENERATED, never hand-written, so it cannot drift from the record.
Re-run it after each task; the ledger is the source of truth.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger.jsonl"
OUT = ROOT / "WORKLOG.md"

PHASES = [
    ("WP0", "Environment & platform verification"),
    ("pre-WP0", "Kinematic slice (pre-gate exploration)"),
    ("WP1", "Joint mechanics"),
    ("WP2", "Joint calibration — static solver"),
    ("WP2b", "Joint calibration — live sim"),
    ("WP3", "Build planner & bracing"),
    ("WP4", "Motion stack & scripted baseline"),
    ("WP5", "RL insertion"),
    ("WP6", "Perception"),
    ("WP7", "Integration"),
    ("WP8", "Experiments & analysis"),
]

# What each entry type means, so the log reads as a narrative not a dump.
GLYPH = {"gate": "GATE", "decision": "decided", "deviation": "deviated",
         "failure": "problem", "result": "result", "branch": "branch",
         "cut": "CUT"}

FIELDS = ("outcome", "decision", "result", "status", "finding", "implication",
          "correction", "rationale", "diagnosis_so_far", "next_step",
          "recommendation", "next", "significance", "gap_to_G3", "fix")


def rows():
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


def one_line(value, limit=400):
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[:limit] + "…"


def render(entries):
    lines = ["# Work log", "",
             "Generated from `ledger.jsonl` by `scripts/worklog.py` — do not "
             "edit by hand.", "",
             "%d entries. Later entries supersede earlier ones; retractions "
             "are marked." % len(entries), ""]

    seen = set()
    for wp, title in PHASES:
        block = [e for e in entries if e.get("wp") == wp]
        if not block:
            continue
        seen.update(id(e) for e in block)
        lines += ["## %s — %s" % (wp, title), ""]
        for e in block:
            kind = GLYPH.get(e.get("type"), e.get("type", "?"))
            head = "- **%s** `%s`" % (kind, e.get("id", e.get("item", "")))
            if e.get("type") == "gate":
                head += " → **%s**" % e.get("result", "?")
            if e.get("retracts"):
                head += "  ⟵ *retracts %s*" % one_line(e["retracts"], 80)
            if e.get("escalate"):
                head += "  ⚑ *needs a decision*"
            lines.append(head)
            for field in FIELDS:
                if e.get(field):
                    lines.append("    - %s: %s" % (field, one_line(e[field])))
            lines.append("")

    loose = [e for e in entries if id(e) not in seen]
    if loose:
        lines += ["## Unfiled", ""]
        for e in loose:
            lines.append("- **%s** `%s` — %s"
                         % (GLYPH.get(e.get("type"), "?"),
                            e.get("id", ""), one_line(e.get("rationale", ""))))
        lines.append("")

    gates = [e for e in entries if e.get("type") == "gate"]
    if gates:
        lines += ["## Gate history (latest wins)", ""]
        latest = {}
        for g in gates:
            latest[g.get("id")] = g
        for gid, g in latest.items():
            lines.append("- **%s**: %s — %s"
                         % (gid, g.get("result", "?"),
                            one_line(g.get("status", ""), 160)))
        lines.append("")

    # An escalation that a later entry retracts, or that a decision answered,
    # is not still open.
    retracted, decided = set(), set()
    for e in entries:
        if e.get("retracts"):
            retracted.update(w.strip("(),") for w in str(e["retracts"]).split())
        if e.get("type") == "decision":
            decided.add(e.get("id"))
    open_items = [e for e in entries
                  if e.get("escalate")
                  and e.get("id") not in retracted
                  and e.get("id") not in decided]
    if open_items:
        lines += ["## Open — needs a decision", ""]
        for e in open_items:
            lines.append("- `%s` (%s): %s"
                         % (e.get("id") or e.get("item", "?"), e.get("wp", "-"),
                            one_line(e.get("implication")
                                     or e.get("finding")
                                     or e.get("rationale", ""), 220)))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    args = ap.parse_args()
    text = render(rows())
    if args.to_stdout:
        print(text)
    else:
        OUT.write_text(text)
        print("wrote %s (%d lines)" % (OUT, text.count("\n") + 1))
