"""Batch trials in the twin, in parallel: G3's benchmark and ablation A6.

    python -m experiments.runner g3  --structures S1 S2 S3 --trials 20
    python -m experiments.runner a6  --structures S3 S5 --seeds 20
    python -m experiments.runner a6  --steps S3:10,12,14 --seeds 20

Every trial appends its summary to results/<exp>/trials.jsonl and its §2.6
insertion episodes to results/<exp>/episodes.jsonl. A trial already in
trials.jsonl (same kind, structure, step, strategy, seed, variant) is skipped,
so an interrupted batch resumes where it stopped.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _key(t):
    return (t["kind"], t["structure_id"], t.get("step"), t["strategy"], t["seed"],
            t.get("variant", ""))


def _run(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from orchestration import twin_executor as X
    kind, sid, step, strat, seed, variant, out = task
    t0 = time.time()
    kw = {}
    if variant == "passive":
        kw["coordinated"] = False
    try:
        if kind == "step":
            r = X.step_trial(sid, step, strat, seed, out=out / "episodes.jsonl", **kw)
        else:
            r = X.assemble(sid, strat, seed, out=out / "episodes.jsonl", **kw)
        r.pop("episodes", None)
    except Exception as e:                          # a crashed trial is a failed trial
        r = {"structure_id": sid, "strategy": strat, "seed": seed, "success": False,
             "fatal": "exception: %r" % e, "traceback": traceback.format_exc()[-2000:]}
    r.update({"kind": kind, "step": step, "variant": variant, "structure_id": sid,
              "strategy": strat, "seed": seed, "wall_s": round(time.time() - t0, 1)})
    with open(out / "trials.jsonl", "a") as fh:
        fh.write(json.dumps(r, default=str) + "\n")
    return r


def critical_steps(sid):
    """Steps where any bracing strategy braces (the A6 unit)."""
    from orchestration import twin_executor as X
    import planner as P
    from sim.mj import scene as S
    S.centre_plan_origin(P.STRUCTURES[sid])
    plan = X.load_plan(sid, "weakest_joint")
    return [s["step"] for s in plan["sequence"] if s["requires_brace"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["g3", "a6", "a6_passive"])
    ap.add_argument("--structures", nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", default=None, help="S3:10,12,14 ...")
    ap.add_argument("--strategies", nargs="*", default=["none", "nearest", "weakest_joint"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--trials", type=int, default=None, help="alias of --seeds")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.trials or a.seeds
    out = Path(a.out) if a.out else RESULTS / a.exp
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "trials.jsonl").exists():
        for line in open(out / "trials.jsonl"):
            try:
                done.add(_key(json.loads(line)))
            except ValueError:
                pass
    tasks = []
    variant = "passive" if a.exp == "a6_passive" else ""
    if a.exp == "g3":
        for sid in a.structures or ["S1", "S2", "S3"]:
            strat = "weakest_joint"
            tasks += [("assembly", sid, None, strat, k, variant, out) for k in range(n)]
    else:
        steps = {}
        if a.steps:
            for spec in a.steps:
                sid, lst = spec.split(":")
                steps[sid] = [int(v) for v in lst.split(",")]
        else:
            for sid in a.structures or ["S3", "S5"]:
                steps[sid] = critical_steps(sid)
        for k in range(n):                       # seed-major: partial batches stay balanced
            for sid, lst in steps.items():
                for st in lst:
                    for strat in a.strategies:
                        tasks.append(("step", sid, st, strat, k, variant, out))
    todo = [t for t in tasks if (t[0], t[1], t[2], t[3], t[4], t[5]) not in done]
    print("%d trials, %d to run, %d workers -> %s" % (len(tasks), len(todo), a.workers, out))
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(_run, todo):
            print("%-8s %s step %-4s %-13s seed %2d  %s  breaks %s  %.0fs  %s" % (
                r["kind"], r["structure_id"], r.get("step"), r["strategy"], r["seed"],
                "OK  " if r.get("success") else "FAIL", r.get("breaks"), r["wall_s"],
                r.get("fatal") or ""), flush=True)


if __name__ == "__main__":
    main()
