"""RL-bracing Phase R0 runner (plan_v4_rl_brace S0.3): job lists per experiment, job de-duplication, a pool of
cell subprocesses, one row file per job, merged episodes.jsonl per experiment, manifest, tables.

    bash scripts/run.sh experiments/v4_brace.py counts                  # job counts per experiment, de-dup counts
    bash scripts/run.sh experiments/v4_brace.py harness --workers 4     # then throughput, placer, signal, ...
    bash scripts/run.sh experiments/v4_brace.py signal --dry-run
    bash scripts/run.sh experiments/v4_brace.py freeze [--band 0.9 1.3] # after signal; band changes are manual
    bash scripts/run.sh experiments/v4_brace.py choose-exe              # after reaction; screen/validate/harm need it
    bash scripts/run.sh experiments/v4_brace.py tables

Run order (plan section 4): harness, throughput, placer, signal, freeze, reaction, choose-exe, screen,
validate, harm. Everything lives under --root (default results/v4/rl): rows/<id>.json (one per job, shared by
every experiment that needs the same job), <exp>/episodes.jsonl (merged, with the experiment's labels),
plans/<structure>.json (the cell's plan cache), logs/<id>.log, r0/manifest.json, r0/tables.json.

A job id is the sha256 (first 16 hex) of the canonical JSON of what determines the run: structure, step,
action, lambda, lateral, executor setting, repeat (a fixture-off job has no lambda/lateral). Experiment, arm,
stage and replicate are labels: the same run in two experiments is run once. Needs only numpy and the
task library: no newton in this process (the cell is a subprocess).
"""

import argparse
import hashlib
import json
import math
import os
import signal as _signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tasks import brace_bandit as BB  # noqa: E402

ROOT = HERE / "results" / "v4" / "rl"
NO_LATERAL = [0.0, 0.0]
NUM_FRAMES = 6000                  # 100 s of sim: a step-episode is ~12-20 s; the null viewer stops it when done
ONSET_TOL_S = 0.1                  # A2: sham valid iff its insert onset is within this of the braced run's
EXPERIMENTS = ["harness", "throughput", "placer", "signal", "reaction", "screen", "validate", "harm"]
STAGES = {"harness": "harness", "throughput": "throughput", "signal": "signal", "validate": "validate"}   # A6 labels
NOMINAL = {"throughput": 160, "placer": None, "signal": 200, "reaction": 8, "screen": 224, "validate": 256,
           "harm": 96, "harness": 49}                                                 # plan section 5
NOMINAL_STARTS = (("cube", 7), ("arch", 10), ("hollow_box", 11))

# --- manifest ------------------------------------------------------------------------------------------------

def manifest_path(root):
    return Path(root) / "r0" / "manifest.json"


def default_manifest():
    return {"master_seed": BB.MASTER_SEED, "stages": STAGES, "band": [0.5, 1.3], "band_history": [],
            "lateral_max_N": 3.0, "no_lateral": NO_LATERAL,
            "pulse": {"up_s": 0.3, "hold_s": 0.3, "down_s": 0.3, "note": "smoothstep; set in the cell (plan 2.2); a recorded value, not a runner input"},
            "actions": {"leans_deg": list(BB.LEANS), "brace_cost": BB.BRACE_COST,
                        "candidates": "bracing.candidates(brick, placed, clearance=lambda *a: True) x leans (28 mm rule off)",
                        "screen": "screen_candidates: LP-best 4 + nearest 2 + random 6 (seeded from stage 'screen', key, step)",
                        "pick_rule": "screen_pick: success with lowest u2_peak; ties within 0.01 -> lp_u, |dy|, index"},
            "split": {"train": BB.SPLITS["train"][1] - BB.SPLITS["train"][0], "val": BB.SPLITS["val"][1] - BB.SPLITS["val"][0],
                      "test": "on demand (R2)"},
            "pools": {"critical_u": BB.CRITICAL_U, "scales": list(BB.SCALES)},
            "exe_settings": BB.EXE, "exe": None, "frozen": False, "job_counts": {}, "selection": {}}


def load_manifest(root):
    p = manifest_path(root)
    return json.loads(p.read_text()) if p.exists() else default_manifest()


def save_manifest(root, M):
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    M["hash"] = manifest_hash(M)
    p.write_text(json.dumps(M, indent=1))


def manifest_hash(M):
    """Of the frozen task: seed, stages, band, pulse, actions, split, pools, executor settings and choice."""
    keys = ("master_seed", "stages", "band", "lateral_max_N", "pulse", "actions", "split", "pools", "exe_settings", "exe")
    return hashlib.sha256(json.dumps({k: M[k] for k in keys}, sort_keys=True, default=float).encode()).hexdigest()[:16]


# --- structures ----------------------------------------------------------------------------------------------

def key_id(key):
    return hashlib.sha256(json.dumps([list(c) for c in key], separators=(",", ":")).encode()).hexdigest()[:12]


_REG = {}


def registry():
    """{structure id: entry} of the train + val structures; id = hash of the canonical key."""
    if not _REG:
        n_tr, n_va = BB.SPLITS["train"][1], BB.SPLITS["val"][1]
        for i, e in enumerate(BB.unique_structures(n_va)):
            _REG[key_id(e["key"])] = dict(e, split="train" if i < n_tr else "val", index=i)
    return _REG


def structure_spec(sid):
    """The cell-spec keys naming the structure: a blueprint sample ('shape:cube') or the inline bricks."""
    if sid.startswith("shape:"):
        return {"shape": sid[6:]}
    return {"bricks": [list(b) for b in registry()[sid]["bricks"]], "name": "rl_" + sid}


_CTX = {}


def structure_contexts(sid):
    if sid not in _CTX:
        _CTX[sid] = BB.contexts(registry()[sid]["bricks"])
    return _CTX[sid]


def get_ctx(sid, step):
    return structure_contexts(sid)[step - 1]


def meta_of(sid, c):
    e = registry()[sid]
    return {"split": e["split"], "pool": c["pool"], "u0": [round(c["u0"][s], 4) for s in BB.SCALES]}


def best_critical(sid):
    """The critical context of structure sid with the highest LP u0 (at 1.3 x design; ties: lower step), or None."""
    cs = [c for c in structure_contexts(sid) if c["pool"] == "critical"]
    return max(cs, key=lambda c: (c["u0"][1.3], -c["step"])) if cs else None


def train_ids():
    return [k for k, e in registry().items() if e["split"] == "train"]       # dict order = split order


# --- jobs ----------------------------------------------------------------------------------------------------

def _act(a):
    if a is None:
        return None
    if a[0] == "sham":
        return ["sham", None if a[1] is None else round(float(a[1]), 3)]
    return [round(float(a[0]), 6), round(float(a[1]), 6), int(a[2])]


def core_of(sid, step, action, lam, lateral, exe, rep):
    return {"s": sid, "n": int(step), "a": _act(action), "lam": None if lam is None else round(float(lam), 6),
            "lat": None if lateral is None else [round(float(x), 6) for x in lateral], "exe": exe, "rep": int(rep)}


def core_id(core):
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def job(exp, sid, step, arm, action, lam=None, lateral=None, exe="E0", rep=0, stage=None, r=None, meta=None):
    """A job: the core (hashed) plus labels (experiment, arm, stage, replicate, meta). A sham whose duration is
    not known yet has id None (pending)."""
    core = core_of(sid, step, action, lam, lateral, exe, rep)
    pending = core["a"] is not None and core["a"][0] == "sham" and core["a"][1] is None
    return {"exp": exp, "arm": arm, "stage": stage, "r": r, "meta": meta or {}, "core": core,
            "id": None if pending else core_id(core)}


def drawn(exp, sid, c, r, R, band, arm, action, exe, rep=0, meta=None):
    d = BB.hidden(STAGES[exp], c["key"], c["step"], r, R, band=tuple(band))
    return job(exp, sid, c["step"], arm, action, d["lam"], d["lateral"], exe, rep, STAGES[exp], r, meta)


# --- the store: one row file per job, shared across experiments -------------------------------------------------

def rows_dir(root):
    return Path(root) / "rows"


def load_rec(root, jid):
    p = rows_dir(root) / (jid + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def plan_path(root, sid):
    return Path(root) / "plans" / (sid.replace(":", "_") + ".json")


def cell_spec(j, root):
    c, jid = j["core"], j["id"]
    a = c["a"]
    brace = None if a is None else {"sham_s": a[1]} if a[0] == "sham" else a
    return {**structure_spec(c["s"]), "plan": str(plan_path(root, c["s"])), "start_step": c["n"], "only_step": True,
            "brace_json": brace, "press_scale": c["lam"], "press_lateral": c["lat"], "exe": c["exe"],
            "out": str(rows_dir(root) / (jid + ".part")), "tag": jid, "repeat": c["rep"]}


def run_one(j, root, timeout, num_frames=NUM_FRAMES):
    """Run the cell on one job; one retry if no row came out. Writes rows/<id>.json (row None = job error)."""
    root, jid = Path(root), j["id"]
    for d in ("rows", "specs", "logs", "plans"):
        (root / d).mkdir(parents=True, exist_ok=True)
    spec = root / "specs" / (jid + ".json")
    spec.write_text(json.dumps(cell_spec(j, root)))
    part, row, rc, t0 = rows_dir(root) / (jid + ".part"), None, None, time.time()
    for attempt in (1, 2):
        part.unlink(missing_ok=True)
        with open(root / "logs" / (jid + ".log"), "w" if attempt == 1 else "a") as log:
            log.write("--- attempt %d\n" % attempt)
            log.flush()
            p = subprocess.Popen(["bash", str(HERE / "scripts" / "run.sh"), "dual_arm_sim.py", "--spec", str(spec), "--viewer", "null",
                                  "--test", "--num-frames", str(num_frames)], stdout=log, stderr=subprocess.STDOUT,
                                 cwd=HERE, start_new_session=True)
            try:
                rc = p.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, _signal.SIGKILL)
                p.wait()
                rc = "timeout"
        lines = part.read_text().strip().splitlines() if part.exists() else []
        if lines:
            row = json.loads(lines[-1])
            break
    part.unlink(missing_ok=True)
    rec = {"id": jid, "core": j["core"], "meta": j["meta"], "first_exp": j["exp"], "returncode": rc, "attempts": attempt,
           "wall_s": round(time.time() - t0, 1), "row": row, "error": None if row else "no row after 2 attempts (rc=%s)" % rc}
    tmp = rows_dir(root) / (jid + ".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(rows_dir(root) / (jid + ".json"))
    return rec


def run_jobs(jobs, root, workers, timeout=300, retry_errors=False, limit=None, log=print):
    """Run the jobs that have no row yet (de-duplicated by id) in a pool of `workers` cell subprocesses. A job whose
    structure has no cached plan yet waits while another job of that structure builds it. Returns stats."""
    seen, todo = set(), []
    for j in jobs:
        if j["id"] is None or j["id"] in seen:
            continue
        seen.add(j["id"])
        rec = load_rec(root, j["id"])
        if rec is None or (rec["row"] is None and retry_errors):
            todo.append(j)
    todo = todo[:limit] if limit else todo
    stats = {"unique": len(seen), "to_run": len(todo), "ran": 0, "errors": 0, "wall_s": 0.0}
    t0, building, running = time.time(), set(), {}
    with ThreadPoolExecutor(max(workers, 1)) as ex:
        while todo or running:
            for j in list(todo):
                if len(running) >= workers:
                    break
                sid = j["core"]["s"]
                needs_plan = not plan_path(root, sid).exists()
                if needs_plan and sid in building:
                    continue
                todo.remove(j)
                if needs_plan:
                    building.add(sid)
                running[ex.submit(run_one, j, root, timeout)] = j
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                j = running.pop(f)
                building.discard(j["core"]["s"])
                rec = f.result()
                stats["ran"] += 1
                stats["errors"] += rec["row"] is None
                ep = (rec["row"] or {}).get("episode") or {}
                log("[%d/%d] %s %s %s step %s %s -> %s (%.0fs)" % (stats["ran"], stats["to_run"], j["id"], j["exp"], j["core"]["s"][:12],
                                                                   j["core"]["n"], j["arm"], "ERROR" if rec["row"] is None else
                                                                   "success" if ep.get("success") else "fail:%s" % (rec["row"].get("failure") or "other"),
                                                                   rec["wall_s"]))
    stats["wall_s"] = round(time.time() - t0, 1)
    return stats


# --- merge ---------------------------------------------------------------------------------------------------

OUTCOME_KEYS = ("t_stage_s", "t_transport_onset", "t_insert_onset", "startup_s", "arm_arm_events", "arm_arm_peak_N", "u2_peak", "reaction")


def flatten(j, rec):
    """One episodes.jsonl line: the job's labels, its core, and the outcome numbers of its row (rows/<id>.json has the rest)."""
    c = j["core"]
    out = {"exp": j["exp"], "arm": j["arm"], "id": j["id"], "stage": j["stage"], "r": j["r"], "sid": c["s"], "step": c["n"],
           "action": c["a"], "lam": c["lam"], "lateral": c["lat"], "exe": c["exe"], "rep": c["rep"], "meta": j["meta"],
           "lost": rec["row"] is None, "wall_s": rec["wall_s"]}
    if rec["row"] is not None:
        row, ep = rec["row"], rec["row"].get("episode") or {}
        out.update(success=bool(ep.get("success")), failure=row.get("failure"), error=row.get("error"), rtf=row.get("rtf"),
                   built=row.get("built"), peak_F_total_N=(ep.get("fixture") or {}).get("peak_F_total_N"),
                   **{k: ep.get(k) for k in OUTCOME_KEYS})
    return out


def merge(exp, jobs, root):
    """episodes.jsonl for the experiment from the rows of its job list; <exp>/merge.json says how complete it is."""
    d = Path(root) / exp
    d.mkdir(parents=True, exist_ok=True)
    n = lost = missing = 0
    with open(d / "episodes.jsonl", "w") as f:
        for j in jobs:
            rec = None if j["id"] is None else load_rec(root, j["id"])
            if rec is None:
                missing += 1
                continue
            lost += rec["row"] is None
            n += 1
            f.write(json.dumps(flatten(j, rec)) + "\n")
    st = {"entries": len(jobs), "with_record": n, "lost": lost, "missing": missing, "unique_ids": len({j["id"] for j in jobs if j["id"]})}
    (d / "merge.json").write_text(json.dumps(st))
    return st


def read_episodes(root, exp, need_complete=True):
    p, m = Path(root) / exp / "episodes.jsonl", Path(root) / exp / "merge.json"
    if not p.exists():
        raise SystemExit("prerequisite missing: %s has no episodes.jsonl -- run `%s` first" % (exp, exp))
    st = json.loads(m.read_text())
    if need_complete and st["missing"]:
        raise SystemExit("prerequisite incomplete: %s has %d of %d jobs without a row -- finish `%s` first" % (exp, st["missing"], st["entries"], exp))
    return [json.loads(line) for line in p.read_text().splitlines()]


def ok_rows(rows):
    return [r for r in rows if not r["lost"]]


# --- experiments: job lists ----------------------------------------------------------------------------------
# Each builder returns (jobs, pending): pending = sham jobs whose matched braced run has no row yet.

def exe_label(M):
    if not M["exe"]:
        raise SystemExit("prerequisite missing: manifest has no executor choice -- run reaction, then choose-exe")
    return M["exe"]


def first_critical(n):
    """The best critical context of the first n train structures that have one: [(sid, ctx)]."""
    out = []
    for sid in train_ids():
        c = best_critical(sid)
        if c:
            out.append((sid, c))
        if len(out) == n:
            break
    return out


def build_harness(M, root):
    exe, band, jobs = "E0", M["band"], []
    for rep in range(3):                                          # nominal start, descriptive: rep-major
        for shape, step in NOMINAL_STARTS:
            jobs.append(job("harness", "shape:" + shape, step, "nominal", None, exe=exe, rep=rep, meta={"group": "nominal"}))
    for i, (sid, c) in enumerate(first_critical(20)):             # 20 triples x 2 runs; alternately unbraced / LP-best-candidate
        act = None if i % 2 == 0 else BB.screen_candidates(c, BB.EXE[exe])[0]
        for rep in (0, 1):
            jobs.append(drawn("harness", sid, c, 0, 1, band, "triple_none" if act is None else "triple_braced", act, exe, rep,
                              dict(meta_of(sid, c), group="triple")))
    return jobs, 0


def build_throughput(M, root):
    """4 batches of 40 (20 critical contexts x {unbraced, braced}); draw r = batch index; + a 20-episode untimed warm-up."""
    exe, band, ctxs = "E0", M["band"], first_critical(20)
    batches = [[] for _ in range(4)]
    for b in range(4):
        for sid, c in ctxs:
            for arm, act in (("none", None), ("braced", BB.screen_candidates(c, BB.EXE[exe])[0])):
                batches[b].append(drawn("throughput", sid, c, b, 4, band, arm, act, exe, meta=dict(meta_of(sid, c), batch=b)))
    warm = [drawn("throughput", sid, c, 4, 5, band, "warm", None, exe, meta=dict(meta_of(sid, c), batch="warm")) for sid, c in ctxs]
    return warm, batches


def build_placer(M, root):
    exe, jobs, easy = "E0", [], []
    for sid, e in registry().items():
        cs = structure_contexts(sid)
        jobs += [job("placer", sid, c["step"], "none", None, exe=exe, meta=meta_of(sid, c)) for c in cs if c["pool"] == "critical"]
        es = [c for c in cs if c["pool"] == "easy"]
        if e["split"] == "train" and es and len(easy) < 20:        # one random easy context per train structure, first 20
            easy.append(es[int(BB._rng("placer_easy", [list(k) for k in e["key"]]).integers(len(es)))] | {"sid": sid})
    jobs += [job("placer", c["sid"], c["step"], "none", None, exe=exe, meta=meta_of(c["sid"], c)) for c in easy]
    return jobs, 0


def signal_selection(root):
    """(critical, easy): critical = for each of the first 40 train structures with a placer-passing critical context, the one
    with the highest LP u0 -> [(sid, step, u0_13)]; easy = the first 20 placer-passing easy contexts. A4 if fewer than 40."""
    pr = [r for r in ok_rows(read_episodes(root, "placer")) if r["success"]]
    by = defaultdict(list)
    for r in pr:
        if r["meta"]["split"] == "train" and r["meta"]["pool"] == "critical":
            by[r["sid"]].append((r["meta"]["u0"][2], -r["step"], r["sid"], r["step"]))
    crit = []
    for sid in train_ids():
        if sid in by:
            u, _, s, n = max(by[sid])
            crit.append((s, n, u))
    if len(crit) < 24:
        raise SystemExit("A4: only %d eligible train structures (< 24): extend the train split once with further unique "
                         "structures (BB.SPLITS) and re-screen the placer; still < 24 is STOP-P. Not automated." % len(crit))
    if len(crit) < 40:
        print("A4: %d eligible train structures (< 40): using all (reduced n)" % len(crit))
    easy = [(r["sid"], r["step"], r["meta"]["u0"][2]) for r in pr if r["meta"]["split"] == "train" and r["meta"]["pool"] == "easy"][:20]
    return crit[:40], easy


def build_signal(M, root):
    crit, easy = signal_selection(root)
    jobs = []
    for sel, R, pool in ((crit, 4, "critical"), (easy, 2, "easy")):
        for sid, n, _ in sel:
            c = get_ctx(sid, n)
            jobs += [drawn("signal", sid, c, r, R, M["band"], "none", None, "E0", meta=meta_of(sid, c)) for r in range(R)]
    return jobs, 0


def reaction_contexts(root):
    return signal_selection(root)[0][:4]            # A5: the first 4 passing train structures in split order, highest LP u0


def reaction_action(c):
    """The LP-best candidate (lean by lean_for) of the context under E0, applied to both settings; bypasses the SF threshold."""
    return BB.pick_lp_shared(dict(c, u0={1.0: float("inf")}), BB.EXE["E0"])


def build_reaction(M, root):
    jobs = []
    for sid, n, _ in reaction_contexts(root):
        c = get_ctx(sid, n)
        act = reaction_action(c)
        for exe in ("E0", "E1"):
            jobs.append(job("reaction", sid, n, "reaction", act, 1.0, NO_LATERAL, exe, 0, meta=meta_of(sid, c)))
            rec = load_rec(root, jobs[-1]["id"])
            if rec and rec["row"] and not ((rec["row"].get("episode") or {}).get("reaction") or {}).get("complete"):
                jobs.append(job("reaction", sid, n, "reaction", act, 1.0, NO_LATERAL, exe, 1, meta=meta_of(sid, c)))   # A5: re-run once
    return jobs, 0


def screen_contexts(root):
    return signal_selection(root)[0][:16]


def screen_cands(sid, n, exe):
    c = get_ctx(sid, n)
    return c, BB.screen_candidates(c, BB.EXE[exe])


def cand_meta(c, exe, i, a):
    f = BB.features(c["bricks"], c["step"], a, BB.EXE[exe])
    return {"index": i, "lp_u": float(f[BB.FEATURE_NAMES.index("u_br_10")]), "dy": float(f[BB.FEATURE_NAMES.index("dy")])}


def build_screen(M, root):
    exe, jobs, pending = exe_label(M), [], 0
    for sid, n, _ in screen_contexts(root):
        c, cands = screen_cands(sid, n, exe)
        base = meta_of(sid, c)
        jobs += [job("screen", sid, n, "screen", a, 1.0, NO_LATERAL, exe, meta=dict(base, **cand_meta(c, exe, i, a))) for i, a in enumerate(cands)]
        jobs.append(job("screen", sid, n, "none", None, 1.0, NO_LATERAL, exe, meta=base))
        rows = _screen_rows(root, sid, n, len(cands))
        pick = None if rows is None else _screen_pick_index(rows)
        T = None if rows is None else {r["meta"]["index"]: r for r in rows}[pick]["t_stage_s"]       # A2: the matched run's measured staging
        jobs.append(job("screen", sid, n, "sham", ("sham", T), 1.0, NO_LATERAL, exe, meta=dict(base, matched=pick)))
        pending += T is None
    return jobs, pending


def _screen_rows(root, sid, n, k):
    """The k screen rows of a context in candidate order (from the last merge), or None if any is missing or lost."""
    p = Path(root) / "screen" / "episodes.jsonl"
    rows = [r for r in map(json.loads, p.read_text().splitlines()) if r["arm"] == "screen" and r["sid"] == sid and r["step"] == n] if p.exists() else []
    rows = sorted(({r["meta"]["index"]: r for r in rows}).values(), key=lambda r: r["meta"]["index"])
    return rows if len(rows) == k and not any(r["lost"] for r in rows) else None


def _pick_rows(rows):
    return [{"index": r["meta"]["index"], "success": r["success"], "u2_peak": r["u2_peak"] if r["u2_peak"] is not None else 99.0,
             "arm_arm": bool(r["arm_arm_events"]), "lp_u": r["meta"]["lp_u"], "dy": r["meta"]["dy"]} for r in rows]


def _screen_pick_index(rows):
    return BB.screen_pick(_pick_rows(rows))["index"]


def screened_best(root, sid, n, M):
    """The screened-best action of context (sid, n) from the screen rows; fails clearly if the screen is incomplete."""
    exe = exe_label(M)
    c, cands = screen_cands(sid, n, exe)
    rows = _screen_rows(root, sid, n, len(cands))
    if rows is None:
        raise SystemExit("prerequisite incomplete: the screen of %s step %d has missing or lost candidate rows -- finish `screen`" % (sid, n))
    return c, cands[_screen_pick_index(rows)]


def _best_rows(root):
    """{(sid, step, r): screened-best row} of the validate experiment, for the sham duration."""
    p = Path(root) / "validate" / "episodes.jsonl"
    if not p.exists():
        return {}
    return {(r["sid"], r["step"], r["r"]): r for r in map(json.loads, p.read_text().splitlines()) if r["arm"] == "screened_best" and not r["lost"]}


def build_validate(M, root):
    exe, band, jobs, pending = exe_label(M), M["band"], [], 0
    have = _best_rows(root)
    for sid, n, _ in screen_contexts(root):
        c, best = screened_best(root, sid, n, M)
        lp = BB.pick_lp_shared(c, BB.EXE[exe])
        meta = meta_of(sid, c)
        for r in range(4):
            args = (sid, c, r, 4, band)
            jobs += [drawn("validate", *args, "none", None, exe, meta=meta), drawn("validate", *args, "lp_shared", lp, exe, meta=meta),
                     drawn("validate", *args, "screened_best", best, exe, meta=meta)]
            b = have.get((sid, n, r))
            T = b and b["t_stage_s"]
            jobs.append(drawn("validate", *args, "sham", ("sham", T), exe, meta=dict(meta, matched=b["id"] if b else None)))
            pending += T is None
    return jobs, pending


def build_harm(M, root):
    exe, jobs = exe_label(M), []
    for sid, n, _ in screen_contexts(root):
        c, best = screened_best(root, sid, n, M)
        lp = BB.pick_lp_shared(c, BB.EXE[exe])
        for rep in (0, 1):
            jobs += [job("harm", sid, n, arm, a, exe=exe, rep=rep, meta=meta_of(sid, c)) for arm, a in
                     (("none", None), ("lp_shared", lp), ("screened_best", best))]
    return jobs, 0


BUILD = {"harness": build_harness, "placer": build_placer, "signal": build_signal, "reaction": build_reaction,
         "screen": build_screen, "validate": build_validate, "harm": build_harm}


# --- running an experiment ---------------------------------------------------------------------------------------

def run_experiment(exp, root, M, args):
    if exp == "throughput":
        return run_throughput(root, M, args)
    while True:
        jobs, pending = BUILD[exp](M, root)
        if args.dry_run:
            return report_counts(exp, jobs, pending, root)
        stats = run_jobs(jobs, root, args.workers, args.timeout, args.retry_errors, args.limit)
        st = merge(exp, jobs, root)
        print("%s: %s | merge %s" % (exp, stats, st))
        M["job_counts"][exp] = {"entries": len(jobs), "unique": st["unique_ids"], "pending_sham": pending, "ran_last": stats["ran"]}
        save_manifest(root, M)
        if args.limit:
            return st
        jobs2, pending2 = BUILD[exp](M, root)                      # sham / re-run jobs appear once their matched runs have rows
        if {j["id"] for j in jobs2} == {j["id"] for j in jobs}:
            if pending2:
                print("%s: %d sham job(s) still without a matched staging time (lost or no-transport run)" % (exp, pending2))
            return st


def run_throughput(root, M, args):
    warm, batches = build_throughput(M, root)
    if args.dry_run:
        return report_counts("throughput", warm + sum(batches, []), 0, root, extra="(%d warm-up + 4 x %d)" % (len(warm), len(batches[0])))
    pth = Path(root) / "throughput" / "batches.json"
    pth.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(pth.read_text()) if pth.exists() else []
    print("warm-up (untimed): %s" % run_jobs(warm, root, 4, args.timeout, args.retry_errors))
    for w, jobs in zip((1, 2, 4, 8), batches):
        if any(b["workers"] == w for b in done):
            continue
        for j in jobs:                                              # a timed batch is run in full, never from stored rows
            (rows_dir(root) / (j["id"] + ".json")).unlink(missing_ok=True)
        stats = run_jobs(jobs, root, w, args.timeout)
        recs = [load_rec(root, j["id"]) for j in jobs]
        ok = [r for r in recs if r and r["row"]]
        done.append({"workers": w, "n": len(jobs), "wall_s": stats["wall_s"], "lost": len(jobs) - len(ok), "ids": [j["id"] for j in jobs],
                     "startup_s": [(r["row"].get("episode") or {}).get("startup_s") for r in ok],
                     "rtf": [r["row"].get("rtf") for r in ok]})
        pth.write_text(json.dumps(done, indent=1))
        print("batch workers=%d: %d episodes in %.0f s = %.0f/h" % (w, len(jobs), stats["wall_s"], len(jobs) / stats["wall_s"] * 3600))
    merge("throughput", warm + sum(batches, []), root)
    M["job_counts"]["throughput"] = {"entries": len(warm) + 160, "unique": len(warm) + 160}
    save_manifest(root, M)


def report_counts(exp, jobs, pending, root, extra=""):
    ids = [j["id"] for j in jobs if j["id"]]
    have = sum(load_rec(root, i) is not None for i in set(ids))
    print("%-10s entries %4d  unique %4d  with-row %4d  to-run %4d  pending-sham %3d %s" % (exp, len(jobs), len(set(ids)), have, len(set(ids)) - have, pending, extra))
    return {"entries": len(jobs), "unique": len(set(ids)), "ids": set(ids), "pending": pending}


# --- executor choice and freeze --------------------------------------------------------------------------------

def reaction_table(root):
    rows = ok_rows(read_episodes(root, "reaction"))
    by, e1_div = defaultdict(dict), False
    for r in sorted(rows, key=lambda r: r["rep"]):                 # a valid rep-1 re-run replaces an incomplete rep 0
        rc = r["reaction"] or {}
        valid = bool(rc.get("complete")) and rc.get("scalar") is not None
        div = r["failure"] == "divergence" or bool(r["error"] and "diverg" in r["error"])
        e1_div |= div and r["exe"] == "E1"
        if r["exe"] not in by[(r["sid"], r["step"])] or valid:
            by[(r["sid"], r["step"])][r["exe"]] = {"scalar": rc.get("scalar"), "valid": valid, "complete": rc.get("complete"), "diverged": div,
                                                   "F_mean": rc.get("F_mean"), "lp_F": rc.get("lp_F"), "M_mean_about_brace": rc.get("M_mean_about_brace"),
                                                   "success": r["success"], "id": r["id"], "rep": r["rep"]}
    t = [{"sid": k[0][:8], "step": k[1], "exe": e, **v} for k, d in by.items() for e, v in sorted(d.items())]
    e0 = sum(1 for d in by.values() if d.get("E0", {}).get("valid") and d["E0"]["scalar"] >= 0.5)
    return {"contexts": len(by), "per_context": t, "n_E0_ge_0.5": e0, "E1_diverged": e1_div,
            "invalid_runs": [[k[0][:8], k[1], e] for k, d in by.items() for e, v in d.items() if not v["valid"]],
            "choice": "E0" if e1_div or e0 >= 3 else "E1", "rule": "E1 divergence -> E0; else E0 iff scalar >= 0.5 in >= 3 of 4 (A5); otherwise E1"}


def cmd_choose_exe(root, M, args):
    t = reaction_table(root)
    if t["contexts"] != 4:
        raise SystemExit("reaction check has %d contexts, not 4 (run `reaction`)" % t["contexts"])
    if t["invalid_runs"]:
        print("note: invalid runs (incomplete plateau after the re-run; they count as below 0.5): %s" % t["invalid_runs"])
    if M["exe"] and M["exe"] != t["choice"]:
        raise SystemExit("executor already chosen (%s); the choice is made once" % M["exe"])
    M["exe"], M["exe_choice"] = t["choice"], {k: t[k] for k in ("n_E0_ge_0.5", "E1_diverged", "rule")}
    save_manifest(root, M)
    print("executor setting: %s (%s)" % (t["choice"], M["exe_choice"]))


def cmd_freeze(root, M, args):
    crit, easy = signal_selection(root)
    sig = read_episodes(root, "signal")                             # signal must be complete
    if args.band:
        M["band_history"].append({"from": M["band"], "to": list(args.band), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": "manual (--band)"})
        M["band"] = list(args.band)
    M["selection"] = {"signal_critical": [list(x) for x in crit], "signal_easy": [list(x) for x in easy],
                      "screen_16": [list(x) for x in crit[:16]], "reaction_4": [list(x) for x in crit[:4]]}
    M["frozen"], M["frozen_at"] = True, time.strftime("%Y-%m-%d %H:%M:%S")
    M["signal_rows"] = len(sig)
    save_manifest(root, M)
    print("frozen: band %s, %d critical + %d easy signal contexts, hash %s" % (M["band"], len(crit), len(easy), M["hash"]))


# --- tables -------------------------------------------------------------------------------------------------------

def rate(rows, key="success"):
    return float(np.mean([bool(r[key]) for r in rows])) if rows else None


def fail_cause(r):
    return r["failure"] or ("arm_arm" if r["arm_arm_events"] else "incomplete")


def _acc_range(shape, step):
    p = HERE / "results" / "v4" / "j" / "acc" / "rows.jsonl"
    v = [json.loads(line)["u_peak_by_step"].get(str(step)) for line in p.read_text().splitlines() if json.loads(line)["structure"] == shape]
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None


def table_harness(root):
    rows = ok_rows(read_episodes(root, "harness", need_complete=False))
    out = {"fixture_wrench_checks": "not run by this runner (static scenes, S0.2/S0.4)"}
    pairs = defaultdict(dict)
    for r in rows:
        if r["meta"].get("group") == "triple":
            pairs[(r["sid"], r["step"], r["arm"])][r["rep"]] = r
    both = [p for p in pairs.values() if 0 in p and 1 in p]
    out["repeat_agreement"] = {"equal": sum(p[0]["success"] == p[1]["success"] for p in both), "pairs": len(both),
                               "gate": ">= 18/20", "u2_peak_pairs": [[p[0]["u2_peak"], p[1]["u2_peak"]] for p in both]}
    nom = []
    for shape, step in NOMINAL_STARTS:
        rs = [r for r in rows if r["sid"] == "shape:" + shape and r["step"] == step]
        u = [r["u2_peak"] for r in rs if r["u2_peak"] is not None]
        nom.append({"shape": shape, "step": step, "runs": len(rs), "success": [r["success"] for r in rs], "u2_peak": u,
                    "full_build_u_peak_range": _acc_range(shape, step)})
    out["nominal_start"] = nom
    return out


def table_throughput(root):
    p = Path(root) / "throughput" / "batches.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text())
    per = [{"workers": x["workers"], "n": x["n"], "wall_s": x["wall_s"], "episodes_per_h": x["n"] / x["wall_s"] * 3600,
            "per_worker_per_h": x["n"] / x["wall_s"] * 3600 / x["workers"], "lost": x["lost"],
            "startup_s_median": float(np.median(x["startup_s"])) if x["startup_s"] else None,
            "rtf_median": float(np.median([v for v in x["rtf"] if v])) if x["rtf"] else None} for x in b]
    best = max(per, key=lambda x: x["episodes_per_h"])
    return {"batches": per, "best": best["workers"], "best_per_h": best["episodes_per_h"],
            "branch": "pass" if best["episodes_per_h"] >= 150 else "A3: 60-149/h continue, record failed" if best["episodes_per_h"] >= 60 else "STOP-T"}


def table_placer(root):
    rows = ok_rows(read_episodes(root, "placer", need_complete=False))
    f = lambda rs: {"n": len(rs), "success": rate(rs), "native_u2_peak_median": float(np.median([r["u2_peak"] for r in rs])) if rs else None,
                    "native_u2_peak_max": max((r["u2_peak"] for r in rs), default=None)}
    crit = [r for r in rows if r["meta"]["pool"] == "critical"]
    easy = [r for r in rows if r["meta"]["pool"] == "easy"]
    return {"all": f(rows), "critical": f(crit), "easy": f(easy), "gate": ">= 0.70",
            "failures_by_cause": dict(Counter(fail_cause(r) for r in rows if not r["success"])),
            "critical_structures_passing": len({r["sid"] for r in crit if r["success"] and r["meta"]["split"] == "train"})}


def table_signal(root, M):
    rows = ok_rows(read_episodes(root, "signal", need_complete=False))
    crit = [r for r in rows if r["meta"]["pool"] == "critical"]
    easy = [r for r in rows if r["meta"]["pool"] == "easy"]
    fails = [r for r in crit if not r["success"]]
    lo, hi = M["band"]
    strata = []
    for k in range(3):
        rs = [r for r in crit if lo + k * (hi - lo) / 3 <= r["lam"] < lo + (k + 1) * (hi - lo) / 3 + (1e-9 if k == 2 else 0)]
        strata.append({"lam": [round(lo + k * (hi - lo) / 3, 3), round(lo + (k + 1) * (hi - lo) / 3, 3)], "n": len(rs), "failure": None if not rs else 1 - rate(rs)})
    return {"band": M["band"], "critical_trials": len(crit), "critical_contexts": len({(r["sid"], r["step"]) for r in crit}),
            "critical_failure_rate": None if not crit else 1 - rate(crit), "gate_critical": "[0.30, 0.90]",
            "failure_causes": dict(Counter(fail_cause(r) for r in fails)),
            "joint_break_share_of_failures": None if not fails else sum(r["failure"] == "joint_break" for r in fails) / len(fails),
            "gate_joint_break": ">= 0.60", "easy_trials": len(easy), "easy_failure_rate": None if not easy else 1 - rate(easy), "gate_easy": "<= 0.10",
            "by_lambda_third": strata, "applied_peak_F_total_N_median": float(np.median([r["peak_F_total_N"] for r in crit if r["peak_F_total_N"]] or [float("nan")]))}


def _by_ctx(rows, arm):
    d = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            d[(r["sid"], r["step"])].append(r)
    return d


def _ctx_diff(a, b, keys):
    """Context-weighted mean of rate(a) - rate(b) over contexts that have both; (mean, better, worse, n)."""
    d = [(k, rate(a[k]) - rate(b[k])) for k in keys if a.get(k) and b.get(k)]
    return {"diff": float(np.mean([x for _, x in d])) if d else None, "better": sum(x > 0 for _, x in d), "worse": sum(x < 0 for _, x in d), "contexts": len(d)}


def table_benefit(root):
    val = ok_rows(read_episodes(root, "validate", need_complete=False))
    harm = ok_rows(read_episodes(root, "harm", need_complete=False))
    arms = {a: _by_ctx(val, a) for a in ("none", "sham", "lp_shared", "screened_best")}
    keys = sorted(arms["screened_best"])
    out = {"validate_rate": {a: rate([r for k in keys for r in arms[a].get(k, [])]) for a in arms},
           "A1_sb_minus_none": _ctx_diff(arms["screened_best"], arms["none"], keys), "gate_A1": ">= 0.20; better in >= 5 of 16, worse in <= 1"}
    h = {a: {(r["sid"], r["step"], r["rep"]): r for r in harm if r["arm"] == a} for a in ("none", "screened_best", "lp_shared")}
    pairs = [(h["screened_best"][k], h["none"][k]) for k in h["screened_best"] if k in h["none"]]
    out["harm_fixture_off"] = {"pairs": len(pairs), "excess_failure_sb_over_none": float(np.mean([(not a["success"]) - (not b["success"]) for a, b in pairs])) if pairs else None,
                               "gate": "<= 0.10", "failure_none": None if not h["none"] else float(np.mean([not r["success"] for r in h["none"].values()])),
                               "failure_sb": None if not h["screened_best"] else float(np.mean([not r["success"] for r in h["screened_best"].values()])),
                               "failure_lp_shared": None if not h["lp_shared"] else float(np.mean([not r["success"] for r in h["lp_shared"].values()]))}
    out["A1_sb_minus_sham_all_pairs"] = _ctx_diff(arms["screened_best"], arms["sham"], keys)
    # A2 onset check: a (context, draw) pair is valid iff the sham's insert onset is within 0.1 s of the braced run's
    sb = {(r["sid"], r["step"], r["r"]): r for r in val if r["arm"] == "screened_best"}
    sh = {(r["sid"], r["step"], r["r"]): r for r in val if r["arm"] == "sham"}
    ok = {k for k in sb if k in sh and sb[k]["t_insert_onset"] is not None and sh[k]["t_insert_onset"] is not None
          and abs(sb[k]["t_insert_onset"] - sh[k]["t_insert_onset"]) <= ONSET_TOL_S}
    v = lambda arm: {c: [r for r in rs if (r["sid"], r["step"], r["r"]) in ok] for c, rs in arms[arm].items()}
    out["A2_onset_check"] = {"pairs": len(sh), "valid_pairs": len(ok), "tol_s": ONSET_TOL_S,
                             "sb_minus_sham_valid_pairs": _ctx_diff(v("screened_best"), v("sham"), keys),
                             "onset_diff_s": [round(sb[k]["t_insert_onset"] - sh[k]["t_insert_onset"], 3) for k in sb if k in sh and sb[k]["t_insert_onset"] and sh[k]["t_insert_onset"]]}
    out["A5_sb_minus_lp_shared"] = _ctx_diff(arms["screened_best"], arms["lp_shared"], keys)
    out["gate_5"] = ">= 0.10"
    out["sham_none_gap"] = _ctx_diff(arms["sham"], arms["none"], keys)
    sc = [r["reaction"]["scalar"] for r in val if r["arm"] == "screened_best" and r["reaction"] and r["reaction"].get("scalar") is not None]
    out["measured_over_LP_reaction_scalar_sb"] = {"n": len(sc), "median": float(np.median(sc)) if sc else None}
    out["arm_arm_rate_by_arm"] = {a: rate([r for k in arms[a] for r in arms[a][k]], "arm_arm_events") for a in arms}
    return out


def table_screen(root):
    rows = ok_rows(read_episodes(root, "screen", need_complete=False))
    ctxs = defaultdict(list)
    for r in rows:
        ctxs[(r["sid"], r["step"])].append(r)
    out = []
    for k, rs in sorted(ctxs.items()):
        cand = [r for r in rs if r["arm"] == "screen"]
        pick = BB.screen_pick(_pick_rows(cand))["index"] if len(cand) else None
        one = lambda arm: next((r["success"] for r in rs if r["arm"] == arm), None)
        out.append({"sid": k[0], "step": k[1], "candidates": len(cand), "successes": sum(r["success"] for r in cand), "pick": pick,
                    "none": one("none"), "sham": one("sham"), "arm_arm": sum(bool(r["arm_arm_events"]) for r in cand)})
    return out


def compute_tables(root, M):
    def safe(f, *a):
        try:
            return f(*a)
        except SystemExit as e:
            return {"unavailable": str(e)}
    t = {"manifest_hash": M.get("hash"), "exe": M["exe"], "band": M["band"],
         "c0_harness": safe(table_harness, root), "c1_throughput": safe(table_throughput, root),
         "c2_placer": safe(table_placer, root), "c3_signal": safe(table_signal, root, M),
         "reaction": safe(reaction_table, root), "screen": safe(table_screen, root), "c4_c5_benefit": safe(table_benefit, root)}
    p = Path(root) / "r0" / "tables.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t, indent=1, default=float))
    return t


def _fmt(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (list, tuple)):
        return [_fmt(x) for x in v]
    if isinstance(v, dict):
        return {k: _fmt(x) for k, x in v.items()}
    return v


def md(title, d):
    out = ["### " + title, ""]
    if d is None or (isinstance(d, dict) and "unavailable" in d):
        return "\n".join(out + ["(no data: %s)" % (d or {}).get("unavailable", "not run")])
    if isinstance(d, list):                                       # a list of records: one table
        keys = list(d[0]) if d else []
        out += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        out += ["| " + " | ".join(json.dumps(_fmt(r.get(k)), default=float) for k in keys) + " |" for r in d]
        return "\n".join(out)
    out += ["| item | value |", "|---|---|"]
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            keys = list(v[0])
            out.append("| %s | %s |" % (k, "; ".join(", ".join("%s=%s" % (kk, json.dumps(_fmt(x.get(kk)), default=float)) for kk in keys) for x in v)))
        else:
            out.append("| %s | %s |" % (k, json.dumps(_fmt(v), default=float)))
    return "\n".join(out)


# --- CLI -----------------------------------------------------------------------------------------------------------

def cmd_counts(root, M, args):
    allids, total = [], 0
    for exp in EXPERIMENTS:
        try:
            if exp == "throughput":
                warm, batches = build_throughput(M, root)
                jobs, pending = warm + sum(batches, []), 0
            else:
                jobs, pending = BUILD[exp](M, root)
            rep = report_counts(exp, jobs, pending, root, "(%s nominal)" % NOMINAL[exp] if NOMINAL[exp] else "")
            allids += jobs
            total += rep["entries"]
        except SystemExit as e:
            print("%-10s needs earlier results (nominal %s): %s" % (exp, NOMINAL[exp], str(e).splitlines()[0][:100]))
    ids = [j["id"] for j in allids if j["id"]]
    print("across the listed experiments: %d entries, %d unique ids, %d duplicates run once" % (len(ids), len(set(ids)), len(ids) - len(set(ids))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=EXPERIMENTS + ["counts", "freeze", "choose-exe", "tables"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300.0, help="wall seconds per job")
    ap.add_argument("--dry-run", action="store_true", help="print job counts, run nothing")
    ap.add_argument("--limit", type=int, help="run only the first N jobs that still lack a row (mini runs)")
    ap.add_argument("--retry-errors", action="store_true", help="also re-run jobs whose stored record is a job error")
    ap.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), help="freeze: set the lambda band (manual, recorded in the manifest)")
    args = ap.parse_args()
    root, M = args.root, load_manifest(args.root)
    if args.cmd == "counts":
        cmd_counts(root, M, args)
    elif args.cmd == "freeze":
        cmd_freeze(root, M, args)
    elif args.cmd == "choose-exe":
        cmd_choose_exe(root, M, args)
    elif args.cmd == "tables":
        t = compute_tables(root, M)
        for title, key in (("Criterion 0: harness", "c0_harness"), ("Criterion 1: throughput", "c1_throughput"), ("Criterion 2: placer", "c2_placer"),
                           ("Criterion 3: signal", "c3_signal"), ("Reaction check", "reaction"), ("Screen (per context)", "screen"),
                           ("Criteria 4 and 5: benefit and headroom", "c4_c5_benefit")):
            print(md(title, t[key]) + "\n")
        print("-> %s" % (root / "r0" / "tables.json"))
    else:
        run_experiment(args.cmd, root, M, args)


if __name__ == "__main__":
    main()
