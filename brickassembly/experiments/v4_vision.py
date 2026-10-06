"""Perception phase V1 runner (plan_v4_perception S1.3b): the experiments E1, E1b, E3/E5 (pool), E2, E2b, E4 and the G-V1 tables.

    bash scripts/run.sh experiments/v4_vision.py e1 --workers 4        # 36 builds: {cube, arch, hollow_box, S3} x {gt, fk_oracle, fk_vision} x seeds 0-2
    bash scripts/run.sh experiments/v4_vision.py e1b --workers 4       # 54 single-footprint step-episodes
    bash scripts/run.sh experiments/v4_vision.py pool                  # E3 + E5 -> pool.json (frozen, hashed); E2 and E2b need it
    bash scripts/run.sh experiments/v4_vision.py e2 --workers 4        # 330 matched-injection step-episodes (fk_oracle)
    bash scripts/run.sh experiments/v4_vision.py e2b --workers 4       # 6 full builds (--continue) with the signed scale-2 vectors
    bash scripts/run.sh experiments/v4_vision.py sweep --workers 4     # ACC's first diagnostic: signed axis sweep, 31 contexts x 12 = 372
    bash scripts/run.sh experiments/v4_vision.py e4 --workers 4        # 40 contexts x {gt no look, fk_vision} timed
    bash scripts/run.sh experiments/v4_vision.py tables                # T-V1a/b/c, F-V1, G-V1 -> tables.json

Everything lives under --root (default results/v4/vision/v1; a re-run round or a smoke run uses another root, e.g. v1_acc, v1_smoke):
rows/<id>.json (one per job, as in experiments/v4_brace.py), recordings/<id>.npz (E1's --record-all), plans/ (the cell's plan cache),
specs/, logs/<id>.log, <exp>/{episodes.jsonl, merge.json, manifest.json}, e4/batches.json, pool.json, tables.json. A manifest carries
the vision PARAMS_HASH and the git HEAD. Every subcommand takes --workers, --limit (run only the first N jobs without a row),
--retry-errors, --dry-run; a finished job is never re-run (resume). The tables read the result files only: an experiment that is
missing or incomplete is "unavailable" and never counts as a pass. No newton in this process (the cell is a subprocess).
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import blueprint as BP  # noqa: E402
import planner as P  # noqa: E402
import vision as V  # noqa: E402
from experiments import j_tables as JT  # noqa: E402
from experiments import v4_brace as VB  # noqa: E402
from tasks import brace_bandit as BB  # noqa: E402

ROOT = HERE / "results" / "v4" / "vision" / "v1"
GATED = ("cube", "arch", "hollow_box")
E1_SHAPES = GATED + ("S3",)                      # S3 is reported, not gated
ARMS = ("gt", "fk_oracle", "fk_vision")
SEEDS = (0, 1, 2)
S5_N = 13                                        # S5[:13]: the feeder holds 24 bricks, S5 has 26
E1B = (("S1", (1, 3)), ("S2A", (4,)), ("S3", (2, 7)), ("S3L", (3,)), ("S5p13", (4, 10, 12)))     # P2's 9 single-footprint contexts
E2_EXTRA = (("S5p13", 12), ("S3", 7))
K_DRAWS = 4
BOOT = 2000
EXPECTED = {"e1": 36, "e1b": 54, "e2": 330, "e2b": 6, "e4": 80}
TIMEOUT = {"e1": 2400, "e2b": 2400, "e1b": 600, "e2": 600, "e4": 600, "sweep": 600}
SWEEP_LEVELS = (0.3, 0.6, 1.0)                   # mm; the ACC axis sweep: {long, short} x {+, -} x levels
NATIVES = HERE / "results" / "v4" / "vision" / "v1_r1" / "tables.json"       # the sweep's common contexts: E2's included contexts of this round
EDGE_MIN = 20                                    # plan 2.4: the edge fallback is admitted after >= 20 single-footprint cases that pass E2 at 2x


# --- structures --------------------------------------------------------------------------------------------------

def bricks_of(name):
    if name == "S5p13":
        return list(P.STRUCTURES["S5"][:S5_N])
    return list(BP.load(name)) if name in BP.SAMPLES else list(P.STRUCTURES[name])


def sid_of(name):
    sid = "shape:" + name if name in BP.SAMPLES else "prefix:S5:%d" % S5_N if name == "S5p13" else "structure:" + name
    if name == "S5p13":
        VB.EXTRA_SPECS[sid] = {"bricks": [list(b) for b in bricks_of(name)], "name": name}
    elif not sid.startswith("shape:"):
        VB.EXTRA_SPECS[sid] = {"structure": name}
    return sid


def nij(bricks):
    """The grid extent exactly as dual_arm_sim.make_plan computes it."""
    return (max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks), max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks))


def exposed_single(order, n, NI, NJ):
    """Whether step n's target is a single footprint (dual_arm_sim.exposed_cells; tests/test_v4_vision.py checks they agree)."""
    i0, j0, k = order[n][2:5]
    nx, ny = P.footprint(order[n][1], order[n][5])
    cells = lambda bs: {c for b in bs for c in P.cells(b)[0]}
    cover = cells([b for b in order[:n] if b[4] == k])
    below = {(i, j) for i in range(-2, NI + 2) for j in range(-2, NJ + 2)} if k == 0 else cells([b for b in order[:n] if b[4] == k - 1])
    cs = sorted(c for c in below - cover if i0 - 3 <= c[0] < i0 + nx + 3 and j0 - 3 <= c[1] < j0 + ny + 3)
    foot = {(i0 + a, j0 + b) for a in range(nx) for b in range(ny)}
    return bool(k > 0 and cs and set(cs) <= foot)


def look_type(name, step):
    bricks = bricks_of(name)
    return "single" if exposed_single(P.sequence(bricks), step, *nij(bricks)) else "course"


def check_s5_prefix():
    """S5[:13] is a faithful prefix: the same bricks in the same order as the full S5's first 13 steps, step 12 included.
    Returns (full NIJ, prefix NIJ): if they differ, the plate-centred world position of the targets shifts (reported)."""
    full, pre = P.sequence(P.STRUCTURES["S5"]), P.sequence(bricks_of("S5p13"))
    assert pre == full[:S5_N], "S5[:%d] differs from the first %d steps of S5" % (S5_N, S5_N)
    assert pre[12] == full[12] and pre[12][0] == "b_012", "S5 step 12 differs"
    return nij(list(P.STRUCTURES["S5"])), nij(bricks_of("S5p13"))


# --- jobs ----------------------------------------------------------------------------------------------------------

def vflags(aim, seed, shadow=False, inject=None, record=False, rr=0, cont=False):
    v = {"look": True, "aim": aim, "seed": seed}
    if cont:
        v["continue_"] = True                                    # the cell's --continue (spec key = its dest): the queue goes on after a failure
    if shadow:
        v["shadow_vision"] = True
    if inject:
        v["aim_inject"] = inject
    if record:
        v["_record"] = True
    if rr:
        v["_rr"] = rr
    return v


def build_e1(reruns=None):
    """36 builds, seed-major; reruns = {shape: round}: that shape's repeats once more in all arms (a new round = new ids)."""
    jobs = []
    for shape in E1_SHAPES:
        for rr in range(1 + (reruns or {}).get(shape, 0)):
            for seed in SEEDS:
                for arm in ARMS:
                    jobs.append(VB.job("e1", sid_of(shape), None, arm, None, rep=seed, meta={"shape": shape, "seed": seed, "rr": rr},
                                       v=vflags(arm, seed, shadow=arm != "fk_vision", record=True, rr=rr)))
    return jobs


def build_e1b():
    check_s5_prefix()
    jobs = []
    for name, steps in E1B:
        for step in steps:
            for seed in SEEDS:
                for arm in ("fk_oracle", "fk_vision"):
                    jobs.append(VB.job("e1b", sid_of(name), step, arm, None, rep=seed,
                                       meta={"shape": name, "step": step, "seed": seed, "look_type": "single"},
                                       v=vflags(arm, seed, shadow=arm == "fk_oracle")))
    return jobs


def e2_contexts():
    return [(s, n) for s in GATED for n in range(len(bricks_of(s)))] + list(E2_EXTRA)


def load_pool(root, partial=False):
    p = Path(root) / "pool.json"
    if not p.exists():
        raise SystemExit("prerequisite missing: %s -- run `pool` first" % p)
    pool = json.loads(p.read_text())
    if pool["hash"] != pool_hash(pool):
        raise SystemExit("pool.json does not match its hash: it was edited after the freeze")
    if pool.get("partial") and not partial:
        raise SystemExit("pool.json was frozen from incomplete E1/E1b rows (a smoke pool); pass --partial to use it anyway")
    return pool


def pool_hash(pool):
    blob = json.dumps({k: pool[k] for k in ("course", "single", "drift_max_mm")}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def draw_vectors(pool, name, step, K=K_DRAWS):
    """K vectors e = (dx mm, dy mm, dyaw deg) from the pool of the context's look type, by the A6 seed contract
    SHA-256(master_seed, "e2", key, step, k); draw 0 is from the top decile of |e_xy|. Distinct while the pool allows."""
    vecs = pool[look_type(name, step)]
    if not vecs:
        raise SystemExit("the %s pool is empty (context %s step %d)" % (look_type(name, step), name, step))
    top = np.argsort([math.hypot(v[0], v[1]) for v in vecs], kind="stable")[len(vecs) - max(1, math.ceil(0.1 * len(vecs))):]
    out, used = [], []
    for k in range(K):
        perm = BB._rng("e2", name, step, k).permutation(top if k == 0 else np.arange(len(vecs)))
        pick = next((int(i) for i in perm if int(i) not in used), int(perm[0]))
        used.append(pick)
        out.append(vecs[pick])
    return out


def scale2(e, drift_mm):
    """Scale 2: 2 e_xy + drift_max e_xy/|e_xy|, 2 dpsi."""
    r = math.hypot(e[0], e[1])
    u = (e[0] / r, e[1] / r) if r > 0 else (1.0, 0.0)
    return [round(2 * e[0] + drift_mm * u[0], 4), round(2 * e[1] + drift_mm * u[1], 4), round(2 * e[2], 4)]


def inj(vec):
    return {"dx_mm": round(vec[0], 4), "dy_mm": round(vec[1], 4), "dyaw_deg": round(vec[2], 4)}


def build_e2(pool):
    """33 contexts x (2 controls + 4 draws x scales {1, 2}) = 330 fk_oracle step-episodes, unbraced, nominal start."""
    check_s5_prefix()
    jobs = []
    for name, step in e2_contexts():
        typ, sid = look_type(name, step), sid_of(name)
        base = {"ctx": [name, step], "look_type": typ, "pool_hash": pool["hash"]}
        for r in (0, 1):
            jobs.append(VB.job("e2", sid, step, "control", None, rep=r, meta=dict(base, kind="control", k=r), v=vflags("fk_oracle", 0)))
        for k, e in enumerate(draw_vectors(pool, name, step)):
            for scale, vec in ((1, [round(x, 4) for x in e]), (2, scale2(e, pool["drift_max_mm"]))):
                jobs.append(VB.job("e2", sid, step, "x%d" % scale, None, rep=k, v=vflags("fk_oracle", 0, inject=inj(vec)),
                                   meta=dict(base, kind="inject", k=k, scale=scale, inj=inj(vec), top_decile=k == 0)))
    return jobs


def build_e2b(pool):
    """fk_oracle full builds of the three gated shapes with E2's signed scale-2 vector at every step, k in {0, 1}. Run with the cell's
    --continue (user decision 2026-10-07, ledger v4_v1_e2b_continue): every step is scored; the ones after a gate miss are after_failure."""
    jobs = []
    for name in GATED:
        for k in (0, 1):
            vec = {str(n): inj(scale2(draw_vectors(pool, name, n)[k], pool["drift_max_mm"])) for n in range(len(bricks_of(name)))}
            jobs.append(VB.job("e2b", sid_of(name), None, "fk_oracle", None, rep=k, meta={"shape": name, "k": k, "pool_hash": pool["hash"], "continue": True},
                               v=vflags("fk_oracle", 0, inject=vec, cont=True)))
    return jobs


def sweep_contexts(natives):
    """E2's contexts minus the native ones ([[name, step], ...]): the common context set (31 of 33 in round v1_r1)."""
    nat = {tuple(c) for c in natives}
    return [c for c in e2_contexts() if c not in nat]


def build_sweep(natives):
    """ACC's first diagnostic: fk_oracle step-episodes (nominal start, unbraced), a persistent vector along the target brick's long axis
    (target-frame x: the mesh's long side; for a square brick the axis is arbitrary) or short axis (y), both signs, |b| in SWEEP_LEVELS
    mm, dpsi 0, 1 trial each: 12 per context."""
    check_s5_prefix()
    jobs = []
    for name, step in sweep_contexts(natives):
        typ, sid = look_type(name, step), sid_of(name)
        for axis, i in (("long", 0), ("short", 1)):
            for sign in (1, -1):
                for b in SWEEP_LEVELS:
                    vec = [0.0, 0.0, 0.0]
                    vec[i] = sign * b
                    jobs.append(VB.job("sweep", sid, step, "fk_oracle", None, rep=0, v=vflags("fk_oracle", 0, inject=inj(vec)),
                                       meta={"ctx": [name, step], "look_type": typ, "axis": axis, "sign": sign, "level": b, "inj": inj(vec)}))
    return jobs


def build_e4():
    """{arm: 40 jobs}: R0's 20 critical contexts (the throughput batch) x 2 hidden draws (r = 2, 3 of 4: the 4- and 8-worker batches),
    unbraced, fixture on; gt without look (R0's configuration: the same job cores as R0's rows) and fk_vision with look."""
    M, ctxs, out = VB.load_manifest(VB.ROOT), VB.first_critical(20), {"gt": [], "fk_vision": []}
    for arm in out:
        for r in (2, 3):
            for sid, c in ctxs:
                d = BB.hidden("throughput", c["key"], c["step"], r, 4, band=tuple(M["band"]))
                out[arm].append(VB.job("e4", sid, c["step"], arm, None, d["lam"], d["lateral"], "E0", 0, "throughput", r,
                                       dict(VB.meta_of(sid, c), draw=r), v=None if arm == "gt" else vflags("fk_vision", 0)))
    return out


# --- running ---------------------------------------------------------------------------------------------------------

def git_head():
    r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def write_manifest(root, exp, **extra):
    d = Path(root) / exp
    d.mkdir(parents=True, exist_ok=True)
    dirty = subprocess.run(["git", "-C", str(HERE), "diff", "--name-only", "HEAD", "--", "*.py"], capture_output=True, text=True).stdout.split()
    M = {"exp": exp, "params_hash": V.PARAMS_HASH, "git_head": git_head(), "dirty_py": dirty, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (d / "manifest.json").write_text(json.dumps(M, indent=1, default=float))
    return M


def flat(j, rec):
    """One episodes.jsonl line: labels, the core's vision flags and the outcome basics (rows/<id>.json has the rest)."""
    c, row = j["core"], rec["row"]
    out = {"exp": j["exp"], "arm": j["arm"], "id": j["id"], "sid": c["s"], "step": c["n"], "rep": c["rep"], "meta": j["meta"],
           "v": c.get("v"), "lost": row is None, "wall_s": rec["wall_s"]}
    if row is not None:
        ep = row.get("episode") or {}
        out.update(failure=row.get("failure"), error=row.get("error"), built=row.get("built"), total=row.get("total"), rtf=row.get("rtf"),
                   startup_s=ep.get("startup_s"), params_hash=(row.get("vision") or {}).get("params_hash"),
                   success=bool(ep["success"]) if ep else row.get("failure") is None and row.get("built") == row.get("total"))
    return out


def run_exp(exp, jobs, args, select=None, **extra):
    """Run the jobs (those `select` keeps) in the pool, merge the full list, write the manifest. The merge covers all of `jobs`, so
    a filtered or limited run shows the rest as missing."""
    root, todo = args.root, [j for j in jobs if select is None or select(j)]
    ids = {j["id"] for j in jobs}
    if args.dry_run:
        have = sum(VB.load_rec(root, i) is not None for i in ids)
        print("%-4s entries %4d  unique %4d  with-row %4d  to-run %4d  (selected %d)" % (exp, len(jobs), len(ids), have, len(ids) - have, len(todo)))
        return None
    stats = VB.run_jobs(todo, root, args.workers, args.timeout or TIMEOUT[exp], args.retry_errors, args.limit)
    st = VB.merge(exp, jobs, root, flat)
    write_manifest(root, exp, jobs=len(jobs), stats=stats, merge=st, workers=args.workers, **extra)
    print("%s: %s | merge %s" % (exp, stats, st))
    return st


def cmd_e1(args):
    root = Path(args.root)
    mp = root / "e1" / "manifest.json"
    reruns = json.loads(mp.read_text()).get("reruns", {}) if mp.exists() else {}
    for s in args.rerun or []:                                   # the plan's re-run rule: that shape's repeats once more in all arms
        if s not in GATED:
            raise SystemExit("only a gated shape is re-run: %s" % list(GATED))
        reruns[s] = 1
    sel = lambda j: (not args.shapes or j["meta"]["shape"] in args.shapes) and (args.seeds is None or j["meta"]["seed"] in args.seeds)
    run_exp("e1", build_e1(reruns), args, sel, reruns=reruns)


def cmd_e1b(args):
    run_exp("e1b", build_e1b(), args)


def cmd_e2(args):
    pool = load_pool(args.root, args.partial)
    run_exp("e2", build_e2(pool), args, pool_hash=pool["hash"])


def cmd_e2b(args):
    pool = load_pool(args.root, args.partial)
    run_exp("e2b", build_e2b(pool), args, pool_hash=pool["hash"])


def read_natives(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit("--natives-from %s does not exist" % p)
    return json.loads(p.read_text())["F_V1_e2"]["native"]


def cmd_sweep(args):
    natives = read_natives(args.natives_from)
    run_exp("sweep", build_sweep(natives), args, natives=natives, natives_from=str(args.natives_from))


def cmd_e4(args):
    """Two timed batches (gt, then fk_vision) at --workers; each is run in full from nothing (never from stored rows)."""
    root, by_arm = Path(args.root), build_e4()
    jobs = sum(by_arm.values(), [])
    if args.dry_run:
        return run_exp("e4", jobs, args)
    for j in jobs:                                               # R0's plan cache: the same plans, no rebuild
        dst, src = VB.plan_path(root, j["core"]["s"]), VB.plan_path(VB.ROOT, j["core"]["s"])
        if not dst.exists() and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    bp = root / "e4" / "batches.json"
    bp.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(bp.read_text()) if bp.exists() else []
    for arm, js in by_arm.items():
        if any(b["arm"] == arm for b in done):
            continue
        if not args.limit:
            for j in js:
                (VB.rows_dir(root) / (j["id"] + ".json")).unlink(missing_ok=True)
        stats = VB.run_jobs(js, root, args.workers, args.timeout or TIMEOUT["e4"], args.retry_errors, args.limit)
        lost = sum(1 for j in js if not (VB.load_rec(root, j["id"]) or {}).get("row"))
        if not args.limit:
            done.append({"arm": arm, "workers": args.workers, "n": len(js), "wall_s": stats["wall_s"], "lost": lost, "ids": [j["id"] for j in js]})
            bp.write_text(json.dumps(done, indent=1))
            print("batch %s workers=%d: %d episodes in %.0f s = %.0f/h" % (arm, args.workers, len(js), stats["wall_s"], len(js) / stats["wall_s"] * 3600))
    st = VB.merge("e4", jobs, root, flat)
    write_manifest(root, "e4", jobs=len(jobs), merge=st, workers=args.workers)
    print("e4: merge %s" % st)


# --- reading results -----------------------------------------------------------------------------------------------------

def load_exp(root, exp):
    """([(episodes.jsonl entry, record)], merge.json) of an experiment, or (None, None) if it has not run."""
    p, m = Path(root) / exp / "episodes.jsonl", Path(root) / exp / "merge.json"
    if not p.exists() or not m.exists():
        return None, None
    ents = [json.loads(line) for line in p.read_text().splitlines()]
    return [(e, VB.load_rec(root, e["id"])) for e in ents], json.loads(m.read_text())


def complete(st, expected):
    return bool(st) and st["missing"] == 0 and st["lost"] == 0 and st["entries"] == expected


def by_step(rec):
    return (((rec or {}).get("row") or {}).get("vision") or {}).get("by_step") or {}


def step_v(rec, n):
    return by_step(rec).get(str(n)) or {}


def row_of(rec):
    return (rec or {}).get("row")


# --- E3 / E5: V5 accuracy, yield, the pool, drift ---------------------------------------------------------------------------

def p95_ci(x, tag):
    """(p95, lo, hi): the 95th percentile with a bootstrap 95 % CI (2,000 resamples, seeded from the A6 contract)."""
    x = np.asarray(x, float)
    if not len(x):
        return None
    rng = BB._rng("bootstrap", tag)
    b = np.percentile(x[rng.integers(len(x), size=(BOOT, len(x)))], 95, axis=1)
    return [float(np.percentile(x, 95)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def collect_v5(sources):
    """sources: [(entries, label)] -> (steps, looks). A step = one with a complete look schedule (n_looks): type from the number of
    looks (a single footprint gets a second look), yielded iff any look was accepted. A look = every scheduled look of such a step;
    accepted ones carry the relative error vector (rel dx, dy in the target frame mm, rel dyaw deg) and the estimator ("edge" = the
    edge fallback; anything else, the stud fit). A step has `yielded` (any accepted look) and `yielded_stud` (without the edge looks)."""
    steps, looks = [], []
    for ents, label in sources:
        for e, rec in ents or []:
            for n, v in by_step(rec).items():
                if "n_looks" not in v or not v.get("looks"):
                    continue
                typ = "single" if v["n_looks"] == 2 else "course"
                steps.append(dict(src=label, arm=e["arm"], shape=e["meta"]["shape"], step=int(n), type=typ, yielded=any(l.get("accepted") for l in v["looks"]),
                                  yielded_stud=any(l.get("accepted") and l.get("estimator") != "edge" for l in v["looks"])))
                for l in v["looks"]:
                    er = l.get("errors")
                    ok = bool(l.get("accepted")) and er is not None
                    looks.append(dict(steps[-1], accepted=ok, estimator=l.get("estimator"), vec=[er["rel_xy_mm"][0], er["rel_xy_mm"][1], er["rel_yaw_deg"]] if ok else None,
                                      radial=er["rel_radial_mm"] if ok else None))
    return steps, looks


def v5_summary(steps, looks, tag="v5", edge=True):
    """edge False: the edge-fallback looks do not count (not qualified): neither their errors nor their yield."""
    if not edge:
        looks = [l for l in looks if l["estimator"] != "edge"]
        steps = [dict(s, yielded=s["yielded_stud"]) for s in steps]

    def block(st, lk, t):
        acc = [l for l in lk if l["accepted"]]
        return {"steps": len(st), "yielded": sum(s["yielded"] for s in st), "yield": (sum(s["yielded"] for s in st) / len(st)) if st else None,
                "looks": len(lk), "accepted_looks": len(acc),
                "radial_mm_p95": p95_ci([l["radial"] for l in acc], tag + t + "radial"),
                "yaw_deg_p95": p95_ci([abs(l["vec"][2]) for l in acc], tag + t + "yaw"),
                "radial_mm_median": float(np.median([l["radial"] for l in acc])) if acc else None,
                "radial_mm_max": max((l["radial"] for l in acc), default=None), "yaw_deg_max": max((abs(l["vec"][2]) for l in acc), default=None)}
    out = {"all": block(steps, looks, "all")}
    for typ in ("course", "single"):
        out[typ] = block([s for s in steps if s["type"] == typ], [l for l in looks if l["type"] == typ], typ)
    for arm in ARMS:
        out["arm_" + arm] = {typ: block([s for s in steps if s["arm"] == arm and s["type"] == typ], [l for l in looks if l["arm"] == arm and l["type"] == typ],
                                        arm + typ) for typ in ("course", "single")}
    g = lambda lst: [x for x in lst if x["shape"] in GATED and x["src"] == "e1"]
    out["gated_e1_only"] = block(g(steps), g(looks), "gated")
    return out


def top_corners(q, btype):
    """(F, 4, 3) world corners of the top face of a brick's body box (the plane the studs stand on) from the recorded poses; j_tables.corners."""
    L, W = sorted((int(x) for x in btype.split("x")), reverse=True)
    return JT.corners(q, L, W)[:, [1, 3, 5, 7]]


def drift_of(d, name, order, v):
    """E5, one build: per step with a snap, the maximum displacement (3-D, mm; and xy only) of the target's supports' top-face corners
    from the capture frame to the snap frame, both read as ex.frame at the call: row i of the record is frame i + 1, the snap_frame
    row is the pre-weld state, the first welded row is snap_frame + 1. Capture frame = the estimate frame t_E of the aim (t_L else)."""
    ids, out = [str(b) for b in d["brick_ids"]], []
    for n, vs in v.items():
        n, snap, aim = int(n), vs.get("snap"), vs.get("aim") or {}
        cap = aim.get("t_E_frame", vs.get("t_L_frame"))
        if not snap or cap is None or n >= len(order):
            continue
        sup = P.supports(order[n], order[:n])
        rows = np.where((d["frame"] >= cap) & (d["frame"] <= snap["snap_frame"]))[0]
        if not sup or not len(rows) or d["frame"][rows[0]] != cap:
            continue
        for sid_, btype in ((b[0], b[1]) for b in order[:n] if b[0] in sup):
            c = top_corners(d["brick_q"][rows, ids.index(sid_)], btype)
            dd = c - c[0]
            out.append(dict(shape=name, step=n, support=sid_, rows=len(rows), cap=int(cap), snap=int(snap["snap_frame"]),
                            d3_mm=float(np.linalg.norm(dd, axis=-1).max() * 1e3), dxy_mm=float(np.linalg.norm(dd[..., :2], axis=-1).max() * 1e3)))
    return out


def e5_drift(root, ents):
    """E5 over E1's FK-arm builds of the gated shapes (their --record-all recordings)."""
    res = []
    for e, rec in ents or []:
        if e["arm"] == "gt" or e["meta"]["shape"] not in GATED or e["meta"].get("rr", 0) != latest_rr(ents, e["meta"]["shape"]):
            continue
        npz = Path(root) / "recordings" / (e["id"] + ".npz")
        if not row_of(rec) or not npz.exists():
            continue
        with np.load(npz) as z:
            d = {k: z[k] for k in ("frame", "brick_ids", "brick_q")}
        res += [dict(x, arm=e["arm"], seed=e["meta"]["seed"]) for x in drift_of(d, e["meta"]["shape"], P.sequence(bricks_of(e["meta"]["shape"])), by_step(rec))]
    return res


def latest_rr(ents, shape):
    return max((e["meta"].get("rr", 0) for e, _ in ents if e["meta"]["shape"] == shape), default=0)


def e1_expected(root):
    mp = Path(root) / "e1" / "manifest.json"
    return EXPECTED["e1"] + 9 * len(json.loads(mp.read_text()).get("reruns", {}) if mp.exists() else {})


def cmd_pool(args):
    root = Path(args.root)
    e1, s1 = load_exp(root, "e1")
    e1b, s1b = load_exp(root, "e1b")
    if e1 is None or e1b is None:
        raise SystemExit("prerequisite missing: pool needs e1 and e1b")
    full = complete(s1, e1_expected(root)) and complete(s1b, EXPECTED["e1b"])
    if not full and not args.partial:
        raise SystemExit("prerequisite incomplete: e1 / e1b have missing or lost jobs -- finish them (or --partial for a smoke pool)")
    cur = [(e, r) for e, r in e1 if e["meta"]["rr"] == latest_rr(e1, e["meta"]["shape"])]      # a re-run round supersedes the first
    steps, looks = collect_v5([(cur, "e1"), (e1b, "e1b")])
    drift = e5_drift(root, cur)
    if not drift:
        raise SystemExit("E5: no recorded FK-arm gated build with a snap -- the pool cannot be frozen without drift_max")
    pool = {"params_hash": V.PARAMS_HASH, "git_head": git_head(), "partial": not full, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_steps": len(steps), "n_looks": len(looks), "n_accepted": sum(l["accepted"] for l in looks),
            "course": sorted([round(x, 4) for x in l["vec"]] for l in looks if l["accepted"] and l["type"] == "course"),
            "single": sorted([round(x, 4) for x in l["vec"]] for l in looks if l["accepted"] and l["type"] == "single"),
            "drift_max_mm": round(max(x["d3_mm"] for x in drift), 4), "drift_xy_max_mm": round(max(x["dxy_mm"] for x in drift), 4),
            "drift_steps": len(drift), "drift_worst": sorted(drift, key=lambda x: -x["d3_mm"])[:5],
            "drift_median_mm": float(np.median([x["d3_mm"] for x in drift])), "v5": v5_summary(steps, looks)}
    pool["hash"] = pool_hash(pool)
    (root / "pool.json").write_text(json.dumps(pool, indent=1, default=float))
    write_manifest(root, "pool", pool_hash=pool["hash"], partial=pool["partial"])
    print("pool %s%s: course %d, single %d vectors (of %d looks, %d steps); drift_max %.4f mm (3-D), %.4f mm (xy) over %d support records"
          % (pool["hash"], " (PARTIAL)" if pool["partial"] else "", len(pool["course"]), len(pool["single"]), pool["n_looks"], pool["n_steps"],
             pool["drift_max_mm"], pool["drift_xy_max_mm"], len(drift)))


# --- E1: builds -----------------------------------------------------------------------------------------------------------

def build_stats(rec):
    """One build: built, total, failure, and per snap the screen / cell-gate outcomes."""
    row = row_of(rec)
    if row is None:
        return None
    snaps = [(int(n), v["snap"]) for n, v in by_step(rec).items() if v.get("snap")]
    return {"built": row.get("built"), "total": row.get("total"), "failure": row.get("failure"), "snaps": sorted(snaps, key=lambda x: x[0]),
            "ok": row.get("failure") is None and row.get("built") == row.get("total") and len(snaps) == row.get("total")}


def e1_latest(ents):
    """E1 entries of the latest round of each shape: {(shape, arm): [(seed, entry, record)]}."""
    out = defaultdict(list)
    for e, rec in ents or []:
        if e["meta"]["rr"] == latest_rr(ents, e["meta"]["shape"]):
            out[(e["meta"]["shape"], e["arm"])].append((e["meta"]["seed"], e, rec))
    return out


def t_v1a(ents):
    t = {}
    for (shape, arm), items in sorted(e1_latest(ents).items()):
        bs = [(seed, build_stats(rec)) for seed, e, rec in sorted(items, key=lambda x: x[0])]
        snaps = [s for _, b in bs if b for _, s in b["snaps"]]
        t["%s/%s" % (arm, shape)] = {
            "builds": len(bs), "built_all": sum(1 for _, b in bs if b and b["ok"]),
            "built": [[b["built"], b["total"]] if b else None for _, b in bs], "failures": [b["failure"] if b else "lost" for _, b in bs],
            "snaps": len(snaps), "screen_pass": sum(s["screen_pass"] for s in snaps), "cell_gate_ok": sum(s["gate_ok"] for s in snaps),
            "lateral_mm_mean": float(np.mean([s["lateral_mm"] for s in snaps])) if snaps else None,
            "lateral_mm_max": max((s["lateral_mm"] for s in snaps), default=None),
            "dz_mm_mean": float(np.mean([s["dz_mm"] for s in snaps])) if snaps else None,
            "dz_mm_range": [min(s["dz_mm"] for s in snaps), max(s["dz_mm"] for s in snaps)] if snaps else None,
            "screen_fail_steps": [[seed, n] for seed, b in bs if b for n, s in b["snaps"] if not s["screen_pass"]],
            "gated": shape in GATED}
    return t


def e1_native(ents, rerun_done):
    """Validity (a): {shape: (True | False | None, reason)}: the gt arm fails the screen or the cell gate (or does not build) at a
    (shape, step) in any repeat of the latest round -> the shape is flagged for one re-run; after the re-run, if it recurs: INCONCLUSIVE."""
    out, latest = {}, e1_latest(ents)
    for shape in GATED:
        items = latest.get((shape, "gt"), [])
        stats = [build_stats(rec) for _, _, rec in items]
        if len(items) < len(SEEDS) or any(b is None for b in stats):
            out[shape] = (None, "gt arm incomplete")
            continue
        bad = [(seed, n) for (seed, _, _), b in zip(items, stats) for n, s in b["snaps"] if not (s["screen_pass"] and s["gate_ok"])]
        bad += [(seed, "build:%s" % b["failure"]) for (seed, _, _), b in zip(items, stats) if not b["ok"]]
        rr = rerun_done.get(shape, 0)
        out[shape] = (True, "clean" + (" after re-run (the first round is superseded)" if rr else "")) if not bad else \
            (False, "native at (seed, step) %s: %s" % (bad, "recurs after the re-run: the shape is INCONCLUSIVE" if rr else "re-run the shape once (`e1 --rerun %s`)" % shape))
    return out


def e1_criterion(ents, arm):
    """Criteria 1 / 4: every gated shape built == n in all 3 repeats with every snap passing the screen, for `arm`."""
    latest, res = e1_latest(ents), []
    for shape in GATED:
        items = latest.get((shape, arm), [])
        stats = [build_stats(rec) for _, _, rec in items]
        if len(items) < len(SEEDS) or any(b is None for b in stats):
            return None, "%s %s incomplete" % (arm, shape)
        res.append(all(b["ok"] and all(s["screen_pass"] for _, s in b["snaps"]) for b in stats))
    return all(res), dict(zip(GATED, res))


# --- E2 / E2b: the matched injection ---------------------------------------------------------------------------------------------

def e2_trial(e, rec):
    """One E2 episode as a plain record: context, kind, scale, the injected vector [dx, dy mm, dyaw deg], the realised xy at the end
    of pre-insert, the screen."""
    m, v = e["meta"], step_v(rec, e["step"])
    snap, real = v.get("snap"), v.get("realised")
    i = m.get("inj")
    return dict(ctx=tuple(m["ctx"]), type=m["look_type"], kind=m["kind"], k=m["k"], scale=m.get("scale"), inj=i and [i["dx_mm"], i["dy_mm"], i["dyaw_deg"]],
                realised=real["xy_mm"] if real else None, screen=bool(snap and snap["screen_pass"]))


def manip_tol(inj_xy):
    return max(0.1, 0.1 * math.hypot(*inj_xy))


def e2_analysis(trials):
    """Native contexts (a control that does not pass the screen; both controls must), the screen pass rate per scale and look type over
    the included contexts, and the manipulation check (amended): the realised offset minus the mean realised offset of the context's
    controls (the descent swing, 1.0-1.2 mm even with no injection) equals the injected xy within max(0.1 mm, 10 %) in >= 95 % of
    injected trials (all contexts; a trial with no realised offset, or whose controls have none, fails)."""
    ctxs = defaultdict(lambda: {"control": [], "inject": []})
    for t in trials:
        ctxs[t["ctx"]][t["kind"]].append(t)
    native = sorted(c for c, d in ctxs.items() if not (len(d["control"]) == 2 and all(t["screen"] for t in d["control"])))
    inc = {c: d for c, d in ctxs.items() if c not in native}
    rate = lambda ts: {"pass": sum(t["screen"] for t in ts), "n": len(ts)}
    passes = {}
    for scale in (1, 2):
        ts = [t for d in inc.values() for t in d["inject"] if t["scale"] == scale]
        passes["scale%d" % scale] = {"all": rate(ts), "course": rate([t for t in ts if t["type"] == "course"]), "single": rate([t for t in ts if t["type"] == "single"])}
    ok = n = 0
    for c, d in ctxs.items():
        cr = [t["realised"] for t in d["control"] if t["realised"] is not None]
        mean = np.mean(cr, axis=0) if cr else None
        for t in d["inject"]:
            n += 1
            if t["realised"] is not None and mean is not None:
                ok += bool(np.hypot(*(np.asarray(t["realised"]) - mean - t["inj"][:2])) <= manip_tol(t["inj"][:2]))
    return {"contexts": len(ctxs), "native": [list(c) for c in native], "n_native": len(native), "included": len(inc), "pass": passes,
            "manipulation": {"ok": ok, "n": n, "frac": ok / n if n else None, "rule": "|realised - mean(controls) - injected| <= max(0.1 mm, 10 %), >= 95 %"}}


def margin_status(k, n, thr=0.95):
    """'pass' | 'marginal' | 'fail' for k passes of n trials: marginal = within 2 trials of the threshold (above, or below by <= 2 trials:
    a marginal result is GO, disclosed; no gate is relaxed)."""
    short = thr * n - k
    return "fail" if short > 2 else "marginal" if short >= -2 else "pass"


def e2b_agreement(e2b_steps, e2_s2):
    """e2b_steps: [(shape, k, step, observed, screen, after_failure, reason)]; e2_s2: {(shape, step, k): screen at scale 2}. Agreement
    over the scored (observed) steps: all of them (the validity check d and criterion 2), and the pre-failure ones only (before the
    first gate miss of the build: a build on a possibly defective structure is after_failure)."""
    def block(obs):
        agree = sum(1 for s in obs if e2_s2.get((s[0], s[2], s[1])) == s[4])
        return {"observed": len(obs), "pass": sum(s[4] for s in obs), "agree": agree, "frac": agree / len(obs) if obs else None}
    obs = [s for s in e2b_steps if s[3]]
    out = block(obs)
    out.update(unobserved=sum(1 for s in e2b_steps if not s[3]), no_e2_match=sum(1 for s in obs if (s[0], s[2], s[1]) not in e2_s2),
               after_failure_steps=sum(1 for s in obs if s[5]), pre_failure=block([s for s in obs if not s[5]]),
               after_failure=block([s for s in obs if s[5]]), unobserved_reasons=dict(Counter(s[6] for s in e2b_steps if not s[3])))
    return out


def e2b_steps_of(ents):
    """Every step of every E2b build (a --continue build): (shape, k, step, observed, screen, after_failure, reason). Observed = the step
    has a snap, scored on the screen. after_failure = after the build's first gate miss (n > that step), or at / after a failure that
    is not a gate miss (a joint break, ...: the step's own structure may already be defective). A step with no snap stays unobserved,
    with the build's failure as the reason."""
    out = []
    for e, rec in ents or []:
        row = row_of(rec)
        snaps = {n: step_v(rec, n).get("snap") for n in range(row["total"] if row else 0)}
        miss = min((n for n, sn in snaps.items() if sn and not sn["gate_ok"]), default=None)
        fail, at = row and row.get("failure"), row and row.get("failed_at_step")
        for n, snap in snaps.items():
            after = (miss is not None and n > miss) or (fail not in (None, "gate_miss") and at is not None and n >= at)
            reason = None if snap else "no_snap: %s at step %s" % (fail, at) if fail else "no_snap (no failure recorded)"
            out.append((e["meta"]["shape"], e["meta"]["k"], n, snap is not None, bool(snap and snap["screen_pass"]), bool(after), reason))
    return out


# --- the ACC axis sweep ------------------------------------------------------------------------------------------------------------

def sweep_analysis(trials):
    """trials: [dict(ctx, type, axis, sign, level, screen)] -> per signed axis ("long+", "long-", "short+", "short-"): the screen pass
    rate per level (all / course / single contexts), per context the largest passing level (None = none passes), the contexts whose
    passes are not monotone in the level (a level passes above one that fails) and the count of largest levels by look type."""
    ctx = defaultdict(dict)
    for t in trials:
        ctx[(t["axis"] + ("+" if t["sign"] > 0 else "-"), tuple(t["ctx"]))][t["level"]] = (t["screen"], t["type"])
    out, rate = {"episodes": len(trials), "contexts": len({c for _, c in ctx}), "levels_mm": list(SWEEP_LEVELS), "axes": {}}, lambda ts: {"pass": sum(ts), "n": len(ts)}
    for sa in ("long+", "long-", "short+", "short-"):
        cs = {c: d for (a, c), d in ctx.items() if a == sa}
        typ = lambda d: next(iter(d.values()))[1]
        by_level = {str(b): {t: rate([d[b][0] for d in cs.values() if b in d and (t == "all" or typ(d) == t)]) for t in ("all", "course", "single")}
                    for b in SWEEP_LEVELS}
        largest = {"%s:%d" % c: max((b for b in SWEEP_LEVELS if d.get(b, (False,))[0]), default=None) for c, d in cs.items()}
        counts = {t: dict(Counter(str(largest["%s:%d" % c]) for c, d in cs.items() if t == "all" or typ(d) == t)) for t in ("all", "course", "single")}
        mono = lambda d: not any(lo in d and hi in d and d[hi][0] and not d[lo][0] for lo, hi in zip(SWEEP_LEVELS, SWEEP_LEVELS[1:]))   # no pass above a fail
        out["axes"][sa] = {"pass_by_level": by_level, "largest_passing_counts": counts, "largest_passing_by_context": largest,
                           "non_monotone": sorted("%s:%d" % c for c, d in cs.items() if not mono(d))}
    return out


def sweep_trial(e, rec):
    m, v = e["meta"], step_v(rec, e["step"])
    snap = v.get("snap")
    return dict(ctx=tuple(m["ctx"]), type=m["look_type"], axis=m["axis"], sign=m["sign"], level=m["level"], screen=bool(snap and snap["screen_pass"]))


# --- the edge fallback's qualification (plan 2.4) ------------------------------------------------------------------------------------

def edge_qualification(looks, pool, trials, native):
    """The edge fallback is admitted after >= EDGE_MIN accepted single-footprint cases whose errors join the pool and that pass E2 at 2x on the
    single-footprint contexts. looks: collect_v5's (estimator "edge" = the fallback); trials: E2's (e2_trial); native: E2's native contexts
    ([[name, step], ...]). The E2 pass is over the included single-footprint scale-2 trials whose injected vector is scale2 of an edge
    vector. qualified: False (fewer than EDGE_MIN cases, or the pass rate < 95 % or no E2 trial used an edge vector), None (E2 unavailable)."""
    edge = [l for l in looks if l["accepted"] and l["estimator"] == "edge" and l["type"] == "single"]
    if not edge:
        return {"unavailable": "no accepted edge-fallback single-footprint look in the rows", "qualified": None}
    vecs = [[round(x, 4) for x in l["vec"]] for l in edge]
    out = {"n_cases": len(edge), "n_steps": len({(l["src"], l["arm"], l["shape"], l["step"]) for l in edge}),
           "rel_radial_mm_p95": p95_ci([l["radial"] for l in edge], "edgeradial"), "rel_yaw_deg_p95": p95_ci([abs(l["vec"][2]) for l in edge], "edgeyaw"),
           "in_pool": None if pool is None else sum(v in pool["single"] for v in vecs), "e2_scale2_single": None}
    if pool is not None and trials:
        s2 = {tuple(scale2(v, pool["drift_max_mm"])) for v in vecs}
        ts = [t for t in trials if t["kind"] == "inject" and t["scale"] == 2 and t["type"] == "single" and list(t["ctx"]) not in native
              and tuple(round(x, 4) for x in t["inj"]) in s2]
        out["e2_scale2_single"] = {"pass": sum(t["screen"] for t in ts), "n": len(ts)}
    e2 = out["e2_scale2_single"]
    out["qualified"] = False if len(edge) < EDGE_MIN else None if e2 is None else bool(e2["n"] and e2["pass"] >= 0.95 * e2["n"])
    return out


# --- G-V1 -------------------------------------------------------------------------------------------------------------------------

def gate_v1(validity, criteria, marginal=False):
    """Validity first, then criteria 1-4, then the outcome. validity / criteria: {name: (True | False | None, reason)}, None = unavailable.
    INCONCLUSIVE takes precedence over everything; any unavailable input -> 'unavailable' (never a pass); ARCH (1 fails), ACC (1 passes,
    2 or 3 fails), LOOP (1-3 pass, 4 fails), GO (1-4 pass; marginal is GO, disclosed)."""
    if any(v[0] is False for v in validity.values()):
        return "INCONCLUSIVE"
    if any(v[0] is None for v in validity.values()):
        return "unavailable"
    c = {k: v[0] for k, v in criteria.items()}
    if c[1] is False:
        return "ARCH"
    if c[1] is None:
        return "unavailable"
    if c[2] is False or c[3] is False:
        return "ACC"
    if c[2] is None or c[3] is None:
        return "unavailable"
    if c[4] is False:
        return "LOOP"
    if c[4] is None:
        return "unavailable"
    return "GO (marginal, disclosed)" if marginal else "GO"


def tri(ok, reason):
    return (ok, reason)


def table_e4(root):
    bp = Path(root) / "e4" / "batches.json"
    ents, st = load_exp(root, "e4")
    if not bp.exists() or ents is None:
        return {"unavailable": "e4 has not run"}
    out = {"complete": complete(st, EXPECTED["e4"]), "arms": {}}
    for b in json.loads(bp.read_text()):
        es = [(e, r) for e, r in ents if e["arm"] == b["arm"] and r and r["row"]]
        med = lambda xs: float(np.median(xs)) if xs else None
        vt = lambda key: [sum(v.get(key, 0.0) for v in by_step(r).values()) for _, r in es]
        out["arms"][b["arm"]] = {"workers": b["workers"], "n": b["n"], "lost": b["lost"], "wall_s": b["wall_s"], "episodes_per_h": b["n"] / b["wall_s"] * 3600,
                                 "episode_wall_s_median": med([r["wall_s"] for _, r in es]), "startup_s_median": med([e["startup_s"] for e, _ in es if e.get("startup_s")]),
                                 "rtf_median": med([e["rtf"] for e, _ in es if e.get("rtf")]), "success": sum(bool(e["success"]) for e, _ in es),
                                 "render_ms_median": med(vt("render_ms")), "vision_ms_median": med(vt("vision_ms")), "mask_ms_median": med(vt("mask_ms")),
                                 "looks_sim_s_median": med([sum(v.get("look_sim_s") or 0 for v in by_step(r).values()) for _, r in es])}
    a = out["arms"]
    if "gt" in a and "fk_vision" in a:
        r = a["fk_vision"]["episodes_per_h"] / a["gt"]["episodes_per_h"]
        out["ratio_vision_over_gt"] = r
        out["cost_ok"] = bool(r >= 0.8 and a["fk_vision"]["episodes_per_h"] >= 150) if out["complete"] else None
        out["rule"] = "fk_vision >= 0.8 x gt episodes/h and >= 150/h (U-P3 only; provisional)"
    return out


def compute_tables(root):
    root = Path(root)
    mp = root / "e1" / "manifest.json"
    e1, s1 = load_exp(root, "e1")
    e1b, s1b = load_exp(root, "e1b")
    e2, s2 = load_exp(root, "e2")
    e2b, s2b = load_exp(root, "e2b")
    sw, ssw = load_exp(root, "sweep")
    reruns = json.loads(mp.read_text()).get("reruns", {}) if mp.exists() else {}
    full = {"e1": complete(s1, e1_expected(root)), "e1b": complete(s1b, EXPECTED["e1b"]), "e2": complete(s2, EXPECTED["e2"]),
            "e2b": complete(s2b, EXPECTED["e2b"])}
    pool = None
    try:
        pool = load_pool(root, partial=True)
    except SystemExit:
        pass
    # params hash of the rows against the current vision parameters
    stale = sorted({e.get("params_hash") for ents in (e1, e1b, e2, e2b, sw) for e, _ in ents or [] if e.get("params_hash") not in (None, V.PARAMS_HASH)})
    T = {"params_hash": V.PARAMS_HASH, "git_head": git_head(), "complete": full, "stale_params_hashes": stale,
         "pool_hash": pool and pool["hash"], "pool_partial": pool and pool.get("partial")}
    # T-V1a
    T["T_V1a_builds"] = t_v1a(e1) if e1 else {"unavailable": "e1 has not run"}
    # T-V1b (the summary is made below, once the edge fallback's qualification is known)
    have_v5 = e1 is not None or e1b is not None
    steps, looks = collect_v5([([x for x in e1 or [] if x[0]["meta"]["rr"] == latest_rr(e1, x[0]["meta"]["shape"])], "e1"), (e1b, "e1b")]) if have_v5 else ([], [])
    T["T_V1b_pool"] = None if pool is None else {k: pool[k] for k in ("hash", "partial", "n_steps", "n_looks", "n_accepted", "drift_max_mm", "drift_xy_max_mm",
                                                                       "drift_steps", "drift_median_mm", "drift_worst")} | {"course": len(pool["course"]), "single": len(pool["single"])}
    # F-V1
    trials = [e2_trial(e, r) for e, r in e2 or [] if r and r["row"]]
    A2 = e2_analysis(trials) if trials else None
    e2_s2 = {(t["ctx"][0], t["ctx"][1], t["k"]): t["screen"] for t in trials if t["scale"] == 2}
    B2 = e2b_agreement(e2b_steps_of(e2b), e2_s2) if e2b else None
    if B2:
        B2["n_native_steps_included"] = sum(1 for s in e2b_steps_of(e2b) if s[3] and [s[0], s[2]] in (A2 or {}).get("native", []))
    T["F_V1_e2"] = A2 or {"unavailable": "e2 has not run"}
    T["T_V1b_edge_fallback"] = edge_qualification(looks, pool, trials, A2["native"]) if have_v5 and A2 else \
        edge_qualification(looks, pool, [], []) if have_v5 else {"unavailable": "e1 and e1b have not run", "qualified": None}
    edge_ok = T["T_V1b_edge_fallback"]["qualified"] is True        # edge-accepted steps count in the yield (and the accuracy) only if qualified
    T["T_V1b_v5"] = dict(v5_summary(steps, looks, edge=edge_ok), edge_counted=edge_ok) if have_v5 else {"unavailable": "e1 and e1b have not run"}
    nsw = json.loads((root / "sweep" / "manifest.json").read_text())["jobs"] if (root / "sweep" / "manifest.json").exists() else None
    T["F_V1_sweep"] = dict(sweep_analysis([sweep_trial(e, r) for e, r in sw if r and r["row"]]), complete=complete(ssw, nsw)) if sw and nsw else \
        {"unavailable": "the sweep has not run"}
    T["F_V1_e2b"] = B2 or {"unavailable": "e2b has not run"}
    T["T_V1c_cost"] = table_e4(root)
    # G-V1
    validity, crit, marginal = {}, {}, False
    nat = e1_native(e1, reruns) if e1 else {s: (None, "e1 has not run") for s in GATED}
    if e1 is not None and not full["e1"]:
        nat = {s: (None if v[0] is True else v[0], "e1 incomplete") for s, v in nat.items()}
    for s, v in nat.items():
        validity["a_E1_native_" + s] = v
    ok2 = full["e2"] and A2 is not None and pool is not None
    validity["b_native_contexts_le_3"] = tri(A2["n_native"] <= 3, "%d of %d native" % (A2["n_native"], A2["contexts"])) if ok2 else tri(None, "e2 unavailable")
    validity["c_manipulation_check"] = tri(A2["manipulation"]["frac"] >= 0.95, "%s" % A2["manipulation"]) if ok2 and A2["manipulation"]["frac"] is not None else tri(None, "e2 unavailable")
    ok2b = full["e2b"] and B2 is not None and B2["frac"] is not None and B2["no_e2_match"] == 0 and ok2
    validity["d_e2_e2b_agreement_ge_90"] = tri(B2["frac"] >= 0.9, "%s of %s scored steps agree (pre-failure only: %s of %s)" % (B2["agree"], B2["observed"], B2["pre_failure"]["agree"], B2["pre_failure"]["observed"])) if ok2b else tri(None, "e2b unavailable")
    c1, why1 = e1_criterion(e1, "fk_oracle") if e1 and full["e1"] else (None, "e1 unavailable")
    c4, why4 = e1_criterion(e1, "fk_vision") if e1 and full["e1"] else (None, "e1 unavailable")
    crit[1], crit[4] = tri(c1, why1), tri(c4, why4)
    m2 = []
    if ok2 and ok2b:
        for name, p in (("e2", A2["pass"]["scale2"]["all"]), ("e2b", {"pass": B2["pass"], "n": B2["observed"]})):
            m2.append((name, margin_status(p["pass"], p["n"]), p["pass"], p["n"]))
        crit[2] = tri(all(s != "fail" for _, s, _, _ in m2), "%s" % m2)
        marginal = any(s == "marginal" for _, s, _, _ in m2)
    else:
        crit[2] = tri(None, "e2 / e2b unavailable")
    v5 = T["T_V1b_v5"]
    if "unavailable" not in v5 and full["e1"] and full["e1b"] and v5["all"]["yield"] is not None and v5["all"]["yaw_deg_p95"]:
        y, yaw = v5["all"]["yield"], v5["all"]["yaw_deg_p95"][0]
        crit[3] = tri(y >= 0.95 and yaw <= 1.0, "yield %.3f, yaw p95 %.3f deg" % (y, yaw))
    else:
        crit[3] = tri(None, "e1 / e1b unavailable")
    G = {"validity": {k: list(v) for k, v in validity.items()}, "criteria": {k: list(v) for k, v in crit.items()}, "marginal_e2_e2b": m2,
         "cost_u_p3_only": T["T_V1c_cost"].get("cost_ok") if "unavailable" not in T["T_V1c_cost"] else None,
         "outcome": gate_v1(validity, crit, marginal)}
    if stale:
        G["outcome"] += " [STALE: rows from other vision parameters %s]" % stale
    T["G_V1"] = G
    (root / "tables.json").write_text(json.dumps(T, indent=1, default=float))
    return T


def print_tables(T, root):
    for title, key in (("T-V1a: builds (E1)", "T_V1a_builds"), ("T-V1b: V5 accuracy and yield (E1 + E1b)", "T_V1b_v5"), ("T-V1b: the frozen pool, drift", "T_V1b_pool"),
                       ("T-V1b: edge fallback qualification", "T_V1b_edge_fallback"),
                       ("F-V1: matched injection (E2)", "F_V1_e2"), ("F-V1: transfer (E2b, --continue)", "F_V1_e2b"),
                       ("F-V1: ACC axis sweep", "F_V1_sweep"), ("T-V1c: cost (E4)", "T_V1c_cost")):
        v = T[key]
        if key == "F_V1_sweep" and "unavailable" not in v:           # per-context detail stays in tables.json
            v = dict({k: v[k] for k in ("complete", "episodes", "contexts")},
                     **{sa + " " + k: d[k] for sa, d in v["axes"].items() for k in ("pass_by_level", "largest_passing_counts", "non_monotone")})
        print(VB.md(title, v if v is not None else {"unavailable": "no pool.json"}) + "\n")
    G = T["G_V1"]
    print("### G-V1 (complete: %s)\n" % T["complete"])
    for part in ("validity", "criteria"):
        for k, (ok, why) in G[part].items():
            print("  %-9s %-28s %-12s %s" % (part, k, "unavailable" if ok is None else "pass" if ok else "FAIL", str(why)[:150]))
    print("\n  OUTCOME: %s   (cost, U-P3 only: %s)\n-> %s" % (G["outcome"], G["cost_u_p3_only"], Path(root) / "tables.json"))


def cmd_tables(args):
    print_tables(compute_tables(args.root), args.root)


# --- CLI ---------------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["e1", "e1b", "pool", "e2", "e2b", "sweep", "e4", "tables"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, help="wall seconds per job (default: %s)" % TIMEOUT)
    ap.add_argument("--dry-run", action="store_true", help="print job counts, run nothing")
    ap.add_argument("--limit", type=int, help="run only the first N jobs that still lack a row (smoke)")
    ap.add_argument("--retry-errors", action="store_true", help="also re-run jobs whose stored record is a job error")
    ap.add_argument("--partial", action="store_true", help="pool / e2 / e2b: accept incomplete E1 / E1b rows (a smoke pool)")
    ap.add_argument("--natives-from", type=Path, default=NATIVES, help="sweep: the tables.json whose F_V1_e2.native contexts are left out (default: %(default)s)")
    ap.add_argument("--shapes", nargs="+", help="e1: run only these shapes (the merge still covers all)")
    ap.add_argument("--seeds", nargs="+", type=int, help="e1: run only these seeds")
    ap.add_argument("--rerun", nargs="+", metavar="SHAPE", help="e1: the plan's re-run rule: that gated shape's repeats once more in all arms")
    args = ap.parse_args()
    {"e1": cmd_e1, "e1b": cmd_e1b, "pool": cmd_pool, "e2": cmd_e2, "e2b": cmd_e2b, "sweep": cmd_sweep, "e4": cmd_e4, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
