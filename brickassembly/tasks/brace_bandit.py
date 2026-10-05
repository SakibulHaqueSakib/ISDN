"""RL bracing task library, plan_v4_rl_brace S0.1 -- pure Python, no simulator.

    python -m tasks.brace_bandit --self-check     # via: bash scripts/run.sh tasks/brace_bandit.py --self-check

A context is (structure, step n). This module makes the structures (corbel_family,
canonical_key, split), the contexts and their LP screen (contexts), the action set
(actions), the observation (features), the hidden load draw (hidden) and the arm
pickers of section 2.8 that need no simulation. It imports planner, blueprint,
bracing and stability only: no newton, no warp, and it never reads results/.

All planner geometry is taken in the cell's frame (the build centred between the
arms, base plate top at Z0, as dual_arm_sim.make_plan sets it) inside cell_frame();
the public functions enter it themselves, so a caller's P.VOXEL_ORIGIN is untouched.
An action is None (no brace) or (y, z, lean_deg): the pad centre in that frame and
the lean about the grip axis, the same number as the degrees of brace_tilt_rotvec[0].
"""

import argparse
import contextlib
import functools
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import blueprint as BP
import bracing
import planner as P
import stability as ST

HERE = Path(__file__).resolve().parents[1]
MASTER_SEED = 20260930
Z0 = 0.0032                       # dual_arm_sim.Z0: base plate top
TRAVEL = 0.08                     # dual_arm_sim.TRAVEL: clearance above the structure while moving
ARM_B_XY = (0.45, 0.0)            # dual_arm_sim.ARM_B base: the feeder ring is measured from it
WRIST_LEN = 0.1034                # m, TCP to the Franka hand flange along the tool axis [A: FR3 datasheet]
LEANS = (-45, 0, 45)              # degrees about the grip axis
SCALES = (0.5, 1.0, 1.3)          # x design press, the LP screen
CRITICAL_U = 0.8                  # pool "critical": LP u0(1.3 x design) >= this
BRACE_COST = 0.05                 # reward = success - BRACE_COST * brace
U_CAP = 10.0                      # LP utilisations are clipped here (infeasible = inf) before they are features
EXE = {"E0": {"name": "E0", "grip_N": 14.3, "mu": 0.7},     # committed hold (dual_arm_sim.BRACE_GRIP_N)
       "E1": {"name": "E1", "grip_N": 28.6, "mu": 0.7}}     # stiff hold; mu 0.7 is the plan's probe value [A]
LEAN_DEG = round(math.degrees(bracing.TILT))                 # 45: lean_for's magnitude


@contextlib.contextmanager
def cell_frame(bricks):
    """P.VOXEL_ORIGIN as dual_arm_sim.make_plan sets it for this structure, restored on exit."""
    NI = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    NJ = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    old = P.VOXEL_ORIGIN
    P.VOXEL_ORIGIN = (-NI * P.PITCH / 2, -NJ * P.PITCH / 2, Z0)
    try:
        yield
    finally:
        P.VOXEL_ORIGIN = old


def feeder_slots(n):
    """The first n feeder slots: a copy of dual_arm_sim.feeder_grid (that module needs newton)."""
    base = np.array(ARM_B_XY)
    dist = lambda g: np.linalg.norm(np.array(g) - base)
    near = lambda g: 0.40 <= dist(g) <= 0.65
    grid = sorted((g for g in ((x, -y) for x in (0.10, 0.17, 0.24, 0.31)
                               for y in (0.22, 0.29, 0.36, 0.43, 0.50)) if near(g)), key=dist)
    grid += sorted((g for g in ((x, -y) for x in (0.03, 0.10, 0.17, 0.24, 0.31, 0.38)
                                for y in (0.22, 0.29, 0.36, 0.43, 0.50, 0.57))
                    if near(g) and g not in grid), key=dist)
    return grid[:n]


# --- structures ---------------------------------------------------------------

def _voxels(cells):
    NJ = max(j for _, j, _ in cells) + 1
    V = np.zeros((max(i for i, _, _ in cells) + 1, NJ, max(k for _, _, k in cells) + 1), bool)
    for c in cells:
        V[c] = True
    return V


def _valid(bricks):
    return (5 <= len(bricks) <= 24 and BP.components(bricks) == 1
            and all(P.footprint(b[1], b[5])[0] == 2 for b in bricks))   # 2 studs deep in x


def _corbel_voxels(rng):
    pw, h, nc = 2, int(rng.integers(3, 7)), int(rng.integers(1, 4))
    ext = [int(e) for e in rng.integers(1, 3, nc)]
    cells = {(i, j, k) for i in range(2) for k in range(h) for j in range(pw)}          # the pier
    tip = pw
    for c in range(nc):                                                                 # corbel layers
        tip += ext[c]
        cells |= {(i, j, h + c) for i in range(2) for j in range(tip)}
    top = h + nc                                              # first free layer above the corbels
    if rng.random() < 0.5:                                    # second pier, bridged by a crown
        gap, pw2, ov = int(rng.integers(0, 3)), int(rng.integers(2, 4)), int(rng.integers(1, 3))
        cells |= {(i, j, k) for i in range(2) for k in range(top) for j in range(tip + gap, tip + gap + pw2)}
        for k in range(top, top + int(rng.integers(1, 3))):                             # crown, 1-2 layers
            cells |= {(i, j, k) for i in range(2) for j in range(max(0, tip - ov), tip + gap + pw2)}
    elif rng.random() < 0.5:                                  # crown alone, on the tip
        cells |= {(i, j, top) for i in range(2) for j in range(max(0, tip - int(rng.integers(2, 5))), tip)}
    return cells


def corbel_family(seed):
    """Bricks (planner tuples, build order) of the seed's structure: a pier of height 3-6 (2 studs wide, as S3's),
    1-3 corbel layers each reaching 1-2 studs further, optionally a second pier and a crown; depth 2 in x;
    5-24 bricks in one component, tiled by blueprint.tile. Deterministic in seed (redraws until valid)."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for _ in range(200):
        try:
            bricks = BP.tile(_voxels(_corbel_voxels(rng)))
        except ValueError:
            continue
        if _valid(bricks):
            return bricks
    raise RuntimeError("corbel_family(%d): no valid structure in 200 draws" % seed)


def _cells(bricks):
    return {(i, j, P.cells(b)[1]) for b in bricks for i, j in P.cells(b)[0]}


def _mirror(cells):
    """y-mirror of a voxel set, both normalised to start at j = 0."""
    j0 = min(j for _, j, _ in cells)
    jmax = max(j for _, j, _ in cells) - j0
    return ({(i, j - j0, k) for i, j, k in cells}, {(i, jmax - (j - j0), k) for i, j, k in cells})


def canonical_key(bricks):
    """The smaller of the brick-cell set and its y-mirror (sorted tuples of (i, j, k)): a structure and
    its mirror share one key."""
    return min(tuple(sorted(c)) for c in _mirror(_cells(bricks)))


def _rng(*parts):
    """PCG64 seeded from the SHA-256 of the canonical JSON of (MASTER_SEED, *parts)."""
    blob = json.dumps([MASTER_SEED, *parts], sort_keys=True, separators=(",", ":"))
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(blob.encode()).digest(), "big")))


def _oriented(key):
    """The one orientation a unique structure is used in, drawn from its key; None if it does not tile."""
    cells = set(key)
    if _rng("orientation", key).integers(2):
        cells = _mirror(cells)[1]
    try:
        bricks = BP.tile(_voxels(cells))
    except ValueError:
        return None
    return bricks if _valid(bricks) else None


_UNIQUE = []                      # [{"seed", "key", "bricks"}] in seed order, grown on demand
_SEEN = set()
_NEXT_SEED = [0]


def unique_structures(n):
    """The first n unique structures (dedup by canonical_key, generated on demand in seed order)."""
    while len(_UNIQUE) < n:
        seed = _NEXT_SEED[0]
        _NEXT_SEED[0] += 1
        key = canonical_key(corbel_family(seed))
        if key in _SEEN:
            continue
        _SEEN.add(key)
        bricks = _oriented(key)
        if bricks is not None:
            _UNIQUE.append({"seed": seed, "key": key, "bricks": bricks})
    return _UNIQUE[:n]


SPLITS = {"train": (0, 72), "val": (72, 90)}


def split(name, n=None):
    """Entries of one split by order of unique structures: train = the first 72, val = the next 18,
    test = the following n (n required; generated on demand)."""
    if name == "test":
        if n is None:
            raise ValueError("split('test') needs n")
        return unique_structures(90 + n)[90:]
    a, b = SPLITS[name]
    return unique_structures(b)[a:b]


# --- contexts and actions -----------------------------------------------------

@functools.lru_cache(maxsize=None)
def _util(bricks, n, scale, grasp=None, mu=None):
    """LP result (ST.insertion_utilisation) for step n at scale x design press; grasp = (y, z, grip_N)
    or None. Call inside cell_frame."""
    brick, placed = bricks[n], list(bricks[:n])
    Q = scale * sum(P.supports(brick, placed).values()) * ST.F_INSERT_PER_STUD
    br = [] if grasp is None else [bracing._brace_obj(placed, grasp[0], grasp[1], grasp[2], mu)]
    return ST.insertion_utilisation(brick, placed, braces=br, press=Q)


@functools.lru_cache(maxsize=None)
def _cands(bricks, n):
    """bracing.candidates with the 28 mm rule off (pad-geometry feasibility only)."""
    return tuple(bracing.candidates(bricks[n], list(bricks[:n]), clearance=lambda *a: True))


def contexts(bricks):
    """Steps 1..N-1 as dicts: bricks, step, brick, placed, key, design_press (N), u0 {0.5, 1.0, 1.3 x
    design press: unbraced LP utilisation}, pool ('critical' iff u0[1.3] >= 0.8, else 'easy')."""
    bricks = tuple(bricks)
    key = canonical_key(bricks)
    out = []
    with cell_frame(bricks):
        for n in range(1, len(bricks)):
            u0 = {s: _util(bricks, n, s).s for s in SCALES}
            out.append({"bricks": bricks, "step": n, "brick": bricks[n], "placed": bricks[:n], "key": key,
                        "design_press": sum(P.supports(bricks[n], bricks[:n]).values()) * ST.F_INSERT_PER_STUD,
                        "u0": u0, "pool": "critical" if u0[1.3] >= CRITICAL_U else "easy"})
    return out


def actions(ctx):
    """A(c) = [None] + [(y, z, lean_deg)] over bracing.candidates (28 mm rule off) x LEANS."""
    with cell_frame(ctx["bricks"]):
        return [None] + [(y, z, l) for (y, z) in _cands(ctx["bricks"], ctx["step"]) for l in LEANS]


# --- features -----------------------------------------------------------------

FEATURES = [
    # context
    ("brick_studs", "studs of the brick being placed"),
    ("brick_len", "its longer side, studs"),
    ("n_support", "number of supporting bricks"),
    ("support_studs", "studs mated with the supports"),
    ("step_frac", "step index / number of bricks"),
    ("layer", "layer k of the brick"),
    ("struct_h", "layers in the placed structure"),
    ("struct_w", "studs the placed structure spans in y"),
    ("place_y", "placement y in the cell frame, m"),
    ("place_z", "placement z (brick bottom), m"),
    ("u0_05", "LP unbraced utilisation at 0.5 x design press"),
    ("u0_10", "... at 1.0 x design press"),
    ("u0_13", "... at 1.3 x design press"),
    ("wj_dy", "y of the weakest joint (LP at 1.0) minus placement y, m"),
    ("design_press", "n_studs x 8.9 N"),
    # candidate
    ("is_none", "1 for the no-brace action"),
    ("dy", "grip y minus placement y, m"),
    ("abs_dy", "|dy|"),
    ("dz", "grip z minus placement z, m"),
    ("height", "grip z above the base plate top, m"),
    ("lean", "lean about the grip axis, degrees"),
    ("lean_rel", "lean / lean_for's lean: +1 = away from the placer (scripted), -1 toward it"),
    ("lean_feeder", "-sin(lean) x (+1 if the feeder slot is at larger y than the grip, else -1): > 0 = hand housing leans toward the feeder"),
    ("n_gripped", "bricks under the pad window"),
    ("max_share", "largest pad-contact share among them"),
    ("grips_wj", "grips a brick of the weakest joint"),
    ("grips_support", "grips a support of the new brick"),
    ("grips_base", "grips a layer-0 brick"),
    ("u_br_10", "LP braced utilisation at 1.0 x design press (optimistic)"),
    ("u_br_13", "... at 1.3 x design press"),
    ("du_13", "u_br_13 - u0_13 (LP gain, negative = helps)"),
    ("brace_util", "LP brace_util at 1.0: |force| / its friction share, worst gripped brick"),
    ("reaction_N", "LP expected reaction force norm at 1.0, N"),
    ("wrist_dz", "wrist point height minus travel height, m (wrist = TCP - 0.1034 m along the tool axis)"),
    ("wrist_seg", "wrist distance to B's transport segment (feeder slot -> target at travel height), m"),
    ("wrist_col", "wrist distance to B's descent column (target xy, travel height -> target), m"),
    ("rule28", "1 iff |dy| >= 28 mm, the committed hand-clearance rule"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
NF = len(FEATURES)


def _seg_dist(p, a, b):
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _cap(u):
    return min(float(u), U_CAP)


def _weakest_bricks(bricks, n):
    w = _util(bricks, n, 1.0).weakest
    return {b for b in (w or ()) if b is not None}


@functools.lru_cache(maxsize=None)
def _context_features(bricks, n):
    brick, placed = bricks[n], list(bricks[:n])
    sup = P.supports(brick, placed)
    nx, ny = P.footprint(brick[1], brick[5])
    j0 = min(b[3] for b in placed)
    j1 = max(b[3] + P.footprint(b[1], b[5])[1] for b in placed)
    r = _util(bricks, n, 1.0)
    wj_dy = 0.0
    if r.weakest:
        c = next(c for c in ST.connections(placed) if (c.upper, c.lower) == r.weakest)
        wj_dy = float(c.centre[1] - P.brick_pose(brick)[1])
    px, py, pz = P.brick_pose(brick)
    return [float(nx * ny), float(max(nx, ny)), float(len(sup)), float(sum(sup.values())), n / len(bricks),
            float(brick[4]), float(max(b[4] for b in placed) + 1), float(j1 - j0), py, pz,
            _cap(_util(bricks, n, 0.5).s), _cap(r.s), _cap(_util(bricks, n, 1.3).s), wj_dy,
            sum(sup.values()) * ST.F_INSERT_PER_STUD]


@functools.lru_cache(maxsize=None)
def _candidate_features(bricks, n, action, grip_N, mu):
    brick, placed = bricks[n], list(bricks[:n])
    px, py, pz = P.brick_pose(brick)
    if action is None:
        return [1.0] + [0.0] * (NF - FEATURE_NAMES.index("is_none") - 1)
    y, z, lean = action
    area = bracing.pad_contact(placed, y, z)
    tot = sum(area.values())
    sup = P.supports(brick, placed)
    by_id = {p[0]: p for p in placed}
    g = grip_N
    r = _util(bricks, n, 1.0, (y, z, g), mu)
    r13 = _util(bricks, n, 1.3, (y, z, g), mu)
    ok = r.feasible and r.brace_wrench
    lean_for = math.degrees(bracing.lean_for(brick, y)[0])
    slot = feeder_slots(len(bricks))[n]
    sgn_feeder = 1.0 if slot[1] > y else -1.0                 # which way along y the feeder is from the grip
    th = math.radians(lean)
    wrist = np.array([P.VOXEL_ORIGIN[0] + P.PITCH, y - WRIST_LEN * math.sin(th), z + WRIST_LEN * math.cos(th)])
    z_travel = max(P.brick_pose(b)[2] for b in bricks) + P.BRICK_H + TRAVEL
    up = lambda x, yy: np.array([x, yy, z_travel])
    tgt = np.array([px, py, pz])
    wj = _weakest_bricks(bricks, n)
    return [0.0, y - py, abs(y - py), z - pz, z - P.VOXEL_ORIGIN[2], float(lean),
            lean / lean_for if lean_for else 0.0, -math.sin(th) * sgn_feeder,
            float(len(area)), max(area.values()) / tot if tot else 0.0,
            float(bool(wj & set(area))), float(bool(set(sup) & set(area))),
            float(any(by_id[i][4] == 0 for i in area)),
            _cap(r.s), _cap(r13.s), _cap(r13.s) - _cap(_util(bricks, n, 1.3).s),
            float(r.brace_util[0]) if ok else 0.0,
            float(np.linalg.norm(r.brace_wrench[0][:3])) if ok else 0.0,
            float(wrist[2] - z_travel),
            _seg_dist(wrist, up(*slot), up(px, py)), _seg_dist(wrist, up(px, py), tgt),
            float(abs(y - py) >= bracing.HAND_CLEARANCE_Y)]


def features(bricks, step, action, exe):
    """The NF observation numbers (FEATURES: name and one-line meaning each) for `action` (None or
    (y, z, lean_deg)) at `step` of `bricks`. A pure function of the plan and cell constants; LP
    features use exe['grip_N'] and exe['mu']. Reads no hidden draw, no simulator state, no file."""
    bricks = tuple(bricks)
    with cell_frame(bricks):
        return np.array(_context_features(bricks, step)
                        + _candidate_features(bricks, step, action, exe["grip_N"], exe["mu"]))


# --- the hidden load draw -----------------------------------------------------

def hidden(stage, structure_key, step, r, R, band=(0.5, 1.3), stratified=True):
    """Draw for replicate r of R at (stage, structure, step): {'lam', 'lat_mag' (N), 'lat_angle' (rad),
    'lateral' (Fx, Fy in N)}. Stratified: lam from the r-th of R equal-width strata of `band` and the
    lateral angle from the r-th of R equal sectors, each in a context-specific random order (the
    angle with a random phase and an independent order); magnitude U(0, 3 N). Random access in r, no
    dependence on call order; shared by every arm of a stage. Unstratified (training rounds): all three
    drawn independently per (context, r)."""
    key = [list(c) for c in structure_key]
    if stratified:
        ctx = _rng(stage, key, step)                       # per-context order and phase, R values each
        order_lam, order_ang, phase = ctx.permutation(R), ctx.permutation(R), ctx.uniform(0, 2 * math.pi)
        g = _rng(stage, key, step, "rep", r)
        lam = band[0] + (order_lam[r] + g.random()) * (band[1] - band[0]) / R
        mag = 3.0 * g.random()
        ang = (phase + (order_ang[r] + g.random()) * 2 * math.pi / R) % (2 * math.pi)
    else:
        g = _rng(stage, key, step, "free", r)
        lam, mag, ang = g.uniform(*band), 3.0 * g.random(), 2 * math.pi * g.random()
    return {"lam": float(lam), "lat_mag": float(mag), "lat_angle": float(ang),
            "lateral": (float(mag * math.cos(ang)), float(mag * math.sin(ang)))}


# --- arm pickers --------------------------------------------------------------

def pick_none(ctx):
    return None


def _lp_key(bricks, n, y, z, exe):
    """bracing.assign's weakest_joint ordering of a grasp (lower is better)."""
    g, mu = exe["grip_N"], exe["mu"]
    r = _util(bricks, n, 1.0, (y, z, g), mu)
    ra = _util(bricks, n, bracing.FF_PRESS_MARGIN, (y, z, g), mu) if r.s < 0.5 else r
    return (round(r.s, 2), round(ra.s, 3), round(r.brace_util[0], 3) if r.brace_wrench else 0.0)


def _scripted_lean(ctx, y):
    return round(math.degrees(bracing.lean_for(ctx["brick"], y)[0]))


def pick_lp_shared(ctx, exe, sf=ST.SAFETY_FACTOR):
    """LP-shared: brace iff LP u0(nominal) x sf >= 1; then the (y, z) of A(c) minimising the LP braced u
    (bracing.assign's key order), leaned by bracing.lean_for. Returns an action."""
    if ctx["u0"][1.0] * sf < 1.0:
        return None
    with cell_frame(ctx["bricks"]):
        yz = _cands(ctx["bricks"], ctx["step"])
        if not yz:
            return None
        best = min(yz, key=lambda c: _lp_key(ctx["bricks"], ctx["step"], c[0], c[1], exe))   # first wins ties
        return (best[0], best[1], _scripted_lean(ctx, best[0]))


def pick_lp_committed(ctx, exe, sf=ST.SAFETY_FACTOR):
    """LP-committed: bracing.assign(strategy='weakest_joint') as committed (28 mm mask, sf, scripted lean).
    Returns the action it chose, snapped to A(c), or None."""
    with cell_frame(ctx["bricks"]):
        r0 = _util(ctx["bricks"], ctx["step"], 1.0)
        b = bracing.assign(ctx["brick"], list(ctx["placed"]), "weakest_joint", safety_factor=sf, unbraced=r0,
                           grip_N=exe["grip_N"], mu=exe["mu"])
    if b is None or not b.get("feasible", True):
        return None
    y, z = b["brace_pose"][1:3]
    lean = round(math.degrees(b["brace_tilt_rotvec"][0]))
    with cell_frame(ctx["bricks"]):
        yz = min(_cands(ctx["bricks"], ctx["step"]), key=lambda c: (c[0] - y) ** 2 + (c[1] - z) ** 2)
    return (yz[0], yz[1], lean)


def pick_random(ctx, rng):
    """A uniformly random braced candidate of A(c) (numpy Generator), or None if there is none."""
    acts = actions(ctx)[1:]
    return acts[int(rng.integers(len(acts)))] if acts else None


def screen_candidates(ctx, exe):
    """The 12 braced candidates of the screened-best procedure: the LP-best 4 (y, z) (assign's key
    order; leans scripted, scripted, 0, opposite), the nearest 2 to the placed brick's centre (leans
    scripted, 0), and 6 random from the rest of A(c) (their own leans). The random draw is seeded from
    (stage 'screen', structure key, step), so the list is reproducible."""
    bricks, n = ctx["bricks"], ctx["step"]
    with cell_frame(bricks):
        yz = list(_cands(bricks, n))
        if not yz:
            return []
        px, py, pz = P.brick_pose(ctx["brick"])
        ranked = sorted(range(len(yz)), key=lambda i: (_lp_key(bricks, n, *yz[i], exe), i))
        s = lambda i: _scripted_lean(ctx, yz[i][0])
        best = ranked[:4]
        out = [(*yz[i], l) for i, l in zip(best, [s(best[0]), s(best[1]), 0, -s(best[3])] if len(best) == 4
                                           else [s(i) for i in best])]
        near = sorted((i for i in range(len(yz)) if i not in best),
                      key=lambda i: (math.hypot(yz[i][0] - py, yz[i][1] - (pz + P.BRICK_H / 2)), i))[:2]
        out += [(*yz[i], l) for i, l in zip(near, (s(near[0]), 0))] if near else []
    have = set(out)
    rest = [a for a in actions(ctx)[1:] if a not in have]
    rng = _rng("screen", [list(c) for c in ctx["key"]], n)
    out += [rest[i] for i in rng.choice(len(rest), min(6, len(rest)), replace=False)]
    return out


def screen_pick(rows):
    """The screened-best pick over result rows, one per candidate: dicts with index (candidate index),
    success (bool), u2_peak, arm_arm (bool), lp_u (LP braced u) and dy. Rule: the success with the lowest
    u2_peak; ties within 0.01 go to the lower lp_u, then the smaller |dy|, then the lower index. With no
    success: the lowest u2_peak among candidates without arm_arm, else the lowest u2_peak (same ties).
    Returns the chosen row."""
    pool = [r for r in rows if r["success"]] or [r for r in rows if not r["arm_arm"]] or list(rows)
    lo = min(r["u2_peak"] for r in pool)
    return min((r for r in pool if r["u2_peak"] <= lo + 0.01),
               key=lambda r: (r["lp_u"], abs(r["dy"]), r["index"]))


# --- self-check ---------------------------------------------------------------

def _best_braced(bricks, n, scale, exe, mask):
    """min over candidates of the LP braced u at scale x design press; mask 'committed' (28 mm rule) or 'relaxed'."""
    brick, placed = bricks[n], list(bricks[:n])
    ok = (lambda y, z: bracing.feasible(brick, placed, y, z)) if mask == "committed" else (lambda y, z: True)
    us = [_util(bricks, n, scale, (y, z, exe["grip_N"]), exe["mu"]).s for y, z in _cands(bricks, n) if ok(y, z)]
    return (min(us) if us else float("nan")), len(us)


def _quantiles(v):
    return (min(v), float(np.median(v)), max(v)) if v else (None, None, None)


def self_check():
    """Reproduce the [P] numbers of plan_v4_rl_brace section 0 (facts 2, 7, 8), print them beside the
    plan's values, and write results/v4/rl/r0/lp_probe.json. Nothing is tuned to match."""
    exe = EXE["E0"]
    rows, out = [], {"exe": exe, "notes": []}

    def row(name, plan, *mine, note=""):
        best = min((abs(m - plan) for m in mine if m == m), default=float("nan"))
        rows.append((name, plan, mine, "" if best <= 0.05 else "DIFF", note))

    s3 = tuple(P.sequence(P.STRUCTURES["S3"]))
    arch = tuple(BP.load("arch"))
    lp = {}
    with cell_frame(s3):
        for n in (13, 14):
            for sc in (1.0, 0.5):
                lp[n, sc] = {"u0": _util(s3, n, sc).s}
                for m in ("committed", "relaxed"):
                    lp[n, sc][m], lp[n, sc][m + "_n"] = _best_braced(s3, n, sc, exe, m)
    with cell_frame(arch):
        for m in ("committed", "relaxed"):
            lp["arch", m] = _best_braced(arch, 10, 1.0, exe, m)[0]
        lp["arch", "u0"] = _util(arch, 10, 1.0).s
    a = lambda n, sc, k: lp[n, sc][k]
    row("S3 step 13 u0, design press", 3.43, a(13, 1.0, "u0"))
    row("S3 step 13 best braced, committed mask", 1.97, a(13, 1.0, "committed"))
    row("S3 step 13 best braced, relaxed mask", 1.56, a(13, 1.0, "relaxed"))
    row("S3 step 13 half press, committed mask", 0.49, a(13, 0.5, "committed"))
    row("S3 step 13 half press, relaxed mask", 0.08, a(13, 0.5, "relaxed"))
    row("S3 step 14 u0, design press", 1.96, a(14, 1.0, "u0"))
    row("S3 step 14 braced, committed (press scale ambiguous: 1.0 | 0.5)", 0.44,
        a(14, 1.0, "committed"), a(14, 0.5, "committed"))
    row("S3 step 14 braced, relaxed (press scale ambiguous: 1.0 | 0.5)", 0.00,
        a(14, 1.0, "relaxed"), a(14, 0.5, "relaxed"))
    row("arch step 10 u0", 0.95, lp["arch", "u0"])
    row("arch step 10 best braced (committed | relaxed)", 0.95, lp["arch", "committed"], lp["arch", "relaxed"])
    out["lp"] = {"%s@%s" % k: v for k, v in lp.items()}

    with cell_frame(s3):                                      # LP solve time
        br = bracing._brace_obj(list(s3[:13]), *_cands(s3, 13)[0], exe["grip_N"], exe["mu"])
        t = time.perf_counter()
        for _ in range(40):
            ST.analyze(s3[:13], loads=ST.insertion_loads(s3[13], list(s3[:13])), braces=[br])
        ms = (time.perf_counter() - t) / 40 * 1e3
    out["lp_solve_ms"] = ms
    row("LP solve, ms (S3 step 13, braced)", 3.0, ms, note="plan: 2-4 ms; 5 solves per utilisation")

    counts = {}
    for name, bricks in (("S3", s3), ("S5", tuple(P.sequence(P.STRUCTURES["S5"]))), ("arch", arch)):
        crit = [c for c in contexts(bricks) if c["pool"] == "critical"]
        with cell_frame(bricks):
            counts[name] = {"critical_steps": [c["step"] for c in crit],
                            "committed": [len([1 for y, z in _cands(bricks, c["step"])
                                               if bracing.feasible(c["brick"], list(c["placed"]), y, z)]) for c in crit],
                            "relaxed": [len(_cands(bricks, c["step"])) for c in crit]}
    allc = [v for c in counts.values() for v in c["committed"]]
    out["candidate_counts"] = counts
    rows.append(("candidates per critical step, S3/S5/arch (committed mask): min-max", "40-178",
                 (min(allc), max(allc)) if allc else (None, None), "", "critical = u0(1.3) >= 0.8; relaxed counts in the json"))

    stats = corbel_statistics(30, exe)
    out["corbel_30"] = stats
    rows.append(("corbel family: structures with >= 1 critical step, of 30", "19 of 30", "%d of 30" % stats["with_critical"], "", "a different generator: not expected to match"))
    rows.append(("corbel family: critical steps", "47", str(stats["critical_steps"]), "", ""))
    rows.append(("corbel family: LP-saveable critical steps (committed mask)", "17 of 47", "%d of %d" % (stats["saveable"], stats["critical_steps"]), "",
                 "saveable = some candidate's braced u(1.3 x design) < 1"))
    rows.append(("corbel family: median share of candidates that save", "30%", "%s%%" % stats["median_save_pct"], "", ""))
    rows.append(("corbel family: median candidates per critical step (committed mask)", "28", str(stats["median_candidates"]), "", ""))
    out["rows"] = [{"item": r[0], "plan": r[1], "reproduced": r[2], "flag": r[3], "note": r[4]} for r in rows]

    for r in rows:
        mine = ", ".join("%.3f" % m if isinstance(m, float) else str(m) for m in r[2]) if isinstance(r[2], tuple) else \
            ("%.3f" % r[2] if isinstance(r[2], float) else str(r[2]))
        print("%-72s plan %-9s reproduced %-22s %s %s" % (r[0], r[1], mine, r[3], r[4]))
    path = HERE / "results" / "v4" / "rl" / "r0" / "lp_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print("-> %s" % path)


def corbel_statistics(n_struct, exe):
    """LP statistics of corbel_family(0..n_struct-1) (raw orientation): critical steps, saveable at the
    executor grip (committed mask; some candidate's braced u at 1.3 x design < 1), share of candidates that
    save among saveable steps, candidates per critical step (committed mask)."""
    crit_steps = saveable = 0
    with_crit, save_pct, ncand = set(), [], []
    keys = set()
    for seed in range(n_struct):
        bricks = tuple(corbel_family(seed))
        keys.add(canonical_key(bricks))
        for c in contexts(bricks):
            if c["pool"] != "critical":
                continue
            with cell_frame(bricks):
                cm = [(y, z) for y, z in _cands(bricks, c["step"])
                      if bracing.feasible(c["brick"], list(c["placed"]), y, z)]
                ok = [_util(bricks, c["step"], 1.3, (y, z, exe["grip_N"]), exe["mu"]).s < 1.0 for y, z in cm]
            crit_steps += 1
            with_crit.add(seed)
            ncand.append(len(cm))
            if any(ok):
                saveable += 1
                save_pct.append(100.0 * sum(ok) / len(ok))
    return {"structures": n_struct, "unique_keys": len(keys), "with_critical": len(with_crit),
            "critical_steps": crit_steps, "saveable": saveable,
            "median_save_pct": round(float(np.median(save_pct)), 1) if save_pct else None,
            "median_candidates": float(np.median(ncand)) if ncand else None,
            "bricks_per_structure": _quantiles([len(corbel_family(s)) for s in range(n_struct)])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
    else:
        ap.print_help()
