"""Static force balance of a (partial) brick structure -- master_report §3.3/§3.6.

    res = analyze(bricks, loads=[...], braces=[...])
    res.s              max clutch utilisation (>= 1: no admissible force distribution)
    res.util[c]        per-connection utilisation in the returned distribution
    res.weakest        the connection closest to failure (the predicted failure joint)
    res.brace_wrench   what each brace must supply, per brace

The formulation is StableLego's [StableLego]: every brick in force and moment
equilibrium under gravity and applied loads, carried by connection forces that
must be admissible -- compression anywhere in the overlap patch, tension only
at the studs, at most f_break each. That per-connection condition is
sim/joint_model/capacity.Patch, so the planner's prediction and the twin's
joint model share one capacity law (and it reproduces BrickSim's static_solve
on S3: u = 2.36 at a 35.6 N tip press). The LP minimises the maximum
utilisation s over all admissible distributions -- the lower-bound theorem: if
s < 1 some distribution holds, so the structure stands.

A brace is a grasp by the stabilizer: a bounded wrench on each gripped brick
(pad friction mu * grip per gripped brick, a pad-couple moment limit, and the
arm's force limit on the total), which the LP may use as it likes. After the
minimax solve a second LP fixes s and minimises brace effort, so the brace
wrench reported is the least the stabilizer must supply -- the
expected_reaction_wrench §2.6 asks the plan to emit.

Not modelled (as in capacity.py): shear and torsion capacity (stud bearing
carries them), and the 0.2 mm gaps between neighbours in a layer (bricks in a
layer interact only through the bricks above and below).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import planner as P
from sim.joint_model.capacity import F_BREAK_PER_STUD, F_INSERT_PER_STUD, Patch

G = 9.81
MASS_PER_STUD = 0.0025 / 8
LATERAL_UNCERTAINTY = 3.0        # N, §3.6 step 1
SAFETY_FACTOR = 3.0         # §3.6 step 4 default is 1.5. v3.1: the twin broke S3's pier
                            # at a step this rigid-plastic model rates 0.40 -- its joints are
                            # brittle and compliant (the pier leans 2 deg, P-delta), which a
                            # lower-bound plastic LP does not see -- so the margin is set from
                            # that measured gap (ledger v31_safety_factor)


@dataclass
class Connection:
    upper: str
    lower: object                 # brick id or None (baseplate)
    cells: frozenset              # grid cells of the overlap
    centre: np.ndarray            # world (x, y, z) of the patch centre (interface plane)
    patch: Patch


@dataclass
class Brace:
    """A stabilizer grasp. `bricks` are the gripped brick ids (1, or 2 when the
    pads straddle a seam); `point` the grasp centre (world, m)."""
    bricks: tuple
    point: tuple
    grip_N: float = 40.0          # per finger
    shares: tuple = None          # fraction of the pad contact on each gripped brick
                                  # (default: equal); pad friction splits by contact
    mu: float = 1.0
    arm_max_N: float = 60.0       # total force the stabilizer holds
    moment_max_Nm: float = 0.3    # pad couple about the grip axis, shared like the force
                                  # (bracing.py sets it from the grip: mu * 2 * grip * 5 mm)
    lateral_max_N: float = None   # bound on the horizontal brace force (the feed-forward:
                                  # a sideways push loads the stud interface being pressed
                                  # and stalled the S3 step-12 insertion at 69 N)


@dataclass
class Result:
    s: float
    feasible: bool
    util: dict = field(default_factory=dict)          # (upper, lower) -> utilisation
    weakest: tuple = None
    brace_wrench: list = field(default_factory=list)  # per brace, world (Fx..Mz) on the structure
    brace_util: list = field(default_factory=list)    # per brace: max over gripped bricks of
                                                      # |force| / its friction share


def brick_centre(b):
    x, y, z = P.brick_pose(b)
    return np.array([x, y, z + P.BRICK_H / 2])


def connections(bricks):
    """Connections of a placed set: each brick on each overlapping brick one
    layer below, and layer-0 bricks on the baseplate."""
    out = []
    by_layer = {}
    for b in bricks:
        by_layer.setdefault(b[4], []).append(b)
    for b in bricks:
        cu, k = P.cells(b)
        supports = [(None, cu)] if k == 0 else [
            (s[0], cu & P.cells(s)[0]) for s in by_layer.get(k - 1, []) if cu & P.cells(s)[0]]
        for sid, patch in supports:
            pts = np.array([[P.VOXEL_ORIGIN[0] + (i + 0.5) * P.PITCH,
                             P.VOXEL_ORIGIN[1] + (j + 0.5) * P.PITCH] for i, j in patch])
            cen = pts.mean(axis=0)
            rect = (pts[:, 0].min() - P.PITCH / 2 - cen[0], pts[:, 0].max() + P.PITCH / 2 - cen[0],
                    pts[:, 1].min() - P.PITCH / 2 - cen[1], pts[:, 1].max() + P.PITCH / 2 - cen[1])
            z = P.VOXEL_ORIGIN[2] + k * P.BRICK_H
            out.append(Connection(b[0], sid, frozenset(patch), np.array([cen[0], cen[1], z]),
                                  Patch(pts - cen, rect, F_BREAK_PER_STUD)))
    return out


def insertion_loads(brick, placed, lateral=(0.0, 0.0), press=None):
    """§3.6 step 1: the insertion wrench of `brick` onto `placed`, as point
    loads on its supports: n_studs x f_insert per patch along the stud axis
    (or `press` total, split by studs), plus a lateral uncertainty term."""
    cu, k = P.cells(brick)
    sup = P.supports(brick, placed)
    by_id = {p[0]: p for p in placed}
    total_studs = sum(sup.values()) if sup else len(cu)
    Q = press if press is not None else total_studs * F_INSERT_PER_STUD
    loads = []
    if not sup:
        return loads                                   # onto the baseplate
    for sid, n in sup.items():
        patch = cu & P.cells(by_id[sid])[0]
        pts = np.array([[P.VOXEL_ORIGIN[0] + (i + 0.5) * P.PITCH,
                         P.VOXEL_ORIGIN[1] + (j + 0.5) * P.PITCH] for i, j in patch])
        cen = pts.mean(axis=0)
        z = P.VOXEL_ORIGIN[2] + k * P.BRICK_H
        share = n / total_studs
        loads.append((sid, np.array([lateral[0] * share, lateral[1] * share, -Q * share]),
                      np.array([cen[0], cen[1], z])))
    return loads


def _cross_mat(p):
    x, y, z = p
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.0]])


def analyze(bricks, loads=(), braces=(), gravity=True, minimise_brace=True, util_cap=None):
    """Minimax utilisation of `bricks` (planner tuples, all placed).

    loads: [(brick_id, force(3), point(3))] in N and m, world frame.
    braces: [Brace].
    util_cap: instead of the least total utilisation at the minimax, the
    least brace effort that keeps every joint at or below max(s*, util_cap)
    -- the wrench a coordinated stabilizer should feed forward (bracing.py).
    """
    bricks = list(bricks)
    ids = [b[0] for b in bricks]
    idx = {bid: n for n, bid in enumerate(ids)}
    conns = connections(bricks)
    nc, nb = len(conns), len(bricks)
    nbr = sum(len(br.bricks) for br in braces)
    # variables: 6 per connection (wrench on the upper brick at its patch
    # centre, forces in N, moments in N*mm), 6 per gripped brick, then s
    MM = 1000.0
    nv = 6 * nc + 6 * nbr + 1
    s_col = nv - 1
    A = lil_matrix((6 * nb, nv))
    b_eq = np.zeros(6 * nb)

    def add_wrench(row0, col0, point, sign):
        """Spatial force (about the origin, moments in N*mm) of the wrench in
        columns col0..col0+5 applied at `point`, into rows row0..row0+5."""
        pm = np.asarray(point) * MM
        for a in range(3):
            A[row0 + a, col0 + a] += sign
            A[row0 + 3 + a, col0 + 3 + a] += sign
        X = _cross_mat(pm)                       # moment of the force about the origin
        for r in range(3):
            for c in range(3):
                if X[r, c]:
                    A[row0 + 3 + r, col0 + c] += sign * X[r, c]

    for c_i, c in enumerate(conns):
        add_wrench(6 * idx[c.upper], 6 * c_i, c.centre, +1.0)
        if c.lower is not None:
            add_wrench(6 * idx[c.lower], 6 * c_i, c.centre, -1.0)
    col = 6 * nc
    brace_cols = []
    for br in braces:
        cols = []
        for gb in br.bricks:
            add_wrench(6 * idx[gb], col, br.point, +1.0)
            cols.append(col)
            col += 6
        brace_cols.append(cols)
    # right-hand side: minus external loads (gravity at centres, applied loads)
    ext = np.zeros(6 * nb)

    def add_ext(bid, force, point):
        f = np.asarray(force, float)
        m = np.cross(np.asarray(point) * MM, f)
        ext[6 * idx[bid]:6 * idx[bid] + 3] += f
        ext[6 * idx[bid] + 3:6 * idx[bid] + 6] += m

    if gravity:
        for b in bricks:
            nx, ny = P.footprint(b[1], b[5])
            add_ext(b[0], [0, 0, -MASS_PER_STUD * nx * ny * G], brick_centre(b))
    for bid, f, p in loads:
        add_ext(bid, f, p)
    b_eq = -ext
    # inequalities: capacity per connection edge, <= s ; brace bounds via variable bounds
    rows = []
    for c_i, c in enumerate(conns):
        for e in range(4):
            coef = c.patch.coef[e]                  # (Fz, Mx, My) with M in N*m
            r = np.zeros(nv)
            r[6 * c_i + 2] = coef[0]
            r[6 * c_i + 3] = coef[1] / MM
            r[6 * c_i + 4] = coef[2] / MM
            r[s_col] = -1.0
            rows.append(r)
    A_ub = np.array(rows) if rows else np.zeros((0, nv))
    b_ub = np.zeros(len(rows))
    # The pads close along x, so the brace's force on each gripped brick in the
    # pad plane (y, z) is friction: |(fy, fz)| <= mu * 2 * grip * share, as an
    # octagon inscribed in that circle. The arm's total force is bounded the
    # same way in (y, z) and by arm_max along x. (A per-axis box let the LP ask
    # one brace for 76 N from a 60 N arm.)
    extra_ub, extra_b = [], []
    octagon = [(math.cos(a), math.sin(a)) for a in np.arange(8) * math.pi / 4]
    inscribe = math.cos(math.pi / 8)
    for br, cols in zip(braces, brace_cols):
        shares = br.shares or tuple(1.0 / len(br.bricks) for _ in br.bricks)
        for c0, sh in zip(cols, shares):
            for cy, cz in octagon:
                r = np.zeros(nv)
                r[c0 + 1], r[c0 + 2] = cy, cz
                extra_ub.append(r)
                extra_b.append(br.mu * 2 * br.grip_N * sh * inscribe)
        for cy, cz in octagon:
            r = np.zeros(nv)
            for c0 in cols:
                r[c0 + 1], r[c0 + 2] = cy, cz
            extra_ub.append(r)
            extra_b.append(br.arm_max_N * inscribe)
        for sgn in (1.0, -1.0):
            r = np.zeros(nv)
            for c0 in cols:
                r[c0] = sgn
            extra_ub.append(r)
            extra_b.append(br.arm_max_N)
    if extra_ub:
        A_ub = np.vstack([A_ub, np.array(extra_ub)])
        b_ub = np.concatenate([b_ub, extra_b])
    bounds = [(None, None)] * (6 * nc)
    for br in braces:
        shares = br.shares or tuple(1.0 / len(br.bricks) for _ in br.bricks)
        for sh in shares:
            fmax = br.mu * 2 * br.grip_N * sh
            mmax = br.moment_max_Nm * MM * sh
            lat = fmax if br.lateral_max_N is None else min(fmax, br.lateral_max_N * sh)
            bounds += [(-lat, lat)] * 2 + [(-fmax, fmax)] + [(-mmax, mmax)] * 3
    bounds += [(0, None)]
    cost = np.zeros(nv)
    cost[s_col] = 1.0
    A_eq = A.tocsr()
    r1 = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r1.status != 0:
        return Result(s=float("inf"), feasible=False)
    x = r1.x
    s_star = float(x[s_col])
    # Stage 2: at the optimal max, the least total utilisation (so connections
    # not on the critical path are not left arbitrarily loaded -- the minimax
    # is degenerate, e.g. every pier joint of S3 pivots on the same edge line)
    # plus a small weight on brace effort, so the brace wrench reported is the
    # least the stabilizer must supply.
    n_aux = nc + (6 * nbr if minimise_brace else 0)
    n2 = nv + n_aux
    cost2 = np.zeros(n2)
    cost2[nv:nv + nc] = 1.0 if util_cap is None else 1e-3
    if minimise_brace and nbr:
        cost2[nv + nc:] = 1e-3 if util_cap is None else 1.0
    A_eq2 = lil_matrix((A_eq.shape[0], n2))
    A_eq2[:, :nv] = A_eq
    ub_rows = [np.hstack([A_ub, np.zeros((A_ub.shape[0], n_aux))])]
    aux = []
    for c_i, c in enumerate(conns):
        for e in range(4):
            coef = c.patch.coef[e]
            r = np.zeros(n2)
            r[6 * c_i + 2] = coef[0]
            r[6 * c_i + 3] = coef[1] / MM
            r[6 * c_i + 4] = coef[2] / MM
            r[nv + c_i] = -1.0
            aux.append(r)
    if minimise_brace:
        for k in range(6 * nbr):
            for sgn in (1.0, -1.0):
                r = np.zeros(n2)
                r[6 * nc + k] = sgn
                r[nv + nc + k] = -1.0
                aux.append(r)
    if aux:
        ub_rows.append(np.array(aux))
    A_ub2 = np.vstack(ub_rows)
    b_ub2 = np.concatenate([b_ub, np.zeros(len(aux))])
    bounds2 = list(bounds)
    bounds2[s_col] = (0, max(s_star, util_cap or 0.0) * 1.0001 + 1e-9)
    r2 = linprog(cost2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2.tocsr(), b_eq=b_eq,
                 bounds=bounds2 + [(0, None)] * n_aux, method="highs")
    if r2.status == 0:
        x = r2.x[:nv]
    util = {}
    for c_i, c in enumerate(conns):
        w = x[6 * c_i:6 * c_i + 6]
        util[(c.upper, c.lower)] = float(c.patch.utilization(w[2], w[3] / MM, w[4] / MM))
    # the predicted failure joint; ties (a column pivoting on one edge) go to
    # the joint nearest the load, where a brace intercepts the path first
    zc = {(c.upper, c.lower): c.centre[2] for c in conns}
    weakest = max(util, key=lambda k: (round(util[k], 6), zc[k])) if util else None
    bw, bu = [], []
    for br, cols in zip(braces, brace_cols):
        tot = np.zeros(6)
        worst = 0.0
        shares = br.shares or tuple(1.0 / len(br.bricks) for _ in br.bricks)
        for c0, sh in zip(cols, shares):
            w = x[c0:c0 + 6] * np.array([1, 1, 1, 1 / MM, 1 / MM, 1 / MM])
            tot += w
            worst = max(worst, float(np.linalg.norm(w[:3]) / (br.mu * 2 * br.grip_N * sh)))
        bw.append(tot)
        bu.append(worst)
    return Result(s=s_star, feasible=True, util=util, weakest=weakest, brace_wrench=bw,
                  brace_util=bu)


def insertion_utilisation(brick, placed, braces=(), press=None):
    """Worst case over the §3.6 lateral uncertainty directions of the minimax
    utilisation of `placed` while `brick` is pressed onto it. Returns the
    Result with the largest s."""
    worst = None
    for lat in ((0, 0), (LATERAL_UNCERTAINTY, 0), (-LATERAL_UNCERTAINTY, 0),
                (0, LATERAL_UNCERTAINTY), (0, -LATERAL_UNCERTAINTY)):
        r = analyze(placed, loads=insertion_loads(brick, placed, lat, press), braces=braces)
        if worst is None or r.s > worst.s:
            worst = r
    return worst


def best_press_point(brick, placed, grasp, press=None, step=0.002):
    """Where along the brick to grip it (so where the press goes in): the
    offset, among those that keep the pads on the brick, that minimises the
    force-balance utilisation with the new brick in the model and the press
    applied at the TCP, over the lateral-uncertainty directions along the
    slide axis. The planner's default -- toward the mating patch's centroid --
    put a corbel's press exactly over the pier's edge (S3 step 11: the pier's
    base joint broke in the twin at a predicted 0.66).
    Returns (tcp_offset_m, utilisation)."""
    nx, ny = P.footprint(brick[1], brick[5])
    off = list(grasp["tcp_offset_m"])
    along = 1 if grasp["yaw_offset_deg"] == 90 else 0      # fingers on x faces: slide along y
    n_along = (nx, ny)[along]
    room = max(0.0, n_along * P.PITCH / 2 - 0.0085 - 0.001)
    sup = P.supports(brick, placed)
    Q = press if press is not None else sum(sup.values()) * F_INSERT_PER_STUD
    if not sup or room <= 0:
        return off, None
    bx, by, bz = P.brick_pose(brick)
    top = bz + P.BRICK_H
    best = None
    for o in np.arange(-room, room + 1e-9, step):
        px, py = (bx, by + o) if along == 1 else (bx + o, by)
        worst = 0.0
        for lat in (0.0, LATERAL_UNCERTAINTY, -LATERAL_UNCERTAINTY):
            f = (0.0, lat, -Q) if along == 1 else (lat, 0.0, -Q)
            r = analyze(list(placed) + [brick], loads=[(brick[0], np.array(f), np.array([px, py, top]))])
            worst = max(worst, r.s)
        key = (round(worst, 3), abs(o - off[along]))
        if best is None or key < best[0]:
            best = (key, float(o), worst)
    new = [0.0, 0.0]
    new[along] = round(best[1], 4)
    return new, best[2]


if __name__ == "__main__":
    # check against BrickSim on the v3.0 cantilever (README: u = 2.36 at a
    # 35.6 N tip press); the benchmark's S3 is now the corbelled bridge
    import sys
    sys.modules.setdefault("stability", sys.modules[__name__])
    order = P.sequence(P.STRUCTURES["S3C"])
    placed, tip = order[:-1], order[-1]
    r = analyze(placed, loads=insertion_loads(tip, placed), gravity=False)
    print("S3C tip press, no gravity: s = %.3f  weakest %s" % (r.s, r.weakest))
    r = insertion_utilisation(tip, placed)
    print("S3C with gravity and +-3 N lateral: s = %.3f  weakest %s" % (r.s, r.weakest))
    sys.exit(0 if abs(analyze(placed, loads=insertion_loads(tip, placed), gravity=False).s - 2.363) < 0.01 else 1)
