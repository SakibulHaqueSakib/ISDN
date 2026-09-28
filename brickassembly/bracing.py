"""Brace assignment -- master_report §WP3.6, the project's primary contribution.

    brace = assign(brick, placed, strategy)     # dict per §2.6, or None

Three strategies (§WP3.6 "baselines that MUST also be implemented"):

  none           no stabilizer.
  nearest        the naive heuristic: grasp the placed brick nearest the
                 placement, at the nearest spot the two hands can share.
  weakest_joint  the proposed method: find the joints the static force model
                 (stability.py -- the StableLego force balance with
                 capacity.Patch) predicts will fail under the insertion
                 wrench, and grasp where it predicts the lowest utilisation
                 among grasps touching a brick of such a joint.

Both bracing strategies brace on the SAME trigger -- predicted utilisation x
safety factor 1.5 >= 1 (§3.6 step 4) -- so ablation A6 varies only WHERE the
stabilizer holds, not whether it does.

A brace is a parallel-jaw grasp by arm A across the structure's x faces
(every brick here is 2 studs wide in x). The Panda pads are 17 x 17 mm, so a
grasp centred on a seam grips both bricks of that joint and one centred on a
brick also catches the edges of its neighbours; the gripped set is whatever
the pad window overlaps. In the force model each gripped brick may receive a
bounded wrench from the stabilizer: pad friction mu * 2 * grip shared among
the gripped bricks, a pad-couple moment limit, and the arm's force limit on
the total. The stabilizer leans 45 deg away from the placer about the grip
axis so the two hands fit (the Newton prototype's measured fix).

D8's Allegro stabilizer is replaced by a second parallel jaw (A9 cut): see
the v3.1 amendment.
"""

import math

import numpy as np

import planner as P
import stability as ST

GRIP_N = 70.0                    # per finger while bracing: the Franka hand's continuous
                                 # maximum (40 N let the pads slip in the S3 step-12 trial)
PAD_MOMENT_ARM = 0.005           # m: mean lever of a 17 x ~7 mm pad contact about its
                                 # centre, so the pad couple about the grip axis is
                                 # mu * 2 * grip * 5 mm = 0.7 N*m at 70 N (the twin's
                                 # pads measured 1.65 N*m: this is the conservative one)
FF_PRESS_MARGIN = 1.3            # the executor presses at 1.3 x n_studs x f_insert
FF_UTIL_CAP = 0.5                # feed-forward: least brace effort keeping joints <= this
PAD_HALF = 0.0085                # Panda pad half-size, m (17 x 17 mm)
MIN_PAD_Z = 0.004                # m of pad height on a brick for it to count as gripped
HAND_CLEARANCE_Y = 0.028         # |y_brace - y_place| so fingertips keep >= 11 mm apart
TILT = math.radians(45)
STRATEGIES = ("none", "nearest", "weakest_joint")


def _span(b):
    """World (y0, y1, z0, z1) of a brick's side face."""
    x, y, z = P.brick_pose(b)
    nx, ny = P.footprint(b[1], b[5])
    return (y - ny * P.PITCH / 2, y + ny * P.PITCH / 2, z, z + P.BRICK_H)


def pad_contact(placed, y, z):
    """{brick id: pad contact area (m^2)} for the pad window centred at (y, z)."""
    out = {}
    for b in placed:
        y0, y1, z0, z1 = _span(b)
        dy = min(y1, y + PAD_HALF) - max(y0, y - PAD_HALF)
        dz = min(z1, z + PAD_HALF) - max(z0, z - PAD_HALF)
        # a brick counts as gripped only with MIN_PAD_Z of pad height on it: a
        # 3.7 mm sliver got a share of the brace in the LP, and in the twin the
        # joint above it broke (S3 step 12)
        if dy > 0.002 and dz >= MIN_PAD_Z:
            out[b[0]] = dy * dz
    return out


def gripped(placed, y, z):
    """Bricks whose side faces the pad window centred at (y, z) overlaps."""
    return tuple(sorted(pad_contact(placed, y, z)))


def lean_for(brick, y):
    """Rotation vector of the brace's 45 deg lean away from the placer."""
    lean = -1.0 if y < P.brick_pose(brick)[1] else 1.0
    return [lean * -TILT, 0.0, 0.0]


def feasible(brick, placed, y, z, clearance=None):
    """Can arm A hold at (y, z) while arm B inserts `brick`? With a twin
    `clearance` check (sim/mj/checks.BraceClearance) the hands are tested
    against each other and the carried brick; without one, the 28 mm rule."""
    z_plate = P.VOXEL_ORIGIN[2]
    if z - PAD_HALF < z_plate + P.STUD_H + 0.001:        # pads above the baseplate studs
        return False
    py = P.brick_pose(brick)[1]
    if clearance is None and abs(y - py) < HAND_CLEARANCE_Y:   # fingers clear of the placer's
        return False
    g = gripped(placed, y, z)
    if not g:
        return False
    # the pad window must be backed by structure over most of its height, or
    # the grasp is on an edge
    covered = 0.0
    for b in placed:
        if b[0] in g:
            y0, y1, z0, z1 = _span(b)
            covered += max(0.0, min(z1, z + PAD_HALF) - max(z0, z - PAD_HALF))
    if covered < 1.2 * PAD_HALF:
        return False
    return clearance is None or clearance(brick, y, z, lean_for(brick, y))


def candidates(brick, placed, clearance=None):
    """Grasp centres: every stud column along y, at every seam and mid-brick
    height of the placed structure."""
    ys = sorted({P.VOXEL_ORIGIN[1] + (j + 0.5) * P.PITCH
                 for b in placed for (_, j) in P.cells(b)[0]})
    kmax = max(b[4] for b in placed)
    zs = [P.VOXEL_ORIGIN[2] + (k + f) * P.BRICK_H for k in range(kmax + 1)
          for f in (0.0, 0.25, 0.5, 0.75)]
    return [(y, z) for y in ys for z in zs if feasible(brick, placed, y, z, clearance)]


def _brace_obj(placed, y, z, grip=GRIP_N):
    """The grasp as the force model sees it: pad friction on each gripped
    brick in proportion to its share of the pad contact. (An equal split let
    a brick caught by 3 mm of pad count as fully held; the LP then preferred
    a grasp that the twin showed pried open 13 times.)"""
    x = P.VOXEL_ORIGIN[0] + P.PITCH          # structures are 2 studs wide: faces at i=0 and i=2
    area = pad_contact(placed, y, z)
    ids = tuple(sorted(area))
    tot = sum(area.values())
    return ST.Brace(bricks=ids, point=(x, y, z), grip_N=grip,
                    shares=tuple(area[i] / tot for i in ids),
                    moment_max_Nm=1.0 * 2 * grip * PAD_MOMENT_ARM)


def predicted(brick, placed, brace=None):
    return ST.insertion_utilisation(brick, placed, braces=[brace] if brace else [])


def assign(brick, placed, strategy, safety_factor=ST.SAFETY_FACTOR, unbraced=None,
           clearance=None):
    """§2.6 brace dict for inserting `brick` onto `placed`, or None."""
    if strategy == "none" or not placed:
        return None
    r0 = unbraced or predicted(brick, placed)
    if r0.s * safety_factor < 1.0:
        return None
    px, py, pz = P.brick_pose(brick)
    over = {j for j, u in r0.util.items() if u * safety_factor >= 1.0}
    joint_bricks = {b for j in over for b in j if b is not None}
    cands = candidates(brick, placed, clearance)
    if not cands:
        return _record(brick, placed, None, r0, strategy, "no_feasible_grasp")
    if strategy == "weakest_joint":
        pool = [(y, z) for y, z in cands if set(gripped(placed, y, z)) & joint_bricks] or cands
        best = None
        press = FF_PRESS_MARGIN * sum(P.supports(brick, placed).values()) * ST.F_INSERT_PER_STUD
        for y, z in pool:
            br = _brace_obj(placed, y, z)
            r = predicted(brick, placed, br)
            # lowest predicted utilisation (§3.6, nominal press); among equals,
            # the lowest at the press the executor applies; then the grasp whose
            # anchor is least loaded relative to its pad share (the LP treats a
            # 3 mm pad overlap as a rigid anchor; the twin does not)
            ra = ST.insertion_utilisation(brick, placed, braces=[br], press=press) \
                if r.s < 0.5 else r
            key = (round(r.s, 2), round(ra.s, 3), round(r.brace_util[0], 3))
            if best is None or key < best[0]:
                best = (key, y, z, br, r)
        _, y, z, br, r = best
        return _record(brick, placed, (y, z, br, r), r0, strategy, "weakest_joint_lp")
    if strategy == "nearest":
        # the nearest placed brick that can be grasped at all (taking the
        # literal nearest and giving up when the hands cannot share it would
        # make the baseline a straw man)
        graspable = [b for b in placed if any(b[0] in gripped(placed, y, z) for y, z in cands)]
        target = min(graspable, key=lambda b: np.linalg.norm(np.array(P.brick_pose(b)) - [px, py, pz]))
        on = [(y, z) for y, z in cands if target[0] in gripped(placed, y, z)]
        tz = P.brick_pose(target)[2] + P.BRICK_H / 2
        y, z = min(on, key=lambda c: (abs(c[0] - py), abs(c[1] - tz)))
        br = _brace_obj(placed, y, z)
        return _record(brick, placed, (y, z, br, predicted(brick, placed, br)), r0, strategy,
                       "nearest_placed_brick")
    raise ValueError(strategy)


def _record(brick, placed, choice, r0, strategy, rationale):
    """The §2.6 brace fields (plus what the executor and the analysis need)."""
    py = P.brick_pose(brick)[1]
    out = {"arm": "A", "rationale": rationale, "strategy": strategy,
           "predicted_failure_joint": list(r0.weakest) if r0.weakest else None,
           "predicted_util_unbraced": round(r0.s, 4)}
    if choice is None:
        out.update({"feasible": False})
        return out
    y, z, br, r = choice
    # the coordinated brace (skills.brace_feedforward): the wrench the LP puts
    # on the brace at the nominal press (no lateral uncertainty), per newton
    # of press, for the stabilizer to feed forward as the placer presses. A
    # passive hold cannot take its share: the welded joints are far stiffer
    # than any arm, so they carry the load and break first (S3 step 12:
    # brace z measured -12 N where the LP needs -35 N; 35 joint breaks).
    # Sized at the press the executor actually applies, as the least effort
    # keeping every joint under FF_UTIL_CAP (the minimax asks for more than is
    # needed and pushes the pads toward their friction limit).
    press_nom = FF_PRESS_MARGIN * sum(P.supports(brick, placed).values()) * ST.F_INSERT_PER_STUD
    nom = ST.analyze(placed, loads=ST.insertion_loads(brick, placed, press=press_nom),
                     braces=[br], util_cap=FF_UTIL_CAP)
    ff = nom.brace_wrench[0] / press_nom if nom.feasible and press_nom else np.zeros(6)
    tilt = lean_for(brick, y)                        # rotation vector about x (the grip axis)
    w = r.brace_wrench[0]
    out.update({
        "feasible": True,
        "target_brick_id": br.bricks[0] if len(br.bricks) == 1 else
        min(br.bricks, key=lambda b: abs(P.brick_pose(next(p for p in placed if p[0] == b))[2]
                                         + P.BRICK_H / 2 - z)),
        "gripped_bricks": list(br.bricks),
        "brace_pose": [round(br.point[0], 5), round(y, 5), round(z, 5)],
        "brace_tilt_rotvec": [round(v, 4) for v in tilt],
        "brace_force_N": GRIP_N,                    # grip per finger
        "brace_axis": [1, 0, 0],                    # the fingers close along x
        # what the stabilizer must supply to the structure (world frame, N and
        # N*m); the arm feels the opposite. WP7 compares this with measurement.
        "expected_reaction_wrench": [round(float(v), 4) for v in w],
        "predicted_util_braced": round(r.s, 4),
        "press_nominal_N": round(press_nom, 3),
        "predicted_util_feedforward": round(nom.s, 4) if nom.feasible else None,
        "feedforward_wrench_per_N": [round(float(v), 5) for v in ff],
        "end_effector": "parallel_jaw",
    })
    return out
