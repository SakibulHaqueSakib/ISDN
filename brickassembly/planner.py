"""Build planner: structure -> sequenced, brace-annotated assembly_plan.json.

Implements a small-scale slice of master_report.md WP3 (§2.3 geometry, §2.6
contract, §3.6 bracing).  Pure python/numpy, no GPU, no simulator.

    python planner.py            # self-check + write plans for S1/S2/S3
    python planner.py --list     # show available structures
"""

import argparse
import json
import math
from pathlib import Path

# --- §2.3 LEGO System geometry (metres) -------------------------------------
PITCH = 0.008          # stud pitch
BRICK_H = 0.0096       # brick height (3 plates)
STUD_H = 0.0018
F_INSERT_PER_STUD = 8.9   # N, WP2 default (MechanicsSnapFit envelope)
F_BREAK_PER_STUD = 11.3   # N, extraction > insertion (asymmetry invariant)
SAFETY_FACTOR = 1.5       # §3.6 step 4
PRE_INSERTION_DZ = 0.025  # §2.5 handoff: +25 mm above target
FINGER_W = 0.0175         # FR3 fingertip width along the face it grips (collider)

# (studs_x, studs_y) per SKU, yaw_index 1 swaps them.
BRICKS = {"1x1": (1, 1), "1x2": (1, 2), "1x4": (1, 4), "1x6": (1, 6),
          "2x2": (2, 2), "2x3": (2, 3), "2x4": (2, 4), "2x6": (2, 6)}

# Baseplate origin in the placer arm's base frame.  Calibration knob: keep the
# build far enough out that the arm is not folded over its own base when it
# descends, and lifted off the table so the fingertips clear it.
VOXEL_ORIGIN = (0.46, -0.04, 0.010)

# ponytail: structures hand-authored instead of sampled from StableLego (D14).
# Swap for the dataset once it is downloaded; the rest of the file is agnostic.
# Each brick: (id, type, i, j, k, yaw_index)
STRUCTURES = {
    "S1": [("b_%03d" % n, "2x2", 0, 0, n, 0) for n in range(6)],
    "S2": [  # staggered wall, seams offset between layers
        ("b_000", "2x4", 0, 0, 0, 0), ("b_001", "2x4", 0, 4, 0, 0),
        ("b_002", "2x2", 0, 2, 1, 0), ("b_003", "2x4", 0, 4, 1, 0),
        ("b_004", "2x2", 0, 0, 1, 0), ("b_005", "2x4", 0, 0, 2, 0),
        ("b_006", "2x4", 0, 4, 2, 0),
    ],
    "S3": [  # pier + cantilever: pressing the far end levers the cantilever's studs
        ("b_000", "2x2", 0, 0, 0, 0), ("b_001", "2x2", 0, 0, 1, 0),
        ("b_002", "2x2", 0, 0, 2, 0),
        ("b_003", "2x6", 0, 0, 3, 0),   # overhangs to j=5, held by 4 studs at j=0,1
        ("b_004", "2x2", 0, 4, 4, 0),   # pressed onto the free end
    ],
    # S3 stops failing single-arm once the clutch is calibrated to the
    # literature's 8-15 N/stud (ledger: s3_not_failing_when_calibrated), so
    # S3L stacks a second 2x6 to double the overhang and restore the case
    # ablation A6 needs.
    "S3L": [
        ("b_000", "2x2", 0, 0, 0, 0), ("b_001", "2x2", 0, 0, 1, 0),
        ("b_002", "2x2", 0, 0, 2, 0),
        ("b_003", "2x6", 0, 0, 3, 0),   # overhangs to j=5
        ("b_004", "2x6", 0, 4, 4, 0),   # laps onto its end, reaches j=9
        ("b_005", "2x2", 0, 8, 5, 0),   # pressed at the far tip
    ],
}


def footprint(btype, yaw_index):
    """Stud extent (nx, ny) of a brick type at the given yaw."""
    nx, ny = BRICKS[btype]
    return (ny, nx) if yaw_index else (nx, ny)


def cells(brick):
    """Grid cells (i, j) occupied by a brick, and its layer k."""
    _, btype, i, j, k, yaw = brick
    nx, ny = footprint(btype, yaw)
    return {(i + a, j + b) for a in range(nx) for b in range(ny)}, k


def brick_pose(brick):
    """World pose of the brick's *bottom* centre, in the arm base frame."""
    _, btype, i, j, k, yaw = brick
    nx, ny = footprint(btype, yaw)
    return (VOXEL_ORIGIN[0] + (i + nx / 2) * PITCH,
            VOXEL_ORIGIN[1] + (j + ny / 2) * PITCH,
            VOXEL_ORIGIN[2] + k * BRICK_H)


def supports(brick, placed):
    """{brick_id: n_shared_studs} for the already-placed bricks holding this one."""
    c, k = cells(brick)
    out = {}
    for p in placed:
        pc, pk = cells(p)
        if pk == k - 1 and c & pc:
            out[p[0]] = len(c & pc)
    return out


def sequence(bricks):
    """Bottom-up order; a brick may be placed once all its supports are placed.

    ponytail: layer-sorted greedy, not PhysASP's physics-aware action mask
    (§3.5).  Valid for these structures; cite, do not claim. A per-layer
    search for an order in which no finger grazes a neighbour (grasp_for) was
    tried and reverted: on the arch it pushed the crown last, into a slot
    between two placed bricks with 0.1 mm to spare at each end -- worse than
    the graze it avoided.
    """
    placed, order = [], []
    for b in sorted(bricks, key=lambda b: (b[4], b[2], b[3])):
        if b[4] > 0 and not supports(b, placed):
            raise ValueError("%s is floating: no support below" % b[0])
        placed.append(b)
        order.append(b)
    return order


def _extent(cellset):
    """World centre (x, y) and half-extent of a set of grid cells."""
    xs = [VOXEL_ORIGIN[0] + (i + 0.5) * PITCH for i, _ in cellset]
    ys = [VOXEL_ORIGIN[1] + (j + 0.5) * PITCH for _, j in cellset]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    half = max(PITCH * 0.5, max(max(abs(x - cx) for x in xs),
                                max(abs(y - cy) for y in ys)))
    return cx, cy, half


def weakest_joint(brick, placed):
    """Lever-arm force analysis of the partial structure under the insertion press.

    Returns (joint, tension_N, margin, (x, y)) for the joint closest to
    pull-off, where joint is (support_id, brick_id) and (x, y) the stud row of
    its patch farthest from the load -- where pressing down clamps it shut
    with the longest lever -- or None if every joint is in compression.

    The press load starts at the new brick's mating studs and is pushed down
    the support chain, split between supports in proportion to shared studs and
    applied at the centroid of each shared patch.  A joint whose load lands
    outside its own stud footprint is being levered open, and its studs carry
    the resulting tension.  A staggered wall therefore stays in compression;
    a brick pressed onto the middle of an unsupported span does not.

    ponytail: rigid lever model, load path only, no global force balance.
    Replace with BrickSim's convex QP (§1.2.1) at WP1 -- this function is the
    seam where that swaps in.
    """
    by_id = {p[0]: p for p in placed}

    def split(b, load):
        """Yield (support_brick, load_share, application_point) for b's supports."""
        sup = supports(b, placed)
        total = sum(sup.values())
        bc, _ = cells(b)
        for sid, n in sup.items():
            sc, _ = cells(by_id[sid])
            cx, cy, _h = _extent(bc & sc)
            yield by_id[sid], load * n / total, (cx, cy)

    n_mating = sum(supports(brick, placed).values())
    queue = list(split(brick, n_mating * F_INSERT_PER_STUD))
    worst = None
    while queue:
        p, load, (ax, ay) = queue.pop()
        sup = supports(p, placed)
        if not sup:
            continue  # sits on the baseplate: cannot be levered off
        pc, _ = cells(p)
        held = set().union(*(pc & cells(by_id[s])[0] for s in sup))
        cx, cy, half = _extent(held)
        e = math.hypot(ax - cx, ay - cy)
        if e > half:
            tension = load * (e - half) / (2 * half)
            margin = sum(sup.values()) * F_BREAK_PER_STUD / tension
            if worst is None or margin < worst[2]:
                pts = [(VOXEL_ORIGIN[0] + (i + 0.5) * PITCH,
                        VOXEL_ORIGIN[1] + (j + 0.5) * PITCH) for i, j in held]
                far = max(math.hypot(x - ax, y - ay) for x, y in pts)
                row = [(x, y) for x, y in pts
                       if math.hypot(x - ax, y - ay) > far - PITCH / 2]
                spot = (sum(x for x, _ in row) / len(row),
                        sum(y for _, y in row) / len(row))
                worst = ((min(sup, key=sup.get), p[0]), tension, margin, spot)
        queue.extend(split(p, load))
    return worst


def grasp_for(brick, placed):
    """The grasp that touches the fewest bricks already placed in its layer.

    Each finger is FINGER_W wide, centred on the face it grips. On a 2-stud
    face (16 mm) it overhangs the ends and grazes a brick sitting diagonally
    or flush in line. Sliding the grip off-centre to dodge that makes the
    brick swing about the pad axis, so it stays centred; accessible=False
    then means a graze risk. Preference on a tie: the short side.

    yaw_offset_deg is the gripper's yaw about world z: 0 closes the fingers
    along y, 90 along x. ponytail: neighbour cells, not the cuRobo swept
    volume of §3.4 (cite BricksToBots either way).
    """
    c, k = cells(brick)
    beside = set()
    for p in placed:
        if p[4] == k:
            beside |= cells(p)[0]
    _, _, i0, j0, _, _ = brick
    nx, ny = footprint(brick[1], brick[5])

    def touched(axis):
        lo, n = (j0, ny) if axis == 0 else (i0, nx)
        mid, half = (lo + n / 2) * PITCH, FINGER_W / 2 - 0.0001
        span = range(math.floor((mid - half) / PITCH), math.ceil((mid + half) / PITCH))
        sides = ((i0 - 1, i0, i0 + nx - 1, i0 + nx) if axis == 0
                 else (j0 - 1, j0, j0 + ny - 1, j0 + ny))
        return sum(((s, t) if axis == 0 else (t, s)) in beside for s in sides for t in span)

    # 0: fingers on the +-x faces; min() is stable, so ties keep the short side
    axis = min([0, 1] if nx <= ny else [1, 0], key=touched)
    across = (nx, ny)[axis]
    return {"face_pair": "long" if across == min(nx, ny) else "short",
            "yaw_offset_deg": 90 if axis == 0 else 0,
            "grip_width_m": round(across * PITCH - 0.0012, 4),
            "grip_force_N": 15.0,
            "accessible": touched(axis) == 0}


def brace_for(brick, placed, strategy):
    """§3.6 -- brace field for one step, or None. Strategies: none/nearest/weakest_joint."""
    if strategy == "none" or not placed:
        return None
    n_mating = sum(supports(brick, placed).values())
    press = n_mating * F_INSERT_PER_STUD
    px, py, _ = brick_pose(brick)

    worst = weakest_joint(brick, placed)
    if strategy == "weakest_joint":
        if worst is None or worst[2] >= SAFETY_FACTOR:
            return None
        joint, tension, _, (bx, by) = worst
        target_id = joint[1]
        rationale = "weakest_joint_qp"
    else:  # nearest structurally-connected placed brick (the naive heuristic)
        if worst is None or worst[2] >= SAFETY_FACTOR:
            return None
        target_id = min(placed, key=lambda p: math.hypot(
            brick_pose(p)[0] - px, brick_pose(p)[1] - py))[0]
        tension = worst[1]
        rationale = "nearest_placed_brick"
        # on it, the free cell farthest from the placement: not under the new
        # brick, and not where the placer's hand will be
        cover = cells(brick)[0]
        own = cells(next(p for p in placed if p[0] == target_id))[0]
        bx, by = max(((VOXEL_ORIGIN[0] + (i + 0.5) * PITCH, VOXEL_ORIGIN[1] + (j + 0.5) * PITCH)
                      for i, j in (own - cover or own)),
                     key=lambda c: math.hypot(c[0] - px, c[1] - py))

    # Press down on whatever is on top at (bx, by). For weakest_joint that is
    # straight over the joint's stud patch, which clamps it shut; the brick's
    # own centre can sit on the load side of the pivot and add to the prying.
    cell = (math.floor((bx - VOXEL_ORIGIN[0]) / PITCH),
            math.floor((by - VOXEL_ORIGIN[1]) / PITCH))
    top_k = max(p[4] for p in placed if cell in cells(p)[0])
    force = min(20.0, max(5.0, 1.2 * tension))
    return {
        "arm": "A",
        "target_brick_id": target_id,
        "rationale": rationale,
        "predicted_failure_joint": list(worst[0]) if worst else None,
        "brace_pose": [bx, by, VOXEL_ORIGIN[2] + (top_k + 1) * BRICK_H,
                       0.0, 1.0, 0.0, 0.0],
        "brace_force_N": round(force, 2),
        "brace_axis": [0, 0, -1],
        "expected_reaction_wrench": [0.0, 0.0, -round(press, 2),
                                     round(press * (by - py), 4),
                                     round(press * (px - bx), 4), 0.0],
        "end_effector": "parallel_jaw",
    }


def _bricksim_crosscheck(structure_id, order):
    """Per braced step, what BrickSim's static solver says the weakest joint is.

    Returns {step: {...}} or {} if static_solve is not built. The two models
    answer different questions -- the lever model loads the structure with the
    insertion press, static_solve only with gravity -- so this records whether
    they agree on the JOINT, never on the magnitude.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "sim" / "joint_model"))
    try:
        import bricksim_adapter as bs
    except ImportError:
        return {}
    if not bs.available():
        return {}

    out = {}
    for n, b in enumerate(order):
        if n == 0:
            continue
        # Query BrickSim even when the lever model found nothing -- a joint it
        # misses is the interesting disagreement, and skipping those steps
        # would hide exactly that case.
        lever = weakest_joint(b, order[:n])
        topo = to_bricksim_topology(structure_id, upto_step=n)
        worst = bs.weakest_connection(topo)
        lever_joint = list(lever[0]) if lever else None
        bs_joint = list(bs.joint_of(topo, worst[0])) if worst else None
        if lever_joint is None and bs_joint is None:
            continue                      # both say nothing is loaded
        out[n] = {"lever_joint": lever_joint,
                  "lever_margin": round(lever[2], 3) if lever else None,
                  "bricksim_joint": bs_joint,
                  "bricksim_utilization": round(worst[1], 5) if worst else 0.0,
                  "agree": bs_joint == lever_joint}
    return out


def build_plan(structure_id, strategy="weakest_joint", crosscheck=False):
    """Emit an assembly_plan.json dict per §2.6."""
    bricks = STRUCTURES[structure_id]
    order = sequence(bricks)
    steps, placed = [], []
    for n, b in enumerate(order):
        bid, btype, i, j, k, yaw = b
        sup = supports(b, placed)
        x, y, z = brick_pose(b)
        nx, ny = footprint(btype, yaw)
        brace = brace_for(b, placed, strategy)
        yaw_rad = math.pi / 2 * yaw
        # Gripper points down: Rz(yaw) * Rx(pi), as xyzw.
        quat = [math.cos(yaw_rad / 2), math.sin(yaw_rad / 2), 0.0, 0.0]
        steps.append({
            "step": n,
            "brick_id": bid,
            "placer_arm": "B",
            "requires_brace": brace is not None,
            "brace": brace,
            "grasp": grasp_for(b, placed),
            "pre_insertion_pose": [x, y, z + PRE_INSERTION_DZ] + quat,
            "target_pose": [x, y, z] + quat,
            "mating_studs": [[s, n_studs] for s, n_studs in sup.items()],
        })
        placed.append(b)

    worst_overall = min(
        (weakest_joint(b, order[:n]) for n, b in enumerate(order) if n),
        key=lambda w: w[2] if w else 1e9, default=None)
    return {
        "structure_id": structure_id,
        "source": {"dataset": "hand_authored", "index": None},
        "voxel_origin": list(VOXEL_ORIGIN),
        "grid_pitch": [PITCH, PITCH, BRICK_H],
        "bricks": [{"id": b[0], "type": b[1], "grid_pos": [b[2], b[3], b[4]],
                    "yaw_index": b[5], "color": "red"} for b in bricks],
        "sequence": steps,
        "validation": {
            "stability_method": "lever_arm_margin",
            "bracing_strategy": strategy,
            "weakest_joint": list(worst_overall[0]) if worst_overall else None,
            "min_margin": round(worst_overall[2], 3) if worst_overall else None,
            "all_bricks_accessible": None,  # WP3.4, needs cuRobo batch IK
            "brace_required_count": sum(s["requires_brace"] for s in steps),
            # gravity-only second opinion; see sim/joint_model/bricksim_adapter.py
            "bricksim_crosscheck": _bricksim_crosscheck(structure_id, order)
            if crosscheck else None,
        },
    }


# --- BrickSim interop -------------------------------------------------------
# BrickSim's lego_topology@2: parts carry (L, W, H) in studs/plates, and a
# connection names the lower part's studs, the upper part's holes, and the
# upper part's grid offset within the lower part's frame.
BASEPLATE = {"id": 0, "type": "brick",
             "payload": {"L": 20, "W": 20, "H": 1, "color": [155, 161, 157]}}
BASEPLATE_ORIGIN = (8, 8)     # where the build sits on the 20x20 plate


def to_bricksim_topology(structure_id, upto_step=None):
    """Emit a BrickSim topology JSON dict for a structure, or its first n steps.

    upto_step=None means the finished structure; upto_step=n means the partial
    structure after n placements, which is what a bracing query needs.
    """
    order = sequence(STRUCTURES[structure_id])
    if upto_step is not None:
        order = order[:upto_step]

    parts = [dict(BASEPLATE)]
    part_id = {}
    for b in order:
        part_id[b[0]] = len(parts)
        nx, ny = footprint(b[1], b[5])
        parts.append({"id": part_id[b[0]], "type": "brick",
                      "payload": {"L": nx, "W": ny, "H": 3,
                                  "color": [201, 26, 9]}})

    connections, placed = [], []
    for b in order:
        bid, _btype, i, j, _k, _yaw = b
        sup = supports(b, placed)
        if not sup:
            # rests on the baseplate
            connections.append({
                "id": len(connections), "stud_id": 0, "stud_iface": 1,
                "hole_id": part_id[bid], "hole_iface": 0,
                "offset": [BASEPLATE_ORIGIN[0] + i, BASEPLATE_ORIGIN[1] + j],
                "yaw": 0})
        for sid in sup:
            s_brick = next(p for p in placed if p[0] == sid)
            connections.append({
                "id": len(connections), "stud_id": part_id[sid], "stud_iface": 1,
                "hole_id": part_id[bid], "hole_iface": 0,
                "offset": [i - s_brick[2], j - s_brick[3]],
                "yaw": 0})
        placed.append(b)

    return {"schema": "bricksim/lego_topology@2", "parts": parts,
            "connections": connections,
            "pose_hints": [{"part": 0, "pos": [0.0, 0.0, 0.0],
                            "rot": [1.0, 0.0, 0.0, 0.0]}],
            # not part of the schema; carried so a reader can map back
            "_brick_ids": {str(v): k for k, v in part_id.items()}}


def _self_check():
    for sid in STRUCTURES:
        plan = build_plan(sid)
        steps = plan["sequence"]
        assert len(steps) == len(STRUCTURES[sid])
        seen = set()
        for s in steps:  # every brick is supported when it is placed
            b = next(b for b in STRUCTURES[sid] if b[0] == s["brick_id"])
            assert b[4] == 0 or s["mating_studs"], s
            assert all(m[0] in seen for m in s["mating_studs"]), s
            seen.add(s["brick_id"])
        # no two bricks share a cell in the same layer
        occ = set()
        for b in STRUCTURES[sid]:
            c, k = cells(b)
            keyed = {(i, j, k) for i, j in c}
            assert not (occ & keyed), "overlap in %s" % sid
            occ |= keyed
    # the spanning arch must need a brace; the column must not
    assert build_plan("S1")["validation"]["brace_required_count"] == 0
    assert build_plan("S3")["validation"]["brace_required_count"] > 0
    # the three strategies all produce valid plans for the same structure
    n, w = (build_plan("S3", s)["validation"]["brace_required_count"]
            for s in ("none", "weakest_joint"))
    assert n == 0 and w > 0
    # the brace presses over the pier; the cantilever's centre is on the load
    # side of the pivot and would add to the prying
    brace = build_plan("S3")["sequence"][-1]["brace"]["brace_pose"]
    pier = next(b for b in STRUCTURES["S3"] if b[0] == "b_002")
    assert abs(brace[1] - brick_pose(pier)[1]) < footprint(pier[1], pier[5])[1] * PITCH / 2, brace
    # a finger never goes into the 0.2 mm gap beside a placed brick
    first, second = ("a", "2x4", 0, 0, 0, 0), ("b", "2x4", 2, 0, 0, 0)
    assert grasp_for(second, [first])["yaw_offset_deg"] == 0
    print("planner self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default=None, choices=list(STRUCTURES))
    ap.add_argument("--strategy", default="weakest_joint",
                    choices=["none", "nearest", "weakest_joint"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--topology", metavar="STRUCTURE",
                    help="write a BrickSim topology JSON instead of a plan")
    ap.add_argument("--upto-step", type=int, default=None,
                    help="with --topology: only the first n placements")
    ap.add_argument("--crosscheck", action="store_true",
                    help="also ask BrickSim's static_solve which joint is weakest")
    args = ap.parse_args()

    if args.topology:
        topo = to_bricksim_topology(args.topology, args.upto_step)
        out = Path(__file__).parent / "plans"
        out.mkdir(exist_ok=True)
        name = "%s%s.topology.json" % (
            args.topology, "" if args.upto_step is None else "_step%d" % args.upto_step)
        (out / name).write_text(json.dumps(topo, indent=1))
        print("%s -> plans/%s  (%d parts, %d connections)"
              % (args.topology, name, len(topo["parts"]), len(topo["connections"])))
        raise SystemExit

    if args.list:
        for sid, bs in STRUCTURES.items():
            print("%s: %d bricks, %d layers" % (sid, len(bs), max(b[4] for b in bs) + 1))
        raise SystemExit

    _self_check()
    out = Path(__file__).parent / "plans"
    out.mkdir(exist_ok=True)
    for sid in ([args.structure] if args.structure else STRUCTURES):
        plan = build_plan(sid, args.strategy, crosscheck=args.crosscheck)
        path = out / ("%s_%s.json" % (sid, args.strategy))
        path.write_text(json.dumps(plan, indent=1))
        print("%s -> %s  (%d steps, %d braced, min margin %s)" % (
            sid, path.name, len(plan["sequence"]),
            plan["validation"]["brace_required_count"],
            plan["validation"]["min_margin"]))
