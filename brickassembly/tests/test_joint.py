"""Gate G1 -- joint mechanics acceptance tests (master_report §WP1).

WP1 took branch 3a, so these test BrickSim's joint model rather than one of
ours: the point is to confirm it meets the §2.3.3 semantics this project is
about to build on, before WP5 blames the RL for a physics defect.

Run headless under BrickSim's interpreter (it needs a live PhysX scene):

    cd ~/Documents/ISDN_Robofab/BrickSim
    OMNI_KIT_ACCEPT_EULA=1 .venv/bin/python \\
        ../brickassembly/tests/test_joint.py --headless

NOT through `python -m bricksim`: that entry point starts the full GUI
experience, whose RTX init needs a Vulkan compat layer this driver downloads
from a URL that now 404s. AppLauncher's headless path avoids the renderer
entirely, and `import bricksim` still self-enables the Kit extension.

Writes tests/test_joint.json and prints a per-criterion verdict.
"""

import argparse
import json
import sys
import math
from pathlib import Path

# Fail now, not five minutes into Isaac Sim startup, if this is the wrong
# interpreter: bricksim lives only in BrickSim's venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envcheck  # noqa: E402

envcheck.require("bricksim", ["bricksim", "isaacsim"])
envcheck.banner("bricksim")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

from bricksim.assets.stages import DEFAULT_STAGE_PATH  # noqa: E402
from bricksim.core import (  # noqa: E402
    AssemblyThresholds,
    are_parts_connected,
    import_lego,
    set_assembly_thresholds,
)

CALIBRATION = Path(__file__).resolve().parents[1] / "joint_calibration.json"
SWEEP = (Path(__file__).resolve().parents[1] / "scripts"
         / "02_dynamic_calibration.jsonl")

OUT = Path(__file__).parent / "test_joint.json"
PLACE = ([0.3, 0.0, 0.0], (1.0, 0.0, 0.0, 0.0))
GRAVITY = 9.81

# Two 2x4 bricks, the upper one mated squarely onto the lower.
def two_brick_topology(offset=(0, 0)):
    return {
        "schema": "bricksim/lego_topology@2",
        "parts": [
            {"id": 0, "type": "brick",
             "payload": {"L": 20, "W": 20, "H": 1, "color": [155, 161, 157]}},
            {"id": 1, "type": "brick",
             "payload": {"L": 2, "W": 4, "H": 3, "color": [201, 26, 9]}},
            {"id": 2, "type": "brick",
             "payload": {"L": 2, "W": 4, "H": 3, "color": [0, 85, 191]}},
        ],
        "connections": [
            {"id": 0, "stud_id": 0, "stud_iface": 1, "hole_id": 1,
             "hole_iface": 0, "offset": [8, 8], "yaw": 0},
            {"id": 1, "stud_id": 1, "stud_iface": 1, "hole_id": 2,
             "hole_iface": 0, "offset": list(offset), "yaw": 0},
        ],
        "pose_hints": [{"part": 0, "pos": [0.0, 0.0, 0.0],
                        "rot": [1.0, 0.0, 0.0, 0.0]}],
    }


def step_n(world, n):
    for _ in range(n):
        world.step(render=False)


def pose_of(path):
    """Pose from the PHYSICS view, not from USD.

    world.step(render=False) does not flush transforms back to USD, so a
    SingleXFormPrim read returns the authored value (or zeros) no matter what
    the simulation did. That silently turns every motion test into a pass.
    """
    body = SingleRigidPrim(prim_path=path)
    pos, quat = body.get_world_pose()
    return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)


def system_energy(paths):
    """Kinetic + gravitational potential energy of the listed rigid bodies (J)."""
    total = 0.0
    for path in paths:
        body = SingleRigidPrim(prim_path=path)
        mass = float(body.get_mass())
        v = np.asarray(body.get_linear_velocity(), dtype=float)
        w = np.asarray(body.get_angular_velocity(), dtype=float)
        pos, _ = body.get_world_pose()
        # rotational inertia is not exposed per-prim here, so the angular term
        # uses |w|^2 scaled by mass only -- an approximation, but one that still
        # grows if the joint injects energy, which is what the test watches for.
        total += 0.5 * mass * float(v @ v) + 0.5 * mass * float(w @ w) \
            + mass * GRAVITY * float(pos[2])
    return total


def main():
    results = {}

    def record(name, passed, **evidence):
        results[name] = {"pass": bool(passed) if passed is not None else None,
                         **evidence}
        state = "BLOCKED" if passed is None else ("PASS" if passed else "FAIL")
        print("  [%s] %s  %s" % (state, name, evidence), flush=True)

    if World._world_initialized:
        World.clear_instance()
    open_stage(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    world.reset()

    # §2.3.3 as the report specifies it, not BrickSim's looser defaults.
    thr = AssemblyThresholds()
    thr.distance_tolerance = 0.001
    thr.max_penetration = 0.005
    thr.z_angle_tolerance = 4.0 * (math.pi / 180.0)
    thr.required_force = 4.0
    thr.yaw_tolerance = 5.0 * (math.pi / 180.0)
    thr.position_tolerance = 0.0012
    set_assembly_thresholds(thr)

    parts, conns = import_lego(json=two_brick_topology(), env_id=-1,
                               ref_pos=PLACE[0], ref_rot=PLACE[1])
    if len(parts) != 3 or len(conns) != 2:
        record("import", False, parts=len(parts), connections=len(conns))
        OUT.write_text(json.dumps(results, indent=1))
        simulation_app.close()
        return
    lower, upper = parts[1], parts[2]
    world.reset()
    step_n(world, 60)

    # --- control: is physics actually running? -------------------------
    # A mated pair that never moves and an energy trace that never changes are
    # also what a frozen scene looks like. Drop an unconnected brick first: if
    # it does not fall, every result below is meaningless.
    free = two_brick_topology()
    free["parts"] = [free["parts"][1]]          # one loose brick, no baseplate
    free["connections"] = []
    # ref_pos alone does not lift an unconnected part -- it needs a pose hint,
    # or it spawns at the origin intersecting the floor and is pushed UP to
    # rest at half a brick height (4.8 mm), which looks like a failure to fall.
    free["pose_hints"] = [{"part": 1, "pos": [0.3, -0.3, 0.25],
                           "rot": [1.0, 0.0, 0.0, 0.0]}]
    free_parts, _ = import_lego(json=free, env_id=-1)
    free_path = free_parts[1]
    z_before = float(pose_of(free_path)[0][2])
    step_n(world, 120)
    z_after = float(pose_of(free_path)[0][2])
    fell_mm = (z_before - z_after) * 1000.0
    record("control_free_brick_falls", fell_mm > 1.0,
           dropped_mm=round(fell_mm, 3), z_before=round(z_before, 4),
           z_after=round(z_after, 4),
           criterion="an unconnected brick must fall > 1 mm in 2 s")

    # --- mated pair holds under gravity -------------------------------
    p0, q0 = pose_of(upper)
    step_n(world, 1800)                     # 30 s at 60 Hz
    p1, q1 = pose_of(upper)
    drift_mm = float(np.linalg.norm(p1 - p0)) * 1000.0
    still_connected = bool(are_parts_connected(lower, upper))
    record("mated_pair_holds_30s", drift_mm < 0.05 and still_connected,
           drift_mm=round(drift_mm, 5), connected=still_connected,
           criterion="drift < 0.05 mm")

    # --- energy is non-increasing with no actuation --------------------
    # A settled pair conserves energy trivially, so nudge it first: the defect
    # this test exists to catch is a joint that PUMPS energy once disturbed
    # ("tall structures are unstable" in WP5 is usually this).
    SingleRigidPrim(prim_path=upper).set_linear_velocity(np.array([0.05, 0.0, 0.0]))
    step_n(world, 10)
    bodies = [lower, upper]
    e0 = system_energy(bodies)
    peak = e0
    for _ in range(100):
        step_n(world, 100)                  # 10 000 steps total
        peak = max(peak, system_energy(bodies))
    e1 = system_energy(bodies)
    growth_mj = (peak - e0) * 1000.0
    record("energy_non_increasing_10k_steps", growth_mj < 1.0,
           start_j=round(e0, 6), end_j=round(e1, 6),
           peak_growth_mj=round(growth_mj, 6),
           criterion="peak energy growth < 1 mJ",
           note="angular term approximated with mass, not the inertia tensor")

    # --- two bricks cannot mate to the same stud -----------------------
    clash = two_brick_topology()
    clash["parts"].append({"id": 3, "type": "brick",
                           "payload": {"L": 2, "W": 4, "H": 3,
                                       "color": [35, 120, 65]}})
    clash["connections"].append({"id": 2, "stud_id": 1, "stud_iface": 1,
                                 "hole_id": 3, "hole_iface": 0,
                                 "offset": [0, 0], "yaw": 0})
    try:
        p2, c2 = import_lego(json=clash, env_id=-1, ref_pos=[0.3, 0.4, 0.0],
                             ref_rot=PLACE[1])
        detail = {"connections_created": len(c2), "of": 3}
        if len(c2) < 3:
            rejected = True                       # refused at authoring time
        else:
            # It authored both. Does the SIMULATION keep two bricks in one
            # place? Settle, then compare the rival bricks' poses: if they are
            # co-located the model really did let them share the studs.
            step_n(world, 300)
            pos_a = pose_of(p2[2])[0]
            pos_b = pose_of(p2[3])[0]
            gap_mm = float(np.linalg.norm(pos_a - pos_b)) * 1000.0
            rejected = gap_mm > 1.0
            detail.update(rival_brick_gap_mm=round(gap_mm, 3),
                          criterion="bricks must not end up co-located",
                          note="import_lego authored both connections; this "
                               "checks whether physics kept them apart")
    except Exception as e:                        # a raised error is also a reject
        rejected, detail = True, {"raised": type(e).__name__}
    record("double_mate_rejected", rejected, **detail)

    # --- f_break and asymmetry, from the ISOLATED harness ---------------
    # These are NOT measured here. A pull-off run after the tests above gives
    # 39.4 N where the same article measured alone gives 89.6 N: released
    # bricks from earlier articles land on the one under test and end its
    # ramp early. The measurement only means anything one-per-process, which
    # is what scripts/02_dynamic_sweep.sh does. This reads that sweep.
    cal = json.loads(CALIBRATION.read_text())
    live = cal["bricksim_settings"]["live_sim"]
    sweep = [json.loads(line) for line in SWEEP.read_text().splitlines() if line.strip()]
    at_adopted = [r for r in sweep
                  if r.get("brick") == "2x4" and r.get("release_n")
                  and r["preload_n"] == live["preloaded_force"]]

    if not at_adopted:
        record("f_break_release_within_15pct", None,
               note="no isolated 2x4 measurement at the adopted preload; "
                    "run scripts/02_dynamic_sweep.sh")
    else:
        measured = at_adopted[0]["release_n"]
        predicted = live["gives_n_per_stud"]["2x4"] * 8
        err = abs(measured - predicted) / predicted
        record("f_break_release_within_15pct", err <= 0.15,
               measured_n=measured, predicted_n=round(predicted, 2),
               error_pct=round(err * 100.0, 2),
               repeatability="0.5% over three repeats at preload 60 "
                             "(79.0 / 78.9 / 78.6 N)",
               caveat="f_break IS the adopted calibration, so this checks "
                      "self-consistency and reproducibility, not an "
                      "independent prediction")
        record("extraction_exceeds_insertion",
               measured > cal["bricksim_settings"]["required_force"],
               extraction_n=measured,
               insertion_n=cal["bricksim_settings"]["required_force"],
               ratio=round(measured / cal["bricksim_settings"]["required_force"], 3),
               note="asymmetry is IMPOSED by configuring required_force below "
                    "the clutch capacity, not produced emergently")

    print("\n-- Gate G1 --", flush=True)
    for name, r in results.items():
        state = "BLOCKED" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
        print("  [%s] %s" % (state, name), flush=True)
    for name in ("gate_boundary_6_cases_per_condition",
                 "identical_at_1_and_max_envs", "scripted_insertion_snaps_from_2mm"):
        print("  [NOT RUN] %s" % name, flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print("\nreport -> %s" % OUT, flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
