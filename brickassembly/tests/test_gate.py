"""G1 -- assembly gate boundaries and scripted insertion (master_report §2.3.3).

Two of G1's open criteria:

  * "Gate fires inside tolerance and not outside -- 6 boundary cases per gate
    condition". Run here for the two conditions this project actually depends
    on: lateral offset and yaw error.
  * "A scripted top-down insertion snaps reliably from <= 2 mm initial error
    and holds 30 s."

Unlike the pull-off tests these do not eject bricks at speed, so the cases can
share one process; a repeat of the first case runs last as a drift control.

    cd ~/Documents/ISDN_Robofab/BrickSim
    OMNI_KIT_ACCEPT_EULA=1 .venv/bin/python \\
        ../brickassembly/tests/test_gate.py --headless
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
parser.add_argument("--press-n", type=float, default=15.0,
                    help="downward press; must exceed the gate's required_force")
parser.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "test_gate.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

from bricksim.assets.stages import DEFAULT_STAGE_PATH  # noqa: E402
from bricksim.core import (  # noqa: E402
    AssemblyThresholds,
    BreakageThresholds,
    are_parts_connected,
    import_lego,
    set_assembly_thresholds,
    set_breakage_thresholds,
)

CALIBRATION = Path(__file__).resolve().parents[1] / "joint_calibration.json"
BRICK_H = 0.0096
PRESS_STEPS = 240
SETTLE_STEPS = 30

# §2.3.3 as the report specifies it.
GATE_LATERAL_M = 0.0012
GATE_YAW_DEG = 5.0

# Six cases per condition, straddling the tolerance.
LATERAL_CASES_MM = [0.0, 0.6, 1.0, 1.15, 1.5, 2.5]
YAW_CASES_DEG = [0.0, 2.0, 4.0, 4.8, 6.0, 12.0]


def base_topology(with_upper):
    parts = [
        {"id": 0, "type": "brick",
         "payload": {"L": 20, "W": 20, "H": 1, "color": [155, 161, 157]}},
        {"id": 1, "type": "brick",
         "payload": {"L": 2, "W": 4, "H": 3, "color": [201, 26, 9]}},
    ]
    conns = [{"id": 0, "stud_id": 0, "stud_iface": 1, "hole_id": 1,
              "hole_iface": 0, "offset": [8, 8], "yaw": 0}]
    if with_upper:
        parts.append({"id": 2, "type": "brick",
                      "payload": {"L": 2, "W": 4, "H": 3, "color": [0, 85, 191]}})
    return {"schema": "bricksim/lego_topology@2", "parts": parts,
            "connections": conns,
            "pose_hints": [{"part": 0, "pos": [0.0, 0.0, 0.0],
                            "rot": [1.0, 0.0, 0.0, 0.0]}]}


def yaw_quat(deg):
    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def run_case(world, slot, dx_m=0.0, dyaw_deg=0.0, hold_steps=0):
    """Drop an offset brick onto a mated one, press, and report whether it snapped."""
    parts, _ = import_lego(json=base_topology(with_upper=True), env_id=-1,
                           ref_pos=[0.25 + 0.2 * slot, -0.55, 0.0],
                           ref_rot=(1.0, 0.0, 0.0, 0.0))
    lower, upper = parts[1], parts[2]

    world.reset()
    for _ in range(SETTLE_STEPS):
        world.step(render=False)

    # The pose must be written through the PHYSICS view, not a SingleRigidPrim:
    # a USD-side write on a simulating body does not move it, which silently
    # left every brick where it was imported and made all six lateral cases
    # read "no snap".
    view = RigidPrim(prim_paths_expr=upper)
    view.initialize()
    low = SingleRigidPrim(prim_path=lower)
    lp, _lq = low.get_world_pose()
    target = np.array([[lp[0] + dx_m, lp[1], lp[2] + BRICK_H + 0.0015]])
    view.set_world_poses(positions=target,
                         orientations=np.array([yaw_quat(dyaw_deg)]))
    view.set_velocities(np.zeros((1, 6)))
    world.step(render=False)

    placed = view.get_world_poses()[0][0]
    gap_mm = (float(placed[2]) - float(lp[2]) - BRICK_H) * 1000.0

    snapped = False
    for _ in range(PRESS_STEPS):
        view.apply_forces(forces=np.array([[0.0, 0.0, -args_cli.press_n]]),
                          is_global=True)
        world.step(render=False)
        if are_parts_connected(lower, upper):
            snapped = True
            break
    held = None
    if snapped and hold_steps:
        for _ in range(hold_steps):
            world.step(render=False)
        held = bool(are_parts_connected(lower, upper))
    return snapped, held, gap_mm


def main():
    if World._world_initialized:
        World.clear_instance()
    open_stage(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    world.reset()

    cal = json.loads(CALIBRATION.read_text())
    bt = BreakageThresholds()
    bt.preloaded_force = cal["bricksim_settings"]["live_sim"]["preloaded_force"]
    set_breakage_thresholds(bt)

    thr = AssemblyThresholds()
    thr.distance_tolerance = 0.001
    thr.max_penetration = 0.005
    thr.z_angle_tolerance = 4.0 * (math.pi / 180.0)
    thr.required_force = cal["bricksim_settings"]["required_force"]
    thr.yaw_tolerance = math.radians(GATE_YAW_DEG)
    thr.position_tolerance = GATE_LATERAL_M
    set_assembly_thresholds(thr)

    results = {"press_n": args_cli.press_n,
               "gate": {"lateral_mm": GATE_LATERAL_M * 1000.0,
                        "yaw_deg": GATE_YAW_DEG,
                        "required_force_n": thr.required_force},
               "lateral": [], "yaw": []}
    slot = 0

    print("-- lateral offset (gate %.2f mm) --" % (GATE_LATERAL_M * 1000.0), flush=True)
    for mm in LATERAL_CASES_MM:
        snapped, _, gap = run_case(world, slot, dx_m=mm / 1000.0)
        slot += 1
        expected = mm < GATE_LATERAL_M * 1000.0
        results["lateral"].append({"offset_mm": mm, "snapped": snapped,
                                   "expected": expected,
                                   "agrees": snapped == expected,
                                   "start_gap_mm": round(gap, 3)})
        print("   %5.2f mm -> snapped %-5s (gate says %s, start gap %.2f mm)"
              % (mm, snapped, expected, gap), flush=True)

    print("-- yaw error (gate %.1f deg) --" % GATE_YAW_DEG, flush=True)
    for deg in YAW_CASES_DEG:
        snapped, _, gap = run_case(world, slot, dyaw_deg=deg)
        slot += 1
        expected = deg < GATE_YAW_DEG
        results["yaw"].append({"yaw_deg": deg, "snapped": snapped,
                               "expected": expected,
                               "agrees": snapped == expected,
                               "start_gap_mm": round(gap, 3)})
        print("   %5.1f deg -> snapped %-5s (gate says %s, start gap %.2f mm)"
              % (deg, snapped, expected, gap), flush=True)

    # Scripted insertion from 2 mm, then hold 30 s at 60 Hz.
    snapped, held, _ = run_case(world, slot, dx_m=0.002, hold_steps=1800)
    slot += 1
    results["insertion_2mm"] = {"snapped": snapped, "held_30s": held}
    print("-- scripted insertion from 2.0 mm -> snapped %s, held 30 s %s"
          % (snapped, held), flush=True)

    # Drift control: repeat the first case last. If the answer moved, the
    # shared process is contaminating results and the table above is suspect.
    ctrl, _, _ = run_case(world, slot, dx_m=0.0)
    results["drift_control"] = {"case": "lateral 0.0 mm repeated last",
                                "snapped": ctrl,
                                "agrees_with_first": ctrl == results["lateral"][0]["snapped"]}
    print("-- drift control: %s (first run gave %s)"
          % (ctrl, results["lateral"][0]["snapped"]), flush=True)

    lateral_ok = all(c["agrees"] for c in results["lateral"])
    yaw_ok = all(c["agrees"] for c in results["yaw"])
    results["verdict"] = {
        "gate_boundary_lateral": lateral_ok,
        "gate_boundary_yaw": yaw_ok,
        "scripted_insertion_2mm": bool(snapped and held),
        "control_clean": results["drift_control"]["agrees_with_first"],
    }
    args_cli.out.write_text(json.dumps(results, indent=1))
    print("\n-- verdict --", flush=True)
    for k, v in results["verdict"].items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k), flush=True)
    print("report -> %s" % args_cli.out, flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
