"""WP2b -- calibrate against the LIVE sim. ONE measurement per process.

WP2 fitted PreloadedForce against static_solve's QP and got 11.3 N/stud. The
dynamic path then released a 2x4 full lap at 36.75 N where that fit predicts
90.4 N. WP5 trains in the dynamic path, so the dynamic number is the one that
has to be right.

Earlier in-process sweeps were contaminated: a released brick leaves at speed
(tens of newtons on a few grams), lands on the next article, and ends its
measurement early -- which showed up as non-monotonic sweeps and an isolated
14.7 N reading among repeats of 79.0 / 78.9 / 78.6 N. Tearing the article down
with deallocate_all_managed() then crashed the physics view with "Failed to get
rigid body velocities from backend". So each process now performs exactly ONE
measurement in a fresh scene and exits.

    bash scripts/02_dynamic_sweep.sh              # the sweep
    # or a single point:
    cd ~/Documents/ISDN_Robofab/BrickSim
    OMNI_KIT_ACCEPT_EULA=1 .venv/bin/python \\
        ../brickassembly/scripts/02_dynamic_calibration.py --headless \\
        --ny 4 --preload 60 --ramp 0.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Fail now, not five minutes into Isaac Sim startup, if this is the wrong
# interpreter: bricksim lives only in BrickSim's venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envcheck  # noqa: E402

envcheck.require("bricksim", ["bricksim", "isaacsim"])
envcheck.banner("bricksim")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ny", type=int, default=4, help="article is a 2 x ny full lap")
parser.add_argument("--preload", type=float, default=60.0)
parser.add_argument("--ramp", type=float, default=0.1,
                    help="N added per physics step; above 0.25 is not quasi-static")
parser.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "02_dynamic_calibration.jsonl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

from bricksim.assets.stages import DEFAULT_STAGE_PATH  # noqa: E402
from bricksim.core import (  # noqa: E402
    BreakageThresholds,
    get_disassembled_connections,
    import_lego,
    set_breakage_thresholds,
)

SETTLE_STEPS = 60
MAX_RAMP_STEPS = 6000


def pair_topology(nx, ny):
    """Baseplate + a brick + an identical brick lapped squarely on top."""
    return {
        "schema": "bricksim/lego_topology@2",
        "parts": [
            {"id": 0, "type": "brick",
             "payload": {"L": 20, "W": 20, "H": 1, "color": [155, 161, 157]}},
            {"id": 1, "type": "brick",
             "payload": {"L": nx, "W": ny, "H": 3, "color": [201, 26, 9]}},
            {"id": 2, "type": "brick",
             "payload": {"L": nx, "W": ny, "H": 3, "color": [0, 85, 191]}},
        ],
        "connections": [
            {"id": 0, "stud_id": 0, "stud_iface": 1, "hole_id": 1,
             "hole_iface": 0, "offset": [8, 8], "yaw": 0},
            {"id": 1, "stud_id": 1, "stud_iface": 1, "hole_id": 2,
             "hole_iface": 0, "offset": [0, 0], "yaw": 0},
        ],
        "pose_hints": [{"part": 0, "pos": [0.0, 0.0, 0.0],
                        "rot": [1.0, 0.0, 0.0, 0.0]}],
    }


def main():
    if World._world_initialized:
        World.clear_instance()
    open_stage(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    world.reset()

    bt = BreakageThresholds()
    bt.preloaded_force = args_cli.preload
    set_breakage_thresholds(bt)

    parts, _ = import_lego(json=pair_topology(2, args_cli.ny), env_id=-1,
                           ref_pos=[0.3, -0.6, 0.0], ref_rot=(1.0, 0.0, 0.0, 0.0))
    view = RigidPrim(prim_paths_expr=parts[2])
    view.initialize()
    world.reset()
    for _ in range(SETTLE_STEPS):
        world.step(render=False)
    get_disassembled_connections(clear=True)      # discard settling events

    mine = {parts[1], parts[2]}
    studs = 2 * args_cli.ny
    force, released, steps = 0.0, None, 0
    t0 = time.time()
    for steps in range(1, MAX_RAMP_STEPS + 1):
        force += args_cli.ramp
        view.apply_forces(forces=np.array([[0.0, 0.0, force]]), is_global=True)
        world.step(render=False)
        if any(i.stud_path in mine and i.hole_path in mine
               for i in get_disassembled_connections(clear=True)):
            released = force
            break

    row = {"brick": "2x%d" % args_cli.ny, "studs": studs,
           "preload_n": args_cli.preload, "ramp_n_per_step": args_cli.ramp,
           "release_n": round(released, 4) if released else None,
           "per_stud_n": round(released / studs, 4) if released else None,
           "ramp_steps": steps, "wall_s": round(time.time() - t0, 1),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with args_cli.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("RESULT 2x%d preload %.3f ramp %.3f -> release %s N (%s N/stud)"
          % (args_cli.ny, args_cli.preload, args_cli.ramp,
             row["release_n"], row["per_stud_n"]), flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
