"""Open a recorded run's USD in Isaac Sim 6.0, which renders on this machine.

Must be the Isaac Sim 6.0 interpreter: BrickSim's 5.1 segfaults in
librtx.scenedb.plugin.so on this GPU (ledger:
rtx_crash_is_isaac_sim_51_not_the_driver).

    ~/Codes/CAIRSS/Issac/bin/python viz/open_usd.py results/s1_run.usda

A window opens with the animation loaded; use the timeline to scrub. Add
--frames N to step without a window instead (useful to check the file loads
before opening a viewport).
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
parser.add_argument("--frames", type=int, default=0,
                    help="step this many frames then exit (0 = stay open)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim 6.0 launches HEADLESS unless a visualizer is named: --headless is
# deprecated and omitting --viz means no window. Without this the script ran
# to completion and printed "window open" while showing nothing at all.
if args_cli.visualizer is None and not args_cli.frames:
    args_cli.visualizer = ["kit"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envcheck  # noqa: E402

envcheck.require("isaaclab", ["isaacsim", "isaaclab"])
envcheck.banner("isaaclab")
envcheck.rtx_compat()

simulation_app = AppLauncher(args_cli).app

# omni.usd is core Kit and always present; isaacsim.core.utils is an
# extension that is not loaded in this app configuration.
import omni.usd  # noqa: E402


def main():
    if not args_cli.usd.exists():
        raise SystemExit("no such file: %s" % args_cli.usd)
    ctx = omni.usd.get_context()
    ctx.open_stage(str(args_cli.usd.resolve()))
    for _ in range(60):                      # let the stage finish loading
        simulation_app.update()
    stage = ctx.get_stage()
    prims = len(list(stage.Traverse())) if stage else 0
    print("opened %s (%d prims)" % (args_cli.usd, prims), flush=True)

    if args_cli.frames:
        for _ in range(args_cli.frames):
            simulation_app.update()
        print("stepped %d frames; the stage loads cleanly" % args_cli.frames,
              flush=True)
    else:
        print("window open -- use the timeline to scrub; ctrl-c to exit",
              flush=True)
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
