"""WP1 batching probe (master_report §WP1 step 2) -- does BrickSim batch?

Steps BrickSim's own assemble_brick environment at several env counts and
records wall-clock throughput plus BrickSim's internal per-step solve time.
The branch this decides:

    3a  batches acceptably (>= 512 envs at usable throughput) -> write
        sim/joint_model/bricksim_adapter.py; contribution (2) is NOT claimed.
    3b  does not batch -> write sim/joint_model/warp_kernel.py and validate it
        against BrickSim on single-env cases; contribution (2) IS claimed.

Run under BrickSim's own interpreter -- it pins Isaac Sim 5.1 and python 3.11,
neither of which matches this project's Isaac env:

    bash brickassembly/scripts/01_bricksim_sweep.sh
"""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "01_bricksim_probe.jsonl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# App first, everything else after -- the order bricksim/cli/assemble_brick_expert.py uses.
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import bricksim.core as core  # noqa: E402
from bricksim.envs.assemble_brick.env import AssembleBrickEnvCfg  # noqa: E402
from bricksim.envs.assemble_brick.expert import AssembleBrickExpert  # noqa: E402


def probe(n_envs):
    """Time `steps` expert-driven steps at n_envs. Returns a result dict."""
    cfg = AssembleBrickEnvCfg()
    cfg.scene.num_envs = n_envs
    # BrickSim's own CLI sets render_interval=1 because it saves camera frames.
    # For a throughput probe that measures the renderer, not the solver -- render
    # once per environment step instead.
    cfg.sim.render_interval = cfg.decimation
    cfg.seed = 0

    t0 = time.time()
    env = ManagerBasedRLEnv(cfg=cfg)
    expert = AssembleBrickExpert()
    obs, _ = env.reset()
    startup_s = time.time() - t0

    solve_ms = []
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = expert.compute_actions(obs["privileged"])
            obs, _, _, _, _ = env.step(actions)
            try:
                solve_ms.append(core.get_last_step_profiling().step_time * 1000.0)
            except Exception:
                pass
    wall_s = time.time() - t0
    env.close()

    solve_ms.sort()
    return {
        "n_envs": n_envs,
        "startup_s": round(startup_s, 1),
        "steps": args_cli.steps,
        "wall_s": round(wall_s, 2),
        "steps_per_s": round(args_cli.steps / wall_s, 1),
        "env_steps_per_s": round(n_envs * args_cli.steps / wall_s, 1),
        "bricksim_solve_ms_median": round(solve_ms[len(solve_ms) // 2], 3) if solve_ms else None,
        "bricksim_solve_ms_max": round(solve_ms[-1], 3) if solve_ms else None,
    }


def main():
    # One env count per process. Building a second ManagerBasedRLEnv inside a
    # live Kit app hangs indefinitely -- that is a harness limit, NOT a BrickSim
    # one, and reporting it as a batching failure would be wrong. The sweep is
    # 01_bricksim_sweep.sh, one process per count.
    print("== probing %d envs ==" % args_cli.num_envs, flush=True)
    try:
        r = probe(args_cli.num_envs)
        print("  %(steps_per_s)s steps/s, %(env_steps_per_s)s env-steps/s, "
              "solve %(bricksim_solve_ms_median)s ms" % r, flush=True)
    except Exception as e:
        r = {"n_envs": args_cli.num_envs, "failed": "%s: %s" % (type(e).__name__, e)}
        print("  FAILED %s: %s" % (type(e).__name__, e), flush=True)

    r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with args_cli.out.open("a") as f:
        f.write(json.dumps(r) + "\n")
    print("  appended -> %s" % args_cli.out, flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
