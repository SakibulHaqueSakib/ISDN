# robofab_sim

## What it is

A dual-arm LEGO brick assembly cell (two Franka FR3 arms) in standalone Newton 1.2.1: Warp, the MuJoCo-Warp solver and
Newton's own collision pipeline, with studded hollow brick meshes. Arm B (the placer, at +x) picks each brick from a
feeder and presses it onto the plate; arm A (the bracing arm, at -x) can grip bricks beside a joint to brace it. RGB-D
cameras (`top`, `wrist_A`, `wrist_B`) render with sensor noise.

Observations are vision only: `top_detect.py` reads every brick's colour, size and pose, the baseplate grid and the
bricks held in a gripper off the top image (plus joint state and the plan). Simulator ground truth is used only in the
reward and returned in `info["privileged"]` / `extras["privileged"]`.

It runs as a viewer simulator (`dual_arm_sim.py`) and as RL environments with an Isaac Lab-style vectorized interface
(`rl_env.py`, `rl_vec.py`, `rl_batched.py`, `train_rl.py`). No `isaaclab` or `isaacsim` import is needed.

## Requirements

- Linux with an NVIDIA GPU and driver. `setup.sh` picks the CUDA 12.8 or CUDA 13 build of torch from `nvidia-smi`
  (a driver older than CUDA 12.8 is refused). `--cpu` runs everything on the CPU and is slow.
- Python 3.12 (`uv` is used if present, else `python3.12 -m venv`).
- Network on the first setup: pip, the torch index, and Newton's FR3 asset (downloaded once, cached afterwards).
- A display for the GL viewer; not needed for `--viewer null` or `--viewer viser`.

## Install

    bash setup.sh                  # .venv (Python 3.12), torch CUDA build picked from nvidia-smi
    bash setup.sh --cuda cu128     # force a torch CUDA build (cu128 | cu130)
    bash setup.sh --cpu            # CPU torch (Newton/Warp then run on the CPU, slowly)
    bash setup.sh --no-torch       # skip torch (dual_arm_sim.py does not import it; the RL files and rsl_rl do)
    bash setup.sh --check          # after installing: also run the tests and a headless smoke build

Idempotent: re-run it to repair or upgrade. `VENV=/path` overrides the venv location (`run.sh` honours it too).
`rl_games==1.6.1` is installed with `--no-deps`: its PyPI metadata pins `wandb<0.13`, which does not build on Python 3.12.
Its real dependencies are in `requirements.txt`; wandb is not installed.

## Run the simulator

`./run.sh [args]` runs `.venv/bin/python dual_arm_sim.py [args]` from this folder; `./run.sh <script>.py [args]` runs
any script here; `./run.sh test [pytest args]` runs the tests. (Docstrings in the code say
`bash scripts/run.sh dual_arm_sim.py ...`; here that is `./run.sh ...`.) `./run.sh --help` lists every flag. Viewer flags
come from Newton: `--viewer gl|viser|null|usd|rerun`, `--num-frames N`, `--test`, `--device`.

    ./run.sh --shape cube                    # live build in the GL window (shapes: cube, hollow_box, arch)
    ./run.sh --shape arch --viewer viser     # browser viewer, http://localhost:8080
    ./run.sh --structure S3                  # a hand-authored structure (S1, S2, S3)

Headless, with a pass/fail exit code (`--test` asserts that every brick snapped; exit 1 otherwise):

    ./run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 7800     # expect "done: 8 / 8 placed"

A drawing instead of a sample shape (black shape on white; `--width` is studs across the front view; `--side` or `--top`
gives the depth, else `--depth`):

    ./run.sh --front blueprints/cube_front.png --side blueprints/cube_side.png --width 4 --name mycube
    ./run.sh --front my.png --width 8 --name mine

Arm A stays parked by default (`--strategy none`). `--strategy nearest|weakest_joint` runs the plan's hand-planned
braces (see Known issues).

Perception and camera windows (GL viewer):

    ./run.sh --shape cube --show-cameras                       # implies --pace (60 frames/s)
    ./run.sh --shape cube --show-cameras --camera-every 5      # refresh the feeds every 5 frames (default 10)
    ./run.sh --shape cube --look --aim fk_vision --show-cameras
    ./run.sh --shape cube --manual --show-cameras              # drive the hands by hand

- `--show-cameras` opens an image window with the top, wrist_A and wrist_B feeds (RGB over depth). The top feed has
  top_detect's plate and brick outlines; the wrist feeds carry each brick's label ("#id colour size", then x, y and yaw).
  With `--look --aim fk_vision` or `--shadow-vision` the last look with V5's result is added.
- The "Workspace (top camera)" window lists what top_detect reads off the top image: the plate (centre, yaw, studs,
  pitch, top z, occupied cells) and a table of bricks (id, colour, size, x, y, z, yaw, layer, plate cell, seen/hidden/held).
- `--look` adds two look waypoints per brick (hand turned 180 deg); `--aim gt|fk_oracle|fk_vision` picks the aim
  controller; `--shadow-vision`, `--aim-inject JSON`, `--seed N` are the other perception hooks.
- `--manual`: side panel and keys `1`/`2` pick the arm, `J L / K I / G T` move, `Z X` yaw, `C V` grip, `R` home.

Single-step episodes (start from a nominal state, run one step, set its brace), also from one JSON with `--spec`:

    ./run.sh --structure S3 --start-step 5 --only-step --brace-json null --viewer null --test --num-frames 20000
    ./run.sh --structure S3 --start-step 5 --only-step --brace-json '{"sham_s": 2}' --viewer null --test --num-frames 20000
    ./run.sh --spec ep.json

`--brace-json null | [y, z, lean_deg] | {"sham_s": T}`; `--exe E0|E1` picks arm A's gains and grip; `--plan PATH.json`
caches the plan. `--spec` keys are the flag names (`out` is the row file); no `plans/` or `results/` writes:

    {"structure": "S3", "start_step": 5, "only_step": true, "brace_json": null,
     "viewer": "null", "test": true, "num_frames": 20000, "out": "out/ep_row.jsonl"}

Outputs (paths relative to this folder):

- `plans/sim_<name>_<strategy>.json`: the plan of each run (not with `--spec`, or when `--plan` is given).
- `results/proto_episodes.jsonl`: one row per placement attempt (appended; not with `--spec`).
- `--p0-out FILE.jsonl` (alias `--out`): one row per run, appended; implies `--monitor` (the contact monitor).
- `--record-all FILE.npz`: per-frame record of the run. `--record-inserts FILE.npz`: joint positions and phases at the insert frames.
- `--save-captures DIR`: the camera captures per step (`stepNN_<cam>.npz`, ground truth in `*_gt.npz`).
- `imgui.ini`: written by the GL viewer in the working directory.

## RL

### Environments (`rl_env.py`)

Both are gymnasium environments over one `dual_arm_sim.Example` (headless Newton). Both take `structure`/`step` or
`contexts`, `images`, `cameras`, `image_stride`, `n_max`, `exe`.

- `BrickStepEnv`: one decision per brick, one env step per episode. Action `Dict(brace Discrete(2), brace_pose Box(3)
  (y, z, lean_deg), press Box(3) (scale, fx, fy))`; reward `s - 0.05 * 1[brace applied]` with `s = 1` iff no failure,
  the step ran to its end and the contact monitor logged no arm_arm event. Adds `features` (15 numbers of the plan's
  context) to the observation. `images` defaults to True.
- `BrickReachEnv`: low-level control of both hands, 10 Hz (`control_every=6` frames), shaped reward, truncation at
  `max_steps=600` ticks. Action `Box(10)` in [-1, 1]: (dx, dy, dz, dyaw, grip) of arm A then arm B; grip is an absolute
  finger target (-1 shut, +1 open). The episode ends when the Example's snap gate passes. `images` defaults to True.

Observation, a Dict (both envs; vision and proprioception only):

| key | shape | content |
|---|---|---|
| `proprio` | (54,) | joint q (18) and qd (18) of both arms, both hands' FK pose (pos + quat), finger openings (4) |
| `bricks` | (16, 20) | one row per top_detect brick, held ones included: valid, x, y, z_top, sin/cos 2*yaw, L, W, colour one-hot (8), hidden, held_by_A, held_by_B, on_plate; padded with valid = 0 |
| `plate` | (8,) | found, x, y, sin/cos 2*yaw, L, W, top_z |
| `task` | (8,) | the current plan step's target x, y, z, sin/cos 2*yaw, brick L, W, step / N |
| `features` | (15,) | BrickStepEnv only |
| `images` | optional | per camera `<cam>_rgb` uint8 (H/stride, W/stride, 3) and `<cam>_depth` float32 (H/stride, W/stride) from the noised capture; full size is 1280x960, `image_stride` defaults to 4; depth 0 = no return |

The observation is a pure function of the camera captures, top_detect's result, the joint state and the plan
(`ObsSpec.encode`); `tests/test_rl_env.py` checks that moving the bricks does not change it.

    ./run.sh rl_env.py --demo        # the scripted reach episode and a timing line

### Vector interfaces

The surface follows Isaac Lab's `DirectRLEnv`: `num_envs`, `device`, `cfg`, `max_episode_length`, `episode_length_buf`,
`single_observation_space` / `observation_space`, `single_action_space` / `action_space`, `reset(seed, options) ->
(obs, extras)`, `step(actions) -> (obs, reward, terminated, truncated, extras)`, `close()`. Torch tensors in and out;
`obs["policy"]` is the flat float32 `(num_envs, D)` concatenation of proprio, bricks, plate, task [, features] (390 for
the reach env), unnormalised (use `obs_normalization` in the trainer config); `obs["images"]` (only with `images=True`) is
`{<cam>_rgb, <cam>_depth}`. Finished sub-envs autoreset inside `step()` and the returned observation is the new
episode's first; `extras["log"]` / `extras["episode"]` carry Episode/return, Episode/length, Episode/success.
`BrickReachEnv` actions are its `Box(10)`; `BrickStepEnv` actions are a flat `Box(7)`: `a[0] > 0` brace on, `a[1:4]`
(y, z, lean_deg) mapped linearly onto `BrickStepEnv.BRACE_BOUNDS` and moved to the nearest candidate, `a[4]` press scale
`1.3 * max(a[4], 0)`, `a[5:7]` lateral force `3 N * a[5:7]`.

- `rl_vec.NewtonVecEnv(env_cls, num_envs, device="cuda:0", seed=0, per_env_kwargs=None, context="spawn", **env_kwargs)`:
  one subprocess per env, one Example each. A reset costs 2-5 s and stepping is synchronous, so one resetting sub-env
  stalls the others.
- `rl_batched.BatchedReachEnv(num_envs, device="cuda:0", structure="cube", start_steps=(0,), control_every=6,
  max_steps=600, images=False, cameras=..., image_stride=4, n_max=16, exe="E0", seed=0, detect_stride=1, res_div=2,
  procs=None, ...)`: `BrickReachEnv` in N worlds of one Newton model (one captured CUDA graph per frame, batched IK,
  per-world reset). `start_steps` are the plan steps a world may start at, drawn per reset; `reset(options={"step": ...})`
  pins them. Also takes the reward weights `w_reach ... w_drop` and `lift_dz`. `BrickStepEnv` has no batched version.
- Adapters (take either env): `rl_vec.RslRlVecEnv(env)` for rsl_rl 5.0.1, `rl_vec.skrl_env(env)` for skrl,
  `rl_vec.RlGamesVecEnv(env)` for rl_games.

```python
import torch
from rl_batched import BatchedReachEnv

if __name__ == "__main__":            # required for N >= 8: detection workers are spawned
    env = BatchedReachEnv(16, device="cuda:0", structure="cube", images=False)    # procs=0: detect in-process
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        a = torch.rand(env.num_envs, 10, device=env.device) * 2 - 1               # Box(10) in [-1, 1]
        obs, reward, terminated, truncated, extras = env.step(a)                  # obs["policy"]: (16, 390)
    env.close()
```

    ./run.sh rl_batched.py --demo 16       # the scripted controller in N worlds: ticks to success per world, env steps/s

### Training (`train_rl.py`)

PPO smoke runs (tiny defaults, not a tuned training run). rsl_rl by default, `--skrl` for skrl, `--rl-games` for rl_games.

    ./run.sh train_rl.py --iterations 3 --num-envs 64                   # rsl_rl on BatchedReachEnv, images off
    ./run.sh train_rl.py --skrl --timesteps 200 --num-envs 64           # skrl PPO
    ./run.sh train_rl.py --rl-games --num-envs 8                        # rl_games PPO (2 epochs)
    ./run.sh train_rl.py --backend process --num-envs 2                 # BrickReachEnv in subprocess workers
    ./run.sh train_rl.py --env step --num-envs 2 --iterations 2         # BrickStepEnv (process backend only)

Flags: `--env reach|step`; `--backend batched|process` (default batched for `reach`, process for `step`);
`--num-envs` (default 64 batched, 2 process); `--images`; `--iterations` (rsl_rl, default 3); `--steps-per-env` (rsl_rl
rollout length, default 24); `--timesteps` (skrl, default 200); `--log-dir` (rsl_rl tensorboard directory, default a
temp dir); `--device` (default `cuda:0`). The configs are dataclasses (`EnvCfg`, `RslRlCfg`, `SkrlCfg`, `RlGamesCfg`).

### Measured throughput

One machine's numbers (this development machine), images off, env steps/s with zero/random actions:

| environment | env steps/s |
|---|---|
| `BatchedReachEnv` N=1 | 15.6 / 14.5 |
| `BatchedReachEnv` N=16 | 112 / 84 |
| `BatchedReachEnv` N=64 | 170 / 122 |
| `BatchedReachEnv` N=128 | 194 / 133 |
| single `BrickReachEnv` | about 7 |
| `NewtonVecEnv`, 2 workers | 9.2 |
| `NewtonVecEnv`, 4 workers | 10.4 |

Trainer smoke runs on a fresh install, same machine: rsl_rl N=16 79.8 steps/s, skrl N=16 82.0, rl_games N=8 57.3;
rsl_rl N=64 109 steps/s.

### Batched vs single-world differences

`BatchedReachEnv` has the same task, spaces and reward as `BrickReachEnv` (one definition of each); deliberate differences:

- The scene is built once; a reset restores a world's state, welds and collision groups in place (0.2 ms instead of a 2 s Example rebuild).
- One camera key light per env (the light lives in the renderer); the hidden extrinsic error, exposure gain and noise stream are per world and redrawn at every reset. Wrist cameras have no co-located ring light.
- The sensor noise is `cell/cameras.sense` in torch on the GPU from a per-world generator: statistically, not bitwise, the same as numpy's.
- All frames of a tick run in every world: a world whose brick snaps keeps stepping to the end of its tick.
- The joint-break rule is `BatchedBreaker`'s: `JointBreaker`'s condition on the same weld wrenches, without its u2 / peak bookkeeping.
- Default `res_div=2`: the top camera renders at half size and top_detect reads that grid (about 1 mm difference from full size).

## Tests

    ./run.sh test        # about 5 min

On a fresh install: 55 passed, 1 skipped (`test_vision`'s probe-regression test, whose research probe is not in this folder).
`tests/test_top_detect.py` needs the GPU. Running the tests is also `bash setup.sh --check`.

## Known issues

- The plan's LP-shared brace breaks the structure (`joint_break`) on arch step 10 and S3 step 13, while the same steps
  unbraced succeed. Bracing is meant to be learned (`BrickStepEnv`), not taken from the plan, so the default is
  `--strategy none`; `--strategy nearest|weakest_joint` reproduces the breaks (`--test` then exits 1).
- `import gym` prints an "unmaintained" warning: gym 0.23.1 is pinned because rl_games checks its spaces with OpenAI gym's classes.
- The first run of each new code path compiles Warp kernels (cached in `~/.cache/warp`).

## Layout

- `setup.sh`, `run.sh`, `requirements.txt`, `pytest.ini`: install, launcher, pinned packages, pytest config.
- `dual_arm_sim.py`: the simulator (entry point): scene, arms, plan execution, joint breaker, viewer aids.
- `rl_env.py`: `Sim`, `ObsSpec`, `BrickStepEnv`, `BrickReachEnv`, `ReachReward`, scripted reach demo.
- `rl_vec.py`: `NewtonVecEnv` and the rsl_rl / skrl / rl_games adapters.
- `rl_batched.py`: `BatchedReachEnv` and its batched cells, breaker and cameras.
- `rl_detect.py`: top_detect over many worlds in worker processes (used by `rl_batched`).
- `train_rl.py`: PPO smoke runs with rsl_rl, skrl or rl_games.
- `top_detect.py`: top-camera read-out (bricks, plate, held-brick tracking); pure numpy/scipy/cv2.
- `vision.py`: V5 relative-alignment estimation from one wrist look (target lattice pose and held-brick pose).
- `blueprint.py`: drawings to brick lists; `planner.py`, `bracing.py`, `stability.py`: plan and brace.
- `cell/`: `cameras.py` (noised RGB-D), `liveview.py` (viewer aids, manual control), `contacts.py` (contact monitor), `record.py` (recorder).
- `sim/joint_model/capacity.py`: weld capacity model used by the joint breaker.
- `tasks/brace_bandit.py`: pure-Python task library for the bracing RL (structures, contexts, actions, features).
- `motion/fr3.yml`: FR3 collision spheres. `blueprints/`: sample drawings. `plans/`, `results/`: outputs.
- `tests/`: `test_top_detect.py`, `test_manual.py`, `test_vision.py`, `test_rl_env.py`, `test_rl_batched.py`.

Not included from the source project: the BrickSim cross-check (`planner.py --crosscheck` returns `{}` here), the MuJoCo
twin, the cuRobo motion code, the experiments.
