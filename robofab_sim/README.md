# robofab_sim

Dual-arm LEGO assembly simulator (`dual_arm_sim.py`): two Franka FR3 arms, Newton 1.2.1 physics (MuJoCo-Warp
solver), studded brick meshes, simulated cameras. Arm B places bricks from a feeder onto a plate, arm A can brace.
Copied verbatim from `brickassembly/` of ISDN_Robofab at commit `7535897a29db4913655fc0a4327561ac923c56d2`
(branch `v4-newton-vision`); the code edits are listed at the end.

## Requirements

- Linux, x86_64.
- NVIDIA GPU and driver with CUDA >= 12.8 (`nvidia-smi` shows the version). Without a GPU use `--cpu` (slow).
- `uv` (https://docs.astral.sh/uv/) or `python3.12`.
- A display for the GL viewer (not needed for `--viewer null` or `--viewer viser`).
- Internet for the first setup (packages, torch, and the Franka asset, which `setup.sh` caches in `~/.cache/newton`).
  After that nothing is fetched. The first run of each new code path compiles Warp kernels (cached in `~/.cache/warp`).

## Install

    bash setup.sh              # .venv (Python 3.12), torch CUDA build picked from nvidia-smi, requirements, FR3 asset
    bash setup.sh --cpu        # CPU torch      |  --no-torch  skip torch  |  --cuda cu128|cu130  force a build
    bash setup.sh --check      # also runs the tests and a headless smoke build

Idempotent: re-run it to repair. The venv is about 5.6 GB (mostly torch + CUDA libraries). `dual_arm_sim.py` itself does
not import torch; it is installed for RL training code.

## Run

`./run.sh [args]` runs `.venv/bin/python dual_arm_sim.py [args]` from this folder. `./run.sh test` runs pytest.
(The docstrings in the code say `bash scripts/run.sh dual_arm_sim.py ...`; here that is `./run.sh ...`.)
`./run.sh --help` lists every flag. Common viewer flags come from Newton: `--viewer gl|viser|null`, `--num-frames N`, `--test`.

### Live build with the shapes (window)

    ./run.sh --shape cube
    ./run.sh --shape arch
    ./run.sh --shape hollow_box

A drawing instead of a sample shape (black shape on white; `--width` is studs across the front view):

    ./run.sh --front blueprints/cube_front.png --side blueprints/cube_side.png --width 4 --name mycube
    ./run.sh --front my.png --width 8 --name mine

### Hand-authored structures (--structure S1..S3)

    ./run.sh --structure S1
    ./run.sh --structure S2
    ./run.sh --structure S3

### Bracing strategy (arm A)

Default is `--strategy none` (arm A stays parked). The hand-planned braces:

    ./run.sh --structure S3 --strategy weakest_joint
    ./run.sh --structure S3 --strategy nearest

Note: S3 with `weakest_joint` or `nearest` ends at `joint_break at step 13` (the build stops there, `--test` exits 1);
the source project's interpreter gives the same. S3 with `--strategy none` places 16/16.

### Camera feeds and the Workspace window

    ./run.sh --shape cube --show-cameras
    ./run.sh --shape cube --show-cameras --camera-every 5

Top, wrist_A and wrist_B feeds (RGB, depth) plus a Workspace window with what `top_detect` reads off the top image.
Implies `--pace` (60 frames/s).

### Camera-guided build

    ./run.sh --shape cube --look --aim fk_vision --show-cameras

`--look` adds the look waypoints (two views per brick, hand turned 180 deg); `--aim fk_vision` aims from the vision
estimate. `--aim gt | fk_oracle | fk_vision`, `--shadow-vision`, `--aim-inject JSON` are the other perception hooks.

### Manual control

    ./run.sh --shape cube --manual --show-cameras

Side panel and keys: `1`/`2` pick the arm, `J L / K I / G T` move, `Z X` yaw, `C V` grip, `R` home.

### Headless (no window) with a pass/fail exit code

    ./run.sh --shape cube --viewer null --test --num-frames 7800
    ./run.sh --shape cube --look --aim fk_vision --viewer null --test --num-frames 40000
    ./run.sh --shape arch --viewer null --test --num-frames 20000
    ./run.sh --shape hollow_box --viewer null --test --num-frames 20000
    ./run.sh --structure S3 --viewer null --test --num-frames 60000
    ./run.sh --shape cube --show-cameras --viewer null --num-frames 300

`--test` asserts that every brick snapped (exit 1 otherwise). Roughly 1 to 3 minutes per build on an RTX 5090.

### Single-step episodes (start step, one step, brace)

    ./run.sh --structure S3 --start-step 5 --only-step --brace-json null --viewer null --test --num-frames 20000
    ./run.sh --structure S3 --start-step 5 --only-step --brace-json '{"sham_s": 2}' --viewer null --test --num-frames 20000
    ./run.sh --structure S3 --strategy weakest_joint --start-step 11 --only-step --viewer null --test --num-frames 20000
    ./run.sh --spec ep.json     # ep.json below

`--start-step N` starts from a nominal state (steps < N welded at their targets); `--only-step` runs step N alone;
`--brace-json null | [y, z, lean_deg] | {"sham_s": T}` sets that step's brace; `--exe E0|E1` picks arm A's gains and grip;
`--plan PATH.json` caches the plan. `--spec` holds all of these in one JSON (keys are the flag names, `out` is the row file):

    {"structure": "S3", "start_step": 5, "only_step": true, "brace_json": null,
     "viewer": "null", "test": true, "num_frames": 20000, "out": "out/ep_row.jsonl"}

### Outputs and where they go

All paths are relative to this folder.

- `plans/sim_<name>_<strategy>.json`: the plan of each run (not written with `--spec`, or when `--plan` is given).
- `results/proto_episodes.jsonl`: one row per placement attempt (appended; not written with `--spec`).
- `--p0-out FILE.jsonl` (alias `--out`): one row per run (appended; implies `--monitor`).
- `--monitor`: the contact monitor (events in the `--p0-out` row).
- `--record-all FILE.npz`: per-frame record of the whole run (cell/record.py).
- `--record-inserts FILE.npz`: joint positions and phases at the insert frames.
- `--save-captures DIR`: the camera captures per step (`stepNN_<cam>.npz`, ground truth in `*_gt.npz`).
- `imgui.ini`: written by the GL viewer in the working directory.

    ./run.sh --shape cube --viewer null --test --num-frames 7800 --monitor --p0-out out/p0.jsonl \
        --record-all out/rec.npz --record-inserts out/ins.npz --save-captures out/caps

### Seed

`--seed N` seeds the cameras (hidden extrinsics, lighting, noise) and the per-step prior draw (default 0).

    ./run.sh --shape cube --look --aim fk_vision --seed 3 --viewer null --test --num-frames 40000

### Browser viewer

    ./run.sh --shape cube --viewer viser      # http://localhost:8080 ; the server stays up after the build, Ctrl-C to stop

## For RL

Facts only.

- Observation sources:
  - `cell/cameras.py`, `Cameras.capture(cam, ee_pose, state=..., frame=..., sim_time=...)`: noised RGB (uint8) and depth
    (float32, m) with intrinsics and the nominal camera pose, for `top`, `wrist_A`, `wrist_B`. `Cameras.preview` is the
    noise-free display version.
  - `top_detect.detect(rgb, depth, intrinsics_or_rays, cam_pos, cam_R)`, `top_detect.remember(prev, res)`,
    `top_detect.track_held(det, prev, hands)`: the plate and the bricks (colour, size, xy, yaw) read off the top image.
  - the hand forward kinematics: `Example.hand_fk(k, q=None)` in `dual_arm_sim.py`.
- Episode rows: `--p0-out FILE.jsonl` (one run row: placed/total, failure, events, per-step snap and brace data);
  `results/proto_episodes.jsonl` (one row per placement attempt: success, `final_error_mm`, `dz_mm`, `tilt_deg`,
  `failure_mode`, brace target); a `--spec` run writes its row, with an `episode` block, to its `out`.
- `tasks/brace_bandit.py`: pure-Python task library for the bracing RL (structures, contexts, action set, observation
  `features`, hidden load draw `hidden`, scripted pickers). It imports only planner, blueprint, bracing and stability
  (no newton, no warp). The bracing-RL harness (`experiments/v4_brace.py`, `experiments/rl_env.py`) of the source project
  uses it with the single-step episodes above; those two files are not in this folder.

## Tests

    ./run.sh test              # tests/test_top_detect.py (GPU), test_manual.py, test_vision.py

## Layout

`dual_arm_sim.py` (entry), `blueprint.py` (images to bricks), `planner.py`, `bracing.py`, `stability.py` (plan and
brace), `vision.py`, `top_detect.py`, `cell/` (cameras, live-view aids, contact monitor, recorder), `sim/joint_model/capacity.py`,
`tasks/brace_bandit.py`, `motion/fr3.yml` (collision spheres), `blueprints/` (shape drawings), `plans/`, `results/` (outputs).

## Differences from `brickassembly/`

Code edits: none (every .py file is a verbatim copy). Not included: the BrickSim cross-check
(`planner.py --crosscheck`, returns `{}` here), the MuJoCo twin, the cuRobo motion code, the experiments and the other tests.
