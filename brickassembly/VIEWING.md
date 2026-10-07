# Seeing the simulation

## Live: the dual-arm prototype (start here)

```bash
cd ~/Documents/ISDN_Robofab/brickassembly
bash scripts/run.sh dual_arm_sim.py --shape arch                 # OpenGL window
bash scripts/run.sh dual_arm_sim.py --shape cube --viewer viser  # browser, :8080
bash scripts/run.sh dual_arm_sim.py --shape hollow_box
bash scripts/run.sh dual_arm_sim.py --structure S3               # an authored one
bash scripts/run.sh dual_arm_sim.py --front my.png --top my_top.png --width 8
```

The whole pipeline in one process: silhouette images → `blueprint.py` (brick
layout) → `planner.py` (order, grasps, brace assignment) → Newton physics with
two FR3 arms, studded hollow bricks and a live viewer. It is up in seconds,
not the 5–10 minutes Isaac Sim takes, and it runs in real time.

- **Arm B** (right, +x) is the placer: feeder → grasp → align over the studs →
  press. **Arm A** (left, −x) is the stabilizer: when the plan says a
  placement pries a joint open, it leans in and presses that joint shut before
  B arrives, until B lets go.
- Green wireframe: where the current brick must go. Green cross: the brace spot.
- Side panel: current step, what each arm is doing, snaps so far, the last
  gate error.
- Camera: WASD/QE to fly, left-drag to look, middle-drag to orbit
  (shift: pan, ctrl: dolly). Space pauses, `.` single-steps while paused.
  Right-drag grabs a brick and pulls it, to disturb the build.
- `--strategy none|nearest|weakest_joint` switches the bracing baseline;
  every placement is logged to `results/proto_episodes.jsonl`.
- `--viewer usd --output-path run.usd` records instead of showing;
  `--viewer null --num-frames 7800 --test` runs headless and asserts every
  brick snapped and stayed put.

What it is not: the clutch is a stiff weld switched on when the §2.3.3 gate
passes, not BrickSim's force model, and motion is IK between waypoints, not
cuRobo. Joint-failure physics stays with the BrickSim runs below.

### Camera feeds, a camera-guided run, manual control (OpenGL viewer)

```bash
cd ~/Documents/ISDN_Robofab/brickassembly
# 1. a camera-guided build, with every camera feed on screen
bash scripts/run.sh dual_arm_sim.py --shape cube --look --aim fk_vision --show-cameras
# 2. drive the arms yourself (no build queue; feeder bricks and plate are there)
bash scripts/run.sh dual_arm_sim.py --shape cube --manual
# 3. the same, with the feeds (the wrist feed follows the hand you move)
bash scripts/run.sh dual_arm_sim.py --shape cube --manual --show-cameras
```

All three are off unless asked for: the physics, the build and the vision results are the same with the panels on (the live feeds are
separate noise-free renders that touch no random stream, so V5 sees what it saw before). `--show-cameras` and `--manual` hold the loop to 60 frames/s so the looks play at their real
duration. They need `--viewer gl` (the default); on the null viewer `--show-cameras` renders and shows nothing.

**Camera window** (`--show-cameras`): one window, titled `cameras: ...`, tiles in two rows: RGB on top, depth below (colour-mapped, near = blue,
black = no return), columns top camera, wrist_B, and, with `--look --aim fk_vision` (or `--shadow-vision`), the **last look**. Drag its corner to
resize; the side panel's *Logged Images* dropdown hides or reopens it. The live feeds refresh every `--camera-every N` frames (default 10, about 6 Hz, ~30 ms a refresh;
a larger N is lighter) and are shown at half size, without the sensor noise. The wrist feed is rendered from B's FK pose. The look tile is
the real thing: the full noised frame V5 was given. The **last look** tile is the wrist frame at the scheduled look, with V5's result drawn on it: the estimated target
footprint (cyan) and held-brick footprint (magenta), a green frame and "ACCEPTED studs" or a red one and "REJECTED <reason>", and the target / held
stud counts. In the side panel, under the usual step lines: per step (the last three) each look's verdict and estimator, the looks the aim used
(and "EDGE-steered"), and the aim source: `fk_vision (V5)`, `fk_oracle` or `gt (ground-truth servo)`.

**Manual control** (`--manual`): a floating *Manual control* window beside the side panel. Pick arm A or B, then
- sliders `x y z` (mm, world), `yaw` (deg, relative to the arm's base; 0 = parked, the gripper is 180-degree symmetric), `grip` (mm per finger,
  0 = shut, 40 = the joint limit), and `-`/`+` buttons that move the target by the `step` sliders (mm, deg);
- `Open` (30 mm), `Close` (0 mm: it squeezes whatever is between the fingers; use the slider for a gentler grip), `Home (park)`;
- readouts: the hand's measured pose and finger opening, the target, the **tracking error** (hand vs the command it is following) and the **IK
  residual** (how far the IK solution is from the commanded hand: it grows at the workspace or joint limits).
- keys (hold = 10 steps a second; none of them is a viewer key): `1` / `2` select arm A / B, `J` `L` x -/+, `K` `I` y -/+, `G` `T` z -/+,
  `Z` `X` yaw +/-, `C` `V` close / open the fingers, `R` home. The camera keys (WASD/QE, space, `.`, `H`, `F`) work as before.

A cross in the scene marks each arm's target (orange = A, cyan = B). Targets are clamped to the workspace (hand 12-550 mm up, within 0.76 m of the
shoulder, 0.18 m from the base column, no more than 0.12 m past the mid-plane) and the command follows the target at a limited speed and never
more than 15 mm ahead of the hand, so a target the arm cannot reach does not wind up. To pick a brick by hand: B is on the right (+x); the feeder
bricks lie on its -y side (the first at x 170, y -290 mm); a 2x4 brick lies along x, so turn the hand to yaw 90 deg to close across its short
side, hover at z about 100 mm, go down to z 13.5 mm, `Close`, lift. Not done in manual mode: the clutch snap (a brick pressed onto the plate
rests on its studs under the physics, but is not welded) and the build queue.

## BrickSim physics runs — recorded, then played back

**You can watch BrickSim runs — recorded, then played back two ways.**

Isaac Sim **5.1** (what BrickSim pins) segfaults in `librtx.scenedb.plugin.so`
on this GPU. Isaac Sim **6.0** renders fine here — an identical probe renders
60 frames under 6.0 and crashes under 5.1 on the same driver, so it is the
Isaac version, not the driver. Physics runs therefore record headless and play
back afterwards.

## 0. Record a run, then watch it

```bash
cd ~/Documents/ISDN_Robofab/brickassembly

# 1. run it, recording poses (about 15 min; 6/6 on S1)
bash scripts/run.sh orchestration/run_plan.py --structure S1 \
     --gate default --record results/s1_run.poses.jsonl

# 2a. watch in the browser -- no Isaac rendering involved
bash scripts/run.sh viz/play_viser.py results/s1_run.poses.jsonl
#     open http://localhost:8080   (play/pause, frame slider, speed)

# 2b. or rebuild it as USD and scrub it in Isaac Sim 6.0
bash scripts/run.sh viz/to_usd.py   results/s1_run.poses.jsonl
bash scripts/run.sh viz/open_usd.py results/s1_run.poses.usda
```

Verified end to end: a 6/6 S1 run recorded 1565 frames over 78.2 s of
simulated time; the USD carries 7 parts, 11 arm links and 1565 time samples
and opens under Isaac Sim 6.0; the Viser player serves the same recording in a
browser with all 9 Franka joints mapped.

Poses are sampled from the **physics view**, not USD — USD transforms are
stale headless, which was the bug that made WP4 place nothing at all.

### The arm is drawn in both players

A recording stores three things per frame: brick poses, the 11 Franka **link
world poses**, and the 9 **joint angles**. Each player uses whichever suits it:

- **Viser** drives the Franka URDF with the joint angles, matched to the URDF's
  actuated joints **by name**. Isaac's articulation order is not the URDF's, so
  positional mapping would silently bend the wrong joints.
- **USD** ignores the joints and time-samples each link's world pose directly,
  referencing `/panda/<link>` out of Isaac's `franka.usd`. No articulation, no
  joint drives, no forward kinematics — so any USD viewer can play it, not just
  Isaac. The referenced link prims carry their own rest transform, which is
  cleared (`SetXformOpOrder([])`) or the pose would be applied twice.

Checked against the recording inside Isaac Sim 6.0: `panda_link0` and
`panda_hand` compose to exactly their recorded world poses, and the link
meshes resolve (they are instance proxies, so a plain `Stage.Traverse()`
will report zero meshes — that is not a missing arm).

The Franka asset streams from the Omniverse S3 bucket, so the USD player needs
network. Override the root with `ISAAC_ASSET_ROOT` if you have a local copy.

### Use `scripts/run.sh`, not a bare interpreter

Each viewer needs a different Python and they are not interchangeable: Viser
lives in `isdnenv`, the USD scripts need Isaac Sim 6.0. `run.sh` picks the
right one and fixes the working directory, so it works from any directory.
Running `viz/open_usd.py` under the cuRobo interpreter fails on `isaaclab`;
running it from the wrong directory fails on a relative `results/` path.

**Isaac Sim 6.0 opens no window unless a visualizer is named.** `--headless`
is deprecated there and omitting `--viz` now means headless, so `open_usd.py`
used to run happily and print "window open" while showing nothing.
`open_usd.py` therefore passes `--viz kit` itself unless you asked for
`--frames N`.

## 1. Watch the arm build a structure (works now, in a browser)

This is the cuRobo + Viser viewer. It doesn't use Isaac Sim, so the renderer
problem doesn't touch it.

```bash
source ~/Documents/ISDN_Robofab/isdnenv/bin/activate     # the `isdn` alias
cd ~/Documents/ISDN_Robofab/brickassembly
python run_assembly.py --plan plans/S3_weakest_joint.json
```

Then open **http://localhost:8080**. You see the Franka, the baseplate, bricks
appearing as they're placed, and a frame marking the brace pose.

```bash
python run_assembly.py --plan plans/S1_weakest_joint.json --speed 0.5   # slower
python run_assembly.py --plan plans/S2_weakest_joint.json --port 8090   # other port
```

**What this is and isn't:** it's the *kinematic* slice — cuRobo plans the
motion and the snap is decided by a geometric gate. There is no contact
physics, no clutch, no gripper. It shows sequence, reachability and geometry.
It does **not** show what the physics runs do.

## 2. Follow a physics run live (text)

The physics runs are the real pipeline: Franka, PhysX, BrickSim's clutch.
Launch one and watch its log.

```bash
cd ~/Documents/ISDN_Robofab/brickassembly
bash scripts/run.sh orchestration/run_plan.py --structure S1 --gate default \
    > /tmp/run.log 2>&1 &
# the launcher picks BrickSim's venv and adds --headless for you
# --gate default is the gate the current inserter can satisfy; --gate spec
# is the correct §2.3.3 one, and places nothing until the inserter can press
# harder (see WORKLOG.md: inserter_cannot_reach_spec_press)

tail -f /tmp/run.log                                  # everything
grep -aE "PLACEMENT|press force|placed " /tmp/run.log # just the outcomes
```

Expect ~5–10 minutes of Isaac Sim startup before the first step. Useful lines:

| line | meaning |
|---|---|
| `=== Assembly Step n / m ===` | starting a placement |
| `grasp returned True; brick rose 181 mm` | the brick is actually held |
| `grasp attempt n failed: …` | with the reason and what was retried |
| `press force: … gate-projected X N` | force the inserter developed (`--measure-press`) |
| `PLACEMENT … OK / FAILED` | verified 90 physics steps after the press |
| `placed 6 / 6` | final tally for the run |

Add `--measure-press` to see press forces, `--structure S2` / `S3` for other
builds.

## 3. Read what happened afterwards

```bash
cd ~/Documents/ISDN_Robofab/brickassembly
python scripts/status.py                 # gates, calibration, latest tests
python scripts/status.py --open          # only blockers
cat WORKLOG.md                           # narrative, generated from the ledger
python -m json.tool results/wp4_episodes.jsonl | head   # per-placement records
```

## Why there's no picture, and how to get one

Isaac Sim's RTX renderer crashes on this driver. BrickSim ships
`isaacsim_rtx_compat` to work around exactly this (driver newer than
590.48.01 reporting an unsafe Vulkan allocation limit), but it downloads a
Vulkan profiles package whose URL now 404s — the Arch package moved from
`1.4.341.0-1` to `1.4.357.0-2` — and it pins SHA-256 for both the archive and
the extracted library, failing closed.

### Driver downgrade to 590.48.01 (diagnosis confirmed)

The condition is confirmed on this machine. `isaacsim_rtx_compat._limit()`
returns 4292870144, meaning the GPU was **not** skipped by its safety checks —
NVIDIA vendor, driver newer than 590.48.01, allocation limit above the
4 GiB − 2 MiB cap. And Ubuntu's `nvidia-driver-590-open` candidate is
**590.48.01**, exactly the module's `_LAST_KNOWN_GOOD_DRIVER`, so the layer
becomes a no-op and the renderer should start.

Currently installed: `nvidia-driver-595-open 595.84-0ubuntu0.24.04.1`.

```bash
# 1. Record what you have, so a rollback is one command
dpkg -l | grep -E "nvidia-driver|libnvidia-gl" | awk '{print $2, $3}' | tee ~/nvidia-before.txt

# 2. Install the 590 open driver; apt replaces the 595 packages
sudo apt update
sudo apt install nvidia-driver-590-open=590.48.01-0ubuntu0.24.04.5

# 3. Stop it being upgraded back on the next apt upgrade
sudo apt-mark hold nvidia-driver-590-open

# 4. Reboot (the kernel module has to be reloaded)
sudo reboot
```

After the reboot, verify in this order — each step is cheap and rules
something out:

```bash
nvidia-smi                              # expect "Driver Version: 590.48.01"

# CUDA still works for both toolchains?
~/Documents/ISDN_Robofab/isdnenv/bin/python -c "import torch; print('cu130', torch.cuda.is_available())"
~/Codes/CAIRSS/Issac/bin/python        -c "import torch; print('cu128', torch.cuda.is_available())"

# the compat layer should now find nothing to fix -> expect None
cd ~/Documents/ISDN_Robofab/BrickSim
RTX_VULKAN_COMPAT_DISABLE=1 .venv/bin/python -c \
  "import sys; sys.path.insert(0,'python'); import isaacsim_rtx_compat as c; print(c._limit())"
```

Then load the viewer — same command as a normal run, minus `--headless`:

```bash
cd ~/Documents/ISDN_Robofab/BrickSim
OMNI_KIT_ACCEPT_EULA=1 .venv/bin/python -u \
    ../brickassembly/orchestration/run_plan.py --structure S1 --trial 100
```

A window opens and you watch the Franka build the column. The first render is
slow — Isaac Sim compiles its shader cache once, which can take several
minutes with no output.

**Rolling back**, if anything is worse:

```bash
sudo apt-mark unhold nvidia-driver-590-open
sudo apt install nvidia-driver-595-open=595.84-0ubuntu0.24.04.1
sudo reboot
```

**What I'd want you to know before running it.** This is a system-wide change:
it touches your desktop session and both CUDA toolchains, and this workstation's
driver/library state has already been broken once by an install (the report's
risk R9). 590.48.01 is newer than the RTX 5090 D's launch driver and Ubuntu
ships it for 24.04, so support is expected — but I can't prove your desktop and
CUDA come back clean until you reboot, which is why the verification steps
above come before the viewer. The alternative that touches nothing system-wide
is repinning the compat layer's URL and hashes.

Three ways to a visual, in the order I'd try them:

1. **Update the compat layer's pin.** One line for the URL, plus new hashes.
   This replaces an integrity check on a binary that gets loaded into the
   process as a Vulkan layer, so it's your decision, not mine. I patched the
   URL to test the idea and reverted it; the vendored file is untouched.
2. **Record the stage to USD** during a headless run and open it afterwards in
   any USD viewer. No renderer needed at run time. Not built yet — maybe an
   hour of work.
3. **Extend the Viser viewer** to replay a physics run from the episode log
   and recorded poses. Keeps the browser workflow, needs pose logging added.
