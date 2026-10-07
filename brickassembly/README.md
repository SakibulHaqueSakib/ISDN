# brickassembly

**v4 (current):** runs in standalone Newton 1.2.1 in the Isaac interpreter
(Newton's own collision pipeline), vision only, with collision-free dual-arm
motion. Plan of record: [`Docs/plan_v4.md`](../Docs/plan_v4.md) (ledger
`v4_plan_of_record`). The v3.1 MuJoCo twin section below is frozen as the record.

Execution of `Docs/master_report.md` (v3.1). `ledger.jsonl` is the record of every
gate, decision, deviation and failure — read it before `git log`.

**To see where the project stands: `python scripts/status.py`.**
**Results: [`Docs/results_report.md`](../Docs/results_report.md).**
Full instructions in [PROGRESS.md](PROGRESS.md).

## Setup

One environment runs everything except the BrickSim / Isaac Sim 5.1 physics runs:

```bash
bash setup_env.sh                 # from the repo root -> .venv (add --with-curobo for cuRobo)
bash brickassembly/scripts/run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 40000
```

Full instructions, verification and troubleshooting: [`../SETUP.md`](../SETUP.md).
`scripts/run.sh` uses `<repo>/.venv` (or `ROBOFAB_PY`) when it exists; the
multi-environment setup below keeps working when it does not.

## v3.1 — the MuJoCo twin (CPU)

WP3–WP8 were executed in a CPU MuJoCo twin of the cell (master_report §0.6,
amendments §7.7). No Isaac or Docker is needed. Physics is CPU MuJoCo, one
process per core (`--workers`, default = physical cores for the runner, 8 for
PPO); the PPO update runs on CUDA when torch sees a GPU (`--device`).

### Legacy multi-environment setup

Before `setup_env.sh` the twin had its own Python 3.11 environment (the v4 code
used the Isaac Sim 6.0 interpreter and cuRobo a third one, see `scripts/run.sh`):

```bash
uv venv ../mjenv --python 3.11
uv pip install --python ../mjenv/bin/python -r requirements-twin.txt --torch-backend cu130
cd brickassembly            # every command below runs from here
export MJ_PY=../mjenv/bin/python     # or use bash scripts/run.sh <script>
```

| what | command | writes |
|---|---|---|
| G1 joint model (11 tests) | `$MJ_PY -m pytest tests/test_clutch.py -m "not slow"` | `tests/test_clutch.json` |
| G2 planner (+ single-arm failure in the twin) | `$MJ_PY -m pytest tests/test_planner.py [-m slow]` | `tests/test_planner.json` |
| G3 controller checks | `$MJ_PY -m pytest tests/test_control.py` | `tests/test_control.json` |
| G4 environment checks | `$MJ_PY -m pytest tests/test_env.py` | `tests/test_env.json` |
| plans (twin frame, hand-clearance checked) | `$MJ_PY planner.py --frame twin` | `plans/twin/*.json` |
| one assembly through the behaviour tree | `$MJ_PY -m orchestration.twin_executor --structure S3 --strategy weakest_joint --seed 0` | `results/twin_episodes.jsonl` |
| one A6 critical-step trial | `... twin_executor --structure S3 --step 12 --strategy nearest --seed 3` | |
| G3 benchmark (S1–S3 × 20) | `$MJ_PY -m experiments.runner g3 --trials 20` | `results/g3/` |
| A6 (every critical step × 3 strategies × seeds) | `$MJ_PY -m experiments.runner a6 --structures S3 --seeds 20` (or `--steps S3:11,13`) | `results/a6/` |
| WP5 runs R1–R4; R1ht, R1hc (A10) | `$MJ_PY -m tasks.ppo --run R1 --samples 300000` | `results/wp5/<run>/` |
| WP5 evaluation (calibrated joint; `--joint handtuned --stages 0` for A10's own joint) | `$MJ_PY -m tasks.ppo --eval-all` | `results/wp5/eval.jsonl` |
| G6 full loop (RL-first inserter, S1–S5 × 3 strategies) | `$MJ_PY -m experiments.runner g6 --policy R1 --seeds 1` | `results/g6/` |
| statistics + figures, report tables | `$MJ_PY -m experiments.analyze && $MJ_PY -m experiments.tables` | `results/analysis.json`, `results/figures/` |

Layout: `sim/mj/` (scene, bricks, impedance control, runtime, plan checks),
`sim/joint_model/` (clutch + capacity), `stability.py` (force-balance LP),
`bracing.py` (none / nearest / weakest_joint), `planner.py` (S1–S5, plans),
`motion/skills.py` (pick, transport, scripted insertion, brace),
`orchestration/twin_executor.py` (py_trees executor, §2.6 episode logs),
`orchestration/rl_inserter.py` (WP7: policy as the tree's inserter),
`tasks/` (WP5 environment + PPO), `experiments/` (batches, statistics).

The sections below this one describe the v3.0 GPU path (Isaac Sim, BrickSim,
Newton) and remain accurate for it.

## Status

| WP | State |
|---|---|
| WP0 environment | **G0 passed** — Newton, cuRobo and the impedance controller all measured |
| WP1 joint mechanics | branch 3a; **G1 closed with exceptions** — 5 pass, 1 guarded, 1 N/A, 2 cut |
| WP3 planner + bracing | lever model drives bracing; BrickSim cross-checks it and agrees |
| WP2 calibration | static fitted + live-sim adopted; per-stud target unreachable above 2×4 |
| WP4 motion stack | **working** — S1 6/6, S2 5/7, S3 5/5 (1 trial each), single arm, BrickSim |
| Dual-arm prototype | images → blueprint → plan → placer + stabilizer in Newton, live viewer — cube 8/8, arch 11/11 (braced crown), hollow box 10/12 (`dual_arm_sim.py`, [VIEWING.md](VIEWING.md)) |

```
env.lock                 pinned versions, both interpreters
ledger.jsonl             append-only project record (§5.5)
tasks.yaml               machine-readable task graph (§5.2)
scripts/00_env_check.py  WP0 / G0 probe -- run under BOTH interpreters
sim/newton_compat.py   ★ every Newton field name the project touches
blueprint.py             silhouette images -> voxel grid -> bonded brick layout
blueprints/              sample images: cube, hollow box, arch
planner.py               structures -> sequence -> grasps -> bracing -> assembly_plan.json
dual_arm_sim.py          plan -> two FR3s in Newton (placer + stabilizer), live viewer
run_assembly.py          plan -> cuRobo Franka in Viser -> snap gate -> episodes
```

### G0, measured 2026-09-20

| Check | Result |
|---|---|
| Newton ≥ 2048 envs, 2-brick scene | **PASS** — 363 406 env-steps/s (2048), 223 steps/s (1 env) |
| Warp kernel writes per-env body state | **PASS** |
| cuRobo plan latency < 50 ms | **PASS** — 31.7 ms median, 10/10 |
| Impedance 50 mm circle < 0.5 mm RMS | **PASS below 8.7 mm/s** — 0.09 mm static, 0.905 mm at 15.7 mm/s, 4.5 mm at 78 mm/s; pure lag, no oscillation. WP4 descends at 5 mm/s. |
| Container rebuild within 10% | waived — native venvs, see ledger |

Decision recorded at G0: Newton backend. **Reverted by WP1** — see below.

### WP1, measured 2026-09-20

BrickSim (`../BrickSim`, sha cbd5de3) cloned, installed and probed. One Kit
process per env count; 60 expert-driven steps each.

| envs | steps/s | env-steps/s | BrickSim solve |
|---|---|---|---|
| 1 | 118.0 | 118 | 0.001 ms |
| 4 | 77.3 | 309 | 0.003 ms |
| 16 | 37.9 | 606 | 0.009 ms |
| 64 | 26.5 | 1 696 | 0.037 ms |
| 256 | 12.1 | 3 109 | 0.15 ms |
| 512 | 6.8 | 3 459 | 0.35 ms |

**It batches.** 512 envs builds and runs, ~12.5M env-steps/hour. And the snap
solve is linear in env count but only **0.2% of step time** — the cost is PhysX
plus USD scene management, so a hand-written Warp kernel would optimise almost
nothing. Contribution (2), "GPU-batched snap mechanics at RL scale", is dropped.

Branch **3a**: adopt BrickSim. D2 reverts to PhysX, D1 to Isaac Sim 5.1, and
`sim/newton_compat.py` is superseded. D4 (direct vs manager-based env) is now
at risk — BrickSim's env is manager-based.

`static_solve` (built from source) answers the bracing query out-of-simulator:
0.4 ms per structure, returns `stable`, `slack_fraction` and
**`clutch_utilizations`** — the per-connection load fraction that replaces
planner.py's lever-arm model. Its clutch parameters are already exposed as
environment variables, which are WP2's calibration knobs.

The Newton-vs-BrickSim ratio is **not** a fair comparison (2-brick scene with no
robot vs a full task env) and must not be reported as one.

### Gate G1 — closed with exceptions

`tests/test_joint.py`, run headless under BrickSim's interpreter (**not** via
`python -m bricksim`, whose GUI experience dies on an RTX compat download that
404s):

| criterion | result |
|---|---|
| control: unconnected brick falls | PASS — 250 mm, proves the harness sees a live sim |
| mated pair holds 30 s | PASS — 0.00222 mm drift (< 0.05 mm) |
| energy non-increasing, 10 000 steps | PASS but **weak** — the nudge is absorbed by the joint, so it doesn't yet test energy pumping |
| two bricks cannot share a stud | **FAIL** — both connections authored, rival bricks co-located (0.000 mm apart) |
| equal-and-opposite wrench | **PASS, reframed** — structural, residual 3.2e-06 unloaded / 1.6e-05 pressed |
| f_break release ±15% | **PASS** — 89.6 N vs 89.6 N adopted, 0.5% repeatability. Self-consistent by construction, not an independent prediction |
| extraction > insertion | **PASS** — ratio 10.1, but imposed by configuration |
| gate boundary, 6 cases × 6 conditions | **cut** — same harness problem, see ledger |
| identical at 1 vs max envs | **N/A** — written for branch 3b; we don't own the solver |
| scripted insertion snaps from 2 mm | **cut** — harness could not place a brick at a controlled pose |

The patch now also emits **`clutch_forces`** — per-connection `{axial_n,
force[3], slopes[6]}` straight out of the QP solution, no debug-dump decoding.
`axial_n` is newtons, verified by linear scaling (8.903 at 35.6 N, 17.802 at
71.2 N). `bricksim_adapter.axial_forces()` and `release_force()` wrap it.

### WP2 — calibration

`sim/joint_model/calibration.py` measures BrickSim's preload → capacity
transfer function by bisecting a **pure axial pull-off**, then inverts it onto
the report's 8–15 N/stud envelope. `joint_calibration.json` is emitted per §2.6
and `bricksim_adapter` applies it to every solve by default.

| family | N/stud per N preload | rmse | free intercept |
|---|---|---|---|
| 2×N | 0.6000 | 0.0021 | 0.0026 N |
| 1×N | 0.7999 | 0.0081 | 0.0091 N |

Fitted `preloaded_force` = **18.832 N** → verified at 11.302 / 11.3019 / 11.3018
N/stud for 2×2 / 2×4 / 2×6. Residual RMSE 0.0019 N.

Three things that fell out of it:

- **1×N bricks carry 33% more per stud than 2×N.** The plan library must not
  assume one per-stud number.
- **Load mode changes the answer.** A half-lapped 2×4 reaches capacity at
  4.06 N/stud against 11.30 concentric, because pulling an overhanging brick
  pries the joint. Fitting both together gave rmse 6.88; separating them, 0.002.
- §WP2 steps 1–2 (derive the peaks from [MechanicsSnapFit]'s closed forms,
  identify the mechanical phase) were **not done** — the relations aren't in
  hand and inventing them would breach guardrail §5.4.5. This is the
  literature-calibrated fallback §WP2 allows; `phase_regime` is `null`, not
  guessed.

### WP2b — the dynamic calibration (clean sweep, adopted)

One Kit process per measurement (`scripts/02_dynamic_sweep.sh`), pure axial
ramp at 0.1 N/step, release filtered to the article's own prims:

| preload | 2×4 (8 studs) | 2×6 (12 studs) |
|---|---|---|
| 10 N | 2.15 | 2.08 |
| 20 N | 3.89 | 3.78 |
| 35 N | 6.80 | 6.06 |
| 50 N | 9.11 | 7.02 |
| 60 N | 9.88 | 7.16 |
| 70 N | 10.69 | 7.15 |
| 90 N | 11.00 | — |
| 120 N | **11.20** | **7.38** |

(N per stud. `per_stud = 0.4801 · preload^0.7104`, rmse 1.386 — sublinear,
unlike the static path.) Both articles are monotonic, so the earlier "2×6
ignores preload" was contamination. The isolated harness reproduced 79.0 N at
preload 60, matching the clean in-process repeats exactly.

**Release saturates at ~88–90 N per connection regardless of stud count.** So
per-stud capacity *falls* as bricks get larger — the opposite of a real clutch.
At the adopted preload a 2×4 sits at 11.2 N/stud (in envelope) while a 2×6 sits
at 7.4 (below the 8 N floor). **§2.3's 8–15 N/stud cannot be met across the
brick library at any single setting.** And the knob barely works up there:
+71% preload buys +4.8% force.

**Two paths, two preloads.** `static_solve` needs 18.832 N and the live sim
needs 120 N to produce the same ~11.3 N/stud on a 2×4. Using one value in both
misrepresents the joint by 6.4×, so `joint_calibration.json` carries
`bricksim_settings.static_solve` and `.live_sim` separately, and
`calibrated_thresholds(path=...)` makes callers choose.

### ⚠ How the static calibration was found not to transfer

The same 2×4 joint that the static fit says releases at **90.4 N** releases at
**36.75 N** in the dynamic path. WP5 trains in the dynamic path, so training on
the static number would hand the policy a joint ~2.4× weaker than intended.

Dynamic measurement is **reproducible to 0.5%** when the scene is clean
(79.0 / 78.9 / 78.6 N at preload 60), and quasi-static at or below 0.25 N per
physics step (1.0 N/step reads ~20% high). So the earlier scatter was harness
contamination, not physics: released bricks leave at speed and land on later
articles. The in-process fix (`deallocate_all_managed`) crashes the physics
view, so **the clean sweep still has to be done one measurement per process** —
the pattern `01_bricksim_sweep.sh` already uses.

Every force measurement in this project must run **one per process**. The same
2×4 article at the adopted preload reads 39.4 N inside a multi-test session and
89.6 N measured alone — 56% low, because released bricks from earlier articles
land on the one under test. `tests/test_joint.py` therefore no longer measures
pull-off itself; it reads the isolated sweep.

### ⚠ An earlier finding, and its correction, were both wrong

Capacity tracks `PreloadedForce` at roughly **capacity ≈ preload / 10.2**:

| preload | capacity per stud |
|---|---|
| 3.5 N (BrickSim default) | 0.34 N |
| 35 N | 3.42 N |
| 100 N | 9.78 N |

The report's §2.3 pull-off is **8–15 N/stud**, so BrickSim's default clutch is
**25–45× too weak**, and preload must land near **82–153 N**.

I first claimed S3 fails single-arm (true, but on BrickSim's uncalibrated
default, so meaningless). I then retracted that using preload 100 N — which was
derived from the **cantilever prying** ratio, not the pull-off ratio, and is
60 N/stud, four times above the envelope. With the measured calibration:

| clutch | S3 max util @ 35.6 N | S3L |
|---|---|---|
| 8 N/stud | 3.34 | 7.80 |
| 11.3 N/stud | 2.36 | 5.52 |
| 15 N/stud | 1.78 | 4.16 |

**S3 does meet the §WP3 preflight** — it fails single-arm across the whole
envelope. The original conclusion stands, now on a measured basis. The lesson
worth keeping: a capacity number is meaningless without its load mode; prying
and pure tension differ by ~2.8× on the same joint.

Caveats on the force readout: only the axial component is verified; and this is
the static path — the in-sim PhysX path shares the solver but is unchecked.

**G1 is closed with exceptions, not cleanly passed.** Two criteria were cut
after three failed harness attempts: placing a free brick at a controlled pose
above a mated one never worked (every case started at a measured −1.225 mm gap,
i.e. interpenetrating, which the gate correctly rejects). Both were measuring
*BrickSim's* gate — upstream code adopted under branch 3a — while the project's
real WP1 risks (energy injection, a joint that won't hold, wrong force
magnitude) are covered by criteria that pass. Gate behaviour gets exercised
end-to-end in WP4 anyway, where a failure shows up as a brick that doesn't
stick; that's the trigger to revisit.

The double-mate failure is only reachable by authoring an invalid topology,
which a real sequence never emits. **Stud exclusivity is ours to guarantee**,
not BrickSim's, and `planner._self_check` asserts it on every run.

A harness lesson worth keeping: the first two runs reported 0.0 mm drift and
perfectly constant energy — both false passes. `world.step(render=False)`
doesn't flush transforms to USD, so `SingleXFormPrim` reads returned authored
values. Every dynamic test here now reads poses from the physics view
(`SingleRigidPrim`) and ships a liveness control that gates the rest.

### WP4 — scripted baseline

`orchestration/run_plan.py` drives **BrickSim's own grasp and assemble skills**
(§0.3: don't rebuild what exists) from our planner's sequence, with our scene
setup and our episode logging. What runs today, on S1:

- Franka scene with the demo's tuned finger pads, drive limits and solver
  iterations; baseplate pre-placed, loose bricks arranged in the workspace
- our planner's order, not the demo's BFS re-sort — bracing is defined against
  our order
- grasp → transport → press → release → retreat for all six steps
- **grasp verified working**: every brick rises 179–183 mm
- one episode row per placement in `results/wp4_episodes.jsonl`

**First trials: S1 6/6, S2 6/8, S3 4/5.** Bricks seat at exactly 9.6 mm above
their stud brick with XY error under 0.5 mm.

#### Grasp reliability — fixed

Instrumenting the failures showed **two** modes needing different fixes:

| symptom | cause | fix |
|---|---|---|
| brick displaced tens of mm | gripper hit it | plain retry — the pose is re-read |
| brick displaced **0.0 mm**, 56 mm of free space, three identical attempts | gripper never reached it | **home reset** |

The second was the real one. RmpFlow is a reactive policy, not a planner, and
settles into configurations from which the next target is unreachable — which
is why early grasps in a run succeeded and later ones failed. Returning to the
home configuration before each attempt removed it. Grasp failures went
**2–3 per run → 0**.

Also fixed: my loop placed a brick once per *connection*, so a brick on two
supports was grasped twice and the second visit logged a spurious failure. The
demo's own loop skipped already-connected pairs; mine didn't. Totals now count
bricks.

#### Accessibility — alternate grasp face

The demo picks its grasp face from the brick's dimensions (`if W <= L` close
across W, else across L). Both branches grip the 2-stud dimension, so swapping
L and W doesn't change the grip width — it **rotates the approach 90°**.
Wrapping `parse_brick_prim_dimensions` gives a second grasp candidate without
touching BrickSim's source: §WP4.1's "two face pairs" at minimum cost. When a
press doesn't connect, the placement is retried on the other face.

Two accounting bugs fixed at the same time: placements were verified the
instant the skill returned (the gate can fire while the scene settles — an
episode was logged FAILED whose brick connected moments later), and a brick on
two supports produced two episode rows. Verification now waits 90 physics
steps and totals count distinct bricks. **Numbers before this are not
comparable with numbers after it.**

**Current, one trial each: S1 6/6, S2 5/7, S3 5/5.** The two remaining S2
failures fail on *both* grasp faces.

#### ⚠ A6 has lost its subject

S3 now assembles **unbraced, 5/5, including the cantilever-tip press** — so
the live sim does *not* reproduce the static prediction that it fails at
utilisation 2.36. The earlier tip-press failure was an accessibility artifact,
not physics, and I withdraw the suggestion that it corroborated the model.

The likely cause: the static analysis assumes a 35.6 N insertion press, while
the scripted inserter presses by commanding a 0.5 mm positional overshoot
through RmpFlow and the force that actually develops is unmeasured — the gate
reported press forces near 0 N earlier. Until a structure demonstrably fails
under the press the inserter really applies, **ablation A6 has no subject**.

#### Getting here: physics poses never reach USD headless

Placements were 0/5 until this was found. BrickSim's skills read every pose
through `SingleXFormPrim` (USD), but physics only flushes transforms to USD on
render — and headless there is no render. Measured: a brick's USD pose sat
**181.7 mm below** its physics pose immediately after a 181.7 mm lift. The
grasp still worked, because it uses the brick's pre-move pose; assembly could
not, because it needs the *current* pose of the carried brick and the gripper.

Ruled out along the way: the WP2b calibration (identical at 120 N and 3.5 N),
grasp failure (bricks lift 180 mm), scene colliders, `sync_to_usd` (already
on), and the `/physics/updateToUsd` carb settings (no effect). BrickSim's own
demo structure failed identically at 0/3, which is what proved it was our
harness rather than our planner's topology.

The fix injects a `SingleXFormPrim` subclass that reads the **physics view**,
plus a wrapper around `rmpflow.get_end_effector_as_prim`, redirecting every
pose read at once without editing BrickSim's source. Enabling the renderer
would also fix it, but `--enable_cameras` segfaults on this driver and the RTX
compat layer pins SHA-256 hashes for a package that has since moved — working
around that would mean disabling an integrity check on a binary loaded into
the process, which is the user's call, not mine.

Four harness facts needed to get this far, all in the ledger: enable
`isaacsim.robot_motion.motion_generation` explicitly; import `bricksim`
**before** any `enable_extension`; don't call `demo.main()` (it sets a viewport
camera that doesn't exist headless); and **play** the world rather than pausing
it, or the first skill waits forever for a physics callback.

### Force model cross-check

`planner.py --topology S3 [--upto-step N]` emits BrickSim `lego_topology@2`;
`sim/joint_model/bricksim_adapter.py` runs `static_solve` on it and returns
per-connection `clutch_utilizations`. `planner.py --crosscheck` records both
opinions per step:

| structure | lever model | BrickSim | agree |
|---|---|---|---|
| S1 column | nothing loaded | nothing loaded | ✓ |
| S2 wall | nothing loaded | nothing loaded | ✓ |
| S3 step 4 | `b_002→b_003`, margin 0.363 | `b_002→b_003`, util 0.00548 | ✓ |

### The insertion press (patched `static_solve`)

`patches/static_solve_applied_load.patch` adds applied loads to BrickSim's
tool, which hard-coded gravity:

```
static_solve <topology.json> [--load id,fx,fy,fz[,px,py,pz]] [--torque id,tx,ty,tz]
```

With no flags the output is byte-identical to upstream. `BreakageInput` already
carried the per-part impulses; only the CLI didn't expose them.

Pressing the brick being inserted on the **unbraced** S3 cantilever:

| press | max utilisation | stable |
|---|---|---|
| 0 N | 0.011 | yes |
| 1 N | 0.531 | yes |
| 2 N | 1.000 | **no** |
| 10 N | 3.576 | no |
| 35.6 N (a real 4-stud insertion) | **12.7** | no |

**S3 fails single-arm at ~6% of the force the task needs** — §WP3's preflight
requirement for ablation A6, now verified rather than assumed, and asserted in
the adapter's self-check so it can't silently regress.

Two things not to over-claim: below capacity the most-loaded joint is
`b_002→b_003` (both models agree), but past capacity the whole column overturns
and its joints sit within 0.03% of each other, so "which joint fails first" is a
near-tie. And the baseplate is the solver's free root, reacting at its centre of
mass rather than being held down by a table over an area.

BrickSim's gate defaults are **looser than §2.3.3 on every condition but yaw**
(2.0 mm vs 1.2 mm lateral, 5° vs 4° tilt, 1 N vs 4 N contact force, no dwell),
and its 5 mm `MaxPenetration` exceeds a 1.8 mm stud. WP2 decides each one.

Two traps found and worked around, both in the ledger:
`panda_instanceable.usd` 404s on the Isaac 6.0 asset tree (rewrite `6.0` → `5.1`),
and the OSC is unstable unless the task frame is placed **at the target** —
identity task frame gave 191 mm RMS with identical gains.

FORGE ships in this Isaac Lab tree (`direct/forge/`), not just as a paper: the
force-budget observation (D10) and the learned success predictor (D11) are
already implemented there. WP5 builds on `forge_env.py` rather than from scratch.

### Resolved and outstanding

1. **cuRobo cannot be imported next to Isaac Lab** (isdnenv py3.11/torch 2.14 vs
   Issac py3.12/torch 2.10). Resolved: plan offline in isdnenv, replay the joint
   trajectories in sim. No live replanning inside an episode.
2. **Docker waived.** §2.4 rule 2 says containerize; the stack is already
   working natively and reproducibility now rests on `env.lock` alone.

---

# The kinematic slice

A two-file vertical slice of `Docs/master_report.md`: a structure goes in, a
brace-annotated `assembly_plan.json` comes out, and a cuRobo-planned Franka
executes it in Viser while every placement is logged as an episode.

```bash
isdn                                          # curobo + viser env
cd brickassembly
python planner.py                             # self-check + write plans/*.json
python run_assembly.py --plan plans/S3_weakest_joint.json
# open http://localhost:8080
```

## What is real

| Report | Here |
|---|---|
| §2.3 LEGO geometry, voxel grid | `planner.py` constants, exact |
| §2.6 `assembly_plan.json` | emitted, same field names |
| §2.6 `insertion_episode.json` | emitted per placement, subset of fields |
| §3.5 sequencing | bottom-up, support-checked |
| §3.6 bracing, 3 strategies | `none` / `nearest` / `weakest_joint` |
| §2.5 coarse motion | cuRobo `MotionPlanner`, world updated per placement |
| §2.3.3 snap gate | lateral / dz / yaw checked against achieved FK pose |

`python planner.py` runs its own assertions: every brick supported when placed,
no cell overlaps, S1/S2 need no brace, S3 does.

## Measured, 2026-09-20

| Plan | Placed | Note |
|---|---|---|
| S3 (cantilever, 1 braced step) | 5/5 | |
| S2 (staggered wall) | 4/7 | 3 × `descend to target` unplanned |

The S2 failures are real, not a bug: the parallel-jaw fingers collide with the
neighbouring brick already in that layer, so cuRobo cannot plan the last 25 mm.
That is §3.4 accessibility (cite [BricksToBots]) arriving early. The fix is
grasp *selection* — the plan already carries `grasp.face_pair`, which
`run_assembly.py` currently ignores — or the clutch tool the report names as
the WP3 fallback. Either is real work; neither belongs in this slice.

## What is deliberately missing

- **No contact physics.** The gate is evaluated against the pose cuRobo
  achieves, so IK error is the only error source and success is near-certain.
  Force, compliance and the real joint model arrive with Isaac Lab + Newton
  (WP1/WP5). The gate and the episode log are the seams they plug into.
- **No second arm.** The brace pose is computed and drawn as a frame; nothing
  applies the force. Add the stabilizer arm when there is a force to resist.
- **Lever-arm force model, not BrickSim's QP.** `weakest_joint()` pushes the
  insertion load down the support chain and flags joints whose load lands
  outside their own stud footprint. It captures cantilevers (S3); it does not
  capture bending, so a symmetric span reads as pure compression.
- **Hand-authored structures, not StableLego** (deviates from D14) — swap
  `STRUCTURES` once the dataset is local.
- No grasp planning, no perception, no RL, no accessibility check.

## Next rung

1. Second arm for the brace pose (reuse `dual_franka.py`'s base offset).
2. Replace `weakest_joint()` with BrickSim's force distribution — same
   signature, same return.
3. Then WP1: real joint mechanics, which is where the gate stops being free.
