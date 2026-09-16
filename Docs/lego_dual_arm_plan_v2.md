# Dual-Arm LEGO Assembly — 7-Week Research Plan (v2: Isaac Lab + Newton)

**Author:** Sakib
**Stack:** Isaac Lab 3.0 Beta 2 · Isaac Sim 6.0.1 · Newton 1.4 · cuRobo · Warp
**Hardware:** Ubuntu 24.04, Ryzen 9 9950X3D, RTX 5090D (32 GB), 128 GB
**Scope:** Simulation only.
**Assumed effort:** 1 researcher + AI coding assistance, ~40 h/week.

> **Changes from v1:** platform moved from MuJoCo/MJX to Isaac Lab with the Newton backend; dual-arm role locked to *stabilizer + placer*; motion planning replaced by cuRobo; clutch model reworked to exploit Newton's Warp-kernel access and VBD deformables; a new contribution added (Section 3.4) that only Newton makes possible.

---

## 0. Scope Verdict

Unchanged from v1, and worth re-reading. Three hard problems; you can do one well.

| Sub-problem | Decision |
|---|---|
| Image → 3D → legal brick decomposition | **De-scoped.** 5 authored voxel structures. Image→voxel is a Week-7 stretch. |
| Contact-rich precision insertion under force + vision | **The contribution.** Full RL treatment + ablations. |
| Dual-arm coordination | **Locked: stabilizer + placer.** Arm A braces, Arm B presses. |

**Gate 2 (end of Week 3) remains the schedule's load-bearing wall:** a complete scripted end-to-end pipeline with ground-truth poses. Everything after is module replacement.

### New risk introduced by the platform change

Isaac Lab's Newton integration is a **beta on a beta**. NVIDIA explicitly warns of breaking changes, limited documentation, and no official support until release. Mitigations, all mandatory:

1. **Pin everything.** Isaac Lab tag `v3.0.0-beta2.patch1`, Isaac Sim 6.0.1, a specific Newton release, a specific Warp release, a specific cuRobo commit. Record hashes in `env.lock`. Never `git pull` on `develop` mid-project.
2. **Containerize.** Use the Isaac Lab Docker image as the base. Your earlier NVIDIA driver/library collision between Isaac Lab apt dependencies and MuJoCo is exactly the class of failure a container prevents. Do not install this into the system Python that your MuJoCo PKM work depends on.
3. **Keep the PhysX path alive.** Isaac Lab 3.0's multi-backend architecture exists precisely so environments run on either engine. Write the scene and task against the backend-agnostic interfaces; select the engine by config flag. If Newton blocks you in Week 4, you flip a flag rather than rewrite.

---

## 1. Platform Decision — Newton vs PhysX, honestly

You're right that the snap constraint is better served by Newton, but for a more specific reason than "Newton is newer." Four concrete capabilities:

**(a) Warp-kernel access to live engine state.** Isaac Lab exposes Newton's `Model`, `State`, `Control`, and `Contacts` as live Warp arrays through `isaaclab_newton.physics.NewtonManager`, with a selection API (`get_attribute` / `set_attribute`) and `NewtonManager.add_model_change(...)` to signal modifications. This means the clutch can be a **GPU kernel operating on all environments in parallel**, with no per-env Python loop and no CPU sync. In MuJoCo/MJX you fight `eq_active` batching; here the batched path is the native one. This alone justifies the switch.

**(b) Differentiability.** Newton is a differentiable engine. You can fit clutch parameters by gradient descent against a reference force–displacement curve instead of hand-tuning. See 3.4.

**(c) VBD deformables.** Newton's VBD solver does thin-shell deformables with rigid-SDF soft contact and a stable Neo-Hookean material. A LEGO clutch *is* an elastic interference fit — VBD can simulate it properly. Too slow for RL, perfect as a **ground-truth reference** to calibrate the fast model against. This is the project's novel angle.

**(d) Native contact and joint-wrench sensors.** F/T sensing is first-class on the Newton backend rather than something you assemble from contact queries.

**What PhysX still gives you that Newton may not (verify in Week 1):** the `Factory`, `FORGE`, and `AutoMate` contact-rich assembly environments. These are Franka peg-insertion / gear-mesh / nut-thread tasks and 100 plug-socket assembly tasks, with SDF collision and force sensing, and they are the best available template for your insertion task. If they do not yet run on Newton, **read them anyway** — the reward structure, action space, and success criteria transfer directly.

**Decision:** Newton (MuJoCo-Warp solver) as primary, PhysX as fallback, both reachable by config flag. VBD used only for offline calibration.

---

## 2. Locked Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Isaac Lab 3.0 Beta 2, **Direct RL env** (not manager-based) | Lower per-step overhead; CUDA-graph support for direct tasks; the Factory envs are Direct, so the template matches. |
| Physics | Newton, `SolverMuJoCo` (MJWarp) | Best-supported Newton solver in Isaac Lab; MJWarp handles articulations + contacts + constraints. |
| Motion planning | **cuRobo MotionGen** | Already installed. GPU batch IK + collision-aware trajectory optimization. Replaces the entire RRT module from v1. |
| RL library | **`rsl_rl`** (PPO + student–teacher distillation) | Native to Isaac Lab, used by the Newton Franka/KUKA-Allegro stacking tasks, and its distillation workflow with a privileged-state critic/teacher is exactly Section 7.4's shape. |
| **Placer arm (B)** | Franka Panda + **parallel jaw, custom thin fingertips** | See 2.1. |
| **Stabilizer arm (A)** | Franka + **Allegro hand** (primary), parallel jaw (fallback) | See 2.1. |
| Renderer | Isaac Lab tiled camera (`TiledCamera`) | Vectorized rendering is what makes vision-in-the-loop RL tractable at 1k+ envs. |
| Assets | OpenUSD, programmatically authored | Bricks generated from a Python spec, not hand-modelled. |

### 2.1 The gripper question — pushing back on the hand

You asked for a human-like hand if available. The honest answer is **yes for the stabilizer, no for the placer**, and this split actually strengthens the project.

**Why a multi-fingered hand is the wrong placer.**
- A press-fit needs 15–40 N along a tightly controlled axis. A multi-finger grasp on a 15.8 × 31.8 mm brick is compliant; under insertion load the brick shifts *in the hand*, which is precisely the pose error the RL policy exists to eliminate. You'd be adding the disturbance you're trying to reject.
- Action space goes from 6-D Cartesian delta to 6-D + 16 finger DoF (Allegro). Sample complexity roughly squares. In a 7-week budget that is the whole budget.
- Contact count per environment explodes → fewer parallel envs → slower training, on top of the harder problem.
- Allegro fingertips are ~15–20 mm across. In a populated brick field the clearance is worse than a 4 mm custom jaw.

**Why a hand is the right stabilizer.** This is the interesting part. A parallel jaw can only pinch along one axis. A multi-fingered hand can **conform around an irregular partial sub-assembly** and resist the insertion reaction wrench from several directions at once — which is the actual functional requirement for bracing an arch or an overhang. That gives the hand a real job, makes ablation A9 (hand vs. jaw stabilizer) a genuine question, and lets you say "dexterous hand" in the abstract without it being decoration.

**Available assets:** Isaac Lab ships Allegro and Shadow Hand, an integrated KUKA + Allegro arm–hand (DexSuite, plus the Newton stacking tasks), and Franka with Robotiq grippers and suction cups. Franka + Allegro is not stock but is straightforward to compose in USD. If composing it costs more than a day, **use KUKA + Allegro for arm A** — the stabilizer's kinematics don't need to match the placer's.

**Documented alternative for the placer: a clutch tool.** A wrist-mounted 2×4 anti-stud plate that clutches onto a brick's studs from above, plus an ejector pin to release. This is close to how real automated LEGO handling works. It gives perfect top-down clearance in dense fields and applies press force directly along the stud axis, which dissolves the accessibility constraint (Section 4.3) entirely. The cost is that picking becomes its own insertion problem. **Not recommended as primary** — the parallel jaw is the honest hard version and keeps the accessibility coupling that makes your planner interesting — but note it in the paper as a design alternative, and keep it as a Week-7 contingency if accessibility turns out to block S4/S5.

### 2.2 Day-1 compute check

Blackwell (`sm_120`) support: Isaac Sim 6.0 and recent Warp builds target it, and NVIDIA's own Isaac Lab benchmarking puts a GeForce 5090 workstation close to a dual RTX Pro 6000 server on the Franka cabinet-drawer task — so your machine is viable, not marginal. Still, verify empirically on day 1 (`scripts/00_backend_check.py`), and note that 32 GB VRAM will cap you well below the 4096-env figure in v1 once tiled cameras are on. Budget **2048 envs state-only, 512–1024 with vision**, and measure rather than assume.

---

## 3. Technical Specifications

### 3.1 LEGO Geometry and Brick Library

Unchanged from v1 — this is physics, not platform.

| Quantity | Value (mm) |
|---|---|
| Stud pitch P | 8.00 |
| Brick height | 9.60 |
| Plate height | 3.20 |
| Stud diameter | 4.80 |
| Stud height | 1.80 |
| Footprint, n studs | 8n − 0.2 |
| Wall thickness | 1.50 |
| Radial interference | ~0.10 |
| Pull-off force per stud | ~8–15 N (tunable; calibrate in 3.4) |

**Library (8 SKUs):** `1x1 1x2 1x4 1x6 2x2 2x3 2x4 2x6`, standard height only.
**Voxel grid:** `8.0 × 8.0 × 9.6 mm`, integer `(i,j,k)`, baseplate top at `k=0`.

### 3.2 Brick Assets in USD

Generate programmatically from `configs/bricks.yaml`; never hand-model. One Python function per SKU emitting a USD prim.

**Visual mesh:** full detail (studs, fillets, tubes) — cheap, render-only.

**Collision geometry — hollow shell, five convex boxes plus stud cylinders:**

```
top plate    : box, 1.0 mm thick, at z = +4.3 mm (half-extent 0.5)
wall +x/-x   : box, 1.5 mm thick, full depth, cavity depth 8.6 mm
wall +y/-y   : box, 1.5 mm thick
studs (n)    : cylinder r = 2.4, h = 1.8, at z = +5.7 mm
```

A solid box is wrong: it forces a mated brick to sit 1.8 mm high on top of the studs. The hollow shell lets studs enter the cavity.

**Physics attributes:**

```yaml
brick:
  mass: 0.0025            # 2x4, scale by stud count
  collision_approx: convexDecomposition   # PhysX path
  friction: {static: 0.6, dynamic: 0.55, restitution: 0.02}
  contact_offset: 0.0003
  rest_offset: 0.0001
  max_depenetration_velocity: 0.5
  solver_iters: {position: 24, velocity: 4}
```

**Key numbers that will save you a week:**
- `contact_offset` / `rest_offset` must be *small* here. Defaults tuned for decimetre objects will make 4.8 mm studs behave like mush. 0.3 mm / 0.1 mm is the right order.
- Physics `dt = 1/240 s` with 4 substeps (effective 1/960) for training; `1/500` with 8 substeps for evaluation. Below ~5 mm features, coarse stepping tunnels.
- A 2×4 brick is ~2.5 g against a ~10 N clutch force. **The robot dominates, not gravity.** Don't scale forces down.
- Contact filtering: once two bricks are clutched, exclude their stud–cavity pairs, or residual contacts fight the clutch force and inject energy.

**SDF collision (PhysX path only):** if you fall back to PhysX, enable SDF mesh collision on studs and cavities. That is the mechanism Factory used to make tight-tolerance assembly tractable, and it's a large accuracy win over convex decomposition for this geometry.

### 3.3 The Clutch Model — Newton implementation

Same state machine as v1, but now a GPU kernel rather than a Python constraint toggle.

```
DISENGAGED ──(gate)──▶ ENGAGING ──(ramp 50 ms)──▶ MATED ──(F_pull > F_break)──▶ DISENGAGED
```

**Gate conditions** (all must hold between an active brick's anti-stud frame and a placed brick's stud frame):

| Condition | Threshold |
|---|---|
| Lateral offset ‖Δxy‖ | < 1.2 mm |
| Vertical gap Δz | ∈ [−0.3, +1.0] mm |
| Stud-axis misalignment | < 4° |
| Yaw error to nearest 90° | < 5° |
| Downward contact force along stud axis | > 4 N |
| Dwell (continuous) | > 40 ms |

**Implementation pattern.** Allocate a persistent per-`(env, connection)` Warp struct array:

```python
@wp.struct
class ClutchState:
    phase:      wp.int32     # 0 DISENGAGED, 1 ENGAGING, 2 MATED
    timer:      wp.float32   # dwell / ramp accumulator
    body_a:     wp.int32     # moving brick body index
    body_b:     wp.int32     # placed brick body index
    frame_a:    wp.transform # anti-stud frame in body A
    frame_b:    wp.transform # stud frame in body B
    n_studs:    wp.int32
    f_break:    wp.float32   # randomized per episode
```

Each substep, launch one kernel over `(num_envs × max_connections)`:

1. Read body transforms and spatial velocities from the Newton `State` arrays.
2. Evaluate the gate; advance/reset `phase` and `timer`.
3. If `ENGAGING` or `MATED`, compute the snapped target pose `T*` (brick exactly on the stud lattice) and accumulate a critically-damped spatial wrench into the body force array:

```
F = ramp · [ k_p (p* − p) − k_d v ]      k_p = 4000 N/m,  k_d = 12 N·s/m
τ = ramp · [ k_r log(R* Rᵀ) − k_dr ω ]   k_r = 4 N·m/rad, k_dr = 0.02
ramp = clamp(timer / 0.05, 0, 1)
```

4. If `‖F‖ > n_studs · f_break`, transition to `DISENGAGED`.

Apply equal-and-opposite wrench to `body_b` so momentum is conserved — a one-sided force is a silent energy source that will destabilize tall structures.

**Integration point:** a pre-physics-step callback in the Isaac Lab Direct env, after `model.collide()` and before `solver.step()`. Acquire the Newton objects through `NewtonManager` **after** backend init, and treat them as invalid across a model rebuild.

> **API caveat:** the *shape* of this API (NewtonManager → Model/State/Control/Contacts as Warp arrays, selection get/set, `add_model_change`) is verified as of 15 Sep 2026, but exact field names for body transforms / applied forces move between Newton releases. Check the installed version's API reference on day 1 and write a thin adapter module (`sim/newton_compat.py`) so a rename costs one file, not fifty.

**Second implementation — loop joints.** Newton converts MuJoCo equality constraints to loop joints or mimic constraints on MJCF/USD import (`convert_mjc_equality_constraints=False` retains legacy arrays). A pre-allocated pool of disabled fixed loop joints, enabled on gate, gives exact rigidity — better for stability/drop tests than a spring. Runtime joint *creation* triggers a model rebuild and is too expensive; pre-allocation is the only viable route. **Test this in Week 1; if it works, use it for evaluation runs and keep the kernel for training.**

### 3.4 Clutch Calibration Against a VBD Reference — *the novel contribution*

This is the part of the project that only exists because you chose Newton. Budget 3 days in Week 2.

**Setup.** Model a single 2×4 brick's cavity walls as a VBD thin shell: ABS, E ≈ 2.3 GPa, ν ≈ 0.35, wall thickness 1.5 mm, Neo-Hookean triangle material, rigid–soft SDF contact against a rigid stud. Mesh the wall at ~0.3 mm so the 0.1 mm interference is resolved.

**Experiment.** Quasi-static insertion and extraction of one stud into one cavity, displacement-controlled at 0.5 mm/s, 64 parallel instances with randomized interference (0.05–0.15 mm) and friction (0.3–0.7). Record the axial force–displacement curve, the lateral stiffness at full engagement, and the peak extraction force.

**Fit.** Newton is differentiable — fit the analytic model's `(k_p, k_d, f_break, ramp_time, gate tolerances)` to the reference curves by gradient descent on an L2 loss. Report the residual.

**Why this matters for the writeup.** Every RL assembly paper that uses a simplified mating model hand-tunes it and hopes. You get to write: *"we calibrate a GPU-scale analytic clutch model against a deformable-shell reference and report a quantified fidelity gap."* That's a real methodological contribution, and it's about half a day of writing once the numbers exist.

**Fallback if VBD is too slow or unstable:** fit against published LEGO clutch-power measurements instead, report it as a literature-calibrated model, and note the VBD attempt as future work. Do not let this consume more than 3 days.

### 3.5 Scene Layout

```
Table    : 1.2 × 0.8 m, top at z = 0.75
Arm B    : placer,     base (+0.45, 0, 0.75), facing −x   [Franka + thin parallel jaw]
Arm A    : stabilizer, base (−0.45, 0, 0.75), facing +x   [Franka/KUKA + Allegro]
Baseplate: 32×32 studs (256 mm), fixed to table, centre (0, 0, 0.75)
Feeder B : tray (+0.30, −0.25), 30 bricks, randomized pose on reset
Feeder A : tray (−0.30, −0.25) — only used if arm A also places (ablation A6)
Cameras  : overhead (0,0,1.35) ↓, 640×480, fovy 55°
           wrist_B on link8, 60 mm behind fingertips, 30° down, 640×480, fovy 60°
           wrist_A equivalent
```

Workspaces overlap over the baseplate. That's intentional; it's what makes coordination non-trivial.

### 3.6 Sensors

| Sensor | Isaac Lab / Newton mechanism | Noise |
|---|---|---|
| Wrist F/T | Newton **joint-wrench sensor** at the wrist joint, or contact sensor aggregation on the hand | σ = 0.15 N / 0.01 N·m; bias random-walk 0.02 N/s per episode; 2nd-order LPF @ 30 Hz; quantization 0.05 N; **1–2 control-step latency** |
| Joint state | Articulation view | σ = 1e-4 rad, 5e-3 rad/s |
| RGB / depth | `TiledCamera` | RGB σ = 3/255, exposure ±20%; depth σ = 0.001 + 0.002·z², 1 px blur, 2% dropout |
| Contact events | Newton contact sensor | used for clutch gating and for the `engagement` reward term |

Do not train on noiseless F/T. A policy given perfect force timing will learn to exploit the exact contact impulse and teach you nothing.

### 3.7 Controller Stack

```
RL policy / scripted skill     20 Hz   → Δpose in EE frame
  ↓ policy-level action integrator (accumulate target, clip)
Task-space impedance control  240 Hz   → joint torques
  ↓
Newton solve                  240 Hz × 4 substeps
```

Isaac Lab ships `OperationSpaceController` with null-space support; use it rather than reimplementing. Gains:

```
K_p translational : diag(1200, 1200, 600) N/m     # deliberately soft in z
K_p rotational    : diag(40, 40, 40) N·m/rad
damping ratio     : 1.0
null-space        : posture task toward a mid-range configuration
```

Compliance in z is what lets the brick find the socket instead of jamming. The **policy-level action integrator** (accumulate pose targets rather than command absolute poses) is the Factory/IndustReal trick that makes small-delta action spaces work under contact — implement it from the start, not as a fix later.

---

## 4. Module M1 — Build Plan Generator

### 4.1 Target Structures (fixed, authored by hand)

| ID | Name | Bricks | Layers | Difficulty introduced |
|---|---|---|---|---|
| S1 | Tower | 6 | 6 | single-column stacking; sanity check |
| S2 | Wall | 12 | 4 | multi-brick layers, seam staggering |
| S3 | Arch | 18 | 5 | **unsupported span → requires bracing** |
| S4 | Staircase | 24 | 6 | accessibility / gripper clearance |
| S5 | Chair | 31 | 7 | overhangs + sub-assembly; full difficulty |

**Design S3 and S5 so single-arm assembly genuinely fails.** Ablation A6 is the only test of your title. If one arm does just as well, the honest move is to say so.

### 4.2 Pipeline

**Step 1 — Voxelize.** `trimesh.voxel.creation.voxelize(mesh, pitch)` at plate resolution → `.fill()` → collapse to brick layers by majority vote. Output boolean `V[i,j,k]`.

**Step 2 — Brick tiling, per layer, bottom-up.** Beam search, width 20, raster order, 10 restarts.

```
J(T) =  10.0 · coverage(T)
      −  0.1 · |T|                          # prefer fewer, larger bricks
      +  2.0 · stagger_score(T, T_{k−1})    # seams must not align across layers
      +  3.0 · connectivity_score(T, T_{k−1})
      −  5.0 · Σ_b inaccessible(b)          # ← cuRobo batch IK, Section 4.3
```

Runs well under a second for 32×32 layers. Do not reach for an ILP.

**Step 3 — Stability repair.** Connectivity graph `G = (bricks, shared-stud edges)`, edge weight = shared studs. Find articulation points and weight-<2 edges; locally re-tile within a 3-stud radius to raise the min-cut. Cap at 20 iterations. (Luo et al., *Legolization*, SIGGRAPH Asia 2015, §4.)

**Step 4 — Physical validation.** Instantiate the whole structure with all clutches forced `MATED`, hold 2 s under gravity, apply a 0.5 N lateral impulse at the top. **Accept if max displacement < 1.0 mm.** This is ground truth; the graph heuristic is not.

**Step 5 — Sequencing.** Precedence graph:
- `b_j` precedes `b_i` if `b_j` supports `b_i`.
- `b_j` precedes `b_i` if `b_i`'s top-down insertion corridor passes through `b_j`.

Greedy topological order with lookahead-3:

```
cost(b) = 1.0 · travel_time(arm_B, current, b.pick, b.place)
        + 1.0 · needs_brace(b) · brace_setup_time
        + 0.3 · brace_repose_penalty     # arm A already in a usable brace pose?
```

**Bracing schedule.** A brick is marked `requires_brace` if, with clutches on its supports disabled, a static analysis shows the local sub-assembly cannot resist the insertion wrench (≈ `n_studs × 12 N` downward plus ±3 N lateral). For each such brick, schedule arm A to a brace pose on the nearest structurally-connected placed brick, applying 10–15 N opposing the insertion axis, **before** arm B begins its approach. Arm A holds until arm B's snap is verified.

### 4.3 Accessibility Check — cuRobo makes this cheap

For candidate brick `b` with placed set `P`:

1. Enumerate 4 grasp configurations (2 face pairs × 2 approach yaws).
2. Build the gripper swept volume: fingertips at grasp offsets, swept from `z_target + 40 mm` down to `z_target`.
3. **Batch-query cuRobo**: collision-check all 4 configurations × all candidate bricks in the layer in a single GPU call against a world model containing `P`. This is the payoff for having cuRobo — what was a serial mesh-collision loop in v1 is now one batched call.
4. If all 4 fail → mark inaccessible → the tiler is penalized and re-tiles.

Update the cuRobo world model incrementally after each placement (add the new brick's collision mesh) rather than rebuilding.

This coupling between gripper geometry and the build plan is genuinely under-explored. Call it out in the writeup.

### 4.4 Optional: Image → Voxel (Week 7 stretch, 2 days max)

Single RGB → off-the-shelf image-to-3D (TripoSR / InstantMesh / Stable Fast 3D) → mesh → scale normalization → voxelize → Step 2.

Expect **poor** results: these models produce noisy, thin-shelled, non-watertight meshes that voxelize into disconnected fragments at 8 mm pitch. Mitigations: morphological closing, largest-connected-component filter, manual scale prompt. **Report a negative result if it fails** — that is a legitimate finding, not a failure. Two days, hard stop.

---

## 5. Module M2 — Perception

### 5.1 Two-stage design

| Stage | Camera | Purpose | Accuracy needed |
|---|---|---|---|
| Coarse | overhead + wrist | which bricks, what type, where | ±3 mm, ±5° |
| Fine | wrist crop | features into the RL policy | not a pose estimate |

**The coarse stage does not need to be accurate**, because the RL policy exists to close the last 3 mm. State this explicitly — it's the architectural justification for the two-phase design.

### 5.2 Coarse detection + pose

- **Data:** 20k renders via Isaac Lab's replicator/tiled-camera path. Randomize brick count (5–30), types, colours, tray poses, 3 lights (position, intensity 0.5–2.0, colour temperature), table/tray materials, camera extrinsics (±10 mm, ±3°), exposure, 0–5 distractor primitives.
- **Model:** YOLOv8s-seg, 640×640, COCO init. ~2 h on the 5090. Target mAP50 > 0.9 (easy in sim).
- **Pose:** mask ∩ depth → point cloud → centroid (position) + top-surface PCA (yaw) → point-to-plane ICP against the known brick mesh. Snap yaw to nearest of 4 symmetric solutions.
- **Deliverable that matters:** the measured error distribution. That distribution *is* the RL curriculum's Stage-4 initial-error spec. Measure it, don't guess it.

### 5.3 Fine-stage visual input

- Crop 128×128 from wrist camera, centred on the projected target socket.
- Channels: RGB (3) + depth (1), depth normalized over 5–200 mm.
- Encoder: 4-layer CNN (32/64/128/256, stride 2, GroupNorm, SiLU) → global pool → 64-d. ~0.4 M params; this runs inside the RL loop, keep it small.
- **Train state-first, add vision second** (Section 7.4). Do not co-train vision and control from scratch in a 7-week project.

---

## 6. Module M3 — Coarse Motion (cuRobo)

### 6.1 Grasp generation

Four antipodal grasps per brick (2 face pairs × 2 yaws), scored:

```
score = clearance_margin_mm − 2.0·|offset_from_centroid_mm| + 5.0·aligned_with_place_orientation
```

Fingertips centred at 4.8 mm above the brick base; grip width = footprint − 1.0 mm (0.5 mm squeeze per side); grip force 15 N. Validate each grasp offline with a shake test (±0.05 m at 3 Hz for 1 s); prune failures permanently.

### 6.2 Motion planning

- **`MotionGen`** for both arms: batched IK + collision-aware trajectory optimization, with graph planning enabled for global motion. Warm up kernels once at startup to avoid the first-plan latency spike.
- **World model:** table, baseplate, feeders, the growing structure (incrementally updated), and **the other arm as a dynamic obstacle**.
- **Dual-arm coordination — prioritized planning.** Arm A (stabilizer) plans first, because its brace pose is a task constraint, not a preference. Arm A's final configuration is then added to arm B's world model as a static obstacle for the duration of the placement. This is far simpler than 14-DoF centralized planning and is adequate here — the stabilizer is nearly static during insertion, which is exactly what makes the prioritized decomposition valid. Say this in the paper; it's a real justification, not a shortcut.
- **On plan failure:** retreat arm A to a secondary brace pose on the same support brick and retry; then serialize; then escalate to the recovery branch.

### 6.3 Phase handoff

The coarse phase delivers the brick to a **pre-insertion pose**: directly above the target socket, `+25 mm` in z, with injected error sampled from the *measured* perception error distribution (5.2). Matching the handoff error distribution to real perception error is what makes the pipeline internally consistent — get this right and the RL curriculum's top stage is automatically the correct one.

---

## 7. Module M4 — RL Precision Insertion

### 7.1 MDP

**Episode:** starts at pre-insertion pose with a grasped brick and arm A already bracing; ends on snap success, force violation, drop, or timeout (200 steps @ 20 Hz = 10 s).

**Actor observation (deployed):**

| Group | Dim | Contents |
|---|---|---|
| Proprioception | 24 | `q`(7), `q̇`(7), EE position(3), EE rotation 6D(6), gripper width(1) |
| F/T history | 60 | last 10 steps × 6D wrench, EE frame, normalized by 30 N / 2 N·m |
| Visual embedding | 64 | CNN over 128×128 RGB-D wrist crop |
| Task context | 21 | Δpose target→current (3+6), brick type one-hot(8), engagement scalar(1), n_studs(1), reserved(2) |
| Brace state | 7 | arm A contact wrench (6) + brace-active flag (1) |
| Previous action | 6 | |
| **Total** | **182** | |

**Privileged (critic only), +26:** true brick→socket relative pose (9), per-stud engagement depth (≤8), sampled friction (1), sampled `f_break` (1), contact count (1), noiseless wrench (6).

The **brace state group is new in v2** and is the observational consequence of locking the dual-arm role. The placer policy can feel whether the structure is actually being held. Ablation A6 zeroes this group.

**Action:** 6-D `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw]` in EE frame, `tanh`-squashed to ±2 mm / ±2°, integrated by the policy-level action integrator. Optional 9-D variant adds per-axis translational stiffness scaling in [0.3, 2.0] — run as ablation A8, not default.

**Reward:**

```python
r_dist   =  1.0  * exp(-‖p_err‖ / 0.005)          # σ = 5 mm
r_align  =  0.3  * (1.0 - angle_err / π)
r_engage =  0.5  * (engagement_depth / 1.8mm)     # ← the term that actually matters
r_force  = -0.5  * max(0, ‖F‖ - 25.0) / 35.0
r_action = -0.01 * ‖a‖²
r_time   = -0.005
r_snap   = +50.0  (terminal)
r_fail   = -10.0  (terminal: ‖F‖>60 N, drop, timeout)
```

A pure distance reward plateaus at "hovering correctly above the socket." `r_engage` is what drives studs into the cavity.

### 7.2 Curriculum

Advance when success rate over the last 100 episodes > 80%; regress below 50%.

| Stage | Pos error | Rot error | Socket occupancy | Bricks |
|---|---|---|---|---|
| 0 | U(0, 2 mm) | U(0, 2°) | isolated | 2x4 |
| 1 | U(0, 5 mm) | U(0, 5°) | isolated | 2x4, 2x2 |
| 2 | U(0, 10 mm) | U(0, 8°) | 1 neighbour | all 2xN |
| 3 | U(0, 15 mm) | U(0, 12°) | 2–3 neighbours | all 8 |
| 4 | measured perception noise (5.2) | — | ≤4 neighbours | all 8 |

### 7.3 Domain randomization (per episode)

| Parameter | Range |
|---|---|
| Brick–brick friction | U(0.3, 0.9) |
| Brick–finger friction | U(0.5, 1.2) |
| Clutch break force / stud | U(6, 16) N *(narrow to the 3.4-calibrated range once available)* |
| Gate lateral tolerance | U(0.9, 1.5) mm |
| Brick mass | ±15% |
| Baseplate pose | ±1 mm, ±0.5° |
| Impedance gain multiplier | U(0.8, 1.25) |
| Brace force (arm A) | U(8, 18) N, 5% chance of brace slip |
| F/T noise, bias | per 3.6 |
| Control latency | 0–2 steps |
| Camera extrinsics | ±2 mm, ±1° |

The **brace-slip randomization** is worth its weight: it forces the placer policy to be robust to a stabilizer that isn't perfect, which is the realistic case and a good talking point.

### 7.4 Training schedule

**Run A — state teacher (Week 4, ~4 h).** Privileged actor, no vision. `rsl_rl` PPO on Newton. Establishes that the task and reward are learnable before perception enters. **If Run A doesn't reach >85% at Stage 3, the bug is in your reward or your clutch model, not your RL.** Stop and fix it.

**Run B — residual policy (Week 4, ~3 h).** Base controller: scripted admittance insertion — descend 5 mm/s; on contact, 2 mm-radius Archimedean spiral search at 3 mm/s holding 8 N down, with ±1.5° wiggle about x/y at 2 Hz. RL outputs a residual. Converges much faster; a strong baseline whether or not it wins.

**Run C — vision student (Week 5, ~6 h).** Use `rsl_rl`'s student–teacher distillation with Run A as the explicit teacher checkpoint. The Isaac Lab Franka/KUKA-Allegro Newton stacking tasks already implement exactly this shape — one fixed external camera, canonical proprioception, per-episode camera randomization, privileged-state critic/teacher. **Copy that env structure rather than inventing one.** If distillation stalls, fall back to asymmetric actor-critic trained from scratch.

**PPO settings** (adjust `num_envs` to measured VRAM headroom):

```
num_envs       2048 (state) / 512–1024 (vision)
rollout_len    32            gamma          0.99
minibatch      16384         gae_lambda     0.95
epochs         4             clip_eps       0.2
lr             3e-4 cosine   entropy_coef   0.002 → 0.0005
value_coef     0.5           max_grad_norm  1.0
network        MLP [512,256,128] ELU, separate actor/critic trunks
total steps    120M (A/B), 200M (C)
```

Enable CUDA-graphing for the direct RL task — it's a meaningful overhead reduction in Isaac Lab 3.0.

Log per-stage success rate, mean peak force, insertion time, action-magnitude distribution. **Watch for slam-and-snap:** if peak force creeps toward 55 N while success rises, raise the force penalty weight.

### 7.5 Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Hovers, never contacts | `r_dist` saturates pre-contact; force penalty too harsh | raise `r_engage`; only penalize above 25 N |
| Success at the force cap | slam-and-snap exploit | raise force penalty; add jerk penalty on actions |
| Learns Stage 0–1, collapses at 2 | neighbour bricks genuinely harder; curriculum jump too big | insert intermediate stage; raise min-stage dwell |
| Structures explode / energy gain | one-sided clutch wrench, or residual contacts fighting the clutch | apply equal-and-opposite wrench; filter mated pairs |
| Bricks tunnel through studs | substep count too low for 4.8 mm features | 8 substeps; reduce `contact_offset` |
| Vision student ≪ teacher | teacher visits states the student can't recognize | DAgger; more camera DR; verify the crop contains the socket |
| Brick squirts out of gripper | torsional friction missing; grip force low | enable torsional friction; raise to 20 N |
| Newton API breaks after update | tracked `develop` | you were warned; `git checkout` the pinned tag |

---

## 8. Module M5 — Orchestration

`py_trees` behaviour tree. Do not hand-roll a state machine; you'll need the retry logic.

```
Sequence: AssembleStructure
 ├── LoadPlan(assembly_plan.json)
 └── Repeat(each step)
      └── Selector: PlaceBrick                        [≤3 retries]
           ├── Sequence: NominalPlacement
           │    ├── [A] MoveToBracePose + ApplyBraceForce   (if requires_brace)
           │    ├── [B] PerceiveFeeder → Grasp → TransportToPreInsertion
           │    ├── [B] RunRLInsertionPolicy                (timeout 10 s)
           │    ├── VerifySnap
           │    ├── [B] ReleaseAndRetract
           │    └── [A] ReleaseBrace
           └── Sequence: Recovery
                ├── LogFailure
                ├── [B] LiftAndRegrasp
                └── [B] ScriptedInsertionFallback
```

Note arm A brace **precedes** arm B's transport, not just its insertion — cuRobo plans arm B around a settled arm A rather than a moving one.

The **scripted fallback on the third retry** is what makes full-structure completion rates non-zero early and lets you report "RL-first with scripted fallback" as a system rather than a gamble.

---

## 9. Data Contracts

```jsonc
// assembly_plan.json
{
  "structure_id": "S3_arch",
  "voxel_origin": [0.0, 0.0, 0.75],
  "grid_pitch":   [0.008, 0.008, 0.0096],
  "bricks": [
    { "id": "b_007", "type": "2x4", "grid_pos": [4,2,1], "yaw_index": 0, "color": "red" }
  ],
  "sequence": [
    {
      "step": 7, "brick_id": "b_007", "placer_arm": "B",
      "requires_brace": true,
      "brace": { "arm": "A", "target_brick_id": "b_005",
                 "brace_pose": [0.12,-0.03,0.81, 0,0.7071,0,0.7071],
                 "brace_force_N": 12.0, "brace_axis": [0,0,1],
                 "end_effector": "allegro" },
      "grasp": { "face_pair": "long", "yaw_offset_deg": 0,
                 "grip_width_m": 0.0148, "grip_force_N": 15.0 },
      "pre_insertion_pose": [0.032,0.016,0.7936, 0,0,0,1],
      "target_pose":        [0.032,0.016,0.7696, 0,0,0,1],
      "mating_studs": [["b_003",2], ["b_005",2]]
    }
  ],
  "validation": { "graph_min_cut": 2, "drop_test_max_disp_mm": 0.42,
                  "all_bricks_accessible": true, "brace_required_count": 4 }
}
```

```jsonc
// insertion_episode.json — logged every attempt; this is your dataset
{ "brick_id":"b_007", "policy":"rl_vision_v3", "backend":"newton", "success": true,
  "steps":48, "duration_s":2.4,
  "initial_error_mm":7.3, "initial_error_deg":4.1,
  "final_error_mm":0.21, "final_error_deg":0.4,
  "peak_force_N":21.7, "mean_force_N":9.2, "impulse_Ns":22.1,
  "brace_active": true, "brace_peak_reaction_N": 14.2, "brace_slipped": false,
  "snap_time_s":1.9, "n_studs_engaged":8, "failure_mode": null }
```

```jsonc
// clutch_calibration.json — output of Section 3.4
{ "reference":"vbd_thinshell", "n_samples":64,
  "fitted": { "k_p":4120.0, "k_d":11.6, "f_break_per_stud_N":11.3,
              "ramp_time_s":0.047, "gate_lateral_mm":1.18 },
  "residual_rmse_N":0.83, "reference_peak_extraction_N":10.9,
  "lateral_stiffness_N_per_mm":42.1 }
```

---

## 10. Repository Structure

```
lego_dual_arm/
├── env.lock                       # ★ pinned: IsaacLab tag, IsaacSim, Newton, Warp, cuRobo
├── docker/                        # ★ derived from the Isaac Lab image
├── configs/
│   ├── scene.yaml  bricks.yaml  clutch.yaml  control.yaml  rl_ppo.yaml
│   └── structures/S1..S5.json
├── sim/
│   ├── brick_usd.py               # programmatic USD generation per SKU
│   ├── scene.py                   # backend-agnostic scene assembly
│   ├── newton_compat.py           # ★ thin adapter over Newton field names
│   ├── clutch_kernel.py           # ★ the Warp kernel
│   ├── clutch_joints.py           # loop-joint pool (evaluation path)
│   ├── calibration/vbd_reference.py   # Section 3.4
│   └── sensors.py
├── planner/
│   ├── voxelize.py  tiler.py  stability.py  sequencer.py
│   ├── accessibility.py           # cuRobo batch IK + collision
│   └── bracing.py                 # static analysis → requires_brace
├── perception/
│   ├── datagen.py  train_detector.py  pose_estimator.py
├── motion/
│   ├── curobo_planner.py          # MotionGen wrappers, world-model updates
│   ├── dual_arm.py                # prioritized planning, A-then-B
│   └── grasps.py
├── tasks/
│   ├── insertion_direct_env.py    # Isaac Lab DirectRLEnv
│   ├── curriculum.py  domain_rand.py  networks.py
├── orchestration/
│   ├── behavior_tree.py  skills.py  verify.py
├── eval/
│   ├── run_benchmark.py  ablations.py  figures.py
├── tests/
└── scripts/
    ├── 00_backend_check.py        # ★ day-1 Newton/PhysX/cuRobo/VRAM benchmark
    ├── 01_calibrate_clutch.py
    ├── 02_build_plan.py
    ├── 03_train_insertion.py
    └── 04_full_assembly.py
```

`★` marks where the project's risk concentrates. Write those yourself; delegate the rest.

---

## 11. Seven-Week Phase Plan

### Phase 0 — Platform (Week 1, Days 1–3)

| Day | Task |
|---|---|
| 1 | Install Isaac Sim 6.0.1 + Isaac Lab `v3.0.0-beta2.patch1` **in Docker**. Write `env.lock`. Run `00_backend_check.py`: steps/s on Newton vs PhysX for a 2-brick scene at 1/512/2048 envs; VRAM ceiling with and without tiled cameras; cuRobo MotionGen warmup + plan latency. |
| 1 | Run `Isaac-Factory-PegInsert-Direct-v0` on both backends. **Read its source properly** — it's your template. |
| 2 | `brick_usd.py`: all 8 SKUs, hollow-shell collision, stud/anti-stud frames as USD attributes. Visual inspection in the viewer. |
| 2–3 | `scene.py`: table, both arms, baseplate, feeders, 3 cameras, backend-selectable. Compose Franka+Allegro for arm A, or fall back to KUKA+Allegro. |
| 3 | `newton_compat.py`: verify the NewtonManager access pattern against the *installed* API; wrap every field name. Impedance controller free-space test: EE tracks a 50 mm circle with <0.5 mm RMS. |

**Gate 0 (Day 3):** Newton runs ≥2048 envs on a 2-brick scene; cuRobo plans in <50 ms; the impedance controller tracks free-space trajectories; you can read/write Newton state from a Warp kernel.
*Contingency:* if Newton blocks → switch the backend flag to PhysX, enable SDF collision on studs, and move the 3.4 VBD calibration to "future work." Cost: the novel contribution, not the project.

### Phase 1 — Clutch + planner (Week 1 Day 4 – Week 2)

| Days | Task |
|---|---|
| 4–6 | `clutch_kernel.py` + full unit tests (12.1). Both directions of the wrench. Energy-conservation check over 10k steps. |
| 6 | `clutch_joints.py` loop-joint pool; compare rigidity vs the spring on a 6-brick tower |
| 7 | Scripted teleop insertion test — drive a brick in, confirm snap fires, holds 30 s, releases at the right force |
| 8–10 | **VBD calibration (3.4).** Hard stop at 3 days. Produce `clutch_calibration.json` or declare the fallback. |
| 10–12 | `voxelize.py`, `tiler.py`, `accessibility.py` (cuRobo batched), `bracing.py`, `stability.py`, `sequencer.py`; author S1–S5; run the planner on all five |

**Gate 1a (Day 7):** scripted insertion snaps reliably from ≤2 mm error, holds 30 s, releases at plausible force, no energy injection.
**Gate 1b (Day 12):** all 5 structures produce valid plans passing the drop test with `all_bricks_accessible: true`, and S3/S5 have `brace_required_count > 0`.
*Contingency:* if S5 fails accessibility, either swap to a simpler 25-brick structure **or** switch the placer to the clutch tool (2.1). Do not redesign the Allegro hand.

### Phase 2 — Scripted end-to-end (Week 3)

| Days | Task |
|---|---|
| 13–14 | `grasps.py` + shake test; `curobo_planner.py` with incremental world-model updates |
| 15–16 | `dual_arm.py` prioritized A-then-B; brace skill with force regulation on arm A |
| 17–18 | Scripted insertion skill (spiral + admittance) |
| 19 | `behavior_tree.py` + `verify.py`; full loop on S1, S2 |
| 20–21 | Full loop on S3–S5 with **ground-truth poses**; debug |

**Gate 2 (end of Week 3) — critical.** With ground-truth poses and scripted insertion:
- S1, S2 complete ≥90% (n = 20 each)
- S3 completes ≥60% **with bracing active**
- Every module's JSON interface exercised

*Contingency if missed:* **stop all forward work.** A complete pipeline beats a brilliant policy with nothing to plug into. Demote S4/S5 to "extended structures."

### Phase 3 — RL (Weeks 4–5)

| Days | Task |
|---|---|
| 22–23 | `insertion_direct_env.py`; reward; verify env throughput; enable CUDA graphs |
| 23–24 | `curriculum.py`, `domain_rand.py`; reward sanity-check against a hand-coded expert |
| 24–25 | **Run A** (state teacher) + reward iteration |
| 26–27 | **Run B** (residual); compare A vs B |
| 28–30 | Sweep the winner: 3 seeds × {lr, entropy, force-penalty weight} |
| 31–33 | **Run C** (vision student) via `rsl_rl` distillation from A |
| 34–35 | Integrate the best policy into the behaviour tree; re-run S1–S3 |

**Gate 3 (end of Week 5):** ≥85% insertion success at Stage 3 (15 mm / 12°), mean peak force <30 N, integrated end-to-end.
*Contingency:* ship the **residual** policy as the main result and report end-to-end RL as an underperforming ablation. That is honest and publishable.

### Phase 4 — Perception (Week 6, Days 36–40)

| Days | Task |
|---|---|
| 36 | `datagen.py`: 20k renders with full DR via tiled cameras |
| 37 | Train YOLOv8s-seg; report mAP and per-class recall |
| 38 | `pose_estimator.py`; **characterize the error distribution** |
| 39 | Swap ground-truth poses for perception; retune curriculum Stage 4 to the measured error |
| 40 | Full loop S1–S5 with perception in the loop |

**Gate 4 (end of Week 6):** end-to-end perception-driven pipeline, ≥70% completion on S1–S3.
*Contingency:* if pose error >5 mm, widen Stage 4, accept lower completion, and report perception as the dominant failure cause.

### Phase 5 — Evaluation and writeup (Week 7, Days 41–47)

| Days | Task |
|---|---|
| 41–42 | Full benchmark: 5 structures × 20 seeds × 5 configurations = 500 assembly trials |
| 43 | Ablations (12.4) |
| 44 | Figures, tables, failure-mode taxonomy from logged episodes |
| 45 | Stretch — pick **one**: image→voxel, parallel dual-arm placement, or text→structure |
| 46–47 | Report draft; demo video; README and repo cleanup |

---

## 12. Evaluation

### 12.1 Unit tests (exist before Week 2 ends)

```
tests/test_clutch.py
  - gate fires inside tolerance, not outside (6 boundary cases per condition)
  - mated pair holds 30 s under gravity, drift < 0.05 mm
  - releases at f_break ± 15%
  - two bricks cannot clutch the same stud
  - ENERGY: total system energy non-increasing over 10k steps with no actuation   ← the one that matters
  - equal-and-opposite wrench: Σ forces ≈ 0
  - kernel result identical at 1 env and 2048 envs
tests/test_planner.py
  - tiling covers 100% of each authored model; no grid overlaps
  - every sequenced brick supported by already-placed bricks or baseplate
  - topological order valid; requires_brace flags match static analysis
tests/test_control.py
  - impedance step response <5% overshoot
  - null-space posture doesn't perturb task space (<0.1 mm)
  - F/T sensor within 5% of a known static load
tests/test_env.py
  - obs dim == 182, no NaNs over 10k random-action steps
  - reward bounded; all terminal conditions reachable
  - Newton and PhysX agree on 100 steps to <1e-3 (positions)     ← backend parity
```

### 12.2 Snap verification (runtime)

Verified if, 200 ms after release: all expected studs `MATED`, brick pose error <0.5 mm / 1°, velocity <1 mm/s. Else → recovery.

### 12.3 Primary metrics

| Metric | Target |
|---|---|
| Per-brick insertion success (incl. retries) | ≥90% |
| First-attempt success | ≥80% |
| Structure completion rate (S1–S3) | ≥70% |
| Mean peak insertion force | <30 N |
| Force impulse per insertion | reported |
| Mean insertion time (pre-insertion → verified) | <4 s |
| Cycle time per brick | <15 s |
| Final RMS brick pose error vs plan | <0.5 mm |
| Post-assembly robustness (1 N lateral impulse) | <2 mm displacement |
| **Brace reaction force (arm A)** | reported; correlate with placer success |

Report **mean ± 95% CI over 20 seeds.**

### 12.4 Ablations

| # | Ablation | Question |
|---|---|---|
| A1 | RL vs scripted spiral search | Does RL earn its place? |
| A2 | **no F/T** (vision only) | Is force sensing necessary? |
| A3 | **no vision** (F/T + proprio) | Is vision necessary at close range? |
| A4 | end-to-end RL vs residual RL | Does the scripted prior help or constrain? |
| A5 | no curriculum | Quantify the curriculum |
| A6 | **single-arm** (no brace) on S3/S5 | **Does the second arm matter?** |
| A7 | no DR, evaluated on held-out physics | Robustness |
| A8 | 9-D learned stiffness vs 6-D fixed | Is variable impedance worth it? |
| A9 | **Allegro stabilizer vs parallel-jaw stabilizer** | Does the hand earn its place? |
| A10 | **calibrated clutch vs hand-tuned clutch** | Does 3.4 change learned behaviour? |

**A6 is the one reviewers will care about most** — it's the only test of your title. **A10 is the one that makes 3.4 a contribution rather than an appendix**; if the calibrated model produces a measurably different policy (different force profiles, different insertion strategy), that's a result.

---

## 13. Risk Register

| # | Risk | P | Impact | Mitigation | Trigger day |
|---|---|---|---|---|---|
| R1 | Isaac Lab Newton beta breaks / lacks a needed feature | **High** | High | Pinned versions, Docker, `newton_compat.py`, PhysX fallback behind a config flag | 1, then continuous |
| R2 | Clutch kernel injects energy / destabilizes tall structures | Med | **Critical** | Equal-and-opposite wrench; energy unit test; loop-joint alternative | 6 |
| R3 | VBD calibration too slow or unstable | Med | Low | 3-day hard stop; literature-calibrated fallback | 10 |
| R4 | RL fails to learn insertion | Med | High | Residual RL primary fallback; Run A gates reward design before vision | 25 |
| R5 | Franka+Allegro USD composition eats days | Med | Med | 1-day cap → switch arm A to stock KUKA+Allegro | 3 |
| R6 | 32 GB VRAM caps envs below useful throughput | Med | Med | Measured day 1; reduce vision resolution to 96×96; 512 envs; longer wall-clock | 1 |
| R7 | Factory/AutoMate envs don't run on Newton | Med | Low | Read them for structure regardless; they're a template, not a dependency | 1 |
| R8 | Vision student ≪ state teacher | Med | Med | DAgger; or report state-based as main result with vision as a stated limitation | 33 |
| R9 | Image→3D unusable | **High** | Low | Already a stretch; report as negative result | 45 |
| R10 | **Scope creep back toward the original spec** | **High** | High | This document. Re-read Section 0 at every gate. | weekly |
| R11 | Isaac Lab install collides with the MuJoCo PKM environment | Med | High | Docker. Do not install into the system Python. | 1 |

R1, R2 and R10 are the three that actually sink projects like this. R11 is specific to your machine and has already bitten you once.

---

## 14. Stretch: Text → Structure (Week 7, if selected)

Prompt an LLM for a structured build spec, not prose:

```
System: Output only JSON: {"name": str, "dimensions_studs":[w,d,h],
"bricks":[{"type":str,"grid_pos":[i,j,k],"yaw_index":0|1}]}.
Types: 1x1 1x2 1x4 1x6 2x2 2x3 2x4 2x6. Every brick above k=0 must
share ≥2 studs with a brick below. No overlaps.

User: "a small chair with a backrest, about 6 studs wide"
```

Then a **validator + repair loop**: check overlap, connectivity, stability; feed violations back; cap at 5 iterations. Report pass rate before and after repair — the gap is the interesting number. Small, clean, good closing section.

---

## 15. Reading List

**Before Week 1:**
- Narang et al., *Factory: Fast Contact for Robotic Assembly* (RSS 2022) — the contact-handling ideas behind the Isaac Lab assembly envs.
- Tang et al., *IndustReal* (RSS 2023) — SDF rewards, policy-level action integrator, success prediction. The most directly applicable paper.
- Noseworthy et al., *FORGE* — force-conditioned policies with dynamics randomization; the closest prior work to your force-sensing story.
- Luo, Yu, Wang, *Legolization* (SIGGRAPH Asia 2015) §4 — the stability graph and repair in 4.2.
- **Source code:** `Isaac-Factory-PegInsert-Direct-v0`, and the Newton Franka / KUKA-Allegro stacking tasks. Read both before writing any env code.

**Before Week 3:**
- Sundaralingam et al., *cuRobo* (2023) — understand MotionGen's world-model representation before you fight it.
- Martín-Martín et al., *VICES* (IROS 2019) — action-space choice for contact tasks.
- Johannink et al., *Residual RL for Robot Control* (ICRA 2019) — the Run B formulation.

**Before Week 4:**
- Lee et al., *Making Sense of Vision and Touch* (ICRA 2019) — multimodal insertion; directly relevant to A2/A3.
- Schoettler et al., *Deep RL for Industrial Insertion with Visual Inputs* (IROS 2020).
- Tang et al., *AutoMate* (2024) — 100-task assembly, specialist→generalist distillation.

**Context:**
- Testuz, Schwartzburg, Pauly, *Automatic Generation of Constructable Brick Sculptures* (Eurographics 2013).
- Chung et al., *Brick-by-Brick* (NeurIPS 2021).
- Wang et al., *Translating a Visual LEGO Manual to a Machine Executable Plan* (ECCV 2022).
- Newton documentation: solver guide, MJWarp/VBD pages, and the Isaac Lab Newton native-data API reference.

---

## 16. If You Only Do Four Things

1. **Pin the stack and containerize on day 1.** Newton and Isaac Lab 3.0 are both moving targets under active development with explicit "expect breaking changes" warnings. An unpinned `develop` checkout will cost you a week at the worst possible moment, and Isaac Lab's dependencies have already broken your MuJoCo setup once.
2. **Get the clutch kernel right in Week 1, and write the energy-conservation test.** A one-sided wrench or an unfiltered mated contact silently injects energy; you'll see it as "tall structures are unstable" in Week 5 and waste days blaming the policy.
3. **Hit Gate 2 (scripted end-to-end, Week 3) no matter what.** Cut structures, cut arms, cut features.
4. **Make the stabilizer arm necessary, and measure it.** Design S3 and S5 so single-arm assembly genuinely fails, log arm A's reaction force every episode, and run A6 and A9 properly. That's what converts "we used two arms" into a result.
