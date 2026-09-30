# Plan of record v4 (r5) — vision-driven, collision-free dual-arm assembly in Newton

**Status:** Adopted 2026-09-28. User decisions: U1 standalone Newton; U2 measure, then decide at D1 between 1× and 2×; U4 decide after M1; U0, U3 and U5–U13 as recommended. **Replaces** the v3.1 *execution platform*; the v3.1 plan and results stay frozen as the record of the CPU twin. **Milestones shown live to the user:** M0 (Newton cell with cameras and a contact HUD, ≈ day 4), **M0.5 (collision-free two-arm build, including S3's braces, ground-truth-driven, ≈ day 6–7)**, M1 (the same, vision only, ≈ day 10 optimistic / 12–15 realistic). **Amended 2026-09-29 (r4, from P0 and P1's first probe runs; user answers in ledger `v4_plan_r4`). M1 moves to ≈ day 10.5 optimistic / 12.5–15.5 realistic.** **Amended 2026-09-30 (r5 execution phases): hand-planned bracing is dropped and left to a separately planned RL phase; the pipeline builds unbraced now (ledger `v4_brace_to_rl`, `v4_pipeline_unbraced_now`, `v4_r5_phases_revised`; see Revision r5, "Execution phases").**

**Amended 2026-09-30 (r5, adopted after two Codex rounds, escalated to the user, adopted with the two blocker fixes; ledger `v4_plan_r5`, `v4_joint_scope_r5`, `v4_pipeline_gt_now`). User answers: U4 = yes now (rigid, breakable joints: WP5a before WP1; WP5b/E5 still decide after M1); image → build on ground truth first (GT-first pipeline, PL); vision-only M1 unchanged; U-r5-1..6 as in the r5 section. New stages J-pre, J-e, J-d (M0-J) and PL run before WP1. M0/M0.5/M1 slip ≈ 4 agent-days (+ ≤ 1 day if the WP1 seating pull fires).** Item U4 above ("decide after M1") is superseded for WP5a.

---

## Revision r5 (rigid breakable clutch, a brace that pushes nothing and is measured, image → structure on ground truth)

The P4 brace/weld probe (`scripts/13_brace_weld_probe.py`, `results/v4/p4_brace_weld/`, ledger probe `v4_p4_brace_weld`) and the user's observation that braced builds bend and spring back (ledger failure `v4_brace_overforce`) showed that r4's clutch weld is soft, that A's brace executor presses far harder than the brace model assumes, and that nothing in the cell can break. The user answered on 2026-09-30: **U4 = yes now** (rigid and breakable joints), **image → build on ground truth now** (a GT-first pipeline, PL), **vision-only M1 unchanged**. Tags: **[M]** measured (orchestrator-verified from `results/v4/p4_brace_weld/`), **[R]** reviewer-reported, **[A]** assumed.

**Review history.** Plan-reviser draft → Codex round 1 CHANGES REQUESTED (9 findings, below) → revised → Codex round 2 CHANGES REQUESTED (two blockers, B1 and B2 below, plus the implementation-time notes) → **escalated to the user after two rounds; the user chose "adopt with the 2 fixes"** (no third round; Codex checks the two fixes in the code-diff reviews). Round 2 confirmed resolved: the hand-off protocol is acyclic (real `Arm` class, ideal tracking: arch 5,940, S3 9,104, all-braced S3 13,355 updates, no cycle), grouping as a scoped model, grip plumbing (LP gets 14.3 N, μ 1, moment limit 0.143 N·m), and `qfrc_constraint` exists (`mujoco_warp` `types.py:1930`, solver writes Jᵀf). **Adopted 2026-09-30.**

### r5 blocker fixes (from round 2; supersede the text below where they conflict)

- **B1 vision drift budget.** r5 does **not** change §2.5.4. The row "structure drift after the look = 0" stays as the r4 design value for now and is flagged **OPEN**. Before D1, drift is remeasured under the final executor over the capture-to-insert interval (V5 wrist-image capture → insertion) and carried as a **separate conservative allowance** (worst case over runs, added linearly, not RSS'd as a zero-mean Gaussian) unless its statistical model is validated. J-d's M4 (pre-insert → weld) is reported as structural information only. U-r5-3 resolves to this.
- **B2 recording before J-pre.** The recording part of S4 (`--record-all`: commanded and executed q of both arms, phases, `b_step`, welded bricks' `body_q`, raw per-frame A–B and arm/held-brick–welded-brick contact flags, weld fire/break frames) moves to a step **S4a** run before J-pre. Order: S0 → S1 → J-a/J-b → S2 → J-c → **S4a** → J-pre → [WP1 seating pull] → S3 → S4 (rest) → J-e → J-d → PL.

### Faults and evidence

| # | fault | evidence |
|---|---|---|
| F1 | arm–arm contact during the brace is the dominant load | arch step 10: A-hand↔B-arm mean 161 N during transport, 220–231 N from pre-insert to release, peak 442 N, correlation with welded-brick displacement 0.938 [M]. S3 steps 13/14: peaks 173/243 N, and 1026 N with a stiff weld [M] |
| F2 | the brace position target is 2 mm inside a brick, with no cap | `queue_brace` presses at `brace_pose + STUD_HEIGHT`, but `grasp_lp`'s `brace_pose` is a mid-height grasp centre (`dual_arm_sim.py:449-473`, `bracing.py:120,221`). S3 step 11, with no arm–arm contact: a 10–18 N press bends S3 by 29.2 mm, back to 0.48 mm [M] |
| F3 | the clutch weld is soft | solimp (0.95, 0.99, …): column lateral stiffness 0.70–1.35 N/mm, a 15 mm set after 50 N [M] |
| F4 | nothing breaks | the M1 weld pool never releases; U4 is now yes |
| F5 | the docstring "a brace cannot change an outcome" is contradicted | arch step 10 brace window: 51.4 mm peak, 21.9 mm remaining 1 s after A retracts, driven mainly by B-arm↔A-hand contact (F1) [M] |

Model (`stability.insertion_utilisation`, unbraced u at 6/12/20 N): arch step 10 0.23/0.29/0.38; S3 step 11 0.40/0.40/0.40; S3 step 13 0.81/1.14/1.58; S3 step 14 0.39/0.59/0.85. B's insert peak is 3–6 N [M]. A brace can therefore change an outcome, through a break, essentially only at S3 step 13 above ≈ 9 N; whether the passive hold lowers u is measured (G5, controls).

### The revision

| # | change (fault) | evidence | what changes |
|---|---|---|---|
| r5-1 | Weld stiffness (F3) | P4 part B, ungrouped [M]: column lateral 23–24 N/mm, vertical ≈ 560 N/mm, cantilever 34 / 65–70 N/mm, set ≤ 0.03 mm, stable at 16/32 substeps | weld `eq_solimp` (0.95, 0.99, 0.001, 0.5, 2.0) → (0.9999, 0.9999, 0.001, 0.5, 2.0); `eq_solref` (0.002, 1.0) unchanged |
| r5-2 | Structure collision group (F3, F4 load path) | readout ≤ 0.12 % collisions off [M/R]; runtime `shape_collision_group.assign` without re-capture (`results/v4/p1_rtf_filter/fexp.py`); pool ↔ `stability.connections` 1:1 on every shape [R] | collision group G_s = −2, **pre-failure only**: plate shapes start in G_s and a brick joins when its welds fire; the build ends at its first failure, so every structure brick is welded and grouped at every scored time. Declared **weld-only idealisation**; it drops (i) bearing/friction between same-course welded neighbours, (ii) non-interpenetration among welded bricks (bounded by G3), (iii) load sharing through contact. External loads on welded bricks (the pressed brick, B's fingers, A's fingers) remain contacts. Post-break states are never scored; the "lower bound" claim is deleted |
| r5-3 | Break model, WP5a now (F4) | — | every frame, for each enabled weld: end-of-frame weld wrench on the upper brick → patch centre → nominal axes → u = `Patch.utilization(Fz, Mx, My)`; break at u ≥ 1 on 2 consecutive end-of-frame samples, from 0.1 s after the weld fires. Restricted: axial and prying only, shear/torsion ignored (U-r5-5), frame-sampled (J-c quantifies what that misses). The build ends at the first break (failure `joint_break`); afterwards the weld is disabled, the upper brick moves to group 1, and the sim is display-only and unscored |
| r5-4 | Brace executor to `grasp_lp` semantics, commanding no push (F2) | — | open-finger approach; the last 10 mm at 5 mm/s with a contact guard (any A shape > 1 N for 2 frames → stop, hold at FK, `guard_stop`); close with a 1.5 mm squeeze; hold at the measured FK pose. The hold is measured and gated by G5; no "cannot overload" claim |
| r5-5 | Hand-off protocol and executed-path hands-fit (F1) | P3 `step5.json` `padding_needed_mm` 12.08 at time scale 1.0 [M] | (a) an acyclic sequence per braced step with timeouts; (b) `build_plan(clearance=…)` runs a kinematic dry run of that step's full queues for both arms, min sphere clearance ≥ `HANDS_FIT_MARGIN` (12 mm provisional, then max(5 mm, p99 + 2 mm) of the waypoint executor's measured deviation); (c) no fitting grasp → unbraced, flagged `brace_infeasible` (U-r5-1); (d) a whole-build dry run in preflight |
| r5-6 | New stages J-pre, M0-J (build acceptance), PL (image pipeline) before WP1 | — | prototype layout, ground truth, before WP1. M0.5 (cuRobo, randomised) is unchanged as WP2, with a pre-registered escalation |

The WP5 contact spike is not needed for WP5a (loads among welded bricks go through welds; only the new brick at 3–6 N and the arms touch the structure). It remains the entry condition for **WP5b** (detent n·f_insert, v3.1 forces, E5), which is still "decide after M1".

**Codex round 1 → round 2 changes (kept as the review record).** 1 HIGH deadlock/park inside R_zone → the TCP/R_zone zone removed for the waypoint executor, acyclic hand-off, a 30 s timeout on every wait, a kinematic harness with the real `Arm` class (`tests/test_arm_protocol.py`). 2 HIGH grouping/break transition → build ends at its first unrecoverable event (gate miss, break, divergence, timeout); a brick is welded only onto grouped welded supports; post-break states never scored; `--continue` diagnostic (break model off after the first failure, rows flagged `after_failure`). 3 HIGH grip → `grip_N` and μ plumbed explicitly. 4 HIGH hands-fit → per-step dry run of the executor's own queues; margin 12 mm then max(5 mm, p99 + 2 mm); recall validated on J-pre legacy runs. 5 HIGH drift → four metrics at the 8 box corners (M1–M4); §2.5.4 drift a measured term (then B1 replaced this: §2.5.4 unchanged). 6 MED guard → "cannot overload" dropped; per-shape contact guard during open-finger approach only; G5 uses max, not p99. 7 MED J-c → phase-shifted pulses, substep vs frame comparison, yawed-90° and 5°-tilted column, `qfrc_constraint` cross-check, two-support bridge. 8 MED order/controls → J-pre added; pre-registered pull of WP1 seating items; n = 3 fixed layout = descriptive acceptance; equal control counts; any retune re-runs J-d fresh. 9 MED invalidates → listed below. Nit: readout "< 0.1 %" → "≤ 0.12 %".

### Implementation steps

Order: S0 → S1 → J-a/J-b → S2 → J-c → **S4a** → J-pre → [WP1 seating pull if needed] → S3 → S4 (rest) → J-e → J-d → PL. No other GPU job during J-pre, J-d or a live demo.

- **S0 (≈ 40 min GPU).** Probe part A: arch `--strategy none` with the default and the stiff weld; S3 `--strategy none` with the stiff weld. Question: is there a crown-failure path needing no brace? (An unbraced miss shows one exists, not that J cannot help braced runs.)
- **S1 weld + grouping.** `WELD_ATTRS` per r5-1; `STRUCT_GROUP = −2`; `self.plate_shapes` (the plate box and stud-proxy `add_shape_*` ids, lines 277–285) and `self.shapes_of[bid]`; write `STRUCT_GROUP` in place after `finalize` and in `snap()` on a pass. Probe: `--group` for parts B/C; B solimp9999 grouped rows; C staircase cantilever, bridge (2x4 on two 2x2 piers), column with the lower brick yawed 90°, column tilted 5°. Checks J-a, J-b.
- **S2 break model.** Move `weld_rows()` / `brick_wrench()` into `dual_arm_sim` (the probe imports them). `class JointBreaker(example)` maps pool welds to `stability.connections` entries and asserts the pair sets are equal. `update()`: F, M on the upper brick about its origin; c = `conn.centre` (plate) or the lower brick's T_now·T_nom⁻¹ applied to `conn.centre`; M_c = M + (p_U − c) × F; R = R_L,nom·R_L,nowᵀ; u = `patch.utilization((RF)_z, (RM_c)_x, (RM_c)_y)`, compression +Fz; records shear |(RF)_xy| and torsion (RM_c)_z (reported, unused). `U_BREAK` 1.0, `BREAK_SAMPLES` 2, `BREAK_SETTLE_S` 0.1. On a break: log {frame, t, step, pair, u, edge, wrench, A/B phase}, disable the weld, notify, regroup the upper brick to 1, `self.failure = "joint_break"`. End at first failure: `self.failure` is also set on a gate miss, divergence and wait timeout; `schedule()` queues nothing more, both arms open and park; the row carries `failure`, `failed_at_step`, built so far. `--continue` skips the stop, the break model is off after the first failure, rows are `after_failure`. Check J-c and a cube `--test` smoke with 0 breaks.
- **S3 brace, protocol, hands-fit, grip.**
  1. The current `queue_brace` becomes `queue_brace_legacy` behind `--legacy-brace`.
  2. New `queue_brace(n, brace)` with `grasp_lp` semantics: TCP at `brace_pose` (pad centre), yaw π/2, tilt `brace_tilt_rotvec`; moves to brace (back = pose − 50 mm along the tool axis, ≥ `z_travel`) → approach → brace-in (pose − 10 mm) → brace-guard (last 10 mm at 5 mm/s, fingers 8 + 2 mm/side) → close (8 − 1.5 mm/side, 0.5 s, then sets A's command to FK ee and adds n to `braced`) → hold (wait n in `retracted`) → open (0.4 s) → retract → park. Guard in `step()` during brace-guard: any A-shape contact normal > 1 N for 2 frames → `Arm.stop_here(ee)` + `guard_stop`. Per-frame A contact readout in A's brace phases (own Contacts buffer): per-finger normal sum (realised grip), contacted brick set, per contacted brick force sum and moment about the origin, total wrench on the structure. `requires_brace` with `brace.feasible` false → skip A, flag. Row: `brace_requested`, `brace_planned_feasible`, `brace_executed`. **Implementation note (round 2, hold at FK):** setting `A.cmd` alone is not enough (`Arm.update` interpolates toward the queued move's pos before checking wait; Codex reproduced z 0.102 → 0.100 in 12 updates). Rewrite the queued hold target (and the later open/retract targets that depend on it) from the measured FK; the same on a guard stop.
  3. Hand-off protocol (replaces the zone; every wait has `timeout` = 30 s → failure `wait_timeout:<arm>:<phase>`): `schedule()` starts B's next step only when B and A are idle; B's braced-step queue … lift (then adds n to `staged`) → stage (zero-motion, wait n in `braced`) → transport … retract (then adds n to `retracted`); A's brace is queued when n is in `staged`, not at step start; unbraced steps unchanged except the idle rule. Acyclic: B-next waits A-idle; A-brace waits B-staged; B-transport waits A-braced; A-open waits B-retracted; each awaited event comes from a move the other arm starts without waiting on the waiter. Check `tests/test_arm_protocol.py`: stub `Example.__new__` with plan-derived attributes; real `Arm`, `queue_place`, `queue_brace`, `schedule`; ee = FK of the IK solution; committed arch and S3 plans plus a synthetic all-braced plan; every step completes within 20 000 updates, no timeout, never both arms in a moving move inside a braced step.
  4. Grip explicit: `bracing.assign(…, grip_N=GRIP_N, mu=1.0)` → `_brace_obj(placed, y, z, grip_N, mu)` at both call sites → `ST.Brace(grip_N=grip_N, mu=mu, moment_max_Nm=mu*2*grip_N*PAD_MOMENT_ARM)`, `brace_force_N = grip_N` in `_record`; `planner.build_plan(…, brace_grip_N=None, brace_mu=None)` passes through, defaults unchanged (the frozen twin is unchanged). In `dual_arm_sim`, after `add_builder`, A's finger dofs `joint_target_ke[7:9] = [9500]*2`, kd = 2√(0.1·ke) → **14.3 N nominal per finger at 1.5 mm** (r4 realised 13.2 N [M]); `make_plan` passes `brace_grip_N=14.3` and `brace_mu` = the finger shape material μ (finger `geom_priority` 1). Checks: `tests/test_planner.py` under its own protocol (committed twin plans, twin `BraceClearance`) with no diff to the committed twin plans; Newton plans show `brace_force_N` 14.3 and μ.
  5. Hands-fit `motion/brace_clearance.py`, `class BraceClearance`, `__call__(brick, y, z, tilt) → bool`: builds the candidate brace dict and runs the S3.3 harness for that step only (A park → full queue; B from the previous `up(target)` through feeder, stage, full queue), with the same `IKSolver` config and warm start as `Example.step`, finger joints as commanded, B's bias at nominal `hand_off`; per frame the minimum distance between A and B `fr3.yml` spheres (`newton.eval_fk` on the IK model) and A spheres ↔ B's carried-brick OBB; pass if min ≥ `HANDS_FIT_MARGIN` (0.012 m initially); cache by (step, y, z, tilt); store the min clearance; if > 60 s per structure, prefilter with a single-frame check. `make_plan` passes `clearance` for `grasp_lp`. Check J-e.
- **S4a recording (B2; before J-pre).** `--record-all PATH.npz`: commanded and executed q of both arms, phases, `b_step`, welded bricks' `body_q`, raw per-frame flags (any A–B pair d < 0; any arm/held-brick contact with a welded brick d < 0), weld fire/break frames.
- **S4 (rest) metrics, preflight, docstring.** `experiments/j_tables.py` offline, at each brick's 8 box corners (rotation counts): **M1** snap correction (fire pose → reference), where the reference is the mean pose over the last 10 frames of the first unloaded window after fire (≥ 30 frames with no arm or held-brick contact on any welded brick); **M2** peak excursion from fire + 0.1 s vs the reference; **M3** unloaded residual at every later unloaded window and at build end; **M4** target drift per step, at the stud-top corners of the target's supports from the end of B's pre-insert to weld fire, per axis; tracking deviation (link-sphere displacement, commanded vs executed, p99 per arm and phase). Row fields: `weld`, `grouped`, `perception "ground_truth"`, `failure`, `failed_at_step`, `built`, `breaks`, `u_peak_by_step`, `shear_torsion_peak_by_step`, `brace_by_step` (requested/planned_feasible/executed, `guard_stop`, `clearance_mm`, realised grip per finger, contacted set, per-brick max |F| and |M|, reaction vs `expected_reaction_wrench`), `rtf`, `preflight`. `--out` is an alias of `--p0-out`. `test_final` asserts failure None and built == n, drift ≤ 1.0 mm. HUD shows breaks and failure. Preflight in `make_plan`: ≤ 24 bricks; `blueprint.components == 1`; braced ⇒ depth in x == 2; a tiler overhang `ValueError` passes through; list `brace_infeasible` steps; whole-build dry run with min A–B clearance ≥ margin every frame and no timeout. Docstring rewritten (stiff, breakable, grouped weld pool; ends at first failure; **a brace can change an outcome through a break OR through sub-critical displacement that changes the snap gate (`dual_arm_sim.py:513`); report stress reduction and completion effects separately**; the planner predicts u < 1 unbraced at 3–6 N except S3 step 13; hold effect measured; waypoint IK + hand-off + dry-run clearance, not planned motion). `cell/contacts.py` `BRACE_PHASES = ("brace-guard","close","hold","open")` plus legacy ("brace","hold") with `--legacy-brace`; A's fingers permitted on `gripped_bricks` only in those phases; self-check updated. **Monitor note (round 2):** add B's new "stage" phase to `cell/contacts.py` `HELD_PHASES` / held-brick recognition and to the monitor self-check, otherwise B's fingers on its carried brick are flagged unintended. `scripts/j_acceptance.sh` → `results/v4/j/`.

### Gates (any retune re-runs the complete affected gate and every later gate, fresh)

- **J-a weld static** (part B `--group`, solimp9999, 16 substeps): column lateral secant ≥ 20 N/mm and vertical ≥ 300 N/mm at 1–50 N; cantilever ≥ 30 N/mm; set ≤ 0.05 mm after 50 N; no divergence. Fail lateral → `torquescale` ∈ {10, 100}; none → plan-reviser.
- **J-b readout, grouped:** (i) column, staircase, yawed-90° column and 5°-tilted column: F and M_patch within 1 % (or 1 mN / 0.1 mN·m abs) of analytic statics for gravity, ±1 N vertical and 0.5 N lateral; u within 1 %; (ii) Jacobian: per column brick, the weld wrenches it receives (its own minus the upper neighbour's, **the neighbouring weld moments shifted to a common origin before subtracting**) equal `mjw_data.qfrc_constraint` of its free joint mapped to world about its origin, within 1 %; (iii) bridge: the two welds' wrenches sum to the analytic total within 1 %, split reported. Fail → plan-reviser.
- **J-c break rule** (probe `--part d`, test-fixture body forces): (i) 2x4 on 2x4 pull-up ramp 60 N/s breaks at 90.4 N (`grid_patch(4,2).pull_capacity()`) ± 5 %; (ii) staircase tip ramp at analytic u = 1 ± 5 %; (iii) column with 1 N lateral for 5 s, 0 breaks; (iv) bridge eccentric ramp: first weld to break and its load vs the LP weakest joint (reported); (v) pulses at u = 1.5 lasting 1, 8, 16, 24, 40, 60 substeps, 4 phase offsets each: pass if every pulse ≥ 32 substeps is detected and 0 pulses ≤ 16 substeps are; (vi) an ungraphed loop reads u every substep on (i), (ii), (v): distribution of max-over-substeps vs end-of-frame u ratio; (vii) per-weld shear/torsion logged. Fail (i)–(iii) or (v) → fix S2; a large (vi) gap → user under U-r5-5.
- **J-pre one-pass build check:** cube, arch, hollow_box, S3 × 1; S1 + S2 + S4a; `--legacy-brace`, no hands-fit, `--continue`, `--record-all`. Outputs: completion and seating misses per step; waypoint tracking deviation p99 per arm → `HANDS_FIT_MARGIN := max(5 mm, p99 + 2 mm)` (an empirical padding; P3's 12.08 mm is cuRobo evidence, 12 mm is only the provisional waypoint margin); positive-control breaks under the legacy brace; recall data for J-e. **Pre-registered branch:** a recurring non-brace seating miss on an image-sample shape pulls WP1 seating items (r4 clutch-at-rest window, §2.3 guarded insert at P1's F_press or the prototype gate if P1 is not ready, stated) into J before J-d.
- **J-e hands-fit validation:** on J-pre legacy runs, the sphere clearance of executed q ≤ 0 on every raw contact frame (recall 100 %), false-positive fraction reported; dry-run predicted vs executed clearance reported. An empty recall dataset is rejected; the whole-build IK history is kept. Fail recall → sphere set, then fingers as exact URDF boxes; re-run J-e.
- **J-d build acceptance → M0-J:** prototype layout, ground truth, 1×, 16 substeps, `grasp_lp`, `weakest_joint`, `--monitor --record-all`; gated: cube, arch, hollow_box, S3 × 3; controls with equal counts and the same initial state: arch and S3 × 3 each with `--strategy none` and `--legacy-brace`; descriptive acceptance.
  - G1 0 raw arm–arm contact frames and 0 monitor `arm_arm` events. G2 0 breaks. G3 M2 ≤ 1.0 mm. G4 M3 ≤ 0.1 mm at every unloaded window and at the end. G5 every executed brace: max per-brick |F_A| ≤ 10 N [A], realised grip ≥ 0.8× nominal **evaluated over a specified closed-hold interval**, contacted set ⊆ `gripped_bricks`, `guard_stop` reported. G6 no divergence, no timeout. G7 built == n on cube, arch, hollow_box 3/3; S3's first failure reported and attributed; an S3 failure attributed to arm–arm, break or brace fails J-d.
  - Reported, not gated: M1, M4 (per-axis p95, max), shear/torsion, reaction vs LP, u_peak braced vs matched none, tracking deviation, cycle/brick, RTF.
  - On failure (one retune each, then plan-reviser, full J-d re-run): G1 → margin re-set; still → WP2's cuRobo route for B's moves near the build (P3 route 1 passed steps 1–7 [M]); G2 in the settle → `BREAK_SETTLE_S` 0.2; during B's insert at S3 step 13 → a finding to the user; G3/G4 → the `torquescale` branch; G5 → A arm ke 400 → 100 during hold; G6 → re-run alone, recurs → the WP1 divergence item; G7 not caused by J → WP1 input.
- **PL image → structure:** drawings in `brickassembly/blueprints/user/`; preflight including the whole-build dry run; 1 live run shown to the user; 3 headless; G1–G7 with built == n; a preflight failure is reported with reason and fix (not a plan failure).

**User-facing command and inputs.** `bash scripts/run.sh dual_arm_sim.py --front F.png [--side S.png | --top T.png] --width W [--depth 2] --name NAME` (live); headless adds `--viewer null --test --num-frames 40000 --out results/v4/pipeline/NAME.jsonl --record-all results/v4/pipeline/NAME_rK.npz --repeat K`. Inputs: a binary silhouette in either polarity, auto-cropped; front columns = +y, rows = layers, top highest; side columns = +x; top columns = j, rows = i; aspect 1 stud 8.0 mm : 1 layer 9.6 mm (e.g. 20 × 24 px); `--width` = studs across the front; depth 2 by default and required when braced; ≤ 24 bricks; one connected piece; overhangs only as the tiler bridges.

### What the report must show (from result files via `experiments/j_tables.py` / `analyze_v4`)

J1 grouped vs ungrouped stiffness/set, and readout errors including the Jacobian and bridge ("the clutch weld is X N/mm in Newton"). J2 predicted vs measured break loads, pulse detection vs duration, substep/frame ratio, bridge first break vs LP ("breaks follow `capacity.Patch` (axial and prying only, 11.3 N/stud) on end-of-frame weld wrenches, restricted as stated"). J3 tracking deviation, margin, hands-fit recall/FP, seating misses. J4 per shape × arm (fixed/none/legacy), range over 3: built/n, first failure, breaks, M1–M4, `arm_arm`, G5 fields, braces requested/feasible/executed, u_peak braced vs none, cycle, RTF ("rigid" = excursion ≤ 1 mm and residual ≤ 0.1 mm under this executor, n = 3, descriptive; brace benefit only if the braced u_peak < none on all matched runs, descriptive). PL table per drawing ("rebuilt the drawn shape in simulation from simulator poses, not vision"). Figure P4: legacy vs J-d traces. Stress reduction and completion effects of the brace are reported separately. Limitations: weld-only idealisation; ends at first failure; restricted break model; mixed force regime (v3.1's regime needs WP5b); passive hold ≠ LP brace wrench; 14.3 N grip and model μ; waypoint motion, with clearance only up to the measured tracking error.

### What r5 invalidates or changes

- **§2.2:** the weld pool is stiff, breakable, grouped, and the build ends at first failure; "welds never release" and "a brace cannot change an outcome once a brick is welded" are replaced; M2 → WP5a now + WP5b after M1 behind the spike.
- **§2.5.4 (per B1): not changed.** The drift row stays at the r4 design value 0, flagged OPEN, to be remeasured capture-to-insert before D1 as a separate conservative allowance (see B1). The chain RSS and D1(i) are recomputed at D1 with that allowance (U-r5-3).
- **U4:** yes, 2026-09-30; WP5b/E5 decide after M1; Q3 only if WP5b.
- **Order:** P0 → S0 → S1/J-a/J-b → S2/J-c → S4a → J-pre → [WP1 seating pull] → S3/S4/J-e → J-d (M0-J) → PL → WP1 (M0) → WP2 (M0.5) ∥ WP3 → WP4 (M1). WP1's `cell/clutch.py` absorbs `JointBreaker`, grouping and the protocol (−0.25 day). M0/M0.5/M1 slip ≈ 4 agent-days (+ ≤ 1 day if the WP1 seating pull fires).
- **Evidence status:** v3.1 is frozen and not comparable (force regime, grip 70 vs 14.3 N, feed-forward unused here, sampling 5 ms vs end-of-frame, collision semantics). P0/P0′ are historical; P0′ is no longer the M0(a) reference; J-d's gated rows become **P0″**. P4 part A is a failure record. P3 step 8 used P0′ cube inserts → re-run on P0″ before use (r6 draft F2 rework). P3 steps 5/7 tracking/padding transfer only to cuRobo paths; steps 1–4 and 6 are unaffected. P1(b)–(e) used a static lower brick (`10_newton_scale_probe.py:450`) → **transfer check before D1, not blocking J**: P1(d) at F_seat and 2·F_seat on a lower brick welded with the r5 weld + grouping, plus capture-boundary trials, chosen-grip trials and the relevant P1 gate verdicts; agree within one sweep level on F_seat and 0.1 mm on sink, else re-run P1(b)–(d) welded. P1(f) rows are void and re-run under r5. P2 is unaffected.
- **Newton plan fields change:** `brace_force_N` 14.3, μ, brace choices, `expected_reaction_wrench`, `predicted_util_braced`; feed-forward fields are emitted but unused (stated); tracked `plans/sim_*.json` change (committed with S3). Monitor A phases change (and B's "stage" phase is added).
- **Checks that must be green:** `tests/test_planner.py` (own protocol), monitor self-check, `tests/test_arm_protocol.py`, `test_final`, an infeasible-brace unit case, J-b/J-c.
- **Unadopted `Docs/plan_v4_r5_draft.md`:** row 1 adopted (pre-failure, weld-only idealisation; "WP5 ungrouped" void); row 3 subsumed by the corner metrics, returns in r6; rows 2, 4–7 unadopted. The file is renamed `Docs/plan_v4_r6_draft.md`, to be rebased, ledger ids `*_r6`.

**Evidence tags.** Measured: ungrouped stiffness/set; readout ≤ 0.12 %; arm–arm forces/correlation; S3 step 11 bending; B insert 3–6 N; 14.3 → 13.2 N grip; P3 padding 12.08 mm; model utilisations. Reviewer-reported: pool ↔ connections 1:1; deadlock repro; P0′ span fractions; 0.112 %. Assumed, gated: grouped stiffness; partial/rotated/two-support readout; break accuracy and sampling; waypoint deviation/margin; hands-fit recall; G3–G5 thresholds; legacy control breaks. Assumed, not gated: 11.3 N/stud at 1× (U3); μ_finger = the finger material μ. **Effort ≈ 4.4 agent-days (+ ≤ 1 day if the WP1 seating pull fires); J-d's 24 runs ≈ 5–7 h GPU, serial.**

### r5 user decisions

User answers 2026-09-30: **U4 = yes now** (rigid and breakable); image → build on ground truth now; vision-only M1 unchanged.
- **U-r5-1** a braced step with no fitting grasp: run unbraced, flagged, the break model decides (alternatives: stop; skip).
- **U-r5-2** joint scope: grouping as a declared weld-only idealisation, the build ends at first failure (alternative: continue after breaks, needs a validated mixed readout).
- **U-r5-3** vision drift budget: per **B1** (§2.5.4 unchanged, OPEN; remeasured capture-to-insert before D1 as a separate conservative allowance; M4 reported as structural information). Alternatives were §2.5.4 taking J-d's measured M4, or keeping 0 and gating M4 p95 ≤ 0.05 mm per axis.
- **U-r5-4** order: J before WP1; M0/M0.5/M1 slip ≈ 4 agent-days (+ ≤ 1 day if the WP1 seating pull fires) (alternative: parallel).
- **U-r5-5** shear/torsion/sampling: axial + prying only, end-of-frame sampling; shear/torsion and the substep gap are reported (alternatives: add limits, with no calibration data; or substep readout, needing an ungraphed step).
- **U-r5-6** the user supplies 1–2 drawings to the input conventions above and watches the PL live run.

**Ledger (r5 adoption):** `probe v4_p4_brace_weld`; `failure v4_brace_overforce`; `decision v4_joint_scope_r5`, `v4_pipeline_gt_now`, `v4_plan_r5`; `deviation v4_clutch_weld_stiffness`, `v4_structure_collision_group`, `v4_clutch_breakable` (2 end-of-frame samples, 0.1 s settle, axial + prying only; replaces the planned `v4_break_hold_33ms`), `v4_build_ends_at_first_failure`, `v4_brace_executor_r5`, `v4_handoff_protocol_r5`, `v4_milestone_order_r5`. After the runs: results `v4_j_s0`, `v4_j_weld_grouped`, `v4_j_break_rule`, `v4_j_pre`, `v4_j_hands_fit`, gate `v4_M0J`, `v4_p0pp_reference`, `v4_pipeline_<name>`, and one deviation per retune/branch.

### Execution phases (2026-09-30, workflow change; phases revised the same day when bracing went to RL)

Under the phases > segments > subtasks workflow (commit 4c8cea7) reviews and plan revisions happen only at phase gates; in-phase check failures are logged and carried to the gate. Applied to r5 without changing any criterion (the revision below removes the brace items):

| phase | segments | gate |
|---|---|---|
| 1 Joint model (closed) | S1 stiff weld + grouping (done, commit a890b68); S2 break model; S4a recording | J-a + J-b + J-c together (ledger `v4_phase1_joint_model`) |
| 2 Unbraced build and image pipeline | (a) pipeline defaults to unbraced + preflight + run row/metrics (S4 rest); (b) acceptance runs cube / arch / hollow_box × 3, unbraced (S3 reported, not gated); (c) PL on the user's drawing(s) | G2 (0 breaks), G3 (peak excursion ≤ 1.0 mm), G4 (unloaded residual ≤ 0.1 mm), G6 (no divergence / timeout), G7 (built == n, 3/3 on the image shapes), on the acceptance runs and on the user's drawing(s) |
| 3 RL bracing | plan in `Docs/plan_v4_rl_brace.md`: R0 (feasibility) adopted 2026-09-30 with amendments A1-A7; R1 and R2 a draft, revised once at the G-R0 gate if R0 passes (ledger `v4_rl_brace_plan`) | G-R0 (R0); R1/R2 gates set at the revision |

J-a..J-c kept their pre-registered criteria and were reviewed together at the Phase 1 gate (J-e is dropped, see the 2026-09-30 paragraph below). The proposed J-b criterion revision "r5.1" was NOT adopted (withdrawn under the new workflow); J-b's result is carried to the Phase 1 gate.

**Phase 1 gate outcome (2026-09-30).** Reviewed once by Codex (code approved). J-a passes and J-c passes on its pre-registered items (2x4 pull-off break at 92.0 N against 90.4 N, +1.77 %; cantilever break 1.11 % below the analytic u = 1 load; 1 N lateral for 5 s, no break; pulses of 40 and 60 substeps detected, of 16 or fewer not). J-b **fails as written**: 66 of 3060 force/moment readings and 12 utilisation readings fall outside tolerance (worst 22.387 mN, 0.191 mN.m), caused by the GPU MuJoCo-Warp solve stopping after 1-2 iterations (solver-side, "float32" inferred). Codex's verdict was CHANGES REQUESTED; **by user decision (asked directly, options: revise J-b once, try a solver fix, accept) J-b is accepted as a documented limitation and Phase 2 proceeds**, so a pre-registered criterion is replaced by decision, not met. The readout is validated only for the tested cases (full 2x4 patch, one cantilever geometry; not partial, rotated or multi-support patches near threshold), and the report's J1/J2 tables must say so. Non-gating follow-ups in Phase 2: 32-substep pulse trials, save the J-c repeat runs, keep reporting the substep-vs-frame gap. Ledger: `v4_j_c_break_rule`, `v4_phase1_joint_model`, `v4_jb_accepted_with_known_cause`.

**Phases revised 2026-09-30 (bracing goes to RL).** User direction, verbatim: "stop trying to manually plan the brace leave it to RL workflow." Asked directly, the user chose: pipeline now = "Build unbraced now"; RL scope = "Where and when to brace" (the policy picks, per step, whether to brace and which grip on the structure; a scripted stabilizer motion carries it out); RL simulator = "Newton v4 cell". Evidence at the time: J-pre (`results/v4/j/pre/j_pre.jsonl`) built the cube 8/8 and the hollow_box 12/12 with no brace required and no break, while the arch with the legacy brace ended in joint_break at step 10 (u 2.0862, 5 arm_arm events) and S3 lost b_001 at step 1 and diverged at t = 37.42 s (step 3); the new grasp-hold brace and hand-off protocol (commit 73a4bd5; `results/v4/j/seg2/`) still ended in joint_break at step 10 on the arch (u 1.1365, 2 arm_arm events, realised grip 25.5 / 7.52 N per finger against 14.3 N nominal). **Dropped:** S3.5 dry-run hands-fit, J-e, the G1/G5 brace gate items, the fixed / none / legacy brace controls; the pipeline runs with no stabilizer (strategy none) and the r5 breakable joints, so an overloaded joint breaks and ends the build. **Stays in the code, unused by default:** the committed brace executor, the hand-off protocol, and `--legacy-brace`. The S3 step-1 loss and step-3 divergence is a non-brace path and is carried (ledger `v4_s3_step1_loss_divergence`). The Phase 2 gate criteria are the r5 ones with the brace items removed; Phase 3 (RL bracing) is planned in `Docs/plan_v4_rl_brace.md` (R0 adopted 2026-09-30; R1 and R2 a draft; ledger `v4_rl_brace_plan`). U12 is superseded for bracing only; U13 is moot until then. Ledger: `v4_j_pre`, `v4_brace_executor_seg2`, `v4_s3_step1_loss_divergence`, `v4_brace_to_rl`, `v4_pipeline_unbraced_now`, `v4_r5_phases_revised`.

---

## Revision r4 (P0 findings and P1 probe findings)

The first P1 runs showed four protocol faults and six probe departures; P0 showed that the monitor's merging fragmented contacts and that the `grasp_lp` contingency fired. The reviewer reproduced each P1 finding with independent runs (`results/v4/p1_review/`; smoke rows `results/v4/p1_smoke/`), recorded as ledger probe `v4_p1_smoke_finger_contact`. **The r3-protocol P1 rows that involve the fingers — (b), (d), (e), (f) — are void.** The (a), (c) and (g) rows involve no fingers and remain valid evidence. §0 fact 6 and `newton_brick_press_tunnelling` still stand, for the same reason.

**Precedence (applies to every r4 item):** an r4 gate item that fails at one scale counts as a P1 gate failure at that scale only, and routes through the existing branch (fails at 1× only → s = 2). A plan-reviser revision is needed only if an r4 item fails at both scales.

| # | change | evidence | what changes |
|---|---|---|---|
| 1 | Finger contacts stiffened | The FR3 finger shapes kept Newton's default ShapeConfig (ke 2500, kd 100 → solref 20 ms, damping ratio 1). The fingers have `geom_priority` 1, so MuJoCo uses the finger's solref alone at every finger contact. The stiffness is per unit acceleration, so N/mm scales with brick mass: ≈ 1.1–1.3 kN/m at 1×. At 1×: realised grip saturated at 1.2–2.0 N per finger for 2.9/14/100 N nominal (`FINGER_KE` had no effect); pads sank 0.9–1.6 mm; press slip 1.1–1.8 mm; press force saturated at ≈ 4–5 N, so F_cap ≥ 10 N never triggered. Carry slip was 1.60 mm at both scales whatever the grip; it is friction creep proportional to the finger time constant (20/6.7/3.3/2.2 ms → 1.60/0.47/0.19/0.095 mm). The wall-thickness explanation was refuted. | §2.1: finger shapes (both arms) use ke_f 2.5e5, kd_f 900 at 16 substeps (450 at 8), keeping 2/kd_f ≥ 2·dt; priority and solimp unchanged. ke 1e6 diverged at kd 500 and 1000 (the only values tried). Measured at 1×: grip 6.9/13.2/24.9 N for 7.1/14.3/28.6 N nominal; pad penetration 0.09–0.24 mm (2×: 0.03–0.06); press slip 0.07–0.19 mm (2×: 0.08–0.23). Side effect: the grip no longer absorbs touchdown by sliding. k_os rose from ≈ 1.5 to 2.5–4.4 at 20 mm/s and F_cap 5 N (up to 3.8 at 1×; peaks 12.4–21.9 N), which squeezes D1(ii); v_press 2 mm/s is added as a pre-registered rescue (user item D4). The pad bound caps the 1× grip near 30 N per finger (extrapolated). New P1 gate items: realised grip, pad penetration, carry slip. D1(iv) uses the chosen grip. P0 is re-run as P0′. New R14. |
| 2 | Clutch tested at rest, unloaded, in windows; dz ∈ [−0.5s, +0.3s] mm | r3's +1.0s edge sat only 0.1–0.37 mm below the 1× stud-top rest (+1.1 to +1.37 mm), which creeps down ≈ 19 µm/s and so closes that margin in 5–20 s. It admitted the 2× stud-top rest after a 2 mm drop (+1.84; +2.69 after a 4 mm drop). The probe scored any frame, so a 20 mm/s descent passed in transit: every (d) row "fired" at +0.71 to +0.96 mm at 1× and +1.92 to +1.99 mm at 2×. Seated and released, bricks rest at ≈ 0 to +0.05 mm within 1 s at 1× (still creeping up: +0.06 to +0.10 after 3–6 s) and +0.05 to +0.18 mm at 2×. Intermediate end states exist at 1×: 10 (b) trials ended in (+0.4, +1.0) mm (`big.jsonl`), and three (d) load trials ended at +0.24/+0.33/+0.35 mm, tilted 1.2–1.7° (`all.jsonl`). P0 (prototype gate \|dz\| < 1.5 mm): 6 of 247 successful snaps welded at +0.47 to +0.95 mm, above r4's band. | §2.2: a 1 s weld window opens when neither B finger touches the brick (d < 0), again for the step's unwelded bricks when A's fingers leave them at Unbrace, and after a V7 re-press. The gate is tested only at rest (speed relative to the partner), on the pose relative to the partner, and the first pass fires. +0.3s is ≥ 0.2s mm above the 1 s seated rest and ≥ 0.6s mm below the stud-top rest (1× and 2×); because intermediate 1× states exist near it, a new P1 band-sanity gate bounds how many bricks rest near the edge. −0.5s matches D1(iii) and V7's `below_seated`. P1 scores success, F_seat and c_held this way; transit passes are logged but never scored, and the probe gains a release phase. c_held is contiguous, with the literal maximum also reported. V7's seated band = the clutch band; V7's label comes from ground-truth dz at rest, not clutch state. M0(b) may see fewer snaps than P0 (tighter band). |
| 3 | P1(c) measures rest stability instead of raw jitter (post-hoc change to a pre-registered gate: user item D2) | Raw 2 s peak-to-peak was 0.037–0.048 mm at 1× (gate 0.02s), on free bricks with no fingers, so the finger change does not void it. Under r3's gate **1× fails P1 → s = 2**. It was linear creep (−19 to +25 µm/s). Detrended it is ≤ 0.0022 mm (1×) and ≤ 0.0043 mm (2×). A seated brick rises ≈ 0.1 mm in 5 s. | (c) is measured on arm-seated bricks after release (no body force; judged on seated bricks; r3 did not specify the resting state, and 2× stud-top rows also exceed 0.02s raw). Gate: detrended jitter < 0.02s mm and drift ≤ 0.1s mm/s. Under this gate 1× passes (c). The 1 s window bounds creep to ≤ 0.03 mm, and a brick left on the stud tops is not re-tested outside a window, so creep cannot weld it. |
| 4 | FK "seated" is gated where it is used | At offsets ≥ 1 mm the soft-gripped hand slid 3 mm down the brick, and FK reported seated with the brick at +1.3 mm. | The spiral trigger is FK-seated's only decision consumer; the FK stop is a motion limit, and V7 stays the authority. Gate: FK false positives ≤ 5 % over (b) at the chosen grip, else the next grip multiple. Row 1 removes the mechanism (press slip ≤ 0.19 mm at 1×, well below the 0.6s mm between the stud-top rest and the FK threshold). |
| 5 | Probe departures blessed; one wording corrected | GRASP_DZ was scaled; the rig was raised; collision ran every substep while the real-time gate used once per frame; c_held was contiguous; the proxy clearance is 0.70s mm in code. Smoke real-time factor on a shared GPU: 0.35 / 0.55 / 0.90 at 16 / 8 / 4 substeps. | GRASP_DZ = 8.9 + 4.1s mm (13.0 at 1×, unchanged). The rig puts the lower brick at z = 0.05 m; the cell's k_os is re-read in M0(b) and M1. P1 physics collides once per frame (the cell's mode), with one cross-check that collides every substep, covering (d) and (e). (f) runs first on an idle GPU and fixes n_sub (16, else 8; 4 is excluded because kd_f ≤ 240), otherwise U14. §0 fact 6 and (h): the proxy stud–wall clearance is 0.70s mm; (h)'s 0.6s mm tube clearance is unchanged. |
| 6 | Contact monitor: P0's two windows adopted; events merge across 0.5 s gaps | P0 needed two windows. The picked brick rests on the table during descend, grasp and lift (the implementer reported 172 spurious "unintended" events on one cube run without this window; that run was not saved). Support and neighbour contact continues through Release. Contiguous-only merging fragments flickering contacts: 26–37 `neighbour_rub` events per cube run; with a 30-frame merge, 18 on every cube run. On the arch `grasp_lp` runs the merge takes 453–507 displaced records to 35–40 events on 8 bricks (unintended 23–25, penetration 10, arm_arm 5–7); on S3 `grasp_lp` r0, displaced 146, unintended 76, penetration 56, arm_arm 8. **What remains is real brace-driven motion, not a counting artefact:** A's `grasp_lp` brace drags bricks a cumulative 1.0 m on arch r0 (largest single move 29.7 mm) and 5.9 m on S3 r0 (largest 319 mm), mostly while A holds and B transports, or during retract. | §2.4 adopts both windows and 30-frame (0.5 s) same-pair merging, keeping single-frame sensitivity. M0(a) and P3 step 8 use P0′. The brace-driven motion is a WP1 input (§3 P0). |
| 7 | P0 recorded; contingency fired | P0 (24 runs) had no gate. `grasp_lp` braces break the prototype (row 6), so r3's contingency fires: WP1 fixes the brace executor and M0 uses `lever_press` until then. Brace-independent failures (both brace models): arch b_010 6/6, hollow_box b_009 and b_010 6/6 (rest ≈ +5 mm, 12–16° tilt), S3 b_010 6/6. S3 `lever_press` diverged in 2/3 runs (t ≈ 145.5 s, step 14). HEAD reaches arch 10/11 and S3 13–14/16 with either brace model, against M0(b)'s 11/11 and ≥ 15/16. | §3 P0: `result v4_p0_baseline` recorded now, plus `failure v4_p0_s3_lever_press_diverged`; the failures are WP1 inputs, with an effort caveat on WP1. |
| 8 | Effort | — | P1 +0.5 agent-day; P0′ ≈ 45 min GPU; P3 step 8 runs on P0′. Total 21.75–22.75 agent-days without WP5. WP1's 1.75 days carry a caveat (§5). |

---

## Revision r3 (on adoption)

Both rows correct r2 row B, which stays below as the record.

| # | finding | what changed |
|---|---|---|
| A | Logic review round 3, verified by the orchestrator against `wr_passthrough/r_1x_fixed_ramp.txt`: r2 said seated bricks carry 71–100 N under a "speed-limited or ramped" load. The ramp runs start an **unseated** brick 2 mm above seated. | Wherever r2 said so (§1.2, §2.2 M2, U4, the WP5 spike), it now reads **"seated first (then stepped), or pressed by a speed-limited stand-in"**. At 1× an unseated brick under a 0.5 s force ramp **passes through at 71 N** (ends −9.2 mm; −11.6 mm at 100 N); at 2× the same ramp holds 100 N (−0.50 mm). U4: WP5's 1-day spike is **expected to pass at 2× and uncertain at 1×**. |
| B | Verified in `press.py` lines 114–122: the servo stand-in is a single z-prismatic joint to the world. | §0 fact 6: the stand-in has **no sideways, tilt or yaw freedom**, while a free brick under a 0.5 s ramp seats at ≤ 0.5 N at 0 offset (−0.072 mm at 0.5 N, −0.114 mm at 1 N). "Impact-driven" now covers only the lateral-offset (capture) runs. The U2 planning expectation changes from "leaning 2×" to **"no lean: P1's arm-held F_seat and c_held decide"**; R2 and the D1 expectation are softened to match. Also: **P1(e) reports finger-into-brick penetration** at each grip level (bound ≤ 0.3s mm), and **U2 names the trade-off** (the plan prefers the real brick mesh at 2× over a designed clearance at 1×; the user may flip it at D1). |

---

## Revision r2 — changes from r1

Rows 1–10 are the logic reviewer's round-2 findings. Row B is the work-reviewer's reproduction of §0 fact 6, which also supersedes r1 row 6 below.

| # | finding | what changed |
|---|---|---|
| 1 | Threaded planning collides with CUDA graph capture. cuRobo captures lazily on first call (`GraphExecutor._initialize_cuda_graph`: `gc.collect`, synchronize, `torch.cuda.graph` in global capture mode). Meanwhile the main thread runs `wp.capture_launch`, `.numpy()`/`.assign`, camera renders and ViewerGL. | New **CUDA-graph discipline** in §2.4. Newton's graphs are captured at startup, as today. **Every cuRobo graph is captured in a warm-up before the loop**: each arm, free-space and brick-attached queries, fixed batch sizes, and the collision cache pre-sized to the largest obstacle count (`MotionPlannerCfg(collision_cache=…)`), so later world updates happen in place. A guard fails the run if any graph is captured after warm-up. **P3 step 7 is a threaded soak**: ≥ 200 plans while the cell steps, both cameras render and ViewerGL runs. It is in the P3 gate. **Fallbacks, in order:** `use_cuda_graph=False` in the thread, synchronous plan calls (≈ 0.1 s viewer pause per plan), route 2. |
| 2 | The coarse gates fail under the plan's own calibration model. The `top` camera's error is about 4 mm per axis, so its radial p95 is about 9.7 mm. The "other chains" rows summed 1σ terms linearly and called the result p95. V4-fine's 1 mm gate was twice as strict as its own derivation. | §2.5.2 and §2.5.4 now **state the distribution**: independent zero-mean Gaussians per axis, with the stated values as 1σ. Terms combine by RSS per axis. Radial p95 = 2.45σ; single-axis or yaw p95 = 1.96σ. All chains are recomputed: `top` ≈ 10 mm radial p95; grasp ≈ 0.9 mm; plate index ≈ 0.8–1.1 mm; insertion 0.23–0.33 mm. **Gates are set at what the next stage needs:** V2 xy ≤ 15 mm (LookFeeder window ±26 mm); V4-coarse ≤ 20 mm / 2° (survey window ±78 mm at 150 mm); V4-fine ≤ 2s mm with 100 % correct lattice index; V3 ≤ half the per-side open clearance (1.3 mm at 1×). If a coarse gate still fails, V1 re-estimates `top`'s tilt from the known table plane (predicted ≈ 4.4 mm p95). |
| 3 | §0 fact 6 was stated as fact before it was reproduced. The case where contacts turn out stiff (capture set by 0.002 mm geometry and SDF blur, c_held ≪ 0.5 mm, so a 0.25c spiral cannot find the window) had no branch. | Fact 6 is rewritten from row B. Every hard-coded soft-contact claim is replaced by the reproduced finding: §1.2, §2.2 M2, R1, R2, U2, U4, the WP5 spike and the ledger probe. **New P1 on-failure branch:** if the prototype mesh fails P1's gate at both scales, or c_held(s) < 2 × predicted chain p95(s) at both scales, adopt the **contact-spike arm P1(h)**. That arm is a designed tube–stud clearance of 0.6s mm, keeping the mesh and slab proxies, because proxies alone cannot hold 0.5 N. It is pre-registered and run inside P1, so it costs no calendar time. The search is the fallback only if 1.5·c_held ≥ chain p95, i.e. the spiral can reach the error; otherwise escalate. |
| 4 | Grip force does not scale with the press. 1.5 N per finger gives about 2.1 N of axial hold, so presses slip. That biases D1 toward 1×, and FK "seated" becomes a false positive if the brick slides up. | §2.3 **grip rule:** per-finger normal force N ≥ F_press/μ, so the hold 2μN ≥ 2·F_press. It is set through `FINGER_KE` at a fixed 1.5 mm squeeze and capped at the 100 N finger effort limit. The real FR3 hand's 70 N continuous rating enters D1(iv). **P1(e) reports slip against grip force.** FK "seated" is advisory; V7 decides. P1 reports the FK-seated false-positive rate. |
| 5 | P1's "seated" is not M1's success criterion. The brick rests at +1.1 to +1.7 mm and needs to sink only about 0.5 mm before the weld snaps it. | **P1(b) and (d) are scored with the M1 clutch active:** success means the weld gate fires. Two lower-brick setups are used: the first course on the plate's proxy studs, and a brick welded to the plate. A free lower brick appears only in the fact-6 reproduction P1(g). **F_seat, F_pt and k_os are defined** in P1(d). c_held is taken at **F_press = 2·F_seat**. D1 adds the **press-window ratio 0.5·F_pt/(k_os·F_seat) ≥ 2**. |
| 6 | The permitted set misses the brace's other gripped bricks. | A's fingers are permitted to touch **every brick in the step's `gripped_bricks`**, from A's brace contact segment to Unbrace. **B's finger window starts at the descend segment**, not at Grasp. |
| 7 | The contact-segment check was either vacuous or always positive. | **Per-link check** (§2.3). Arm and finger spheres, with finger spheres fitted to ≤ 0.5 mm and 0–1 mm padding, are checked against the world minus only the grasped or braced bricks. The held brick is a numpy OBB checked against the world minus its supports and same-course neighbours. **P3 measures the false-positive rate on the cube's insert segments.** |
| 8 | The wrist view is likely occluded at the look pose. The held brick hides target points on the camera side; single-footprint targets show about 2 studs; a lower look makes this worse. | **P2 sweeps look height (25s–60s mm) and camera offset.** It reports visible unmasked target-course studs by grasp orientation and target type (towers and S3 piers separately). V5 uses P2's chosen look height. The budget's rotation × offset term now spans 0.04–0.10 mm. P2(c) allows the V5(a) edge fallback for single-footprint targets. WP3's lower-look remedy applies **only if P2 shows it keeps the stud count**. **R3 is raised to high.** |
| 9 | Tray refill timing and honesty. | **Refill happens only at step start, before PerceiveTray, with B outside the tray sector** (an `[env]` node in §2.6). §2.2, U6, U10 and the report call the feeder a **kitting dispenser stocked from the plan**; vision-only is unaffected. **The sector is clipped to bearings 50°–85°**, inside the measured feeder-slot bearings of ≈ 30°–85°, giving ≈ 0.064 m². |
| 10 | Nits. | M0(a) now uses a tolerance: within P0's run-to-run spread (P0 runs its config 3×) or ±1 event per class. D1 names F_press. V7 flags **`below_seated`**. P1(f) runs with the full weld pool allocated. **`build_plan(…, safety_factor=)` plumbing** is added to §2.8 and WP1. The effort was rebuilt from the table: **≈ 21–22 agent-days without WP5**. |
| B | Work-reviewer's reproduction (`wr_passthrough/press.py`, 29 result files). The numbers roughly reproduce r1's protocol, but **the conclusion was wrong**. | **§0 fact 6 is rewritten.** r1's "2 N pushes a brick through another" is a **tunnelling artefact** of a step body force on a light free brick. Seated bricks carry **71–100 N**, sinking 0.9–2.7 mm at 1× and 0.45–0.6 mm at 2×. A speed-limited stand-in press never passed through, but it **stayed on the stud tops up to ≈ 50 N at 1×**. The **free-brick 1 mm capture was impact-driven.** Consequences: every press is **arm-held and speed-limited**, never a raw body force on a free brick. P1 measures F_seat, F_pt, sink under load and c_held with the arm's own press profile. **D1(iii): sink ≤ 0.5s mm at the peak press** (2×'s smaller sink counts for 2×). The **U2 expectation moves from 1× to "leaning 2×"**. The WP5 spike is **expected to pass** (U4 reworded). The "ke ∝ s³" variant is deleted, because Newton's solref is already mass-scaled. The ledger probe is kept only as the reproduced, corrected finding. |

---

## Revision r1 — changes from the reviewed draft

| # | finding | what changed |
|---|---|---|
| 1 | accuracy gates impossible under the calibration model | New **§2.5.4 error budget** (per-term, 1σ, with its source), written before D1; every WP3 gate is re-derived from it. `top` is now **coarse only** (plate ≈ 5 mm / 0.5°, tray ≈ 5 mm / 5°). Fine alignment is **one wrist image containing both the held brick and the target neighbourhood (V5)**. That cancels the hand-eye translation, leaves hand-eye rotation × a ≤ 30 mm look→insert offset (≈ 0.05 mm), and predicts ≈ 0.24 mm radial p95. **LookFeeder is mandatory.** A **wrist plate survey** (V4-fine) resolves the lattice index. **Slip gate ≤ 0.2·c.** A **spiral search is part of the baseline Insert**. The `up` camera is dropped from the baseline; it is now the fallback if P2's wrist-visibility check fails. |
| 2 | s = 3 infeasible, tray too small or out of reach | Scale candidates are **{1, 2}** only (§0 fact 9: the cube's 30.8 mm grips need about 70 mm of the 80 mm stroke at s = 2). "s+1" is removed. If WP3 fails, the remedies are resolution, a closer look, a second look, then the search; there is one D1 revisit, 1 → 2, only if D1 chose 1. **Tray refill:** at most 6 bricks lie in the tray; spares are parked off-stage and teleported in as the tray empties. The tray is an **annular sector of B's measured pick ring (r 0.42–0.62 m)**. |
| 3 | gates barely exercise arm–arm avoidance | **S3 is added to the M0, M0.5 and M1 gates** (3 braced steps). A **zone stress test** runs S3 with `--brace-all` (bracing.assign with safety_factor = ∞, so A braces on every step that has a support brick). Arm–arm results are **reported separately for braced placements**. S5 (10 braces) enters E3 and E4 if s = 2; at 1× its flanked 2x2 grasps are inaccessible (G2 xfail). |
| 4 | contact-monitor definition broken | An event must involve an **arm link or the held brick**. The held brick may touch its supports and same-course neighbours only in the final Insert segment; neighbour contact is logged separately as `neighbour_rub`. **Arm–arm counts at any d < 0 in a single frame.** A new class counts brick–brick penetration (d < −1 mm), to catch pass-through. Distance comes from `mjw_data.contact.dist`; forces go into a **separate Contacts buffer** via `update_contacts`. |
| 5 | scale decision circular; E2 cannot show the null | D1 uses P1's **held-brick** capture c(s) and P2's **V5 prototype pixel terms, both at s ∈ {1, 2}**, fed into the §2.5.4 budget. E2 is relabelled a **confirmation** and runs at both scales. E1 renders V5 at both scales. Q1 is rewritten so that "1× suffices" is a reachable outcome. |
| 6 | §0 fact 6 wrong | **Verified, and it goes further.** The prototype's brick SDF mesh has collision on (`BRICK_CFG`, `has_shape_collision` default True) and contains the tubes (r 3.255 mm, 2-wide bricks) and 2.4 mm studs, next to r 2.2 mm stud proxies. My probe (new §0 fact 6) shows the contact is **soft**. Unpressed bricks rest on the stud tops even at 0 offset. A 0.25–1 N press seats 9/9 up to 1.0 mm offset. **2 N or more pushes a brick through another.** So neither the 0.6 mm nor the 0.002 mm geometric clearance governs, and c is measured, not derived. P1(b) now uses random direction and yaw, 20 trials per level, a held brick, and a **guarded (force-limited) press**. |
| 7 | weld pool bound to steps | The **weld pool is over body pairs**: N(N−1)/2 brick–brick plus N brick–plate (210 for S3 with 4 spares). The gate is checked against the **nearest lattice pose**, whose relpose is written at engagement (Newton allows runtime relpose and enabled changes only). |
| 8 | V5 obstruction check vs A's brace | A still braces first, which keeps prioritised planning deadlock-free. **A's links are masked in B's wrist image** by projecting A's padded collision spheres through A's FK (proprioception). The missing-stud check counts only unmasked studs. If the view is too occluded, a second look is taken with the hand yawed 180°. I chose this over "look before A braces", which would have B hovering in the zone while A moves in. |
| 9 | executed vs planned clearance | cuRobo trajectories are **time-scaled** and the planned joint velocity is fed forward (`control.joint_target_vel`). P3 **measures executed link deviation**, and the padding is set from its p99. **Executed** minimum clearance is logged. The final 30 mm contact segments are **collision-checked** with `curobo.collision_checking.RobotCollisionChecker`. |
| 10 | P2 gate on an untuned pipeline; Lambertian renderer | P2 is **gated on information content**: shape_index visibility and depth contrast. Studs are found **in depth**, not with RGB Hough. The wrist camera carries a **co-located point light** (ring light), since the renderer supports several spot, directional and point lights. The `up` camera is dropped; V6 is folded into V5. |
| 11 | scope | **Oblique cameras cut.** U4 default is **"decide after M1"**. **U12** is new (RL out of v4). **Bracing default `nearest`** (A6 verdict) becomes U13. |
| 12 | schedule | **M0.5** added. WP1 starts before D1 because scale is a parameter. Realistic M1 is day 12–15. P3 now generates the **FR3 sphere config** (cuRobo ships only Panda `franka.yml`). The riskiest assumptions are probed first: the alignment budget (P1 and P2), then cuRobo in-process (P3). |
| nits | | The FOV is vertical (`top` ≈ 1.32 mm/px). Depth is ray distance, converted through the ray directions, and 0 means invalid. `log_image` is ViewerGL only. The R5 relaxation is rewritten. The firewall poisons a **read-only ground-truth mirror**, not `body_q`. M0 is compared with P0 on P0's exact config. §0 fact 3 is corrected (the prototype does get safety factor 3.0). The U5 default is stated. The plate has (NI+4)×(NJ+4) visual studs (64 for the cube). E2 n = 40 gives Wilson ±13 % at 90 %, so the logistic fit is primary. Worker count is measured in P3 (expected 3–4, not 6–8). The `up` camera is moot. Plan calls run in a worker thread. |

---

## 0. What I checked while planning

1. **cuRobo 0.8 needs no torch build.** The checkout `~/Documents/ISDN_Robofab/curobo` (78fd485, `nvidia_curobo 0.8.0.post1.dev43`) is pure Python with runtime-compiled CUDA (`cuda-core`). Under the Isaac interpreter (torch 2.10+cu128, warp 1.13), `import curobo` fails only on missing modules: `setuptools_scm`, `yourdfpy`, `numpy-quaternion`, `cuda.core`. Only **`franka.yml` (Panda)** ships, so an FR3 sphere config must be generated with `curobo.sphere_fit`. `curobo.collision_checking.RobotCollisionChecker` exists for checking arbitrary configurations. Still unknown: warp 1.13 against code developed on 1.17, and cuda-core cu12 on this driver (P3).
2. **Not in the Isaac env:** `py_trees`, `open3d`, `ultralytics`, `sklearn`, `skimage`. **Present:** `cv2 4.13`, `torchvision 0.25`, `scipy 1.17`, `trimesh`, `lxml`, `viser`, `networkx`.
3. **The prototype has drifted from the planner.** `plans/sim_*.json` (24–25 Sep) are lever-model plans. `planner.build_plan` now defaults to `brace_model="grasp_lp"`: `bracing.assign` with safety factor 3.0 (`stability.SAFETY_FACTOR`), which the prototype does use. But `grasp_lp`'s `brace_pose` is a mid-height grasp point, and `dual_arm_sim.queue_brace` still presses down at `brace_pose + STUD_HEIGHT` and ignores `grasp.tcp_offset_m`. So the ledger's cube 8/8, arch 11/11 and hollow_box 10/12 are **not known to reproduce at HEAD**; P0 re-establishes them.
4. **Arms already collide physically in Newton** (every shape is in collision group 1). The user's complaint is about the MuJoCo twin, whose Panda links do not collide. In the Newton prototype, collisions happen as physical contacts that nothing plans around or counts.
5. **Contact readback.** Contacts are generated up to 10 mm apart (`gap = SDF_MARGIN`). Signed distance is `solver.mjw_data.contact.dist`, with `contact.geom` mapped to shapes by `solver.mjc_geom_to_newton_shape`. `SolverMuJoCo.update_contacts(c, state)` **overwrites** `c`'s count, shapes, points, normals and force from MuJoCo, so the monitor passes its own buffer (`pipeline.contacts()` allocated a second time).
6. **Brick contacts under a press.** This is r1's planning probe, corrected by the work-reviewer's independent reproduction: `press.py`, 29 result files, prototype brick model and solver settings, 60 fps × 16 substeps. dz = upper bottom − lower top, in mm.
   - **Model.** Brick shapes are the example's collision-enabled SDF mesh (studs r 2.4 mm; interior tubes r 3.255 mm on 2-wide bricks) plus invisible proxies (stud cylinders r 2.2 mm; wall and top-slab boxes).
     - Nominal clearances: tube–stud 0.002 mm (mesh); stud–wall 0.4 mm (mesh) or 0.70 mm (proxy: stud proxy r 2.2 mm against a wall proxy inset 0.1 mm; r3 said 0.6). The combined contact margin is 0.08 mm.
     - Newton maps the brick ke/kd to a MuJoCo solref (time constant 12.9 ms, damping ratio 0.53). That is a stiffness per unit acceleration, so **contact force per mm scales with body mass (∝ s³)**.
   - **At rest:** an unpressed brick rests on the stud tops (dz +1.1 to +1.7) even at 0 offset. The contact is mesh–mesh at the tube–stud rims, because the 0.002 mm clearance is below the 0.08 mm margin.
   - **r1's "≥ 2 N pushes a brick through another" is a loading artefact.**
     - r1 applied a constant body force from t = 0 to a free 1.7 g brick starting 2 mm above seated, i.e. about 590 m/s² per newton.
     - Past the stud rims (≈ 0.25 N) the brick reaches the seat at ≈ 1.3 m/s at 2 N and 7.5 m/s at 71 N. That is 1.3–8 mm per 1.04 ms substep against 1–1.5 mm walls, so it tunnels.
     - Reproduced at 1×: 0 N rests at +1.1 to +1.3; 0.5–1 N seats (−0.07 to −0.17); 2 N ends at −2.4 to −4.1; 3 N at −8.9 to −9.2; ≥ 5 N passes fully through.
   - **Seated bricks carry v3.1's forces (1×):** seated first at 0.5 N, then a step to F: 71 N sinks 0.91 mm on a fixed lower brick and 2.24 mm on a free one or on the baseplate's proxies; 100 N sinks 1.17 and 2.5–2.7 mm.
   - **An unseated brick under a force ramp (1×):** free upper brick from +2 mm, F ramped from 0 over 0.5 s, fixed lower brick, 0 offset.
     - It **seats at ≤ 0.5 N**: −0.072 mm at 0.5 N, −0.114 mm at 1 N.
     - It holds to 50 N (final −0.78, transient −4.3) and **passes through at 71 N** (ends −9.2 mm; −11.6 mm at 100 N).
   - **At 2× (8× the mass):** seated-first, ramped and free-lower cases all hold 100 N at −0.45 to −0.6 mm; the same 0.5 s ramp ends at −0.50 mm at 100 N. A step from a scaled +4 mm start passes through at ≥ 50 N.
   - **A speed-limited stand-in never passed through, but it barely seated from the stud tops.**
     - Stand-in for an arm: the brick on a single z-prismatic joint to the world (0.5 kg armature, ke 2e4 N/m, kd 200, force-capped), so it has **no sideways, tilt or yaw freedom**; 1×, fixed lower brick, 0 offset.
     - At 20 mm/s the brick stays on the stud tops up to 50 N (+1.61), then reaches +0.59 at 71 N and +0.05 at 100 N.
     - At 100 mm/s it is at +1.0 at 10 N, +0.13 at 71 N and +0.01 at 100 N.
     - So this locked stand-in comes within r3's weld gate (dz ≤ +1.0s, under load) only at tens of newtons at 1×; r4's gate (≤ +0.3s, unloaded, §2.2) is stricter, while the free brick under a ramp seats at ≤ 0.5 N at the same 0 offset. An arm-held brick, with some sideways and tilt compliance through the grip, is neither; P1 measures it.
   - **The lateral-offset (capture) runs were impact-driven.**
     - Free brick, 0.2–1 N step, offsets ≤ 1.0 mm in x, y and diagonal: 19/21 seat (dz −0.04 to −0.11; lateral residual 0–0.33 mm). The other 2 jam, tilted 2.7°, at a 0.5 mm x offset. At 1 N the brick dips 1.2–2.5 mm into the lower brick on the way in, then recovers.
     - These bricks seat because they hit the rims at ≈ 0.3 m/s. An arm press does not, so **this ≈ 1 mm is not an arm-held capture**. (At 0 offset, the slow ramp above seats without an impact.)
   - **Load path when seated and loaded:** the lower brick's top-slab proxy against the upper SDF mesh carries ≈ 75–95 %; mesh–mesh carries 10–30 %. **Proxies alone cannot hold even 0.5 N.**
   - **Settings (1×, step protocol):**
     - 32 or 64 substeps raise the pass-through threshold to 3 N or 5–10 N;
     - ke × 10 with kd unchanged is worse; ke × 100 is unstable;
     - ke × 10 with kd × √10 seats up to 5 N;
     - solimp dmax 0.99 gives no gain.
   - **Caveats:**
     - The servo is a stand-in, not the prototype's gripper press. The prototype drives 1 mm past seated over 1.2 s with an unmeasured force, then welds at |dz| < 1.5 mm, possibly while the brick is still on the stud tops.
     - A free lower brick also sinks into its support under load: 8.2 mm into the ground at 71 N, and 2.0–2.6 mm on the baseplate's proxies.
   - **Consequences:**
     - **Every press is arm-held and speed-limited**, never a raw body force on a free brick. The only exception is P1(g), which reproduces this fact.
     - P1 measures F_seat, F_pt, sink under load and c_held **with the arm's own press profile and the M1 clutch active**.
     - The grip must carry the press (§2.3).
     - 2×'s smaller sink under load enters D1.
     - v3.1's 71 N regime is representable in principle once a brick is seated; the WP5 spike is expected to pass at 2× and is uncertain at 1×.
7. **Welds:** `SolverMuJoCo` exposes `mjw_data.efc` (type, id, force) and updates weld `relpose` and `enabled` at runtime through `notify_model_changed(CONSTRAINT_PROPERTIES)`, which the prototype already uses. **A weld's bodies cannot change after `finalize()`.** `newton.eval_jacobian`, `Control.joint_f`, `Control.joint_target_vel` and `newton.ik` custom objectives also exist.
8. **Cameras** (`SensorTiledCamera`):
   - FOV is **vertical**. Depth is **ray-hit distance**, not z. `clear_depth = 0`, so background pixels are 0.
   - Shading is Lambertian with ambient light and optional shadows. From a nadir view, stud tops share the top face's shade, so studs must come from depth.
   - Lights can be **spot, directional or point, several at once**, with arrays updatable per capture.
   - Only **ViewerGL** implements `log_image`.
9. **Scale is capped at 2 by the gripper.** The cube's plan grips 30.8 mm across the 4-stud axis (`grip_width_m 0.0308`). At s = 2 that is 61.6 mm, plus 1.2 mm and 2s mm of opening per side ≈ 70 mm of the FR3's 80 mm stroke; s = 3 needs 92 mm. Every other shape's maximum grip is 14.8 mm at 1×.
10. **Braced steps** (`grasp_lp`, identical under `nearest` and `weakest_joint`, because the strategy changes where A braces, not whether):

    | shape | bricks | braced steps |
    |---|---|---|
    | cube | 8 | none |
    | arch | 11 | 1 (step 10) |
    | hollow_box | 12 | none |
    | S1 | 6 | none |
    | S3 | 16 | 3 (steps 11, 13, 14) |
    | S5 | 26 | 10 |

    `bracing.assign(..., safety_factor=...)` makes a brace-everything stress mode a one-argument change.
11. **A project rule conflicts:** `CLAUDE.md` says "Physics stays on CPU MuJoCo". Only the user can amend it (U0).

---

## 1. Question, hypotheses, and what happens to v3.1

### 1.1 What this phase must establish

v4 is **mostly engineering**. It delivers the user's four requests as one pipeline: collision avoidance, Newton visuals, a scale chosen for precision, and vision-only bricks and placement locations. It keeps the standing steers: prototype first, images → blueprint input, stabilizer A + placer B, ledger discipline.

- **Q1 (precision; the main v4 finding).** With vision only and position-controlled arms, how large is the chained alignment error at insertion, how does placement success depend on it at s = 1 and s = 2, and is 1× enough?
  - *Outcome A (1× suffices):* at s = 1 the chain's p95 lies where E2's success is ≥ 95 %. The user's "larger only if precision requires" is answered: it does not.
  - *Outcome B:* 1× falls short and s = 2 reaches ≥ 95 %.
  - *Outcome C:* neither scale reaches 95 % on alignment alone. The spiral search is the precision mechanism, and the report quantifies what it buys.
- **Q2 (coordination; engineering, measured).** Does prioritised dual-arm planning remove unintended contacts at acceptable throughput? The stabilizer plans first and is static during insertion, and at most one arm moves in the shared zone.
  - *Positive:* zero arm–arm contacts (single-frame sensitivity), including on braced placements and in the zone stress test; ≤ 1 unintended arm/held-brick event per 50 placements; cycle time ≤ 1.5× the prototype's.
  - *Negative:* residual events traced to the planner's world model or to tracking error.
- **Q3 (only if U4 = yes and WP5's contact spike passes).** Does v3.1's bracing direction hold in a second simulator? This is a small-n cross-simulator check, not a new bracing claim.

### 1.2 What happens to v3.1

| v3.1 item | v4 treatment |
|---|---|
| `Docs/results_report.md`, `results/{a6,g3,g6,wp5}`, gates G0–G7 | **Frozen.** Not re-stated, re-run or re-interpreted. Any cross-simulator comparison is labelled as such. |
| `planner.py`, `blueprint.py`, `stability.py`, `bracing.py`, `sim/joint_model/capacity.py`, §2.6 JSON contracts | **Carried over** (pure numpy/scipy). Geometry constants get one `SCALE` (WP1). |
| `experiments/analyze.py` statistics | **Reused** by `experiments/analyze_v4.py`. |
| `orchestration/twin_executor.py` | **Reference only:** tree shape (retries, LiftAndRegrasp) and episode fields. |
| `sim/mj/*`, `tasks/*` (RL), `sim/joint_model/clutch.py` | **Superseded** as the execution path; kept runnable and frozen. **RL is out of v4 (U12).** |
| cuRobo decision `curobo_isaac_split` | **Reversed** if P3 passes, and only with the user's OK (U5). |
| v3.1 M7 force-controlled press, 71 N | **Representable in principle.** §0 fact 6: bricks carry 71–100 N when seated first (then stepped), or pressed by a speed-limited stand-in. At 1× an unseated brick under a 0.5 s force ramp passes through at 71 N (ends −9.2 mm; −11.6 mm at 100 N); at 2× the same ramp holds 100 N (−0.50 mm). r1's 2 N pass-through was a loading artefact. M1 uses an arm-held, speed-limited, force-capped press at P1's F_press, with a spiral search only where it can reach the error. Joint breaking returns only through WP5 (U4). |

---

## 2. Architecture

### 2.1 Simulator — standalone Newton in the Isaac 6.0 interpreter (U1)

Standalone Newton 1.2.1 (`~/Codes/CAIRSS/Issac/bin/python`), extending `dual_arm_sim.py`, **not Isaac Lab 6.1-on-Newton**:
- It is what the user liked: live ViewerGL, about 10 s to start, about real time.
- The cell is driven frame by frame from Python: a behaviour tree each tick, runtime weld toggling, captures on demand. Isaac Lab's manager layer adds batched RL and RTX cameras, which v4 does not need.
- `SensorTiledCamera` is native Newton and gives RGB, depth, normals and per-pixel shape id.
- Isaac Lab's Newton backend is beta, and the project has a history of Isaac boot and rendering failures.
- "Isaac" is still met: the run uses the Isaac interpreter, and an optional WP7 hero render replays a `--viewer usd` recording in Isaac Sim 6.0 RTX.
- **Escalation:** if P2 finds the images uninformative (depth *and* RGB), the user is offered Isaac Lab with an RTX TiledCamera.

**Physics** stays as the prototype's except at the fingers (r4): `SolverMuJoCo(solver="newton", integrator="implicitfast", iterations=15, cone="elliptic", impratio=50, use_mujoco_contacts=False)` with Newton's CollisionPipeline (SDF plus proxies, `sap`, `reduce_contacts=True`), 60 fps × n_sub substeps, collision once per frame. n_sub = 16 unless P1(f) picks 8 for real time (U14).
- Brick contacts keep the example's ke, kd and margin at both scales.
  - Newton maps ke/kd to a MuJoCo solref: time constant 2/kd, damping ratio (kd/2)·√(1/ke). That is a stiffness per unit acceleration, so force per mm scales with brick mass (∝ s³). There is no separate brick stiffness variant.
- Finger contacts (r4): the FR3 finger collision shapes (bodies 12/13 in `dual_arm_sim.build_arm`, both arms) use ke_f = 2.5e5, with kd_f = 900 at 16 substeps or 450 at 8.
  - The fingers keep their existing `geom_priority` 1 and solimp (0.7, 0.95, 1e-4, 0.5, 2.0) (`dual_arm_sim.py:131–135`). With the higher priority, MuJoCo uses the finger's solref and solimp alone at every finger contact, so the brick's ke/kd do not enter.
  - That gives time constants of 2.2 / 4.4 ms (≥ 2·dt) and damping ratios of 0.9 / 0.45. ke 1e6 diverged at kd 500 and 1000 (the only values tried).
  - r3 left them at Newton's default ShapeConfig (ke 2500, kd 100 → 20 ms, damping ratio 1): ≈ 1.1–1.3 kN/m against a 1× brick.
  - With that default, the grip saturated at 1.2–2.0 N per finger, the pads sank 0.9–1.6 mm, and the brick crept 1.6 mm in the hand during a 0.5 m/s carry.
  - With the r4 values at 1×: grip 6.9/13.2/24.9 N for 7.1/14.3/28.6 N nominal; pad penetration 0.09–0.24 mm; press slip 0.07–0.19 mm; carry creep 0.095 mm. At 2×: penetration 0.03–0.06 mm, press slip 0.08–0.23 mm.
  - The pads remain mass-scaled: about 8× softer on a 1× brick than on a 2× one.
- The collision mesh is the prototype's, unless P1's branch adopts the designed-clearance variant P1(h).

### 2.2 Cell and brick model

**Layout.** The prototype's, randomised so that vision is necessary:
- FR3 A (stabilizer) at (−0.45, 0, 0) facing +x; FR3 B (placer) at (+0.45, 0, 0) facing −x.
- Baseplate at a **random pose per seed** (±30 mm x/y, ±15° yaw). Its visual mesh has (NI+4)×(NJ+4) studs (64 for the cube, 84 for S3, 100 for hollow_box); collision studs cover only the NI×NJ build area.
- **Tray:** the annular sector of B's measured pick ring, r ∈ [0.42, 0.62] m from B's base.
  - It spans bearings **50°–85°** from B's facing direction towards −y, inside the measured feeder-slot bearings (≈ 30°–85°). That is about 0.064 m², entirely outside every shape's shared zone.
  - WP1 may extend the sector to 100° only after it verifies picks there.
- **Refill:** the tray is a **kitting dispenser stocked from the plan**. It belongs to the environment, not the robot.
  - It holds at most 6 bricks: the next ≤ 4 needed plus ≥ 2 distractors of other types when available. They are scattered studs-up with uniform yaw and ≥ 25 mm apart.
  - Every brick body is pre-allocated. Bricks not in the tray are parked off-stage (x = 3 m, spaced on the ground plane, outside every frustum).
  - Refilling teleports bricks in by writing `body_q` and zeroing `body_qd`. **It happens only at the start of a step, before PerceiveTray, while B is outside the tray sector.** Nothing appears under B's fingers, and nothing appears after the tray was perceived.
  - The robot still finds every brick by vision. The executor re-perceives the tray before every pick.

**Scale s ∈ {1, 2} (U2, decided at D1 by §2.5.4's rule).**
- Every brick dimension scales by s: pitch 8s mm, height 9.6s mm, stud 4.8s mm.
- s = 2 is exactly the DUPLO grid. At s = 2 the 17.5 mm finger pads stop overhanging a 2x2 face, which removes v3.1's flanked-2x2 inaccessibility (so S5 becomes feasible).
- s ≤ 2 by the gripper stroke (§0 fact 9).
- Implementation: one `SCALE` drives `planner.PITCH`, `BRICK_H`, `STUD_H` and `stability.MASS_PER_STUD` (s³). `FINGER_W` stays 17.5 mm.
  - `GRASP_DZ` = `TIP_BELOW_TCP` + 4.1s mm (8.9 + 4.1s: 13.0 mm at 1×, the prototype's value; 17.1 mm at 2×). The fingertips therefore end 2.4s mm above the lower course's stud tops (r4). A fixed 13 mm would leave ≈ 2 mm of wall between the pads at 2×.
  - Mesh vertices are × s; `_build_mesh_with_sdf(..., scale=s)` for the narrow band and margin; proxies as in `example_brick_stacking.add_board_floor`.
  - The tube collision radius is one parameter; P1(h)'s designed-clearance variant changes only that.

**Clutch, M1 — a weld pool over body pairs; r5: stiff, breakable (WP5a), grouped.**
- Pool: one disabled weld per unordered brick pair plus one per brick to the plate. N bricks including spares give N(N−1)/2 + N welds (S3 with 4 spares: 210; S5 with 4: 465).
- When it is tested (r4): only inside a weld window, on an unloaded brick at rest. It is never tested in transit or under the press. This is on the physics side, where ground truth is allowed; the executor never sees weld state.
  - Windows last 1 s each and open, for a brick that is still unwelded:
    - when neither B finger touches it (no finger contact row with d < 0; rows at d ≥ 0 inside the 10 mm SDF margin do not count), i.e. after Release;
    - at Unbrace, for every unwelded brick of the step (the placed brick and A's `gripped_bricks`), when neither A finger touches it (d < 0). A still holds during B's release window, and P0 shows a `grasp_lp` brace moves bricks by millimetres to hundreds of millimetres, so a brick braced in motion gets a second chance once A lets go;
    - after a V7 re-press, as after Release.
  - "At rest" and the pose are taken relative to the partner (the lower brick, or the plate): on every frame where the brick's speed relative to the partner has been < 1s mm/s for ≥ 6 consecutive frames (0.1 s), the scene tests it against each unwelded partner whose footprint overlaps.
  - The first passing frame fires the weld. If no window passes, the brick stays unwelded. Each evaluation (window, fired, dz, |Δxy|, tilt) goes into the episode row.
  - A brick that V7 calls seated but that ends the step unwelded is logged as `seated_unwelded` (ground truth, scoring only). It does not count as placed: M0(b) and M1 count welded bricks.
- Gate (in the partner's frame, relative to the nearest lattice pose, i.e. offset rounded to the pitch and yaw to 90°): |Δxy| < 1.2s mm, dz ∈ [−0.5s, +0.3s] mm, tilt < 4°, yaw < 5°.
  - Why this band: seated and released after a small force, bricks rest at ≈ 0 to +0.05 mm within 1 s at 1× (creeping up to +0.06–+0.10 over 3–6 s) and +0.05 to +0.18 mm at 2×. On the stud tops they rest at +1.1 to +1.37 mm (1×) and +1.84 to +2.69 mm (2×, depending on the drop). +0.3s is ≥ 0.2s mm above the seated rest and ≥ 0.6s mm below the stud-top rest, at both scales. At 1× some presses end in between (r3 smoke: +0.24 to +0.35 mm tilted 1.2–1.7°, and several in +0.4 to +1.0 mm), so P1 gates how many bricks rest near the edge (band sanity). −0.5s is D1(iii)'s sink bound and V7's `below_seated` edge. r3's +1.0s edge sat 0.1–0.37 mm below the creeping 1× stud-top rest and admitted the 2× one.
  - Creep: seated bricks drift ≤ 25 µm/s (≤ 0.03 mm within a window). A brick left on the stud tops creeps down at ≈ 19 µm/s (1×) to 32 µm/s (2×); it reaches the band only after ≈ 40 s, far outside any window, so creep cannot weld it.
- On pass: the weld's `relpose` is set to that lattice pose and it is enabled. The snap is independent of the plan, so any tray brick of the right type can be used for any step.
- **r5 (replaces r4's "welds never release; a brace cannot change an outcome once a brick is welded"):**
  - The weld is stiff (`eq_solimp` (0.9999, 0.9999, 0.001, 0.5, 2.0); `eq_solref` (0.002, 1.0) unchanged) and **breakable**: each frame, `Patch.utilization` on the end-of-frame weld wrench, u ≥ 1 on 2 consecutive samples from 0.1 s after the weld fires, axial and prying only (WP5a, now).
  - Welded bricks and the plate share structure collision group −2 (pre-failure only): a declared weld-only idealisation (drops bearing/friction between same-course welded neighbours, non-interpenetration among welded bricks, load sharing through contact). External loads on welded bricks stay contacts.
  - **The build ends at its first failure** (gate miss, break, divergence, timeout); every scored state is pre-failure, so every structure brick is welded and grouped.
  - A brace can now change an outcome, through a break or through sub-critical displacement that changes the snap gate. Before its weld, a brace can also move a brick (P0). See the r5 section.

**Joint model, M2 (WP5; r5: U4 = yes, split into WP5a now and WP5b after M1).** WP5a is the r5 break model above (no spike needed: loads among welded bricks go through welds). WP5b, the detent and the v3.1 force regime, keeps the text below and the spike. From the draft: a detent weld holds the brick at the stud tops until the axial reaction exceeds n·f_insert, and the clutch weld's wrench → `capacity.Patch.utilization()`, with a break when u ≥ 1 for 2 frames. **Not blocked:** §0 fact 6 shows bricks carry 71–100 N when seated first (then stepped), or pressed by a speed-limited stand-in. At 1× an unseated brick under a 0.5 s force ramp passes through at 71 N (ends −9.2 mm; −11.6 mm at 100 N); at 2× the same ramp holds 100 N (−0.50 mm). WP5 still **starts with a 1-day contact spike** (§3 WP5), which is expected to pass at 2× and is uncertain at 1×. It measures sink and pass-through under ≈ 92 N applied with the arm's Jᵀ·F press. E5 runs only if the spike passes.

### 2.3 Arm control

- **Free-space motion:** planned trajectories (§2.4), **time-scaled** so that peak joint velocity and acceleration are ≤ the P3-chosen fraction of limits (start 50 %), resampled to frame rate. Written to `control.joint_target_pos` **and `joint_target_vel`** (velocity feed-forward cuts PD lag). Prototype gains (ke 400, kd 40) with gravity compensation.
- **Contact segments** (last 30 mm of grasp, insert and brace): straight Cartesian segments through the batched `newton.ik.IKSolver`. They are **collision-checked per link before execution**, at 2 mm steps:
  - **arm links and finger spheres** vs the world model minus only the brick being grasped (B) or the braced `gripped_bricks` (A), using cuRobo's `RobotCollisionChecker`. Finger spheres are fitted to ≤ 0.5 mm over-approximation and padded 0–1 mm, because by `GRASP_DZ` design (§2.2) the fingertips end 2.4s mm above the lower course's stud tops.
  - **the held brick**, a numpy OBB test against the world's cuboids, vs the world minus its supports and same-course neighbours.
  - A failure triggers a replan with the next grasp yaw or approach. P3 measures the false-positive rate on the cube's insert segments.
- **Align:** an FK integral servo moves the TCP by V5's correction (§2.5) and settles until the FK error is < 0.03 mm. It uses no ground truth.
- Every press is arm-held and speed-limited (§0 fact 6). The TCP descends at v_press ∈ {5, 20} mm/s (P1 picks; 2 mm/s only as P1(d)'s pre-registered rescue for D1(ii), r4), and the press force is capped. No executor path, and no experiment except P1(g), applies a body force to a brick.
- Grip: the per-finger normal force must satisfy N ≥ F_press/μ, so the axial hold 2μN ≥ 2·F_press.
  - It is set through `FINGER_KE` at the prototype's 1.5 mm squeeze. Today's 1000 N/m gives 1.5 N per finger, i.e. ≈ 2.1 N of hold at μ 0.7, which is too little for any press above ≈ 1 N.
  - This works only with stiff finger contacts (r4, §2.1). With r3's defaults, the realised grip saturated at 1.2–2.0 N per finger whatever `FINGER_KE` was.
  - The realised N falls short of nominal by the pad penetration over the 1.5 mm squeeze. P1(e) checks realised N ≥ 0.8× nominal and pad penetration ≤ 0.3s mm. At 1× that bound caps N near 30 N per finger (extrapolated).
  - P1 picks the grip multiple m ∈ {1, 2, 4}, with N = m·F_press/μ: the smallest m that meets the slip gate.
  - N is capped at the 100 N finger effort limit. D1(iv) compares the chosen N with the real FR3 hand's 70 N continuous rating.
- Insert (guarded press), baseline:
  - Descend at v_press from 3s mm above the stud tops. Stop when the simulated wrist wrench |F_z| ≥ F_press, or when FK z ≤ seated − 0.2s mm. Hold 0.3 s, then Release. The M1 clutch tests the released brick at rest (§2.2, r4).
  - F_press = 2·F_seat(s), from P1(d). It must satisfy D1(ii): k_os·F_press ≤ 0.5·F_pt.
    - With stiff finger contacts, the grip no longer absorbs the touchdown by sliding. The review measured k_os 2.5–4.4 at 20 mm/s and F_cap 5 N (up to 3.8 at 1×), so v_press is the lever.
  - FK "seated" (FK height ≤ seated + 0.5s mm) is advisory, because a brick that slides up in the fingers fools it. V7 is the authority.
    - Its only decision consumer is the spiral trigger below. The FK stop above is a motion limit.
    - With r3's soft grip, the hand slid ≈ 3 mm down bricks left at +1.3 mm. P1 gates the false-positive rate at the chosen grip at ≤ 5 %.
  - **If not seated:** a spiral search — lift 1 mm, then an Archimedean spiral r ≤ 1.5c in 0.25c steps, at most 12 guarded probes (a port of v3.1 M7's search). It is **enabled only if 1.5·c_held(s) ≥ the chain p95(s)**; otherwise its window is too small to find.
- **Sensing (proprioception, allowed):** joint q and qd, gripper width, FK, and a simulated wrist wrench from Newton contact forces on the fingers (labelled a sensor model).

### 2.4 Collision-free motion and coordination (U5)

**Planner: cuRobo 0.8 MotionPlanner, one per arm; P3 decides the route.**
1. **In-process** from an overlay: `Issac/bin/python -m pip install --no-deps --target brickassembly/.vendor <P3's exact list>`, appended to `sys.path` (never prepended). The Isaac env is untouched; the list goes in `env.lock` and `.vendor/` is gitignored.
2. **Out of process:** `motion/curobo_server.py` under isdnenv behind `multiprocessing.connection.Listener` (stdlib).
3. **Newton-native** only if both fail: IK with sphere-avoidance objectives plus a numpy RRT-Connect (about 150 lines) against the sphere model.

**Robot model:** the same `fr3_franka_hand.urdf` Newton loads. FR3 collision spheres are generated once in P3 (`curobo.sphere_fit`) and saved to `motion/fr3.yml`. **Padding comes from P3's measured executed deviation** (p99 + 2 mm, at least 5 mm).

**World model — from perception, never ground truth** (WP2 may use ground truth; WP4 switches):
- table and plate cuboids at the registered pose;
- one cuboid per placed brick at its **verified** pose;
- tray bricks (minus the one being grasped);
- the held brick attached to B;
- **the other arm** as its link spheres at its current or hold configuration.

**Plan calls never block the tick.** They run in a `concurrent.futures.ThreadPoolExecutor(1)`. The behaviour-tree leaf returns RUNNING until the future resolves, while the arms hold position and the viewer keeps rendering.

**CUDA-graph discipline.** cuRobo captures its graphs lazily on first call: `GraphExecutor._initialize_cuda_graph` runs `gc.collect`, synchronizes, then `torch.cuda.graph` in global capture mode. A capture in the planning thread would invalidate the main thread's `wp.capture_launch`, `.numpy()`/`.assign`, camera renders and ViewerGL calls, or be invalidated by them. So no graph is ever captured while the loop runs:
1. Newton's graphs (simulate, IK) are captured at startup, as the prototype does now.
2. **A warm-up before the loop captures every cuRobo graph:** one plan per arm, for both free-space and brick-attached queries, at fixed batch sizes. The collision cache is pre-sized to the largest obstacle count of the run (`MotionPlannerCfg(collision_cache=…)`), so world updates only write in place.
3. A guard counts `GraphExecutor` initialisations after warm-up and fails the run if the count is not zero.
4. P3's threaded soak test (step 7) runs this setup under load.
5. **Fallbacks, in order:** `use_cuda_graph=False` in the thread, if P3's latency allows; synchronous plan calls on the main thread (≈ 0.1 s viewer pause per plan); route 2.

**Arm–arm coordination:**
- **Shared zone:** a vertical cylinder around the build, radius = structure half-diagonal + 0.12 m. **At most one arm moves inside it.** An arm static inside it is an obstacle for the other's plan.
- The stabilizer plans and moves first and is static during insertion, so the scheme is deadlock-free: B waits for A to be static or out of the zone.
- The tray is B's alone; A parks outside B's paths and B's camera frusta.

**Contact monitor** (`cell/contacts.py`; used for gates, the HUD and evaluation, never by the executor):
- **Each frame:** read rows `i < nacon` of `mjw_data.contact` (`dist`, `geom` → shape → body → class). Classes: arm A link, arm B link, finger, held brick, placed brick, tray brick, plate, table/ground. The held brick is ground-truth-defined (touching both B fingers); scoring only.
- **Only pairs with an arm link or the held brick are candidate events.** Resting brick–brick, brick–plate and brick–table contacts are ignored, except for penetration (below).
- Permitted, with their phase windows:
  - B fingers ↔ the brick being grasped, from the start of the descend segment to Release;
  - the brick being grasped ↔ the table or ground it rests on at its pick-up spot, from the start of the descend segment to the end of Lift (r4);
  - A fingers ↔ every brick in the step's `gripped_bricks` (`grasp_lp` pads grip several bricks by design), from A's brace contact segment to Unbrace;
  - held brick ↔ its supports and ↔ same-course neighbours, from the start of the final Insert segment through Release, including the search and a V7 re-press (r4: the brick still sits on its supports while the fingers open). Neighbour contacts are logged as `neighbour_rub` (max penetration, duration) and not counted as unintended.
  - Permitted contact is not permitted motion: a braced brick that moves is still counted in the displaced class.
- **Arm–arm event:** any d < 0 between A and B links, in any single frame.
- **Unintended event:** any non-permitted arm or held-brick pair with d < 0 in any single frame.
- **Penetration event:** any brick–brick pair with d < −1·s mm (catches pushed-through bricks). D1(iii) keeps the legitimate sink under the press ≤ 0.5s mm, so the threshold stays meaningful.
- **Displaced brick:** any non-held brick moved > 1·s mm from its last rest pose.
- Merging (r4): the d < 0 frames of one pair merge into one event while the gaps between them are ≤ 30 frames (0.5 s). For displaced bricks the unit is the brick rather than a pair.
  - Each event is logged with phase, max penetration, peak force (from the monitor's own Contacts buffer via `update_contacts`), first and last frame, contact-frame count and fragment count. Displaced events also log the brick's path length and largest single move.
  - A single frame still opens an event, so single-frame sensitivity is kept.
  - Why: r3 merged only contiguous frames, so a contact flickering at d ≈ 0 split into many events. On P0 the merge takes 26–37 `neighbour_rub` events per cube run to 18 on every run, and the arch `grasp_lp` runs' 453–507 displaced records to 35–40 events on 8 bricks.
  - The merged counts that remain on braced runs are real: A's `grasp_lp` brace drags bricks a cumulative 1.0 m on arch r0 (largest single move 29.7 mm) and 5.9 m on S3 r0 (largest 319 mm), mostly while A holds and B transports, or during retract. Merging changes the count, not that finding.

### 2.5 Perception — vision only (U6, U7)

**2.5.1 The rule.**
- **Allowed:** camera images (RGB, depth, exact intrinsics, calibrated extrinsics with hidden error); proprioception of both arms (q, FK, gripper width, simulated wrist wrench); brick CAD; the blueprint and plan in the plate frame.
- **Never:** brick or plate `body_q`, the `shape_index` image, or clutch states. Those score, calibrate at design time, or label training data.
- **Enforcement:**
  - The executor gets a `Robot` facade: copies of arm q and qd, gripper width, wrench, `capture(cam)`, and A's FK for masking.
  - Ground truth for the scorer and viewer lives in a separate **read-only mirror** (a numpy copy made after each step).
  - `tests/test_cell.py::test_firewall` runs one placement twice on the same seed, the second time with the mirror randomised, and asserts identical command traces. It also asserts that no `Robot` attribute references a `Model` or `State` (attribute-type whitelist). Physics is never touched.

**2.5.2 Cameras** (`cell/cameras.py`). The FOVs below are vertical, as the API takes them.

| cam | mount / pose | res, VFOV | mm/px | role |
|---|---|---|---|---|
| `top` | fixed, 1.10 m above (0.10, −0.20), looking down | 1280×960, 60° | ≈ 1.32 at the table | coarse only: tray inventory and pose, coarse plate, final as-built check |
| `wrist_B` | on B's hand, +45 mm in hand x (the non-finger side; P2's sweep may move it to 60 or 75 mm), 60 mm above the TCP, tilted 25° toward the tool axis; **co-located point light** | 640×480, 55° (HFOV ≈ 70°) | ≈ 0.22 at 0.10 m | LookFeeder, plate survey, V5 relative alignment, V7 verification |
| `up` (**fallback only**) | pedestal at (0.30, −0.12, 0.05) looking up, own point light | 640×480, 50° | ≈ 0.2 at 0.12 m | in-hand pose, only if P2(c) fails |

- **Noise model:** depth σ = 2e-3·r² m on the ray distance (20 µm at 0.1 m, 2.4 mm at 1.1 m) plus 1 % dropout at depth edges; RGB σ = 2/255; lighting randomised per seed (intensity 0.5–2.0, direction).
- **Hidden per-seed extrinsic error:** static cameras 1 mm / 0.2°; hand-eye 0.2 mm / 0.1°. **Distribution:** drawn once per seed as independent zero-mean Gaussians per axis (translation x, y, z; rotation about x, y, z), with these values as **1σ per axis**.
- Captures are scheduled so no arm is in the frustum of interest, apart from A during braced V5 looks, which is masked.
- **Renders are ray-traced, not photoreal; no sim-to-real claim.**

**2.5.3 Pipeline** (`vision.py`; numpy, cv2, scipy; testable offline on saved captures). Depth → 3-D points via the per-pixel rays (`compute_pinhole_camera_rays`); depth 0 is invalid.
- **V1 table plane:** RANSAC on `top` points.
- **V2 tray instances (top, coarse):**
  1. mask of points > 4.8s mm above the plane;
  2. connected components (tray bricks are ≥ 25 mm apart by construction, so no colour split is needed);
  3. `minAreaRect` → type = round(L/8s) × round(W/8s) matched to the library, yaw up to symmetry, centre.
- **V3 LookFeeder (wrist, mandatory before every grasp):** B hovers 100 mm above V2's estimate. Top-face segmentation in depth, stud blobs (height > 0.5 stud height above the face), then a lattice fit → fine xy and yaw of the brick to be grasped.
- **V4 plate:**
  - *coarse* — `top` segmentation by height and known colour → `minAreaRect`;
  - *fine* — a **wrist survey at build start**: B looks at two diagonal plate corners from 150 mm; the outline corner plus the stud lattice give T_world_plate. The outline fixes the lattice index. The blueprint grid maps to world through this pose.
- **V5 relative alignment (wrist, one image per step):** B carries the brick to a **look pose**, with the brick's bottom at **P2's chosen look height h_look ∈ [25s, 60s] mm** above the target's stud tops. In that single image:
  - **(a) target:** stud blobs, from depth, of the course the target sits on, within 3 pitches of the footprint. A's links are masked on braced steps: A's padded spheres are projected through A's FK and the calibrated hand-eye. Least-squares lattice fit gives the phase and yaw; the integer index comes from V4-fine (error ≪ half a pitch). If fewer than 4 studs are visible, the fit uses the supports' top-face edges (depth discontinuities).
  - **(b) held brick:** its camera-side face (depth plane fit → offset along the grip and tilt about the finger axis) plus its end studs or bottom corners (→ lateral offset and yaw).
  - **(c) output:** the TCP correction (xy, yaw) in the hand frame, and tilt. Tilt > 3° → regrasp.
  - **Missing expected studs** (unmasked only) → obstruction → abort the step.
  - **Too occluded** → a second look with the hand yawed 180° (a brick footprint is 180°-symmetric).
- **V7 verification (wrist, from the look pose after retract):**
  - top-face height over the footprint:
    - seated ⇒ z_target + 9.6s mm, where the seated band is the clutch band, dz ∈ [−0.5s, +0.3s] mm (r4);
    - above +0.3s ⇒ not seated (on the stud tops ≈ 1.7s mm more, or hung between);
    - more than 0.5s mm below seated ⇒ `below_seated` (pass-through, or a crushed lower course), and the step fails as a penetration outcome.
    - σ ≈ 20 µm. Stud-top vs seated is unambiguous. At 1× a brick that ends just above +0.3s (r3 smoke: +0.33 and +0.35 mm, 1.5–2.5σ above the edge) is not; P1's band-sanity gate bounds how often that happens.
  - the new brick's lattice → its pose error.
  - The as-built model is updated from this measurement.
- **Final as-built (top):** coarse check against the blueprint.
- **Learned fallback (U7) for V2 only:** torchvision `maskrcnn_resnet50_fpn_v2` fine-tuned on about 5k synthetic renders labelled from `shape_index`.

**2.5.4 Error budget** (drives D1 and the WP3 gates; at s = 1. Pixel terms are predictions that P2 replaces with measurements.)

**Error model.** Every term is an independent zero-mean Gaussian per axis, and the tabled values are **1σ per axis**. Calibration terms follow §2.5.2's per-seed draw.
- A chain's per-axis σ is the **RSS** of its terms.
- **Radial xy p95 = 2.45σ** (Rayleigh). A single-axis or yaw p95 is 1.96σ.
- A rotation error δθ at range L moves a point by δθ·L; 0.1° = 1.75e-3 rad.

*Insertion alignment chain (V5 → Align → Insert):*

| term | 1σ | source | note |
|---|---|---|---|
| target lattice fit (wrist depth, ≥ 4 studs, ≈ 22s px stud diameter) | 0.03 mm | P2 measures | depth σ ≈ 20 µm at 0.1 m |
| held-brick pose in the same image | 0.05 mm | P2 measures | face plane plus end studs/corners |
| wrist hand-eye translation (0.2 mm) | **0** | analytic | brick and target are measured in one camera frame |
| wrist hand-eye rotation (0.1°) × look→insert offset d_look (≈ h_look, 25s–60s mm) | 0.04–0.10 mm | analytic | 1.75e-3 × d_look; P2 picks h_look |
| FK servo residual at the insert pose | 0.03 mm | settle criterion | FK is exact in simulation |
| slip during final descent and press | 0.05 mm | P1(e) measures | gate ≤ 0.2·c p95, at the §2.3 grip |
| structure drift after the look | 0 | design (**OPEN**, r5 B1) | A is static before the look; M1 welds are rigid. r5 leaves this value unchanged; before D1 drift is remeasured capture-to-insert (V5 capture → insertion) under the final executor and carried as a separate conservative allowance (worst case, added linearly), unless its statistical model is validated. J-d's M4 is structural information only |
| **total (RSS per axis)** | **≈ 0.09–0.13 mm** | | **radial p95 ≈ 0.23–0.33 mm** (h_look 25s–60s mm); relative yaw p95 ≈ 0.3° |
| *draft chain, for comparison* | | | `top` plate ≈ 4 mm per axis (≈ 10 mm radial p95); wrist+up hand-eye floor ≈ 0.8 mm p95 |

*Other chains* (RSS per axis, then radial p95):

| chain | per-axis 1σ | predicted p95 | requirement → gate |
|---|---|---|---|
| **`top` coarse** (tray, plate; range 1.10 m) | tilt 0.2° × 1.1 m = 3.8 mm; translation 1 mm; yaw 0.2° × ≤ 0.4 m off-axis ≤ 1.4 mm; pixel < 0.5 mm → **≈ 4.2 mm** | **≈ 10 mm** radial; yaw ≈ 0.4–0.6° | V2: the brick must fall inside the LookFeeder window (±26 mm at the 100 mm hover, reviewer's geometry; P2 confirms) → **≤ 15 mm**. V4-coarse: the corner must fall inside the survey window (±78 mm at 150 mm) → **≤ 20 mm, yaw ≤ 2°**. If V1 re-estimates `top`'s tilt from the known table plane, ≈ 1.8 mm per axis → ≈ 4.4 mm p95 (the remedy if a coarse gate fails). |
| **Grasp** (LookFeeder → fingers; camera ≈ 160 mm from the brick) | hand-eye 0.2 mm; 0.1° × 160 mm = 0.28 mm; pixel ≈ 0.05 mm → **≈ 0.35 mm** | **≈ 0.86 mm** | half the per-side open clearance, 0.6 mm + 2s mm → **≤ 1.3 mm at 1×, ≤ 2.3 mm at 2×** |
| **Plate index** (V4-fine; two corners from 150 mm) | per corner: 0.2 mm; 0.1° × 150 mm = 0.26 mm → 0.33 mm; plate-yaw lever to a footprint ≤ 70 mm from the centre ≈ 0.25 mm; pixel ≈ 0.05 mm → **≈ 0.33–0.45 mm** | **≈ 0.8–1.1 mm** | index correct ⇔ error < 0.5·pitch; gate at 0.25·pitch → **≤ 2s mm, and 100 % correct index**. `top` alone (≈ 10 mm) cannot do this, hence the survey. |

**D1 rule.** For each s ∈ {1, 2}, and each contact variant P1 measured, s qualifies if all four hold:
- **(i) alignment:** predicted chain p95(s) ≤ 0.5·c_held(s). The prediction uses this table with P2's measured pixel terms and look height and P1's measured slip. **c_held is taken at F_press(s) = 2·F_seat(s)** (P1(b)).
- **(ii) press window:** 0.5·F_pt(s) / (k_os·F_seat(s)) ≥ 2. k_os is P1's p95 ratio of peak to commanded wrist force.
- **(iii) sink:** the sink below seated at the peak press k_os·F_press ≤ 0.5s mm (P1(d)). This keeps V7 and the penetration class meaningful. Under 71–100 N, §0 fact 6 measured 0.45–0.6 mm of sink at 2× against 0.9–2.7 mm at 1×; this criterion is where that counts for 2×.
- **(iv) grip:** the chosen per-finger grip N = m·F_press/μ (§2.3) ≤ 70 N (the FR3 hand's rated continuous force; the simulation caps at 100 N).

The variants are the prototype mesh, and P1(h) only if P1's branch fired.
- Choose the smallest qualifying s.
- If neither scale qualifies: s = 2, with the spiral search as the precision mechanism, **only if 1.5·c_held(2) ≥ chain p95(2)**. This is recorded as a deviation, and the search's success rate becomes part of the M1 gate. Otherwise escalate (U1).
- **Planning-time expectation: no lean; P1's arm-held F_seat and c_held decide.**
  - With the prototype mesh, the locked stand-in reaches the weld gate only at tens of newtons at 1×, which would put (ii)–(iv) at risk, but a free brick under a 0.5 s ramp seats at ≤ 0.5 N at 0 offset (§0 fact 6); the arm is measured, not inferred. 2× sinks about half as much under load, which counts in (iii).
  - 1× qualifies only if the arm-held c_held(1) ≥ 0.46–0.66 mm with a window. The 1 mm of the free-brick lateral-offset runs was impact-driven and does not count.

### 2.6 Executor (`cell/executor.py`, py_trees ticked once per sim frame; leaves never block)

```
Build (Sequence)
├─ PerceiveCell[top]              → coarse T_world_plate, tray inventory
├─ SurveyPlate[wrist_B]           → fine T_world_plate
├─ for each plan step: Retry(3) Sequence
│   ├─ [env] RefillTray (kitting dispenser; only here, B outside the tray sector)
│   ├─ PerceiveTray[top] → SelectBrick(type; in reach; nearest)
│   ├─ Parallel(all)
│   │   ├─ B: Plan(feeder hover) → Execute → LookFeeder[wrist] → Grasp → Lift
│   │   └─ A (if requires_brace): Plan(brace, as-built) → Zone.acquire → Execute → Brace(hold)
│   ├─ B: Plan(look pose, brick attached, A static = obstacle) → Zone.acquire → Execute
│   │     → V5[wrist] (A masked) → {tilt>3°: Regrasp | obstruction: abort | occluded: 2nd look}
│   │     → Align(FK servo) → CheckSegment(per link) → Insert(arm-held press at F_press;
│   │       spiral if not seated and enabled) → Release → Retract(look pose)
│   ├─ Verify[wrist] → update as-built; not seated → re-press once → LiftAndRetry;
│   │     below_seated → fail the step (penetration outcome)
│   └─ A: Unbrace → Park (unless the next step braces); Zone.release
└─ PerceiveCell[top] → as-built vs blueprint
```

**Brace semantics** follow the planner (`grasp_lp`): a grasp across the x faces at `brace_pose`, leaning by `brace_tilt_rotvec`, using `grasp.tcp_offset_m`. The prototype's press-down brace stays behind `--brace-model lever_press`. **The default strategy is `nearest`** (A6 verdict; U13). `--brace-all` (safety_factor = ∞) is the zone stress test.

**Episode log** `results/v4/episodes.jsonl`, one row per attempt, a superset of §2.6 `insertion_episode`:
- per-stage perception estimates and ground truth (errors computed at write time, labelled `eval_*`);
- plan time and count, path length;
- **planned and executed** minimum clearance, executed max link deviation;
- contact events by class (arm_arm / unintended / neighbour_rub / penetration / displaced), with a `braced` flag;
- commanded and peak press force, grip force, search probes, V7 outcome (seated / on stud tops / below_seated), weld windows and their evaluations, `seated_unwelded`, timing, success, `failure_stage`.

### 2.7 Live viewer (ViewerGL; `--viewer viser` works without image panels, since `log_image` is ViewerGL-only)

- **Image panels:** `top` with detections; `wrist_B` with the fitted lattice, held-brick outline and A's mask.
- **Lines:**
  - perceived brick boxes (yellow);
  - vision target (green);
  - ground-truth target (grey), behind a toggle **"EVAL: not used by robot"**, off by default;
  - planned end-effector paths (cyan B, magenta A);
  - the shared-zone cylinder.
- **HUD:** behaviour-tree node (`py_trees.display.unicode_tree`), each arm's phase, snapped/attempted, last V5 error, **contact-event counters by class** (red above 0).
- `--record out.mp4`: `get_frame()` every 2nd frame into `cv2.VideoWriter`.

### 2.8 Files

- **New:**
  - `cell/`: `__init__.py`, `scene.py` (from `dual_arm_sim.py`: arms, table, plate, scaled bricks, randomisation, tray refill, camera mounts, weld pair pool, ground-truth mirror), `clutch.py`, `contacts.py`, `cameras.py`, `executor.py`;
  - `vision.py`;
  - `motion/plan_curobo.py`, `motion/fr3.yml` (+ `motion/curobo_server.py` only for route 2);
  - `scripts/10_newton_scale_probe.py` (grown from the work-reviewer's `press.py`), `scripts/11_camera_probe.py`, `scripts/12_curobo_probe.py`;
  - `tests/test_vision.py`, `tests/test_cell.py`, `tests/test_motion.py`;
  - `experiments/v4.py` (`--workers`), `experiments/analyze_v4.py`;
  - `Docs/results_v4.md`.
- **Changed:**
  - `dual_arm_sim.py` becomes a thin CLI. Existing commands keep working. New flags: `--scale`, `--perception {vision,gt}`, `--motion {curobo,waypoint}`, `--seed`, `--record`, `--brace-model`, `--brace-all`; the `--strategy` default becomes `nearest`.
  - `planner.py` gets `SCALE`, and `build_plan(…, safety_factor=)` is passed through to `bracing.assign`. Today `build_plan` has no such argument, and `--brace-all` needs it.
  - `scripts/run.sh` routes `cell/*`, `vision.py`, `scripts/1[0-2]_*`, `experiments/v4*.py` and the new tests to the Isaac interpreter (it must match `v4*` before its `experiments/*` → mjenv rule).
  - `.gitignore` gets `.vendor/`; `env.lock` gets a `[v4]` section.

---

## 3. Work packages, in order

Every gate and result gets a ledger entry, and `WORKLOG.md` is regenerated after each. Tests restore tracked `tests/*.json` unless the change is meant to update them. **The riskiest assumptions are probed first:**
1. the press physics and the alignment budget: P1(b), (d), (e) and P2(c);
2. cuRobo in-process under threading: P3, including the soak, on day 1.

### P0 — re-establish the baseline (0.5 day)
- **Objective:** find out what the prototype does at HEAD, and measure the collision problem with the §2.4 monitor.
- **Deliverable:** `dual_arm_sim.py --viewer null --test` on cube, arch, hollow_box and S3, with current `grasp_lp` plans and with `lever_press` plans, on the **prototype's exact config** (1×, fixed feeder slots, fixed plate).
  - The config is run **3×**, to give the run-to-run spread under GPU contact nondeterminism that M0(a) needs.
  - Output `results/v4/p0_baseline.jsonl`: completion, cycle per brick, events by class, displaced bricks, and **dz at snap**. The prototype's |dz| < 1.5 mm gate can weld a brick that is still resting on the stud tops.
- Gate: none; ledger `result v4_p0_baseline` (r4: recorded from `results/v4/p0_baseline.jsonl`, 24 runs). If `grasp_lp` braces break the prototype, WP1 fixes the brace executor, and M0 uses `lever_press` until then.
- Outcome (r4): the contingency fired. On arch and S3, `grasp_lp` produced 453–507 raw displaced records per run on arch and 2345–2684 on S3 (merged, §2.4: 35–40 on arch, 146–156 on S3), and arm_arm 24–39 raw (5–9 merged). WP1 inputs, each to be fixed or explained in WP1:
  - brace-driven motion (§2.4): A's brace drags bricks while A holds and B transports, and during retract;
  - brace-independent failures, with both brace models: arch b_010 fails 6/6 runs; hollow_box b_009 and b_010 6/6 (rest ≈ +5 mm, 12–16° tilt); S3 b_010 6/6;
  - S3 `lever_press` diverged in 2/3 runs (t ≈ 145.5 s, step 14), so the fallback brace is not clean on S3 either;
  - at HEAD arch reaches 10/11 and S3 13–14/16 with either brace model, against M0(b)'s 11/11 and ≥ 15/16.
  - Snaps logged at +56/+79/−72 mm are `ok: false` (lost or dropped bricks), not welds. Of 247 successful snaps, 6 welded at +0.47 to +0.95 mm under the prototype's |dz| < 1.5 mm gate, above r4's band, so M0(b) may see fewer snaps than P0.
- P0′ (r4): the implementer first changes the finger ShapeConfig in `dual_arm_sim.build_arm` (§2.1: ke_f, kd_f; priority and solimp unchanged), then re-runs P0's config 3× (≈ 45 min, prototype weld gate) with the r4 monitor → `results/v4/p0_stiff.jsonl`, insert segments in `results/v4/p0_stiff_inserts/`, ledger `result v4_p0_baseline_stiff_fingers`. M0(a) and P3 step 8 use P0′. P0 stays the record of the prototype at HEAD.

### P1 — Newton contact physics, arm-held press and capture vs scale (2.5 days GPU; parallel with P2, P3)
- Deliverable: `scripts/10_newton_scale_probe.py`, grown from the work-reviewer's `press.py` (same brick model, solver settings and output columns), for s ∈ {1, 2} on 2x4 and 2x2 bricks. The r3-protocol smoke rows for (b), (d), (e) and (f) are void; (a), (c) and (g) rows (no fingers) remain valid (r4).
- Order (r4): (f) first, with no other GPU job running, to fix n_sub. Then (a), (d), (b), (c), (e) and (h) at n_sub; (g) any time.
- Common setup for (b)–(e) and (h):
  - Brick contacts use the example's ke, kd and margin at both scales (mass scaling is automatic, §0 fact 6). Finger contacts use §2.1's ke_f, and kd_f for n_sub.
  - Collision runs once per frame, as in the prototype and the cell.
    - Cross-check: (d) at 0 offset and 20 mm/s, and (e) at the chosen grip, are repeated with collision every substep.
    - If F_seat differs by more than one sweep level, the sink by > 0.1s mm, k_os by > 25 %, or any (e) gate item (realised grip, pad penetration, carry or final-descent slip) changes verdict, the cell collides every substep and (f) is re-read for that mode.
  - The press is arm-held.
    - B holds the brick in the FR3 grip at the prototype arm gains, with the TCP at §2.2's `GRASP_DZ`, and the grip set by the §2.3 rule.
    - The TCP descends at v_press ∈ {5, 20} mm/s with a force cap F_cap, holds 0.3 s, then opens the fingers.
  - The lower brick is either a first-course brick on the plate's proxy studs, or a brick welded to the plate (M1's state).
    - The rig puts its base at z = 0.05 m for a cell-like arm posture.
    - k_os depends on posture, so M0(b) and M1 re-read k_os from the episode log (commanded vs peak press force), and D1(ii) is re-checked with the larger value.
  - Scoring uses the M1 clutch as §2.2 defines it (r4): success means the weld fires within the 1 s post-release window, at rest relative to the lower brick (|Δxy| < 1.2s mm, dz ∈ [−0.5s, +0.3s] mm, tilt < 4°, yaw < 5°).
    - A pass in transit or under the press is logged as `gate_transit` and never scores.
    - The probe scores the weld but does not enable it, so the brick stays free for (c).
- (a) Contact generation: stud proxy vs wall at 50 µm overlap (`mjwarp_insertion_probe`'s test on this pipeline).
- (b) Capture at F_press(s) = 2·F_seat(s), rounded up to a sweep level:
  - offsets from 0 to 2s mm in 0.1s mm steps, with random direction and random yaw error U(−1°, 1°), 20 trials per level;
  - c_held(s) = the largest level L such that every level ≤ L succeeds in ≥ 19/20. This is a contiguous capture window, which is what the spiral and D1 need. The largest single level with ≥ 19/20 is also reported (`c_held_max`).
  - also logged: dz, |Δxy| and tilt at weld evaluation, and the FK-seated false-positive rate: FK seated at the end of the hold, but no weld after release.
- (c) Rest stability (r4; replaces "resting jitter"):
  - Setup: 10 of (d)'s trials at F_seat per s, observed for 5 s after release.
  - Measures, over the last 2 s: detrended jitter (peak-to-peak dz about a linear fit) and drift (the fit's slope). The raw peak-to-peak is also reported, for comparison with r3's gate.
  - Stud-top creep, from any (d) trial left unseated, is reported but not gated, because §2.2 never tests such a brick outside a window.
- (d) Force sweep at 0 offset, run before (b). F_cap ∈ {1, 2, 5, 10, 20, 50, 100} N, 20 trials each. Levels whose grip would need N > 100 N run at the cap and are flagged as grip-limited.
  - F_seat(s): the smallest F_cap whose press, from the Insert's start height, succeeds (scored as above) in ≥ 19/20.
  - F_pt(s): the smallest F_cap at which, with the brick seated first and the cap held for 1 s, the brick sinks > 1s mm below seated, or a brick–brick contact reaches d < −1s mm, in any trial.
  - Sink under load at every level.
  - k_os: the p95 ratio of peak to commanded wrist force. Pre-registered rescue (r4): if D1(ii) fails at 5 mm/s, (d) is repeated at v_press = 2 mm/s before (ii) is declared failed at that s; if it then passes, v_press = 2 mm/s at that s.
- (e) Slip and grip:
  - carrying at 0.5 m/s must slip < 0.5 mm;
  - slip during the 30 mm final descent plus press, relative to the gripper, against grip multiple m ∈ {1, 2, 4} (N = m·F_press/μ). The chosen grip is the smallest m that meets the slip gate;
  - realised per-finger grip vs nominal, and finger-into-brick penetration, at each level;
  - the grip each press needs, against the 100 N simulation cap and the real hand's 70 N.
- (f) Real-time factor, run first:
  - Configurations: 16 / 8 / 4 substeps, with collision once per frame and every substep; 2 arms and 30 bricks, with the full weld pool allocated (S3 plus 4 spares: 210; S5: 465).
  - n_sub = 16 if it reaches 0.8× real time at 465, else 8 if 8 does.
  - 4 is excluded: kd_f ≤ 240 gives an 8.3 ms finger time constant, and carry creep scales with it (≈ 0.6 mm predicted, over the gate).
  - If neither 16 nor 8 reaches 0.8× → U14.
- (g) Reproduction of §0 fact 6: free brick, step body force. This is the only place v4 applies a body force to a brick, and it is labelled as such.
- (h) Contact-spike arm, pre-registered and run in the same batch: (b) and (d) repeated with the collision mesh's tube radius reduced to a designed tube–stud clearance of 0.6s mm.
  - The mesh and the slab proxies are kept, because proxies alone cannot hold 0.5 N (§0 fact 6). The walls stay at 0.4s mm (mesh) and 0.70s mm (proxy).
  - A margin below the 0.002 mm clearance is not a candidate. It would return capture to geometry plus SDF blur (c ≪ 0.5 mm), which the search cannot recover.
  - This arm is adopted only if the branch below fires.
- Gate P1 (per s, prototype mesh, at n_sub):
  - (a) produces > 0 contacts;
  - finger contacts (r4): realised grip ≥ 0.8× nominal and pad penetration ≤ 0.3s mm at the chosen grip; carry slip < 0.5 mm;
  - D1 criteria (ii) window (after the 2 mm/s rescue, if used), (iii) sink and (iv) grip hold;
  - rest stability (r4): detrended jitter < 0.02s mm and |drift| ≤ 0.1s mm/s, so a seated brick moves ≤ 0.1s mm within the weld window;
  - final-descent slip p95 ≤ 0.2·c_held at the chosen grip;
  - FK-seated false-positive rate (r4) ≤ 5 % over (b) at the chosen grip, because the spiral trigger relies on it;
  - clutch band sanity (r4): ≤ 2 % of (b) and (d) trials rest at dz ∈ [+0.2s, +0.4s] mm in their weld window, i.e. few bricks rest near the +0.3s edge;
  - ≥ 0.8× real time with the full weld pool at n_sub (else U14).
- On failure (branch). Every item above, r4 items included, is judged per s:
  - The gate fails at s = 1 only → s = 2. This includes an r4 item that fails at 1× only (for example pad penetration near the 30 N bound at m = 2–4, or band sanity).
  - The gate fails at both scales, or c_held(s) < 2 × predicted chain p95(s) at both scales → adopt (h), if (h) passes the gate and D1(i) at some s. This is recorded as deviation `v4_brick_collision_clearance`, a stated modelling assumption; capture numbers then belong to the designed clearance.
  - If (h) also fails: s = 2 with the search, only if 1.5·c_held(2) ≥ chain p95(2). Otherwise escalate (U1: PhysX via Isaac Lab).
  - r4 items, before the branch: FK false positives > 5 % → next grip multiple, and only if m = 4 still fails is it a gate failure at that s. If a finger-contact, rest-stability or clutch-band item fails at both scales on the prototype mesh → plan-reviser revision before D1, instead of the (h) branch (which may choose (h) for a band-sanity failure), because (h) does not change those mechanisms. No ad hoc retuning. Real time → U14.

### P2 — cameras: speed, information content, look pose, and a V5 prototype (1.75 days; parallel)
- **Deliverable:** `scripts/11_camera_probe.py`:
  - render timing for `top` (1280×960) and `wrist_B` (640×480) with 2 arms, plate and 30 bricks;
  - 100 random tray scenes with noise, **saved as WP3's offline set**;
  - **look-pose sweep, run first:**
    - look height h_look ∈ {25, 35, 45, 60}·s mm × camera offset along hand x ∈ {45, 60, 75} mm;
    - for each grasp orientation (14.8 mm grasp, long axis pointing at the camera; 4-stud-axis grasp) and target type (a course with a 3-pitch neighbourhood; a single-footprint support, such as towers and S3 piers), with and without A braced;
    - reports the **count of visible, unmasked target-course studs** and whether the support's top-face outline is visible;
    - picks h_look and the mount for everything below, including the budget's rotation × offset term;
  - **50 look-pose renders per scale s ∈ {1, 2}** at the chosen look pose (random structures, 20 with A braced) through a minimal V5 prototype: depth stud blobs, lattice fit, held-brick face fit, about 100 lines, the seed of `vision.py`. It reports the §2.5.4 pixel terms per scale.
- **Gate P2 (information, not pipeline quality):**
  - **(a) speed:** `top` ≤ 50 ms; wrist ≤ 10 ms per capture.
  - **(b) top:** ≥ 99 % of tray bricks have ≥ 50 visible top-face pixels (`shape_index`), with top-face height ≥ 3σ_depth above the table.
  - **(c) wrist:** at the chosen look pose, in ≥ 95 % of renders at the candidate s:
    - V5(a) is constrained: **≥ 4 unoccluded target-course studs** after A's mask, or, for single-footprint targets, ≥ 2 studs plus the support's top-face outline on ≥ 2 sides (V5(a)'s edge fallback);
    - stud-bump/σ_depth ≥ 10;
    - the held brick's camera-side face ≥ 500 px.
    - Single-footprint targets are reported separately.
  - **(d)** the V5 prototype's pixel terms are reported; they feed D1 and are not gated.
- **On failure:** (a) → smaller images or fewer captures. (c) → change mount or tilt, then the `up` fallback (its hand-eye term re-enters the budget, which likely forces s = 2). (b)/(c) uninformative in both depth and RGB → escalate U1 before WP3.

### P3 — cuRobo in-process, FR3 model, executed tracking, threaded soak (1.75 days; parallel)
- **Deliverable:** `scripts/12_curobo_probe.py` under the overlay:
  1. import;
  2. **generate `motion/fr3.yml`** with `sphere_fit` on the FR3 link meshes. Every mesh vertex must lie within 2 mm of the sphere union, and **within 0.5 mm on the finger links**, which the contact-segment check uses with 0–1 mm padding;
  3. warm up: **capture every cuRobo graph** (§2.4 discipline) with the collision cache pre-sized;
  4. 100 plan queries for B, with A's spheres in random static configurations plus a plate and 10 cuboids: latency and success;
  5. **execute 20 plans in Newton at time scales {1.0, 0.5, 0.3}** with velocity feed-forward: max executed-vs-planned link deviation (p99), executed minimum clearance, and events from the monitor;
  6. GPU memory per process (two planners) → the worker count for WP6;
  7. **threaded soak:** ≥ 200 plans from the worker thread while the cell steps at 60 fps, renders `top` and `wrist_B` and runs ViewerGL. It counts CUDA errors and post-warm-up graph captures, and reports frame-time p99;
  8. **contact-segment false positives:** the §2.3 per-link check runs on every insert segment of P0′'s cube runs (`results/v4/p0_stiff_inserts/`). A false positive is a segment that is flagged although the monitor saw no event.
- **Gate P3:**
  - runs alongside Newton and warp 1.13 in one process;
  - median plan ≤ 100 ms warm; success ≥ 95 % on feasible queries;
  - at the chosen time scale, deviation p99 + 2 mm ≤ padding;
  - **0 arm–arm events** (single-frame) on the 20 executed plans;
  - **the soak runs 0 CUDA errors and 0 post-warm-up captures**;
  - **0 contact-segment false positives** on the cube's inserts.
- **On failure:**
  - Soak failure → `use_cuda_graph=False` in the thread (the ≤ 100 ms gate re-checked), then synchronous plan calls (≈ 0.1 s viewer pause per plan), then route 2.
  - False positives → finger padding 0 mm, or a tighter finger fit.
  - Anything else → route 2 (socket; gate ≤ 150 ms including IPC), then route 3.
- **Ledger:** `v4_motion_route`; `curobo_isaac_split_reversed` if the route changes (user-authorised).

**Decision point D1** (after P1–P3; the orchestrator presents it to the user):
- scale by §2.5.4's rule (i)–(iv), from P1's c_held(s) at the named F_press(s) = 2·F_seat(s), window ratio, sink and grip, and P2's pixel terms at both scales;
- the contact variant (prototype mesh, or P1(h) if the branch fired), v_press, F_press and grip;
- the camera set (baseline, or with the `up` fallback) and P2's look height and mount;
- motion route and time scale.
- Ledger: `v4_scale`, `v4_cameras`, `v4_motion_route`.

### WP1 — cell refactor → **M0** (1.75 days; starts after P0, before D1, because scale is a parameter)
- **Deliverable:**
  - `cell/scene.py`: randomisation; the reach-sector tray (50°–85°), refilled only at step start with B outside the sector; weld pair pool; ground-truth mirror; parked spares. **Picks are verified across the whole sector.** The sector is extended to 100° only if picks there verify.
  - `cell/clutch.py`: lattice-snap gate, scaled;
  - `cell/contacts.py`: the §2.4 definition, with the `gripped_bricks` and descend-segment windows and its own Contacts buffer;
  - `cell/cameras.py`: `top` and `wrist_B` with ring light, live panels;
  - `dual_arm_sim.py` as the CLI; brace executor aligned to `grasp_lp`; `--strategy nearest` default; `--brace-all`, which needs **`build_plan(…, safety_factor=)` plumbed through to `bracing.assign`**;
  - the Insert with P1's arm-held press and §2.3 grip rule at 1×;
  - `tests/test_cell.py`: pair-pool snap with an out-of-plan brick; contact classifier on scripted collisions (arm–arm, finger graze, one-frame swipe, neighbour rub, penetration, a brace gripping two bricks); firewall skeleton.
- **Gate M0:**
  - **(a)** on **P0's exact config** (1×, fixed layout, waypoint IK) with r4's finger contacts, the new monitor's per-class event counts (merged as §2.4 defines) fall **within P0′'s run-to-run spread over its 3 runs, or ±1 event per class, whichever is larger**;
  - **(b)** ground-truth-driven, waypoint IK, random plate and refilled tray, 3 seeds each: cube 8/8, arch 11/11, S3 ≥ 15/16 per seed. This uses the M1 clutch gate and P1's press at 1×; if P1 is late, the prototype's gate is used, and the run says so;
  - **(c)** camera panels live;
  - shown live to the user.
- **On failure:** fix within WP1. A regression below P0 blocks WP2.

### WP2 — collision-free motion → **M0.5** (3 days; depends on WP1, P3; parallel with WP3)
- **Deliverable:**
  - `motion/plan_curobo.py`: world model from the cell's ground truth in WP2, switched to perception in WP4; time scaling and velocity feed-forward; planning thread; zone reservation; the pre-loop CUDA-graph warm-up and post-warm-up capture guard (§2.4); the per-link contact-segment checks (§2.3); executed-clearance logging;
  - `tests/test_motion.py`: 200 random queries per arm with the other static, plus an executed subset.
- **Gate M0.5** (ground-truth-driven; cube, arch, hollow_box, S3 × 5 seeds = 235 placements, 20 braced; **plus the zone stress test**: S3 `--brace-all` × 3 seeds, about 40 braced placements):
  - **0 arm–arm events**, reported separately for braced and unbraced placements;
  - ≤ 1 unintended event per 50 placements (`neighbour_rub` reported, not counted);
  - 0 displaced bricks; 0 penetration events; executed minimum clearance > 0 on every free-space segment;
  - completion ≥ M0; mean cycle per brick ≤ 1.5× P0's;
  - **shown live to the user (the "arms no longer collide" demo).**
- **On failure:** classify each event by pair and phase. World model wrong → fix it. Deviation → slower time scale or more padding. Contact segment → shorter segment, or plan it with cuRobo. Throughput > 1.5× → the R5 relaxation (only with the user's OK).

### WP3 — perception (4 days; depends on P2, D1; parallel with WP2)
- **Deliverable:** `vision.py` (V1–V5, V7); `tests/test_vision.py` on held-out renders.
- **Datasets** (through `cell/cameras.py` with noise and hidden extrinsic error):
  - 500 tray scenes;
  - 200 LookFeeder hovers;
  - 100 plate surveys;
  - **300 look-pose V5 scenes at the chosen s (≥ 100 with A braced) plus 200 at the other s ∈ {1, 2}** (for E1's scale claim);
  - 200 verification cases.
- **Gate WP3** (p95 at the chosen s; c = c_held(s) from P1; each criterion derived in §2.5.4):

| stage | criterion | derived from |
|---|---|---|
| V2 tray detection (top) | recall, precision, type accuracy ≥ 99 % | per-step selection |
| V2 tray pose (top, coarse) | radial xy ≤ 15 mm, yaw ≤ 5° (predicted ≈ 10 mm) | brick inside the LookFeeder window (±26 mm at the 100 mm hover) |
| V3 LookFeeder | radial xy ≤ 1.3 mm at 1× (2.3 mm at 2×), yaw ≤ 2° (predicted ≈ 0.86 mm) | ≤ half the per-side open clearance (0.6 + 2s mm) |
| V4 coarse / fine | coarse radial xy ≤ 20 mm, yaw ≤ 2° (predicted ≈ 10 mm / 0.5°); fine ≤ 2s mm at every footprint with **100 % correct lattice index** (predicted ≈ 0.8–1.1 mm) | survey window ±78 mm at 150 mm; index margin 0.25·pitch |
| V5 relative alignment | radial xy ≤ 0.35·c, relative yaw ≤ 1°; tilt > 3° flagged with recall ≥ 95 %; obstruction recall ≥ 95 % with ≤ 2 % false aborts (braced scenes reported separately) | chain budget |
| chained, end to end (V5 + FK servo + P1's slip model) | radial xy ≤ 0.5·c | D1 rule |
| V7 verification | accuracy ≥ 98 % over seated / on stud tops / below_seated (labels from ground-truth dz and tilt at rest against the §2.2 band, not from clutch state; `seated_unwelded` cases are reported separately) | as-built integrity |

- **On failure:**
  - V2 or V4-coarse pose → V1 re-estimates `top`'s tilt from the table plane. V2 detection → learned detector (U7).
  - V5 or chained → wrist at 1280×960; a mandatory second look with the hand yawed 180°, with the two estimates averaged; a lower look pose **only if P2's sweep showed it keeps the stud count** at that height.
  - Still failing, and D1 chose s = 1 → **one** D1 revisit to s = 2 (re-render ≈ 1 day; deviation recorded).
  - At s = 2 → the spiral search carries precision (Q1 outcome C), if 1.5·c_held(2) ≥ the measured chain p95, and M1 gates the search's success. Otherwise escalate.

### WP4 — integration → **M1** (2–3 days; depends on WP2, WP3)
- **Deliverable:** `cell/executor.py` (the §2.6 tree) with the world model from perception; `test_firewall`; viewer overlays; `--record` videos of cube, arch and S3.
- **Gate M1** (randomised plate, tray, lighting and extrinsics):
  - `test_firewall` passes;
  - cube 5/5 seeds (40/40 bricks); arch ≥ 4/5; **S3 ≥ 4/5, including its 3 braced steps**;
  - **0 arm–arm events** (braced placements reported separately); ≤ 1 unintended per 50; 0 displaced; 0 penetration;
  - the user watches one live run.
- **On failure:** `failure_stage` and per-stage perception-vs-ground-truth errors say where. Alignment → WP3 remedies. Contact → WP2. Stuck on the stud tops → F_press and v_press (against P1's window and grip), then the search parameters. `below_seated` → sink under load against P1(d).

### WP5 — breakable joint → **M2** (r5: U4 = yes, split)
- **WP5a (r5, now, before WP1):** the stiff, breakable, grouped weld pool and its break model (S1–S2, gates J-a/J-b/J-c of the r5 section), accepted in the build stage J-d/M0-J. No contact spike needed.
- **WP5b (after M1; only if the spike passes; 1-day spike + 3 days), the text below:**
- **Spike first (1 day):** can brick–brick contacts carry 1.3·n·f_insert (≈ 92 N for a 2x4), applied with the arm's Jᵀ·F press profile (speed-limited, ramped), without pass-through, and with the sink ≤ 0.5s mm at the chosen s and contact variant?
  - **Expected yes at 2×, uncertain at 1×.** §0 fact 6: bricks seated first (then stepped) hold 100 N at 1× with 1.2–2.7 mm of sink, and at 2× with 0.45–0.6 mm; a speed-limited stand-in also carried 100 N without passing through. But at 1× an unseated brick under a 0.5 s force ramp passes through at 71 N (ends −9.2 mm; −11.6 mm at 100 N), while at 2× the same ramp holds 100 N (−0.50 mm). The 1× sink may also fail the 0.5s mm bound, which is one more point for 2×.
  - The spike also checks that the detent weld releases without the brick dropping > 1s mm.
  - Tuning only if it fails: 32 or 64 substeps; ke × 10 with kd × √10.
  - **If no:** report "v3.1's force regime is not representable in the Newton cell". WP5 stops, or the user accepts per-stud forces scaled to what contacts carry (a stated assumption, U3). E5 is not run.
- **If yes:** the draft's deliverable and gate — detent and break welds, the Jᵀ·F press and brace feed-forward via `control.joint_f`, the G1-lite checks, and "bracing can change an outcome" on the scaled S3 critical step, 5 seeds.

### WP6 — experiments (§4) (2 days compute + 1 day analysis)

### WP7 — report and records (1.5 days)
`Docs/results_v4.md`. `Docs/master_report.md` gets §0.7 "v4 execution note" and §7.8 "v4 amendments", cross-referenced to ledger entries. README and VIEWING updates. Optional Isaac Sim 6.0 RTX hero render from a `--viewer usd` recording.

---

## 4. End-of-phase experiments, and what the report must show

All headless via `experiments/v4.py --workers N` (N from P3's memory measurement, expected 3–4); every table comes from `experiments/analyze_v4.py`.

| exp | question | design | n and why | metrics / test | feeds |
|---|---|---|---|---|---|
| **E1** perception accuracy | how good is each stage, and V5 at both scales? | WP3 datasets, noise on/off; V5 at s = 1 and 2 | 100–500 per stage (p95 ±2–3 %) | median, p95, max per stage; detection P/R; verification accuracy; measured vs §2.5.4 predicted terms | Table 1, Fig 1 (V5 error vs c(s), both scales) |
| **E2** error → success (Q1; **confirms D1**) | how does success depend on alignment error at each scale? | single-brick insertions with the arm-held press at F_press(s), ground truth otherwise; injected Gaussian xy error σ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2} mm; **s = 1 and s = 2**; search on/off; matched seeds | 40 per cell (Wilson ±13 % at 90 %); **the primary estimate is a logistic fit per scale × search** (bootstrap CI) | success per cell; fitted curve; where E1's V5 p95 falls on each curve | Fig 2 ("why this scale"), Table 2 |
| **E3** collision-free motion (Q2) | does planned, prioritised motion remove contacts? | prototype waypoint IK vs v4 planned, both ground-truth-driven; cube, arch, hollow_box, S3 (+ S5 if s = 2) + zone stress; same seeds | 10 seeds per shape per arm (paired) | events per placement by class (Wilcoxon, paired); runs with ≥ 1 event (McNemar); braced vs unbraced; executed clearance; cycle time | Table 3, a video still |
| **E4** end-to-end, vision only | does the full system build? | cube, arch, hollow_box, S1, S3 (+ S5 if s = 2), randomised; vision vs a ground-truth-perception control on the same seeds | 20 seeds per shape | completion (Wilson CI), per-brick and first-attempt success, search use, cycle time, final RMS pose error, failure stage; paired McNemar vision vs GT | Table 4, Fig 3 (failure stages) |
| **E5** bracing in Newton (Q3) | only if WP5's spike and M2 pass | v3.1 M11 protocol on scaled S3; none / nearest / weakest_joint | 20 seeds per cell | success without a break, placer peak force; McNemar, Wilcoxon | Table 5, "cross-simulator check" |
| **E6** timing | is it usable live? | from E4 logs | all | startup, × real time, per-stage perception latency, plan latency | Table 6 |

**The report** contains:
- gates P1–P3, M0, M0.5, WP3, M1 (and M2 if run), with verdicts;
- the §2.5.4 budget, predicted vs measured;
- the architecture and the vision-only definition (U6);
- Tables 1–6 and Figures 1–3;
- videos of cube, arch and S3;
- limitations, including:
  - §0 fact 6: mass-scaled soft contacts, the stud-top rest, sink under load, and capture that depends on the press profile — and the designed clearance, if P1(h) was adopted; the r4 finger contacts (§2.1), still mass-scaled (softer on a 1× brick);
  - the tray is a kitting dispenser stocked from the plan (the environment's, refilled only at step start); every brick is still found by vision;
- reproduction commands.

**Allowed claims:**
- simulation only; "vision only" in the U6 sense;
- "collision-free" only as measured by the §2.4 monitor;
- the scale justified by E1 and E2 at both scales;
- classical CV on ray-traced renders with synthetic noise — **no sim-to-real claim, and no generalisation beyond the randomisation ranges**;
- capture and press-force numbers belong to Newton's soft brick contacts (and to the designed clearance, if adopted), the r4 finger contacts, the §2.2 clutch band evaluated at rest after release and the stated press profile, not to real LEGO or DUPLO;
- no RL claims; E5, if run, only says whether v3.1's direction replicates.

---

## 5. Order, dependencies and effort

```
(r5: P0 → S0..S4a → J-pre → S3/S4 → J-e → J-d (M0-J) → PL run before WP1; see the r5 section)
P0 ─── WP1 (M0) ──────┬─ WP2 (M0.5) ─┐
P1 ─┐                 │              ├─ WP4 (M1) ─ WP6 ─ WP7
P2 ─┼─ D1 (user) ─────┴─ WP3 ────────┘      │
P3 ─┘  (WP2 also needs P3)                  └─ [WP5b, decide after M1] WP5 spike → WP5b → E5
```

| WP | effort (agent-days) | compute | runs in parallel with |
|---|---|---|---|
| P0 | 0.5 | ~45 min GPU (3 repeats) + ~45 min for P0′ (r4, after the finger change) | P1–P3 |
| P1 | 2.5 | ~3.5 h GPU ((f) first on an idle GPU; release windows; collision-mode cross-check) | P0, P2, P3 |
| P2 | 1.75 | ~1 h (look-pose sweep added) | P0, P1, P3 |
| P3 | 1.75 | ~1.5 h (soak, false-positive check added) | P0–P2 |
| WP1 (M0) | 1.75 | small | P1–P3 (after P0) |
| WP2 (M0.5) | 3 | ~1.5 h | WP3 |
| WP3 | 4 | ~2.5 h renders | WP2 |
| WP4 (M1) | 2–3 | ~1 h | — |
| J stages (r5: S0–S4a, J-pre, J-e, J-d, PL; WP5a) | ≈ 4.4 (+ ≤ 1 if the WP1 seating pull fires); J-d 24 runs ≈ 5–7 h GPU serial | see r5 section | before WP1; WP1's `cell/clutch.py` absorbs `JointBreaker`, grouping, protocol (−0.25 day) |
| WP5b (M2), if decided yes after M1 | 1 + 3 | ~1 h | after M1 |
| WP6 | 3 | E4 ≈ 220 builds × 3–5 min ≈ 11–18 h serial, 3–5 h at 4 workers; E2 ≈ 1100 insertions ≈ 1 h at 4 workers; E3 ≈ 2 h | — |
| WP7 | 1.5 | — | — |

**Calendar:**
- day 1–2.5: P0–P3, with P1(f) first on an idle GPU, and P0′ (≈ 45 min) as soon as the finger change lands in `build_arm`;
- day ≈ 3–4: M0 (M0(a) needs P0′; see the WP1 caveat);
- D1 answered by the user by day 3.5 (each day of waiting slips WP3 and M1);
- day ≈ 6–7: M0.5, live;
- WP3 days 3.5–7.5;
- M1 ≈ day 10.5 optimistic, 12.5–15.5 realistic.
- **r5 order (2026-09-30):** P0 → S0 → S1/J-a/J-b → S2/J-c → S4a → J-pre → [WP1 seating pull] → S3/S4/J-e → J-d (M0-J) → PL → WP1 (M0) → WP2 (M0.5) ∥ WP3 → WP4 (M1). **M0, M0.5 and M1 slip ≈ 4 agent-days** (+ ≤ 1 day if the WP1 seating pull fires); the calendar dates above are r4's and are not re-stated.
- WP1 caveat (r4): WP1's 1.75 days assumed the prototype met M0(b) apart from braces. P0 shows brace-independent failures (arch b_010, hollow_box b_009/b_010, S3 b_010) and an S3 `lever_press` divergence, so M0 may slip. The orchestrator reports the cause of these failures at WP1's midpoint; if they do not share one cause, the user is told the new WP1 estimate.
- whole phase: **the table sums to 21.75 agent-days (WP4 at 2) to 22.75 (WP4 at 3) without WP5**, and 25.75–26.75 with WP5 (1 + 3), before the WP1 caveat. r1's "≈ 22" did not match its own table (20–21).

One 32 GB GPU is shared; no headless batches during a live demo.

---

## 6. Risks and user decisions

### 6.1 Risks

| # | risk | likelihood / impact | mitigation |
|---|---|---|---|
| R1 | the arm-held capture c_held(1) is much smaller than the 1 mm of the free-brick lateral-offset runs, which was impact-driven (§0 fact 6) | **high** / high | P1(b) with the arm's press profile on day 1–2; the designed-clearance arm P1(h); s = 2; the search only where 1.5·c_held ≥ chain p95 |
| R2 | the arm cannot seat a brick with a press the grip can carry. With the prototype mesh, the servo stand-in (no sideways, tilt or yaw freedom) stayed on the stud tops up to ≈ 50 N at 1×, while a free brick under a 0.5 s ramp seats at ≤ 0.5 N at 0 offset; an arm-held brick, with some compliance through the grip, may fall anywhere between the two. At 71–100 N the sink under load is 0.9–2.7 mm at 1×. The stand-in did not pass through. | **medium** (the two bounds disagree; P1 measures the arm) / high | every press arm-held and speed-limited; P1(d), (e) with the scaled grip; D1 (ii)–(iv); P1(h); s = 2 (half the sink); the monitor's penetration class; V7 `below_seated` |
| R3 | the wrist cannot see the held brick and the target neighbourhood in one image: the held brick hides the camera side, and single-footprint targets show about 2 studs | **high** / high | P2's look-height and mount sweep, reported by grasp orientation and target type; the V5(a) edge fallback; a second look yawed 180°; `up` fallback (+ likely s = 2) |
| R4 | renders too clean → classical CV over-fits | medium / medium | noise, lighting and extrinsic randomisation; depth-first detection; stated limitation; U1 escalation |
| R5 | cuRobo 0.8 fails with warp 1.13 / cuda-core cu12, the FR3 sphere fit is poor, or cuRobo's lazy CUDA-graph capture collides with the main thread | medium / medium | P3 on day 1, including the threaded soak; pre-loop graph warm-up with a pre-sized cache; non-graph or synchronous plan calls; socket route; Newton-native fallback |
| R6 | the overlay shadows Isaac packages | low / high | `--no-deps`, appended to `sys.path`, exact list in `env.lock` |
| R7 | zone reservation costs > 1.5× throughput | medium / low | relaxation (a genuine one — the base rule already lets A move while B is outside): shrink the zone margin from 0.12 to 0.06 m and let B pre-stage at the zone boundary with its brick while A braces; only with the user's OK |
| R8 | executed trajectories deviate beyond the padding | medium / medium | time scaling, velocity feed-forward, padding from P3's measured p99, executed clearance logged |
| R9 | ground truth leaks into the executor | medium / high | `Robot` facade, ground-truth mirror, firewall test in the M1 gate |
| R10 | `grasp_lp` braces don't transfer to the Newton cell | high / medium | P0 measures; WP1 aligns; `lever_press` fallback |
| R11 | s = 2 changes plans (lever arms, brace triggers) | certain if s = 2 / low | v3.1 frozen; cross-simulator labels only |
| R12 | GPU memory limits parallel workers | medium / low | measured in P3; 3–4 workers; E4 runs overnight |
| R13 | "Isaac" meant Isaac Sim RTX visuals | medium / medium | U1 asked up front; WP7 hero render |
| R14 | the r4 finger contacts transmit touchdown impacts (k_os 2.5–4.4 at 20 mm/s, up to 3.8 at 1×), which squeezes D1(ii), and make finger grazes push neighbours harder; at 1× the pad bound caps the grip near 30 N | medium / medium | v_press 5 → 2 mm/s (pre-registered rescue); grip multiple from P1(e); the monitor's displaced class; s = 2 (8× stiffer pads) |

### 6.2 Decisions the user must make (recommended default in bold)

- **U0 Project rule:** amend `CLAUDE.md` "Physics stays on CPU MuJoCo" to *"v4 runs in standalone Newton (Isaac interpreter, Newton's own collision pipeline); the MuJoCo twin is frozen for v3.1."* **Default: approve.**
- **U1 Simulator:** **standalone Newton 1.2.1 in the Isaac 6.0 interpreter, ViewerGL live, optional Isaac Sim RTX hero render.** Alternatives: Isaac Lab 6.1 on Newton (RTX TiledCamera); Isaac Sim RTX as the live viewer.
- **U2 Brick scale:** **chosen at D1 from {1×, 2×} by the §2.5.4 rule (i)–(iv); the planning expectation is "no lean: P1's arm-held F_seat and c_held decide".**
  - With the prototype mesh, the locked servo stand-in reaches the weld gate only at tens of newtons at 1×, but a free brick under a 0.5 s ramp seats at ≤ 0.5 N at 0 offset. The arm is measured in P1.
  - 2× (the DUPLO grid) sinks about half as much under load.
  - The 1 mm capture of the free-brick lateral-offset runs, which suggested 1×, was impact-driven.
  - **Trade-off:** P1's branch adopts the designed clearance P1(h) only if the prototype mesh fails at both scales, so the plan prefers the real brick mesh at 2× over a designed clearance at 1×. The user may flip this at D1.
  - 1× (LEGO System) is chosen if P1's arm-held c_held(1) ≥ 2 × chain p95 (≈ 0.46–0.66 mm) with a press window, sink and grip in bounds.
  - More than 2× is infeasible (FR3 stroke vs the cube's 4-stud grips). The user may fix 1× or 2× regardless.
- **U3 Joint forces at s ≠ 1 (only matters if U4 = yes):** **keep 8.9 / 11.3 N per stud; gate distances scale with s, angles do not.** Alternatives: ∝ s or ∝ s². Or, if WP5's spike fails, per-stud forces scaled down to what Newton contacts carry, as a stated assumption.
- **U4 Joint breaking in v4:** **decided yes, 2026-09-30 (r5): rigid, breakable joints now as WP5a (before WP1); WP5b and E5 (detent, v3.1 forces, contact spike) still "decide after M1".** Original text: decide after M1 — mainly a scope question; feasibility is open only at 1×.
  - The work-reviewer's reproduction shows bricks carry 71–100 N when seated first (then stepped), or pressed by a speed-limited stand-in (seated first: sink 0.9–2.7 mm at 1×, 0.45–0.6 mm at 2×). At 1× an unseated brick under a 0.5 s force ramp passes through at 71 N (ends −9.2 mm; −11.6 mm at 100 N); at 2× the same ramp holds 100 N (−0.50 mm). r1's "2 N pass-through" was a tunnelling artefact of a free brick under a step body force.
  - WP5's 1-day spike is expected to pass at 2× and is uncertain at 1×.
  - Alternatives now: yes (WP5 = spike + 3 days after M1), or no (a perception-and-motion phase only).
- **U5 Motion planner route:** **cuRobo 0.8 in-process from the `.vendor` overlay (Isaac env untouched), replanning between segments, which reverses `curobo_isaac_split`.** Plans run in a worker thread after a pre-loop CUDA-graph warm-up. Fallbacks, automatic without asking: within route 1, non-graph or synchronous plan calls if P3's soak fails; then a socket to isdnenv; then Newton-native. Any install *into* `~/Codes/CAIRSS/Issac` itself is not requested and would need a separate OK.
- **U6 What "vision only" means:**
  - **Brick, plate and placement poses come from cameras only.**
  - **Allowed:** proprioception of both arms (joint angles/FK, including projecting A's links to mask them in B's images; gripper width; simulated wrist wrench); calibrated intrinsics and extrinsics with hidden calibration error; brick CAD; the blueprint (images → plan) as the task input.
  - **Cameras:** one static top RGB-D camera and one wrist RGB-D camera on B with a co-located ring light; an upward station camera only as a fallback.
  - **The tray is the environment's kitting dispenser, stocked from the plan** (the next ≤ 4 needed bricks plus distractors). It refills only at step start, before the robot perceives the tray. This decides which bricks are present, not where they are, so the robot still finds every brick by vision.
  - Alternatives: RGB only (no depth); a wrist camera on A too; keep the oblique static cameras.
- **U7 Detector:** **classical CV first; torchvision Mask R-CNN only if WP3's V2 gate fails** (COCO weights download; about 1–2 GPU-hours).
- **U8 Treatment of v3.1:** **frozen as the record; v4 gets its own report; E5 (if run) is a labelled cross-simulator check.**
- **U9 Evaluation shapes:** **cube, arch, hollow_box (image blueprints) + S1, S3; S5 added if s = 2** (at 1× its flanked 2x2 grasps are inaccessible, G2 xfail; tray refill removes the capacity limit). The S3 zone stress test (`--brace-all`) is in M0.5 and E3.
- **U10 Randomisation:** **plate pose ±30 mm / ±15°; tray scatter in the 50°–85° sector, refilled by the kitting dispenser at step start only; lighting; extrinsic error as per-axis Gaussians (static 1 mm / 0.2°, hand-eye 0.2 mm / 0.1°, 1σ) — all on for M1 and E4.**
- **U11 Git:** **commit the uncommitted MuJoCo `--view` hook (`twin_executor.py`, `sim/mj/runtime.py`, view-only) on `local-gpu-twin`, then branch `v4-newton-vision` from it.** BrickSim submodule edits stay uncommitted per CLAUDE.md.
- **U12 RL in v4 (new):** **out of scope. v3.1's RL results (A1/A3/A4/A10, WP5) stand as recorded, and no RL runs in Newton this phase.** Alternative: an RL insertion or bracing phase after M1, planned separately. **Superseded for bracing on 2026-09-30 (ledger `v4_brace_to_rl`): bracing becomes a separately planned RL phase in the Newton v4 cell; all other RL stays out of v4. The plan is `Docs/plan_v4_rl_brace.md`; Phase R0 adopted 2026-09-30 (ledger `v4_rl_brace_plan`).**
- **U13 Bracing strategy in v4 runs (new):** **`nearest` by default**. A6 found pooled nearest beats weakest_joint on S3 (p = 0.04) and S5 (p = 3e-7). `weakest_joint` stays available as a flag. This changes the prototype's CLI default.
- U14 Real time vs substeps (r4; matters only if P1(f) misses 0.8× at 16 substeps): n_sub = 16 if it reaches 0.8× real time with the full pool; else 8 if 8 does (kd_f 450, P1 measured at 8); else 16, with live runs playing below real time (smoke ≈ 0.35×) and `--record` videos for real-time playback. Headless batches are unaffected. Alternatives: 4 substeps (a full P1 re-run; carry creep predicted over the gate); or checking real time at the shown shape's pool (S3: 210) instead of S5's 465.

### 6.3 Ledger entries to record on adoption (doc-writer)

- **On adoption:**
  - `decision v4_plan_of_record` (this plan plus the user's U0–U13 answers);
  - `deviation v4_platform_newton` (cites U0, U1);
  - `decision v4_vision_only_definition` (U6);
  - `decision v4_joint_scope` (U4);
  - U12 (RL out of scope) and U13 (`nearest` bracing default) are recorded inside `v4_plan_of_record`;
  - `result newton_brick_press_tunnelling` — **only as the reproduced, corrected finding.** It records the work-reviewer's reproduction (`press.py`, 29 result files; kept as the base of `scripts/10_newton_scale_probe.py`):
    - pass-through under a step body force on a free brick is a tunnelling artefact;
    - seated bricks carry 71–100 N (sink 0.9–2.7 mm at 1×, 0.45–0.6 mm at 2×);
    - an unseated brick under a 0.5 s force ramp passes through at 71 N at 1× and holds 100 N at 2×;
    - a speed-limited stand-in press (no sideways, tilt or yaw freedom) never passed through and stayed on the stud tops up to ≈ 50 N at 1×;
    - the free-brick lateral-offset capture is impact-driven.
    - r1's "≥ 2 N passes through" conclusion is noted as superseded.
- On r4 adoption: `result v4_p0_baseline` (P0, §3); `failure v4_p0_s3_lever_press_diverged`; `probe v4_p1_smoke_finger_contact` (r4 rows 1–5; evidence in `results/v4/p1_review/` and `results/v4/p1_smoke/`); `decision v4_plan_r4` (this revision plus the user's answers on items D1–D8); `deviation v4_finger_contact_stiffness`, `deviation v4_clutch_gate_at_rest`, `deviation v4_p1_protocol_r4`, `deviation v4_contact_monitor_r4`.
- After P0′: `result v4_p0_baseline_stiff_fingers`.
- **After D1:** `decision v4_scale` (with the contact variant, F_press, the grip multiple, n_sub, kd_f and v_press, including whether the 2 mm/s rescue was used), `decision v4_cameras` (with the look height), `decision v4_motion_route`; `deviation curobo_isaac_split_reversed` if applicable; `deviation v4_brick_collision_clearance` if P1(h) was adopted.
- **If WP3 forces a D1 revisit:** `deviation v4_scale_revisited`.
- **If WP5b runs:** `result v5_contact_spike`. (`deviation v4_break_hold_33ms` is replaced by r5's `deviation v4_clutch_breakable`, whose rule is 2 end-of-frame samples and a 0.1 s settle.)
- On r5 adoption (2026-09-30): `probe v4_p4_brace_weld`; `failure v4_brace_overforce`; `decision v4_joint_scope_r5` (U4 = yes now), `decision v4_pipeline_gt_now`, `decision v4_plan_r5`; `deviation v4_clutch_weld_stiffness`, `v4_structure_collision_group`, `v4_clutch_breakable`, `v4_build_ends_at_first_failure`, `v4_brace_executor_r5`, `v4_handoff_protocol_r5`, `v4_milestone_order_r5`.
