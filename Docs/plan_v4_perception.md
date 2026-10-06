# Plan v4, Phase 4: Perception (sub-phases V1, V2, V3), revision r2

**Status: adopted 2026-10-06 by the user's decision "Adopt with 5 fixes", after two Codex rounds.** There is no third Codex round; Codex checks the five fixes in the code-diff review at G-V1. **V1 (feasibility) is adopted. V2 and V3 stay a draft, revised once at G-V1.**

The user's request (2026-10-06, verbatim): "first build the perception pipeline so that the arm gets instructions from the image about the existin bricks. the input that the controller can use during RL training." This work goes before the placer track and before the RL-bracing R0 runs still pending; both wait until G-V3 (U-P6).

Tags: **[M: file]** measured, **[D]** derived from [M], **[R]** reviewer-reported and not recomputed, **[A]** assumed. Paths are relative to `brickassembly/`. Review history: r0 draft → Codex round 1 CHANGES REQUESTED (10 findings) → r1 → Codex round 2 CHANGES REQUESTED (1 blocker, 4 major) → escalated to the user, who chose "Adopt with 5 fixes" → r2. (earlier texts r0 and r1 are not kept in the repo; their review findings are tabled in the r1/r2 change tables)

Orchestrator checks:
- **r1 facts:** `feeder_grid()` gives 24 slots; S5 has 26 bricks; at 1×, single looks pass 21/24 at (35, 60) and 23/24 at (45, 60); the r4 gate is at plan_v4.md:350.
- **r2 facts:** `GRASP_DZ` = 0.013 (`dual_arm_sim.py:84`). `queue_place` pushes pre-insert, insert and release at `tgt + [0, 0, GRASP_DZ…]` with the original yaw (:846-850). `snap()` is the insert waypoint's `then` callback, so it runs at insert end with the hand still gripping, before release (:848-849).
- **r2 controller algebra:** with waypoint `ee = tgt + [0,0,GRASP_DZ] + bias` and the §2.2a bias, `B_est = ee − R_hand·d_h = T̂ + [i_xy, 0]` in xy; the 13 mm lever cancels.

## r2 changes (Codex round 2 finding → fix)

| # | finding | fix | where |
|---|---|---|---|
| 1 | BLOCKER: the FK controller dropped the 13 mm lever arm, and the yaw correction did not persist | **Full hand-frame vector**: `d_h = R_Lᵀ(ee_L − B̂)` and `B_est = ee − R_hand·d_h`. The nominal `GRASP_DZ` stays in the waypoints and the bias carries only the measured deviation. **δψ is added to every remaining waypoint of the step.** **The look schedule is fixed by the plan**, so a shadow-V5 rejection never aborts or changes gt or oracle control. The gt arm re-measures its GT offset at the final look. | §2.2a, §2.4 |
| 2 | MAJOR: two at-rest rules | **One screen, used everywhere** (E1, E1b, E2, E2b, native classification, G-V1, G-V2): r4's pose thresholds on the pre-weld state `snap()` reads, **with no at-rest clause** (snap happens while gripping; Codex reports 18.04 / 18.46 mm/s median / p95 relative speed at snap [R]). A screen after release would read the weld's pull, not the seat. The conditional exception is removed. | §0 fact 3, D6, gate |
| 3 | MAJOR: E2b scoring | E2b is scored **per step on the screen**, not by build completion. Injected vectors are signed. With no level grid there is no censoring. Predictions come from E2's matched (context, draw). Disagreement → **INCONCLUSIVE**. "Take the smaller value" is removed. | E2b, gate |
| 4 | MAJOR: per-axis windows do not support a radial xy + yaw margin | **The margin is now a matched injection.** Measured V5 error vectors (xy + yaw, from E1/E1b) are injected as persistent setpoint bias through fk_oracle on a common set of 33 contexts, at 1× and at 2× (plus the worst drift). The gate is the screen pass rate at 2×. **The axis-window sweep is dropped** (it becomes ACC's first diagnostic), so E2 shrinks from about 627 to 330 episodes. **A V1 yaw threshold is added**: estimates with \|δψ\| > 3° are rejected, and the gate requires yaw p95 ≤ 1°. Excluded contexts are reported separately. | §2.2a, E2, gate, G-V2 V5 row |
| 5 | MAJOR: gate precedence | **Validity is decided first**; INCONCLUSIVE takes precedence over ARCH, ACC and LOOP; a valid E2/E2b is a prerequisite of GO. **LOOP gets the same re-freeze and re-run rule as ACC.** **The E5 frame convention is corrected.** | gate, E5 |

## r1 changes (Codex round 1 finding → fix), kept as the record; r2 supersedes items 1, 5 and 7 where they conflict

| # | finding | fix |
|---|---|---|
| 1 | E2 had no valid capture window | A persistent bias through the FK controller, per-context windows, the E2b transfer check (superseded in r2 by the matched injection) |
| 2 | Fault detection not validated; branches contradicted GO; no native outcome | V7/V8 certification moved to V2; G-V1 is criteria 1–4; an INCONCLUSIVE outcome |
| 3 | Edge-fallback accuracy unqualified (S5 step 12) | The V5 acceptance rule; no fallback in V1; S5 step 12 as a regression case |
| 4 | Controller not matched across arms | One FK aim controller; ARCH reworded |
| 5 | "Strict seating" was not the r4 criterion | The r4 pre-weld screen (in r2, without the at-rest clause) |
| 6 | Sample sizes support only a screen | Rule-of-three and Wilcoxon removed; descriptive |
| 7 | Drift and remedy details | E5 to the last pre-weld row (convention corrected in r2); remedy re-runs; a "marginal" rule |
| 8 | V3 prefix, aborts, held-offset label | Whole-prefix verification; post-decision failures kept; `v3_pred` label |
| 9 | Firewall provenance | A single snapshot producer; replay under a perturbed mirror; one-way callbacks |
| 10 | Numbers | 0.0548 mm target x; the 35 mm look; rendered training provisional |

## 0. Facts and what they force

| # | fact | tag | consequence |
|---|---|---|---|
| 1 | **P2 was run but never recorded.** **(a) PASS** at 1× and 2× (top 8.8 ms, wrist 3.3 ms per capture). **(b) PASS** (600 tray bricks, 100 %, minimum 144 px). **(c) PASS at 1×**, 48/50 (single footprint 9/10; misses S5 step 12 and S3L step 4). **(c) FAILS at 2×**, 44/50. **Look pose**: h_look 45 s mm, offset 60 mm. **(d) at 1×**: yield 45/50. Relative error with hand-eye error: rms 0.114 / 0.090 mm, yaw 0.057°, radial p95 0.234 mm, max 0.378 mm (n = 45). With the 2-stud S5 step 12 row, n = 46, p95 0.285 mm, max 0.678 mm; that row has x error 0.669 mm at fit RMS 9.65 µm. Pixel-only target x is 0.0548 mm with ≥ 4 studs, against the budget's 0.03 mm. Target fit RMS: median 21.5 µm, p95 268 µm. | M: `results/v4/p2/`; orchestrator check | Record P2 (`v4_p2_cameras`). Yield is below 95 %. Fit RMS is not an accuracy measure. |
| 2 | The executor reads ground truth in three places. (i) `aim_brick` measures the in-hand offset when B leaves its lift, then runs an integral servo on the brick's GT xy (gain 4/s, clip ±4 mm). (ii) Picks go to fixed feeder slots at the final yaw. (iii) Targets are the plan on a fixed plate. The EE pose comes from `body_q`. Waypoints use `tgt + [0, 0, GRASP_DZ]` and keep their own yaw; `Arm.update` interpolates toward each waypoint's yaw. | M: `dual_arm_sim.py:84, 346-370, 1011-1036, 551-571, 819-852` | The GT servo is what perception must replace. Any yaw correction has to be written into the queued waypoints. |
| 3 | `snap()` runs at insert end with the hand still gripping. It uses the loose gate (lateral < 2.0 mm, \|dz\| < 1.5 mm, yaw and tilt < 5°) and welds at once to the nominal relpose. A gate miss ends the build. The 1× stud-top rest is +1.1 to +1.37 mm. The r4 gate is \|Δxy\| < 1.2 mm, dz ∈ [−0.5, +0.3] mm, tilt < 4°, yaw < 5°, on a released brick at rest. GT acceptance (93 snaps): lateral ≤ 0.28 mm, dz −0.38 to −0.16 mm; relative speed at snap median 18.04 / p95 18.46 mm/s [R]. | M: `dual_arm_sim.py:136, 773-778, 1045-1070`; `Docs/plan_v4.md:342-350`; `results/v4/j/acc/`; R: Codex round 2 | The cell gate can count an unseated brick. After release the weld has already pulled the brick to nominal. **The screen (D6)** is the r4 pose thresholds on the pre-weld state at snap, with no at-rest clause. Executor-real V7 negatives need a dataset mode (V2). |
| 4 | No capture tolerance has been measured: P1(b) was never run, D1 never held, and the cell runs 1×. Cavity clearance is 0.6 mm a side; open fingers 2.6 mm a side. | M | The tolerance that matters is tested directly against measured V5 errors (E2, D1). |
| 5 | RL inputs are plan-pure (37 brace features, which depend on the whole structure through the LP). Step-episodes start from a nominal start. R0 has harness and throughput done; the rest is pending and the manifest is not frozen. Nominal start vs full builds: hollow_box step 11 u is 0.149–0.152 vs 0.318–0.424. | M: `tasks/brace_bandit.py:239-345`; `r0/tables.json` | The perception field can enter before the freeze. A full-build check is needed (E2b). |
| 6 | Throughput is GPU-bound: 136.8 / 174.3 / 187.0 / 173.0 episodes/h at 1 / 2 / 4 / 8 workers. | M: `r0/tables.json` | Rendering is cheap; extra motion is the cost. |
| 7 | GPU nondeterminism: identical specs give different u2_peak values (1.1726 vs 1.4948). | M: `r0/tables.json` | The firewall uses replay (D4). |
| 8 | The feeder holds 24 bricks; S5 has 26. | M | S5 contexts run as S5[:13], with an assert that step 12 and its prefix match. |
| 9 | Recorder rows are post-simulate. The row labelled with the frame number `snap()` sees (`ex.frame`) is the state it read, i.e. pre-weld; the first welded row is the next frame. | M: `cell/record.py:3`; `experiments/j_tables.py:70` | Fixes E5's window. |

## 1. Question, hypotheses, outcomes

**Q.** Can the cell build the acceptance shapes when every brick, plate, placement and as-built pose comes from simulated cameras? And can the same perceived state be the RL controllers' input at an affordable cost?

| | hypothesis | positive | null | negative |
|---|---|---|---|---|
| H1 (V1) | One look per step (V5) plus the FK aim controller seats bricks as the GT servo does, with a 2× margin on the measured V5 errors. | fk_vision 3/3 per shape passing the screen; screen pass ≥ 95 % under 2× injected V5 errors plus drift | fk_oracle passes, fk_vision or the margin does not | fk_oracle fails: this controller is insufficient |
| H2 (V2) | Fully vision-only, on randomised scenes, it builds within 1 seed of GT per shape. | G-V2 passes | a stage fails its held-out gate | an end-to-end gap with every stage passing |
| H3 (V3) | Perception-in-the-loop step-episodes match GT within 0.05 at ≥ 150 episodes/h, with a pure observation. | G-V3 passes | throughput < 150/h | success gap > 0.05 |

## 2. Design

**2.1 Output contract: `WorldModel`** (a dataclass in `vision.py`, built in V2; JSON into rows)

| field | content | from |
|---|---|---|
| `plate` | T_world_plate (x, y, yaw) | V4 coarse, then fine |
| `tray[]` | type, pose | V2, V3 |
| `as_built[]` | per plan brick in the prefix: id, type, lattice (i, j, k, yaw index), status ∈ {verified, inferred, raised, below, missing, unseen}, residual (dx, dy, dz mm; dyaw °), source, t | V7 after each placement; V8 at step start and at episode t = 0 |
| `unexpected[]` | occupied cells the plan lacks | V8 |
| `held` | in-hand pose (`d_h`, tilt) in the hand frame; `src` ∈ {`v3_pred` (grasp-model prediction from the pre-grasp V3 estimate), `v5` (measured at the look)} | V3, V5 |
| `target` | perceived target lattice pose, studs used, looks | V5 |

Uncertainty is one σ per stage and look type, equal to the held-out per-axis p95 / 1.96. The as-built model is **plan-anchored** (U-P1).

**2.2 Consumers**

**(a) Executor: one FK aim controller**, shared by fk_oracle, fk_vision and E2.

*Look schedule* (identical in all arms, a pure function of the plan):
- one look per step with the brick bottom at h_look above the target's stud tops;
- a second look with the hand yawed 180° (bypassing the gripper's 180° wrap) iff the target is single-footprint (P2's `exposed()` rule);
- t_L = the frame of the final scheduled look.

*Estimate pair (T̂, B̂)*: the target pose and the held-brick pose (3D position, yaw), from the look frame t_E at which the estimate was made.

*Controller*:
- `d_h = R_hand(t_E)ᵀ (ee_FK(t_E) − B̂)` is the full 3-D vector in the hand frame. It keeps the 13 mm lever, so hand tilt and yaw are modelled.
- `B_est(t) = ee_FK(t) − R_hand(t)·d_h`.
- From t_L on, `bias(t) = (R_hand(t)·d_h − [0, 0, GRASP_DZ]) + [(T̂ − tgt)_xy, 0] + [i_xy, 0]`, where `i_xy` integrates `T̂_xy − B_est_xy(t)` in pre-insert and insert with today's gain and clip.
- The nominal `GRASP_DZ` stays in the waypoints; the bias carries only the measured deviation, the setpoint shift and the integral.
- T̂_z is not used in V1 (plan z).
- δψ = wrap(T̂_yaw − B̂_yaw) is **added to the yaw of every remaining waypoint of the step** (pre-insert, insert, release, retract) once, at t_L. R_hand(t) then follows it through FK.

*Arms*:
- **fk_oracle**: T̂ = the scoring target (plan pose), B̂ = the GT brick pose at t_E = t_L.
- **fk_vision**: V5's pair from the last accepted scheduled look. If no scheduled look is accepted → `perception_v5` (the build fails).
- **E2/E2b**: fk_oracle's pair with a persistent injected vector (Δx, Δy in the target frame, Δψ) added to T̂. It is a setpoint, so the integral cannot remove it.
- **gt**: today's GT servo, with the GT in-hand offset re-measured at t_L (after any scheduled rotation).

*Shadow V5* runs in the gt and fk_oracle arms on the same scheduled looks. It is logged only and **never aborts or changes gt or oracle control**.

*From V2 onward*, the rest of the executor:
- Pick ← `tray`. Plate ← `plate`. Verify ← V7.
- Before each step, a whole-prefix verification gate: every prefix brick must be verified, or inferred (fully covered by verified bricks whose heights need it, and not a bridged overhang).
- The footprint and approach column must be free.
- Anything unresolved → two oblique side views → still unresolved → `as_built_mismatch:<class>`.

**(b) RL observation: `obs = features(verified prefix) ⊕ P`.** It equals today's features once the gate passes. The P block has 6 numbers [A]: `held_dx, held_dy, held_dyaw` (`v3_pred` at a brace decision, `v5` at a placer decision), `sup_dz_max, sup_lat_max`, and `n_inferred`. P is recorded in every row and used only after an offline ablation shows a gain (U-P8).

**(c) Disagreement (U-P2)**:
- *Before the decision*: a mismatch means no policy query and the episode ends. The row is excluded from training rows, counted, and counts as a failure in success metrics.
- *After the decision*: a perception failure (e.g. A occludes the look) is a failed outcome of the action and stays in training rows.

**2.3 Firewall** (§2.5.1's lists unchanged; A's contact guard counts as finger force sensing)
- **Mode labels**: `gt`, `fk_oracle` (uses GT), `fk_vision` (alignment from V5; pick and plate from how the scene was built), `vision` (V2, the only mode that may be called vision-only).
- **V2** moves the executor into `cell/executor.py`, fed by one snapshot producer.
  - The producer receives only `joint_q`, `joint_qd`, finger targets, sim time and the sensor outputs.
  - Whitelisted fields: t, q/qd, EE FK computed on the IK model, grip widths, `stop` (the scorer's failure flag, on the stop path only), and captures (images plus the nominal camera pose).
  - Callbacks are one-way events.
- **`test_firewall`**:
  - (i) object-graph reachability: no `Model`/`State`, warp array or `Example`;
  - (ii) producer field and source whitelist;
  - (iii) replay of 3 traces with snapshots regenerated as recorded and under a randomised GT mirror. With newton and warp blocked, the snapshots and command traces must be identical.
- **RL**: rewards come from the scorer and observations from the executor, joined by job id. The observation must be unchanged under the randomised mirror.

**2.4 Stages and reuse**

| stage | reuse | phase |
|---|---|---|
| Cameras (top fixed; wrist_B +60 mm along hand x, 60 mm above the TCP, 25° tilt; §2.5.2 noise and hidden extrinsics per seed) | From `11_camera_probe.py`, move to `cell/cameras.py` (the probe imports them): `CAMS, cam_pose, sense, to_points, project, intrinsics, set_lights, mask_A`, sphere models | V1 |
| V5 at the scheduled looks. **Acceptance rule**: ≥ 4 non-collinear target studs (spanning ≥ 1 pitch on both lattice axes), ≥ 4 held studs, and **\|δψ\| ≤ 3°** (the V1 yaw acceptance threshold; §2.5.3's regrasp angle). Otherwise the look is rejected. Fit RMS is logged, not used. | The estimation half of `v5()` (`stud_blobs, top_disc, fit_rigid, held_height, plane_fit`) | V1 |
| Edge fallback | new; an ACC remedy only. Admitted after ≥ 20 single-footprint cases whose errors join the injection pool and pass E2 at 2× on the single-footprint contexts | if triggered |
| V7, V8, V1-plane, V2 tray, V3 LookFeeder, V4 plate | §2.5.3; V1's saved captures and P2's offline set | V2 |

**2.5 Departures from plan_v4 §2.5, WP3 and WP4**
1. **D1:** c_held (P1(b)) is replaced by a direct test: matched injection of measured V5 errors at 2× (E2/E2b), in the executor.
2. **D2:** scale 1× only; no D1 decision; no 2× sets.
3. **D3:** waypoint executor + plan-scheduled looks + the FK aim controller; no py_trees, no cuRobo; `--aim gt` kept.
4. **D4:** the firewall is the provenance test of §2.3.
5. **D5:** V8 is a new, plan-anchored stage.
6. **D6:** **the screen** — r4's pose thresholds (\|Δxy\| < 1.2 mm against the scoring target, dz ∈ [−0.5, +0.3] mm, tilt < 4°, yaw < 5°) on the pre-weld state at snap, gripped, with **no at-rest clause** — is the success criterion in every perception experiment and gate. It is a pre-weld screen, not physical seating. The cell gate is reported.
7. **D7:** the feeder grid with jittered poses.
8. **D8:** fallbacks are built only when their triggers fire.
9. **D9:** the B1 drift is measured (E5) and added linearly to the injected vectors.
10. **D10:** V7/V8 certification happens in V2.

## 3. Phase V1: feasibility (adopted; about 2.75 agent-days, about 3.8 GPU-h)

**Segments and subtasks**
- **S1.1 Cameras in the cell** (implementer, 0.5 d)
  - Write `cell/cameras.py`. `Example` gets `Cameras(model, seed)` when a vision flag is on; `capture(cam)` renders the current state, with the wrist mount following B's EE FK.
  - The probe imports from it; `--selfcheck` and `--summary` output is unchanged.
  - Check: capture time in a running build ≤ 1.5× P2's [A]. Captures are saved from S1.
- **S1.2 `vision.py` V5** (implementer, 0.75 d)
  - Estimation only, with the acceptance rule.
  - `tests/test_vision.py`: the synthetic self-check plus V5 on S1.1's captures.
  - Parameters are frozen and the hash goes into the rows before any V1 run.
- **S1.3 Executor hooks** (implementer, 0.75 d), all off by default
  - Flags: `--seed`; `--look` (the plan schedule of §2.2a); `--aim gt|fk_oracle|fk_vision` (§2.2a, including the lever arm and the δψ waypoint rewrite); `--aim-inject JSON` (fk_oracle only: one vector, or `{step: vector}` for full builds); `--shadow-vision`; `--save-captures` (raw top at step start and wrist after retract, for V2). The same fields go in `--spec`.
  - A scorer-only screen in `snap()` (D6), recording `snap_frame = ex.frame`.
  - Row fields: mode, V5 estimates and errors per look, accepted flag, the injected vector, the realised offset at the end of pre-insert, screen and cell-gate values, timings, capture and snap frames.
  - `experiments/v4_vision.py` with `e1 e1b pool e2 e2b e4 tables`. It reuses `v4_brace.run_jobs`/`merge`; new job-core fields only when not default.
  - Regression: all flags off, cube `--test` 8/8 with u_peak in range; `tests/test_arm_protocol.py` green.
  - Unit check: with the hand yawed and tilted, `B_est` equals the GT brick position to < 0.05 mm while the grasp is rigid.
- **S1.4 Runs** (orchestrator, in the background; scout digests the logs) **and records** (doc-writer).

**Experiments** (a feasibility screen, descriptive; the gate uses the thresholds below)

| exp | question | design | n | metrics | command, cost |
|---|---|---|---|---|---|
| **E1** builds | Does the FK aim controller keep the builds with oracle and with V5 estimates? | cube, arch, hollow_box (gated) and S3 (reported) × {gt, fk_oracle, fk_vision}; `--look --shadow-vision --record-all`; repeat r uses seed r in every arm | 3 per shape and arm | built == n, the screen and the cell gate per snap, lateral / dz, V5 errors | `bash scripts/run.sh experiments/v4_vision.py e1 --workers 4` → `results/v4/vision/v1/e1/`; 36 builds, about 1 h |
| **E1b** single footprint | Yield and accuracy on the hardest look type | P2's 9 single-footprint contexts (S1 1, 3; S2A 4; S3 2, 7; S3L 3; S5[:13] 4, 10, 12), nominal start × {fk_oracle, fk_vision} × 3 | 54 episodes | yield, V5 error, screen | `… e1b`; about 0.3 h |
| **E3** accuracy and the pool | How accurate is V5, and what errors get injected? | Accepted V5 estimates at every scheduled look in E1 (all arms) and E1b. The error vector is e = (Δx, Δy in the target frame, Δψ) of (target − held), estimate minus GT. Pool by look type (course vs single footprint). | about 480 steps | Radial and yaw p95 with bootstrap CI; yield = steps with an accepted estimate / all steps. **Pool frozen with drift_max (E5) under a hash before E2.** | `… pool` |
| **E5** drift (B1) | How far do the targets move between capture and weld? | `--record-all` of E1's FK arms: maximum displacement of the target supports' stud-top corners over rows `capture_frame … snap_frame` (both read as `ex.frame` at the call; the snap_frame row is the pre-weld state `snap()` read, and the first welded row is snap_frame + 1) | every gated step | worst case = drift_max | `… pool` |
| **E2** margin by matched injection | Does the executor tolerate twice the measured V5 errors plus the worst drift? | fk_oracle step-episodes (nominal start, unbraced) on a **common context set** of 33: all 31 steps of cube, arch, hollow_box, plus S5[:13] step 12 and S3 step 7. Per context: a control (no injection) × 2; K = 4 vectors drawn from the pool of its look type (A6 seed contract: SHA-256 of seed, "e2", key, step, k; one of the 4 drawn from the top decile of \|e\| [A]), each at scale 1 (e) and scale 2 (2e + drift_max·ê_xy; Δψ doubled) | 330 episodes | **Screen pass rate per scale** over included contexts. A context whose two controls do not both pass is native: excluded and listed separately. **Manipulation check**: the realised (GT brick − target) xy at the end of pre-insert equals the injected xy within max(0.1 mm, 10 %) [A] in ≥ 95 % of trials; otherwise the injection path is fixed and E2 re-run before any row counts | `… e2 --workers 4`; about 1.8 h |
| **E2b** transfer | Do E2's outcomes hold in full builds? | fk_oracle full builds of cube, arch and hollow_box; at each step, the same signed scale-2 vector E2 used for that (context, k), for k ∈ {0, 1} | 6 builds (up to 62 scored steps) | **Per-step screen outcome** vs E2's outcome for the same (context, k). A screen failure does not stop the build (the loose gate welds); a cell-gate miss does, and the steps after it are unobserved (counted, not scored) | `… e2b`; about 0.2 h |
| **E4** cost | Is perception in the loop affordable? | 40 unbraced contexts × {gt without look (R0's configuration), fk_vision} at 4 workers; timings | 80 episodes | episodes/h ratio and breakdown (provisional) | `… e4`; about 0.5 h |

Order: E1 → E1b → pool (E3, E5) → E2 → E2b → E4.

**Gate G-V1.** It is decided in this order; the first step that applies gives the outcome.

**Step 1, validity.** INCONCLUSIVE if any of these holds; INCONCLUSIVE takes precedence over everything below:
- (a) E1 native: the gt arm fails the screen or the cell gate at a (shape, step) in any repeat. That shape's repeats are re-run once in all arms; if it recurs, the shape is INCONCLUSIVE.
- (b) E2: more than 3 of 33 native contexts.
- (c) The E2 manipulation check fails after one fix.
- (d) E2/E2b agreement < 90 % of the observed steps [A].

INCONCLUSIVE goes to the user, with no GO. A shape-level INCONCLUSIVE lets G-V1 be decided on the other shapes only with the user's OK. **A valid E2 and E2b is required for GO.**

**Step 2, criteria:**

| # | criterion | threshold |
|---|---|---|
| 1 | controller | fk_oracle: built == n in 3/3 on cube, arch and hollow_box, every snap passing the screen |
| 2 | margin | the screen passes in ≥ 95 % of scale-2 trials in E2 (included contexts) **and** of the observed scale-2 steps in E2b [A]; the scale-1 rate is reported |
| 3 | V5 | yield ≥ 95 % of steps in E1 + E1b; relative yaw error p95 ≤ 1° (WP3) |
| 4 | in the loop | fk_vision: built == n in 3/3 on each of the three shapes, every snap passing the screen |
| — | cost (U-P3 only) | fk_vision ≥ 0.8 × gt episodes/h and ≥ 150/h [A] |

**Step 3, outcomes** (each branch that fires gets a ledger `deviation`):
- **ARCH** (1 fails): the FK aim controller as specified does not reproduce the GT servo. This does not show that continuous visual servoing is needed. Plan-reviser, with candidates: re-looks in pre-insert, an in-hand estimate refreshed after contact, or a different servo.
- **ACC** (1 passes, 2 or 3 fails): WP3's remedies once (wrist 1280×960; a mandatory second look on every step, averaged; a lower look at 35 s mm, which passes 24/24 best of two flips at 60 mm across both scales and 21/24 single looks at 1× [M]; the qualified edge fallback). The first diagnostic is a signed axis sweep on the common contexts (±long, ±short at {0.3, 0.6, 1.0} mm).
- **LOOP** (1–3 pass, 4 fails): one fix from the failure diagnosis.
- **GO**: 1–4 pass. **Marginal** (scale-2 pass rate within 2 trials of the threshold) is GO, disclosed; V2 makes the averaged second look the default; no gate is relaxed.

**Re-run rule, the same for ACC and LOOP:**
- A fix re-freezes the parameters (new hash).
- It re-runs E1 (all arms), E1b and E4, and re-derives E3, E5 and the pool.
- It re-runs E2 and E2b if the pool, the controller, the look schedule, pose or timing changed.
- Earlier rows are superseded. If the fix still fails, it goes to the user (ACC: accept a lower success rate, add an insertion search, or stop) or to the plan-reviser (LOOP).

## 4. Phase V2: vision-only executor (draft; about 5 agent-days, about 3.5 GPU-h)

**Segments**
- **S2.1** Executor separation, snapshot producer, `test_firewall`; `tests/test_arm_protocol.py` green.
- **S2.2** Randomisation (U-P4, U-P5 answered): plate ±30 mm / ±15° (halved if the 20-seed reach preflight fails); tray jitter ±10 mm / ±15°; lighting and extrinsics; targets through T_world_plate.
- **S2.3** V1-plane, V2 tray, V3 LookFeeder (hover 100 mm above the V2 estimate), V4 coarse and fine.
- **S2.4** `WorldModel`, V7, V8 (top survey, wrist survey, side views, inference), the whole-prefix gate.
- **S2.5** Held-out sets at 1×:
  - 500 top captures, 200 hovers, 100 plate surveys, 300 looks (≥ 60 braced);
  - V7: ≥ 200 cases with ≥ 40 each raised / below / missing;
  - V8: 100 surveys with ≥ 30 each missing / raised / extra, plus nominal-start surveys.
  - Sources: render-only scenes with injected faults (`cell/cameras.py --sets`) and executor-real cases. V7's executor-real negatives come from the dataset-only `--weld-after-verify` mode, never used as build evidence.
  - Development: S1, S2, S4, cube, P2's offline set, V1's captures. Held-out: arch, hollow_box, S3, S5[:13], S2A, S3C, S3L, new seeds. Parameters are frozen before evaluation.
- **S2.6** Runs, plus a live-run command for the user to run.

**Gate G-V2**

| item | threshold | derived from |
|---|---|---|
| firewall | the three tests green | §2.3 |
| V2 tray | recall, precision, type ≥ 99 %; radial p95 ≤ 15 mm, yaw ≤ 5° | LookFeeder window ±26 mm |
| V3 | radial p95 ≤ 1.3 mm, yaw ≤ 2° | half of the 2.6 mm open clearance [M] |
| V4 | coarse ≤ 20 mm / 2°; fine ≤ 2 mm, 100 % correct index | survey window; 0.25-pitch margin |
| V5 | held-out radial p95 and yaw p95 ≤ the V1 pool's (the envelope E2 tested at 2×), yield ≥ 95 %. If they are larger, E2 is re-run with the V2 pool (about 1.8 h) before this item is judged | the tested envelope |
| V7 | ≥ 98 % overall, recall ≥ 95 % per negative class | as-built integrity |
| V8 | 100 % class on visible cells; 0 injected faults reported as verified; whole-prefix resolution ≥ 95 % of nominal-start surveys | RL nominal start |
| end to end | 5 randomised seeds × {gt, vision}: cube 5/5, arch ≥ 4/5, hollow_box ≥ 4/5, every snap passing the screen; vision ≥ gt − 1 seed per shape; S3 reported; failures attributed | M1 gate |

On failure: WP3's remedies once, then plan-reviser.

## 5. Phase V3: RL observation and training (draft; about 2 agent-days, about 2.5 GPU-h)

**5.1 Segments**
- **S3.1** `tasks/brace_bandit.features_from_world(...)`: the gate, `features`, then P. The contract hash includes P. Purity tests.
- **S3.2** Vision-mode step-episodes: nominal start, then V8 at t = 0. The runner and R0's manifest gain a `perception` field before the freeze.
- **S3.3** The error model, under its branch only.
- **S3.4** Runs matched by `hidden()` draw across {gt, vision}: 40 critical × 2 (fixture on, unbraced); 20 easy × 2 (fixture off); 16 critical × LP-shared brace × 2; 20 repeat pairs; a 40-episode throughput batch. About 384 episodes.

**Gate G-V3**
- (a) purity and contract tests green;
- (b) discrete features equal plan features on 100 % of gated contexts;
- (c) |success(vision) − success(gt)| ≤ 0.05 per subset [A], with post-decision failures counted; pre-decision mismatches ≤ 5 % [A];
- (d) ≥ 150 episodes/h;
- (e) repeat agreement ≥ 18/20.

On pass, the remaining R0 runs and placer P0 run with `--perception vision`.

**5.2 Training cost (provisional)**: captures about 22 ms per episode [D]; the look about +5 %; sensor set-up +2–4 % [A]. V2's acquisition motions are not in E4. Predicted about 160–175 episodes/h [D].

**5.3 U-P3 answered: in-loop rendered perception.** The error model is the fallback only if G-V3 (d) fails: it samples from V1/V2's held-out errors, including aborts, and applies them to GT; rows are labelled `perception: error_model`, with no vision-only claim; it must agree with real perception within 0.05 on 60 matched contexts × 2 draws; final evaluation always uses real perception.

## 6. What to report (from result files via `experiments/v4_vision.py tables`)

| item | source | allowed claim |
|---|---|---|
| T-V0 P2 gate | `results/v4/p2/` | "at 1× the cameras carry the information; not at 2×" |
| T-V1a builds by arm and shape (screen, cell gate, lateral / dz) | E1 | "with V5 and the FK aim controller the acceptance shapes built in k/3 repeats (descriptive)" |
| T-V1b V5 errors and yield by look type; drift; the pool | E3, E1b, E5 | "relative error p95 = X mm, yaw p95 = Y°" |
| F-V1 screen pass rate by scale and look type; native contexts listed separately; E2b per-step agreement | E2, E2b | "with V5 errors doubled plus the worst drift, the screen passed in X % of matched nominal-start trials (33 contexts × 4 draws) and Y % of full-build steps" |
| T-V1c cost | E4 | provisional |
| T-V2a held-out table; T-V2b randomised builds and the firewall | S2.5, S2.6 | "built from simulated camera images alone" |
| T-V3 feature equality, matched success, aborts, throughput | S3.4 | "RL step-episodes run on perceived state with no measurable loss (feasibility level)" |

**Not claimable:** sim-to-real; physical seating (from the cell gate or the screen); reliability bounds from clustered placements; a capture window (not measured in V1); tolerance to error vectors outside the tested pool; an open-world as-built model; 2×; tight steps; vision under bracing beyond the measured subset.

## 7. Order, dependencies, compute
1. The Phase 2 Codex gate review runs first.
2. S1.1 ∥ S1.2, then S1.3.
3. E1 → E1b → pool → E2 → E2b → E4, serial, with no other GPU job.
4. G-V1, with Codex's code-diff review checking the r2 fixes, then one revision, then V2, then V3.

Totals: about 9.75 agent-days and about 9.8 GPU-h. R0 and the placer wait until G-V3 (U-P6).

## 8. Risks

| risk | likelihood / impact | handling |
|---|---|---|
| The executor tolerates less than 2× the V5 errors | medium / high | E2; ACC |
| The GT servo also corrected stud-contact deflection that FK cannot see | medium / high | fk_oracle; ARCH |
| V5 yield (90 % in P2) | high / medium | scheduled second look, acceptance rule, ACC |
| Renderer interacts with graph capture | low / medium | S1.1 first |
| Nominal start does not transfer | medium / medium | E2b; INCONCLUSIVE |
| Top camera occluded for V8 | medium / low | side views; a second static camera |
| The loose gate inflates success | known | the screen |
| The look collides with taller neighbours | low | look capped below the travel height |
| The plate exceeds reach | medium / low | preflight; halve the ranges |

## 9. Decisions for the user (answered 2026-10-06)
- **U-P1 As-built: plan-anchored verification with inference and side views** (recommended default taken).
- **U-P2 Disagreement:** a pre-decision mismatch aborts, with no query, and is excluded and counted; a post-decision perception failure is a failed outcome kept in training rows (default taken).
- **U-P3 RL perception: in-loop rendered perception** (user's answer).
- **U-P4 Tray: jitter ±10 mm / ±15°** (user's answer).
- **U-P5 Plate: randomised ±30 mm / ±15°, halved if reach fails** (user's answer).
- **U-P6 Order: V1→V3 first; R0 and the placer wait** (user's answer).
- **U-P7 Success criterion:** the screen of D6 (r4 pose thresholds on the pre-weld state at snap, no at-rest clause), with the cell gate reported (default taken, as fixed in r2).
- **U-P8 P block:** recorded; used after an offline ablation gain (default taken).

## 10. Ledger entries

**On adoption:**
- result `v4_p2_cameras`;
- decision `v4_perception_priority` (the user's words, 2026-10-06);
- decision `v4_perception_plan` (V1 adopted, V2/V3 draft; Codex rounds 1 and 2 CHANGES REQUESTED; the user's "Adopt with 5 fixes" with no third round, the fixes checked in G-V1's code-diff review; the U-P answers);
- deviations `v4_scale_1x_matched_injection` (D1, D2), `v4_perception_executor` (D3, D7, D8), `v4_preweld_screen` (D6: r4 pose thresholds, no at-rest clause), `v4_asbuilt_plan_anchored` (D5), `v4_b1_drift_measured` (D9);
- a row in plan_v4.md's phase table pointing to `Docs/plan_v4_perception.md`;
- then regenerate WORKLOG.

**After V1:** results `v4_v1_builds`, `v4_v1_single_footprint`, `v4_v1_v5_accuracy_pool`, `v4_v1_drift_b1`, `v4_v1_margin_injection`, `v4_v1_transfer`, `v4_v1_cost`; gate `v4_G_V1`; one `deviation` per branch fired, naming the rows it supersedes.

**At V2 adoption:** deviation `v4_firewall_provenance` (D4). Then gates `v4_G_V2` and `v4_G_V3`.

## 11. Files

**New:** `cell/cameras.py`, `vision.py`, `tests/test_vision.py`, `experiments/v4_vision.py`; in V2, `cell/executor.py`.

**Changed:** `dual_arm_sim.py` (flags off by default; the scorer-only screen in `snap()`; in V2, the executor moves out); `scripts/11_camera_probe.py` (imports from `cell/cameras.py`); `experiments/v4_brace.py` (optional job-core fields); `tasks/brace_bandit.py` (V3); `tests/test_arm_protocol.py` (V2). `scripts/run.sh` already routes the new files.

**Not built:** the py_trees tree, cuRobo, the `up` camera, the learned and edge fallbacks unless triggered, 2× sets, and an axis-window sweep unless ACC fires.

## G-V1 outcome (2026-10-07)

Computed by `experiments/v4_vision.py tables` on `results/v4/vision/v1_r1` (commit 713acad); ledger `v4_G_V1`. Outcome **ACC**.
- Validity: pass (E1 native clean on cube/arch/hollow_box; 2 of 33 E2 contexts native; manipulation check 252/264 = 0.9545 under the amended rule; E2/E2b agreement 34/36).
- Criterion 1 PASS: fk_oracle 3/3 on cube, arch and hollow_box, every snap passes the screen.
- Criterion 2 FAIL: E2 scale 2 pass 72/124 (58 %); E2b 22/36 observed steps. Criterion 3 FAIL: V5 yield 0.929 (yaw p95 0.194 deg passes).
- Criterion 4 FAIL: fk_vision built-all cube 3/3 (2 screen fails in seed 0), arch 1/3, hollow_box 0/3. Cost: fk_vision 301.6/h vs gt 276.2/h; U-P3 stands, provisional.
- Codex round 1 (CHANGES REQUESTED): ACC is correct; the scale-2 failure through fk_oracle shows an insertion-margin limitation, so the ACC remedies may not reach criterion 2.
- Codex round 2 (CHANGES REQUESTED): fixes 1, 2, 4, 5 confirmed in code; fix 3 (E2b) not confirmed: all 6 builds stopped early, 26 of 62 steps unscored, so 22/36 is selected by early stopping. No firewall problem (`v4_G_V1_review`).
- User decisions: (i) run the ACC branch (`v4_v1_acc_branch`); (ii) E2b keeps building after jams, every step scored, later steps flagged (`v4_v1_e2b_continue`); (iii) a settle hover before the close on every pick, regression on the acceptance shapes required (`v4_pick_settle_hover`).
- ACC branch in progress; V2/V3 stay a draft until the ACC re-run's gate.
