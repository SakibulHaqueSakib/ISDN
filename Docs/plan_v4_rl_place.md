# Plan v4, Phase 4: RL placement primitives for the placer (sub-phases P0, P1, P2)

Status: **draft, written by the orchestrator (user, 2026-09-30: "dont do planner just write yourself"); not reviewed, not adopted.**
Decision behind it: ledger `v4_placer_rl_primitives` (user: the placer learns per brick how to place in tight spaces; the
pyramid goes into the training set; the policy controls placement primitives, one decision per brick).
Failure it answers: ledger `v4_pipeline_pyramid_inaccessible_grasp`.

Tags: **M** measured (file or code cited), **A** assumption (a number chosen here, revisable at the P0 gate), **D** derived from M.

## 0. Facts and what they force

| # | fact | tag | consequence |
|---|---|---|---|
| 1 | The user's pyramid (5x5, 3x3, 1x1; 11 bricks) placed steps 0-1, then step 2 (b_002, 1x4) missed the gate with dz +7.84 mm, lateral 0.12 mm, tilt 0.9 deg. | M: `results/proto_episodes.jsonl` rows `pyramid_none` | The brick is aimed well; it stops 7.8 mm high. |
| 2 | The placer pinches two side faces with its fingertips 4.1 mm above the brick bottom (`GRASP_DZ` 13 mm − `TIP_BELOW_TCP` 8.9 mm, `dual_arm_sim.py:71,74`). A same-layer neighbour flush against a pinch face has its studs 9.6 + 1.7 mm above the layer bottom, so the fingertips reach them when the gripped brick is 9.6 + 1.7 − 4.1 = 7.2 mm above seated. | D from M (code constants; proxy stud height `ex.STUD_HEIGHT`) | Matches fact 1 (7.84 mm). Any side pinch whose fingers must descend beside a flush neighbour cannot finish the insert. This is gripper geometry, not a motion skill. |
| 3 | Counting same-layer placed cells **flush against the two pinch faces, under the finger window** (FINGER_W 17.5 mm about the grasp centre, `tcp_offset_m` included; `planner.pinch_blocked`, Phase 2 close-out): pyramid steps 2, 3, 4, 6, 7, 8, 9 have 2, 4, 4, 1, 1, 1, 1; every other pyramid step 0; cube, arch, hollow_box and S3 0 at every step. The planner's older `grasp.accessible` also flags cube steps 1, 3, 5, 7 (in-line/diagonal cells under the 0.75 mm finger overhang), and a whole-face count flags hollow_box steps 3, 6, 11 (1x6 bricks: flush cells beyond the finger), yet both shapes build in 3/3. | M (orchestrator check, 2026-10-05; `v4_phase2_acceptance`) | `pinch_blocked > 0` is the operative definition of a **tight step**; `accessible` is too strict. |
| 4 | In a filled layer of 2 or more rows each way, the last brick placed always has a flush neighbour on at least one face of each pinch pair. | D (geometry) | Re-sequencing and the grasp-axis choice cannot remove every tight step; a different primitive is required. |
| 5 | The cell already presses with closed fingertips on stud tops: the legacy brace (`queue_brace_legacy`, `dual_arm_sim.py:764`) drives closed tips 2 mm into stud tops. | M (code) | A press-from-top move is an existing motion, not new control. |
| 6 | Brick–brick contact is carried by box/cylinder proxies inset by `ex.COLLIDER_INSET`; the clutch is a weld engaged when the snap gate passes (lateral < 2 mm, \|dz\| < 1.5 mm, yaw and tilt < 5 deg; `dual_arm_sim.py:120, snap`). | M (code) | A primitive succeeds when it brings the brick inside the gate; it does not need an interference fit. Whether a released brick slides down between flush neighbours under gravity is unmeasured. P0 measures it first. |
| 7 | Step-episodes from a nominal start (`--spec`, `--start-step`, `--only-step`) are being built as RL-bracing segment S0.2. | M (plan `Docs/plan_v4_rl_brace.md` §4) | The placer track reuses them: one tight step = one episode, no full build per sample. |

## 1. Question and outcomes

**Q.** Can the placer seat bricks at tight steps (fact 3) with a top-press primitive, and does choosing its parameters per brick (learned) beat one fixed setting enough to be worth learning?

- **Q0 (P0, feasibility):** does a drop-and-press primitive seat tight-step bricks at all, and how much do its parameters matter per context?
- **Q1 (P1, learn; only if P0 says so):** does a learned per-brick selector (primitive and parameters) beat the best fixed rule on held-out filled-layer structures?
- **Q2 (P2):** does the user's pyramid build end to end, and do the acceptance shapes still pass (no regression)?

"A fixed rule suffices" is an accepted outcome of P0. The user then decides whether learning is still wanted.

## 2. Task

**2.1 Primitives (one choice per brick).**
- `insert` (committed): the current grasp-insert (`queue_place`), unchanged.
- `drop_press` (new, in `queue_place`, default off): grasp at height `h_g`; transport; descend until the brick bottom is `c` above seated, where `c` = the fingertip clearance over the flush neighbours' stud tops plus a margin `m`; servo xy as now; open (release); rise `r_up`; close the fingers fully; press the closed tips on the brick's stud tops (at point `p`) to `d` past seated over `t_p` s; snap gate; retract. Parameters:

| parameter | values screened in P0 | note |
|---|---|---|
| grasp height `h_g` (TCP above brick bottom) | 13 (committed), 15, 16.5 mm | higher = smaller drop; at 16.5 mm the tips are 2 mm below the brick top (pad contact 2 mm tall [A]) |
| release margin `m` | 0.5, 1.5 mm | above the geometric minimum of fact 2 |
| press point `p` | brick centre, centroid of the supported studs | the existing grasp offset rule (`grasp_for`) uses the same centroid |
| press depth `d` | 1, 2 mm past seated | `PRESS` 1 mm, legacy brace 2 mm |
| press time `t_p` | 0.4, 1.0 s | |
| grasp axis | planner's, the other | the planner's axis is chosen by `touched`, not `pinch_blocked` |

The action set is {insert} ∪ the drop_press grid (3·2·2·2·2·2 = 96). The P0 screen uses a fixed 16-setting subset (§4) rather than the full grid.

**2.2 Episode.** A step-episode (RL-bracing S0.2 hooks): bricks of steps < n start at their nominal targets, welded; step n runs alone with B from park; A parked (no brace in this track). Success s = 1 iff the snap gate passes, no weld breaks, and the monitor logs no `arm_arm` contact. Records per episode: gate numbers, the brick's dz trace during release and press, peak finger–neighbour contact force, displacement of the flush neighbours (max over the episode, mm), breaks.

**2.3 Structures.**
- **Training/reference:** the user's pyramid (user instruction: it is in the training set, so it is **not** a held-out test), and the tight steps of the acceptance shapes, if any (fact 3 says the cube has none).
- **Generated family `filled_family(seed)`:** filled-layer shapes drawn as front/side silhouettes and carved with `blueprint.carve` + `tile`, the same path the user's drawings take: stepped pyramids (bases 3–6 studs, 2–4 layers), filled boxes (3–5 x 3–5 x 1–3), ziggurats with an off-centre top. One component, fits the feeder (`feeder_slots`). The canonical key and mirror de-duplication are reused from `tasks/brace_bandit.py`. Split by order of unique structures: train first 30, validation next 10, test next 20 [A].
- **Contexts:** every tight step (pinch_blocked > 0) of every structure; easy steps are only used for the regression check.

**2.4 Observation (for P1).** A pure function of the plan, like the brace features: brick type and length; layer; flush cells per pinch face (both axes); same-layer neighbours per side; supported studs and their centroid offset; whether the brick closes a row or a ring; step / N. No simulator state, no result file (tested as in the brace track).

## 3. Algorithm (P1, draft until the P0 gate)

The brace track's batch contextual bandit (`Docs/plan_v4_rl_brace.md` §3): an ensemble of 5 small MLPs predicting P(s = 1) per (context, action), decisions by argmax mean, UCB collection in rounds. It is shared code, not a second implementation. One decision per brick; reward r = s.

## 4. Phase P0: feasibility (≈ 1.5 agent-days after S0.2)

Segments.
- **S0.1 primitive:** `drop_press` in `queue_place` behind a `--place-json` flag (null = insert; or a parameter dict), default off. With the flag off, a cube `--test` run is unchanged (8/8, u_peak in the acceptance range).
- **S0.2 task library:** `tasks/place_bandit.py`: `filled_family`, contexts (tight steps via `planner.pinch_blocked`), the action grid, the features, and the 16-setting screen subset: the default (h_g 15, m 0.5, centroid, d 1, t_p 0.4, planner axis), its 6 one-factor variants, and 9 random grid points drawn once from a fixed seed [A]. Tests: generator validity and de-duplication, features pure, pyramid tight steps = [2, 3, 4, 6, 7, 8, 9].
- **S0.3 runner:** reuse the brace runner (`experiments/v4_brace.py`) with a `place` experiment: job list, de-duplication, process pool, one row file per job merged into `results/v4/rl/place/<exp>/episodes.jsonl`, tables.
- **S0.4 runs.**

**P0 run order.**
1. **Mechanism probe** (5 runs, descriptive): pyramid step 2 with the default drop_press, watching the release (does the brick fall into the gap, rest on the neighbours' edges, or tip?) and the press (does it seat; do the neighbours move?).
2. **Screen:** every pyramid tight step (7) plus the tight steps of the first 15 train structures (expected ≈ 30–60 contexts [A]) × the 16 settings × 1 run, plus `insert` × 1 as the control.
3. **Validate:** the best fixed setting (highest success over the screen; ties → lower neighbour displacement) and the per-context screened best, each × 3 fresh runs on the same contexts.
4. **Regression:** cube, arch and hollow_box full builds with the rule "drop_press (best fixed setting) iff pinch_blocked > 0, else insert": must stay 3/3 each (they have no tight steps, so this checks the flag-off path end to end).
5. **Pyramid full build** with the same rule, 3 repeats.

**Gate G-P0.**

| # | criterion | threshold |
|---|---|---|
| 0 | harness | flag off: cube `--test` 8/8 and acceptance u_peak range; step-episode repeat labels equal in ≥ 18/20 [A] |
| 1 | primitive works | best fixed setting seats ≥ 70% of tight-step contexts (validate runs) [A]; neighbour displacement ≤ 0.5 mm in ≥ 90% of successes [A] |
| 2 | pyramid | 11/11 in ≥ 2 of 3 full builds with the fixed rule |
| 3 | no regression | cube, arch, hollow_box 3/3 each |
| 4 | headroom for learning | (per-context screened best) − (best fixed setting) on the validate runs, context-weighted |

**Outcomes (pre-registered).**
- **FIXED-RULE:** 1–3 pass and criterion 4 < 0.10. The rule is handed to the pipeline (drop_press iff pinch_blocked > 0). P1 is not run unless the user asks for it.
- **LEARN:** 1 passes and criterion 4 ≥ 0.10. P1 runs on the filled family (P1/P2 are revised once at the gate with the measured numbers).
- **STOP-G:** no setting seats ≥ 50% of tight contexts. With this gripper the tight step is not solvable by top-press. Reported with the mechanism probe. The options go to the user: a stud-gripping end effector, sequencing that leaves tight bricks with a free pinch pair where one exists, or accepting partial builds.
- Criterion 1 between 50% and 70%: one in-phase change to the primitive (e.g. a small xy wiggle during the press, or a second press at the other end), then re-screen once. Still < 70% → STOP-G wording with the measured rate.

## 5. Phases P1 and P2 (outline, draft until G-P0)

- **P1 learn:** bandit rounds on train contexts of the filled family plus the pyramid (≈ 1,000–1,500 step-episodes [A]); validation for early stopping and for freezing the best fixed rule as the comparator.
- **P2 evaluate:** held-out test structures of the family: learned vs best fixed rule vs insert-only, step success rate, cluster bootstrap by structure; full builds of 5 test structures and of the pyramid with the learned selector. Hand-back to the pipeline only if learned ≥ fixed rule.

## 6. Budget and order

- After RL-bracing S0.2 (nominal-start step-episodes), which both tracks need. P0 ≈ 16 × 45 + 2 × 3 × 45 + 9 + 3 + 5 ≈ 1,000 episodes at most [A]; at the brace plan's assumed 250–400 episodes/h at 4 workers that is ≈ 3–4 h of compute.
- P0 runs before RL-bracing R0's sweep if the user wants the pyramid first (it is the user's own shape); otherwise interleaved by worker count.

## 7. Report

The mechanism probe frames; the screen table (contexts × settings, success, neighbour displacement); the fixed rule vs per-context best; the pyramid full builds; the regression runs; the G-P0 outcome.

## 8. Risks

- A released brick may tip onto a neighbour and the press then jams it. The mechanism probe measures this first; a lean-free release (open slowly) is the first in-phase fix.
- The press may push a flush neighbour or break its welds. Displacement and breaks are recorded and gated (criterion 1).
- At h_g 16.5 mm the 2 mm pad contact may slip in transport. Grip slip (brick-in-hand offset after lift) is recorded; that setting drops out if it slips.
- The pyramid is in training (user), so it cannot be evidence of generalisation; P2's test structures carry that claim.

## 9. Decisions for the user (recommended default first)

1. Order: placer P0 before the RL-bracing R0 sweep (the pyramid is your shape) / after it / interleaved.
2. If G-P0 says FIXED-RULE: hand the rule to the pipeline and stop / still train the selector.
3. Family scope: stepped pyramids, filled boxes and ziggurats (default) / also hollow shapes with filled walls.

## 10. Ledger entries

`v4_place_plan` (decision, on adoption); `v4_place_p0_probe` (probe); `v4_place_p0_screen` (result); `v4_place_g_p0` (gate); a `deviation` per pre-registered branch that fires.

## Files

To build: `tasks/place_bandit.py`, `tests/test_place_bandit.py`. To extend: `dual_arm_sim.py` (`queue_place` drop_press behind `--place-json`), `experiments/v4_brace.py` (a `place` experiment), `planner.py` (`pinch_blocked`, Phase 2 close-out). Reused unchanged: the S0.2 step-episode hooks, `blueprint.carve/tile`, the brace bandit's canonical key and learner.
