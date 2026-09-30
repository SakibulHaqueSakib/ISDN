# Plan v4, Phase 3: RL bracing in the Newton v4 cell (sub-phases R0, R1, R2)

**Status (2026-09-30): Phase R0 (feasibility) ADOPTED with the amendments of §11; Phases R1 and R2 remain a DRAFT** and are revised once, at the G-R0 gate, only if R0 passes. Review history: Codex round 1 CHANGES REQUESTED (12 findings, revised once), round 2 CHANGES REQUESTED (6 blocking, 1 for R0); escalated to the user after two rounds; the user chose "Start feasibility phase only" and approved the compute budget (U-RL-7, overnight). User decisions stand (ledger `v4_brace_to_rl`, `v4_pipeline_unbraced_now`): where-and-when policy, scripted stabilizer, Newton v4 cell, pipeline unbraced.

Evidence tags: **[M: source]** measured (result file, ledger, code); **[P]** planner probe (CPU-only LP runs, scripts not in the repo; S0.1 must reproduce these before any report cites them); **[R]** reviewer-reported (Codex's LP recomputation); **[A]** assumed. Paths are relative to `brickassembly/`.

## Change log (Codex round 1 finding → fix)

| # | finding | fix | where |
|---|---|---|---|
| 1 | "oracle-pick" is not a ceiling | Renamed **screened-best** (12-candidate nominal-load screen, re-run). STOP-B and STOP-H say "this screening procedure found…". Signed differences against it replace "regret". | §2.8, §4 R0, §7 |
| 2 | fixture validity | Explicit wrench about the COM from a body-fixed application point; re-added after every `clear_forces`; axial, eccentric and lateral wrench checks; measured A-reaction; B's native load declared additive; timing-matched **sham** control. LP braced values re-labelled optimistic. | §2.2, §2.5, §4 R0 |
| 3 | inference | Matched blocks resampled whole by structure; context-equal weighting stated; precise `hidden()` map with per-context stratified λ and continuous lateral; power arithmetic as planning only, E1 size fixed at G-R1 from measured variance; one confirmatory test plus a fixed-sequence secondary (no Holm); R0 re-labelled a feasibility screen. | §2.2, §6 |
| 4 | verdict rule | Exhaustive categories built on tests, "not demonstrated" wording, no equivalence claim, observed gain separated from evidence of gain. | §1 |
| 5 | action spaces differ | **LP-shared** heuristic over the same action set, executor and nominal-load information; the committed heuristic kept as an extra baseline; the tuned comparator chosen on validation data; `learned` dispatched before `bracing.assign`'s LP threshold. | §2.8, §4 S0.1 |
| 6 | threshold sensitivity | Per-frame consecutive-sample record per weld; "recorded-prefix threshold sensitivity", reported apart from simulated outcomes and from the LP-model discrepancy; worst-case relabelling cut. | §2.5, §6 |
| 7 | teleport is not independence | Independence claim removed; the step-episode is the task definition; preserve/reset lists; harness check against same-seed repeat ranges, descriptive; E3 descriptive transfer evidence only. | §2.1, §4 |
| 8 | purity | Mirror-invariant de-duplication before splitting; normaliser frozen on training rows; invariance tests; E2 re-labelled "previously inspected reference benchmarks". | §2.6, §4 |
| 9 | G-R0 branches | All branches defined; task frozen before the expensive sweep; throughput measured early; paired fixture-off excess harm; sampling and tie-break rules. | §4 R0 |
| 10 | E0/E1 selection | Setting chosen by a mechanical reaction check before the benefit sweep, which runs under one setting only; disclosed; both settings' data saved; all gains frozen; U-RL-3 scoped to A's gains. | §2.7, §4 R0 |
| 11 | budget | Manifest reconciled incl. start-up, the throughput batch, fixture checks and the E1 ceiling; wall-hours and worker-hours separated; U-RL-7 restated. | §5, §9 |
| 12 | boundaries | STOP-C and G-R2 reconciled; E4 loads a frozen contract; preserved items listed; manifest and invalidations go to the ledger; native-regime wording corrected. | §0, §4, §7, §10 |

## 0. Facts and what they force

| # | fact | tag | consequence |
|---|---|---|---|
| 1 | The tested unbraced builds show low joint loads and no breaks: cube 8/8, arch 11/11, hollow_box 12/12 in 3/3 runs; u_peak cube 0.033–0.036, arch 0.057–0.059, hollow_box 0.318–0.424. S3 builds 15/16 in 3/3 with a gate miss at step 15 (a placer failure, not a break). B presses 3–6 N. | M: `results/v4/j/acc/rows.jsonl`; press from `v4_p4_brace_weld` | On the tested shapes the native regime gives a brace policy almost no break signal. Not a general claim that nothing fails natively. A load source is needed for training (§2.2). |
| 2 | With a design-press load the LP predicts a "where" problem. S3 step 13: u0 3.43; best braced 1.97 with the committed mask [P], 1.56 with the relaxed mask [R]; at half press 0.49 / 0.08. S3 step 14: 1.96 → 0.44 / 0.00. Arch step 10: 0.95 → 0.95. | P, R | The LP lets the brace supply any bounded wrench; the executor holds passively: optimistic LP values, not executed benefits. |
| 3 | A holds with joint ke 400 N·m/rad; at ~0.5 m reach ≈ 1.6 N/mm at the hand. The welded structure is 23–24 N/mm (column lateral) and 34–70 N/mm (cantilever). | ke M (code); 1.6 N/mm A; stiffness M (J-a) | The committed hold is expected to carry a few percent of the load. R0 measures the reaction before anything else is spent. |
| 4 | The executed brace collides. Arch step 10: A link vs B hand during B's transport, peak 1019 N, then a break at u 1.14; A touched 5 bricks while gripping 2; grip 25.5 / 7.5 N. The feeder is on −y and that brace leaned −y. | M: `results/v4/j/seg2/arch_new.jsonl` | Lean must be part of the action; arm–arm contact part of the reward. |
| 5 | Row `rtf` is 1.0–1.45 on the acceptance runs; it is taken from `wall0`, set at the end of `Example.__init__`, so it excludes the full-plan LP, `solve_park` (720 IK iterations), the mesh/SDF build, model and solver creation and graph capture. Step cycle 8.8–9.3 s unbraced, 16.2–18.4 s braced. One process uses 1.0–1.2 GB of 32.6 GB VRAM; 32 cores. | M | Per-episode start-up is unmeasured; assumed 15–30 s [A] → ≈ 100–140 episodes/h per worker, 250–400/h at 4 workers [A]. R0 measures it early. |
| 6 | The cell is single-world throughout. `SolverMuJoCo(separate_worlds=True)` needs identical worlds. | M (source read) | Fresh `Example` per episode is the design; in-process reset and multi-world are fallbacks only. |
| 7 | The LP costs 2–4 ms per solve, five per utilisation. Candidates per critical step: 40–178 on S3/S5/arch; median 28 on a corbel family (committed mask). | P | LP features for every candidate cost ≈ 1–4 s per decision. |
| 8 | A pier-and-corbel generator gave 19 of 30 structures with ≥ 1 LP-critical step (47 steps); 17 of 47 LP-saveable at 14.3 N; among those a median 30% of candidates save. | P | The task family must be corbel-biased. |
| 9 | Break decisions land within ≈ 2% of the model on a full 2x4 patch and one cantilever only (J-b is an accepted failure). A two-support bridge broke at 0.70 of the LP load. | M: `v4_j_c_break_rule`, `v4_jb_accepted_with_known_cause` | Two separate issues: readout error near u = 1 (recorded-prefix sensitivity, §6) and an LP load-distribution mismatch (reported separately as model discrepancy). |
| 10 | `simulate()` clears body forces every substep (`dual_arm_sim.py:947`). `bracing.assign` returns no brace when u0 × safety factor < 1 (`bracing.py:150`). | M (code) | The fixture must be re-added after each clear. The learned strategy must be dispatched before that threshold. |

## 1. Question and verdicts

**Q.** In the Newton v4 cell, on one-step placement episodes under a design-level press load, does a learned selector that decides whether to brace and which grasp (position and lean) have a higher step success rate on held-out structures than never bracing? The selector sees only plan-derived information. Secondary: is it higher than an LP heuristic over the same actions?

Sub-questions, each able to end the work: **Q0 (R0)** is there a task where bracing matters, and does the scripted brace deliver anything on a screen? **Q1 (R1)** can a selector be learned within budget? **Q2 (R2)** does it hold on held-out structures?

**Quantities (R2, E1).** D1 = success(learned) − success(none). D2 = success(learned) − success(tuned LP heuristic). Each has an unadjusted 95% cluster-bootstrap CI (descriptive) and a structure-level sign-flip permutation p-value. The verdict uses the p-values only; p2 is tested only if the primary is significant and positive (fixed sequence, no multiplicity correction needed).

**Primary verdict (exhaustive).** **P-SUP:** p1 < 0.05 and D1 > 0 ("learned bracing is superior to never bracing on this task"). **P-INF:** p1 < 0.05 and D1 < 0. **P-ND:** p1 ≥ 0.05 ("superiority over never bracing not demonstrated"; reported with the CI and the achieved minimum detectable effect; no equivalence claim). Magnitude labels, reported separately under P-SUP: "observed gain ≥ 0.15" iff D1 ≥ 0.15; "evidence the true gain exceeds 0.15" iff the CI lower bound ≥ 0.15. A significant gain below 0.15 is P-SUP with neither label.

**Secondary verdict (only under P-SUP).** **H-SUP:** p2 < 0.05, D2 > 0. **H-INF:** p2 < 0.05, D2 < 0. **H-ND:** p2 ≥ 0.05 ("superiority over the heuristic not demonstrated"; neither equivalence nor non-inferiority is claimed). Without P-SUP, D2 is reported descriptively and untested.

**Hand-back.** The learned policy is handed back only under P-SUP and not H-INF. Under P-SUP with H-INF the recommendation is the heuristic. Under H-ND the user chooses.

**R0 stop outcomes** are answers too: STOP-A, STOP-B, STOP-T, STOP-P mean RL bracing is not shown to be worth pursuing in this cell with this task and screen; STOP-H means the screen found a benefit from bracing but no room above the LP heuristic.

## 2. Task definition

**2.1 Episode: one placement step from a nominal start.** A context is (structure, step n). A step-episode is a fresh `Example` in which the bricks of steps < n start at their nominal targets, welded and in `STRUCT_GROUP`, `breaker.fire` at t = 0; step n runs alone. Preserved from the full plan: brick n's original feeder slot; the later bricks in their slots (feeder occupancy); the full-plan travel height; the original step index; the full weld pool and topology; the plan's grasp fields for step n (the plan is computed once per structure and cached; episodes load it). Fresh by construction (each episode is a new `Example`): both state buffers; breaker counts and timestamps; IK seed; arm queues; servo state (`hand_off`, `i_xy`, `bias`); the protocol sets; monitor and recorder. The in-process-reset fallback must reset exactly this list and re-pass criterion 0. **What this is not:** a step-episode is the task definition, not a claim that build steps are independent; it differs from a full build in B's start pose (parked), arm joint state and IK branch, residual velocities, structure deflection and contact history. E3 gives descriptive transfer evidence only.

**2.2 Learning signal: the design-press fixture.** Load: for each support i of the new brick, the force F_i of `stability.insertion_loads(brick, placed, lateral, press=λ·n_studs·8.9 N)` (stud-proportional, at the support-patch centre). Application: on support body i as the wrench (F_i, τ_i) with τ_i = (p_i − x_COM,i) × F_i, where p_i is a body-fixed point (the nominal patch centre expressed in the support's frame, so it moves with the brick), F_i keeps a world-fixed direction, both recomputed once per frame from the start-of-frame pose; the spatial-vector ordering of `body_f` is established by the eccentric check, not assumed. Persistence: a Warp kernel inside the captured graph adds a persistent wrench array to `body_f` after every `clear_forces()`; the host assigns the array once per frame (`Static.set_force` from probe 13 is not copied; it would be erased). Pulse: from the start of B's `insert`, 0.3 s smoothstep up, 0.3 s hold, 0.3 s down; zero when the snap gate is read at 1.2 s (hold = 288 substeps; J-c detected every pulse ≥ 40 substeps [M]). Declared limits: RL experiments only; capacity, `U_BREAK`, `BREAK_SAMPLES`, `BREAK_SETTLE_S` and weld attributes untouched; B's native contact load (3–6 N [M]) is additive and not subtracted (λ labels the fixture part only; each context's native-only u2_peak is recorded in the placer screen). Hidden draw, precise map: `hidden(stage, structure_key, step, r, R)` seeds a generator from (master seed 20260930, stage label, canonical structure key, original step index); within a context, replicate r of R takes λ from the r-th of R equal-width strata of the band in a context-specific random order; lateral magnitude U(0, 3 N); lateral angle from the r-th of R equal sectors with a context-specific random phase and an independent random order; draws are shared across arms within a stage and independent across contexts, structures and stages; default band λ ~ U(0.5, 1.3) [A]; training rounds use unstratified draws from the same generator.

**2.3 Action.** A(c) = {none} ∪ {(y, z, lean)}; (y, z) from `bracing.candidates(brick, placed, clearance=lambda *a: True)` (pad-geometry feasibility only; the 28 mm hand rule is a feature, not a mask); lean ∈ {−45°, 0°, +45°} about the grip axis. This one set is shared by the learned selector, the LP-shared heuristic, the random arm and the screen. `queue_brace` and the hand-off protocol execute the choice unchanged.

**2.4 Observation and privileged information.** Observed: `features(bricks, step, action, exe)`, ≈ 34 numbers, a pure function of the plan and cell constants — context (brick studs and length; support count and studs; step index / N; layer / height; structure height and width; placement y, z; LP u0 at 0.5 / 1.0 / 1.3 × design press; weakest-joint offset; design press) and candidate (is_none; dy, |dy|, dz; height above plate; lean; lean relative to the placement; lean relative to the feeder corridor; gripped count; largest pad share; grips a weakest-joint brick / a support / a base-layer brick; LP braced u at 1.0 and 1.3 and LP Δu (optimistic LP values); LP `brace_util`; LP reaction; wrist-point height minus travel height; its distance to B's transport segment and to B's descent column; the 28 mm-rule flag). The heuristic uses the same nominal-load LP information. No feature reads a hidden draw, a simulator pose or a result record (tested, §4 S0.1). Privileged (reward and scoring only): weld wrenches, utilisations, breaks, contacts, body poses, λ and the lateral draw.

**2.5 Outcome, reward, records.** success s = 1 iff the step ends with `failure` None and the monitor logged 0 `arm_arm` events. reward r = s − 0.05·1[brace] [A]. Auxiliary target u2_peak = the maximum, over welds that are enabled and past their settle time and over frames, of min(u_t, u_{t−1}). Recorded per episode: per-weld u2 peak; a per-frame series of the maximum eligible two-sample u (for §6); A's total reaction wrench on the structure averaged over the pulse plateau, next to the LP's `expected_reaction_wrench`; realised grip; contacted set; guard stop; arm_arm peak.

**2.6 Structures, de-duplication, split.** Generator `corbel_family(seed)`: depth 2; pier of height 3–6; 1–3 corbel layers extending 1–2 studs; optionally a second pier and a crown; tiled with `blueprint.tile`; 5–24 bricks; one component. De-duplication before splitting: the canonical key is the smaller of the brick-cell set and its y-mirror (a structure and its mirror are one entry; each unique structure gets one orientation, drawn from its key; no mirrored relatives in any split). Split by order of unique structures: train = first 72, validation = next 18, test = the following ones, as many as §6's size rule needs. Pools: critical = LP u0(1.3 × design) ≥ 0.8; easy = the rest. Exclusion, applied before any braced run and identically for all arms: a context whose unbraced, fixture-off step-episode fails is dropped as placer-infeasible (count reported). Reference benchmarks: S3 steps 13 and 14, S5 steps 15, 20–23 and 25, arch step 10 — previously inspected, so E2 is descriptive, not held-out.

**2.7 Executor setting.** One setting, fixed before the benefit sweep and used by every arm. E0 (committed): A arm ke 400 / kd 40; A fingers 9500 N/m (14.3 N nominal). E1 (stiff hold): A arm ke 4000 / kd 126; A fingers 19000 N/m (28.6 N nominal); values frozen now [A] (r4 measured 24.9 N realised at 28.6 N nominal [M]). Choice rule (mechanical, not outcome-gated): a reaction check on 4 contexts × {E0, E1} at the LP-best shared candidate, λ = 1, no lateral; E0 is chosen if A's measured plateau reaction reaches ≥ 50% of the LP's expected reaction in ≥ 3 of 4 [A]; otherwise E1; if any E1 run diverges, E0. Both settings' rows are saved and the choice is disclosed. The benefit sweep runs under the chosen setting only. Nothing else about B, weld stiffness, capacity or break thresholds changes.

**2.8 Arms.**

| arm | definition |
|---|---|
| none | A parked; B does not stage |
| sham | A parked; B stages for 5.6 s, the brace's nominal approach-to-close time [M: code durations]; the timing-matched no-contact control |
| LP-shared | over A(c): brace iff LP u0(nominal) × SF ≥ 1; (y, z) minimising LP braced u with `bracing.assign`'s key order; lean by `lean_for` |
| LP-committed | `bracing.assign(strategy="weakest_joint")` as committed: 28 mm mask, SF, scripted lean |
| tuned LP heuristic | the best of {LP-shared, LP-committed} × SF ∈ {1.0, 3.0} by validation on-policy success, frozen before R2; ties go to LP-committed at SF 3.0 |
| random | a uniformly random braced candidate from A(c) |
| learned | §3 |
| learned-sham | the learned policy's when-decisions with sham staging in place of the brace |
| screened-best | per context: 12 braced candidates (LP-best 4, nearest 2, random 6, leans mixed) run once at λ = 1 with no lateral; the pick is re-run on fresh draws. The output of this screening procedure, not an oracle or ceiling |

Screen pick rule: the success with the lowest u2_peak; ties within 0.01 go to the lower LP braced u, then the smaller |dy|, then the lower candidate index; with no success: the lowest u2_peak among candidates without arm_arm, else the lowest u2_peak.

## 3. Algorithm

**A batch contextual bandit with a learned outcome model** (horizon 1; discrete variable-size action set; ≈ 2,000 training episodes, so every sample is reused off-policy). `tasks/ppo.py` is on-policy, continuous, and used 150–300k samples in WP5; reused from it: `mlp`, `RunningNorm`, the JSONL log convention; no PPO head. Model: ensemble of 5 MLPs (2×64, ELU) on normalised features; outputs logit P(s = 1) and û2_peak; loss BCE + 0.5·Huber(log u2_peak); the normaliser is fitted on training rows only, then frozen and stored in the policy file; early stopping on validation structures. Decision: argmax over A(c) of mean P̂(s) − 0.05·1[brace]. Collection: round 0 is R0's bound data on train contexts (existing rows, not counted as new); rounds 1–4 are 400 new episodes each on train contexts (critical pool + 25% easy): 50% UCB picks (mean + 1·ensemble std), 25% uniform random candidates, 25% none and LP-heuristic picks on matched draws; identical (context, action, draw, setting) jobs are run once and reused. The LP enters as features; the pure-LP policies are the heuristic arms; no LP-label pre-training. Offline ablations (no extra simulation): logistic regression; the MLP without LP features.

## 4. Phases, segments, gates

### Phase R0: feasibility (≈ 2.5 agent-days)

Segments. **S0.1** task library, pure Python — `corbel_family`, canonical key and de-duplication, `split`, `contexts`, `actions`, `features`, `hidden`, the arm pickers of §2.8; `--self-check` writes `results/v4/rl/r0/lp_probe.json` reproducing the [P] numbers; in `bracing.py`: `_record(..., tilt=None)`, `brace_at(...)`, and in `assign` the `learned` strategy is dispatched immediately after the `none` / empty check and **before** the `r0.s * safety_factor < 1.0` return (new `tasks/brace_bandit.py`; `bracing.py`; new `tests/test_brace_bandit.py`). **S0.2** cell hooks, all default off — `--spec`, `--plan`, `--start-step`, `--only-step`, `--brace-json` (null, a candidate, or `{"sham_s": T}`), `--press-scale`, `--press-lateral`, `--exe`; nominal start; the fixture of §2.2; the records of §2.5 (`dual_arm_sim.py`). **S0.3** runner — job lists per experiment, job de-duplication, a process pool, one row file per job merged into `results/v4/rl/<exp>/episodes.jsonl`, tables and statistics, `manifest.json` (new `experiments/v4_brace.py`; `scripts/run.sh` test routing). **S0.4** runs → `results/v4/rl/r0/`.

Tests in `tests/test_brace_bandit.py`: generator validity and de-duplication; features identical across hidden draws and with result files absent; features computed with `newton` and `warp` imports blocked; `assign(strategy="learned")` returns a brace on a step where u0 × SF < 1 when the policy stub says brace. Requirements on S0.2: `--spec` runs write no `plans/sim_*.json` and do not append to the tracked `results/proto_episodes.jsonl`; with every new flag at its default, a cube `--test` run gives 8/8 and u_peak within the acceptance range 0.033–0.036 ± 0.01.

**R0 run order** (the expensive sweep comes after the task is frozen):
1. Harness. Fixture wrench checks on static weld-held scenes: axial on a column; eccentric (a support loaded at a patch centre off its COM on a cantilever); lateral 3 N. 20 (context, action, draw) triples run twice. Descriptive: three nominal-start repeats each of cube step 7, arch step 10 and hollow_box step 11 against the per-step u_peak range of the three acceptance full builds.
2. Throughput, early. Four batches of 40 mixed episodes (half braced) at 1, 2, 4 and 8 workers on an otherwise idle GPU; start-up time recorded separately from `rtf`.
3. Placer screen. Unbraced, fixture off, every critical train and validation context plus 20 easy ones, once.
4. Signal. 40 critical contexts × 4 draws, unbraced, fixture on (sampling: the first 40 train structures with a screened critical context; from each, the one with the highest LP u0); 20 easy contexts × 2 draws, fixture on. Any band or pulse change happens here (branches below). Then the task manifest is frozen: band, pulse, action set, split, seeds.
5. Reaction check and executor choice (§2.7): 8 runs.
6. Screen. The first 16 of the 40 signal contexts × 12 candidates, plus none and sham, at λ = 1, no lateral.
7. Validate. Those 16 contexts × {none, sham, LP-shared pick, screened-best} × 4 fresh draws.
8. Harm. The same 16 × {none, LP-shared pick, screened-best} × 2 repeats with the fixture off.

**Gate G-R0.** Criteria 4 and 5 are a feasibility screen (64 paired trials from 16 structures), not a statistical test.

| # | criterion | threshold |
|---|---|---|
| 0 | harness | fixture: weld Fz, patch moment and shear within 5% of statics (floors 25 mN, 0.25 mN·m [A]) on the axial, eccentric and lateral checks, and zero with the fixture off; ≥ 18/20 repeat labels equal [A]; regression guard and tests green. The nominal-start vs full-build comparison is reported, not gated. |
| 1 | throughput | ≥ 150 step-episodes/h at the best worker count on the mixed batches [A] |
| 2 | placer | ≥ 70% of screened contexts succeed unbraced with the fixture off [A] |
| 3 | signal | unbraced failure rate on critical contexts in [0.30, 0.90]; ≥ 60% of those failures are `joint_break`; easy contexts with the fixture on fail ≤ 10% [A] |
| 4 | brace benefit on the screen | screened-best − sham ≥ 0.20 (context-weighted point estimate); better in ≥ 5 of 16 contexts and worse in ≤ 1; paired fixture-off excess failure of screened-best over none ≤ 0.10 [A]. The difference against none and A's measured reaction against the LP's are reported alongside. |
| 5 | headroom on the screen | screened-best − LP-shared pick ≥ 0.10 (point estimate) [A] |

**Branches (pre-registered; each that fires gets a ledger `deviation`).** 0 fails → fix in phase; no task run counts until it passes. 1 fails → in-phase fix: persistent workers with a fresh `Example` per episode, then in-process reset (must re-pass criterion 0); still < 60/h → **STOP-T**, or a user-approved cut with claims reduced to descriptive. 2 fails → drop failing contexts; if ≥ 60 train critical contexts remain, continue and carry the placer issue to the orchestrator; otherwise **STOP-P**. 3, rate < 0.30 → switch once to λ ~ U(0.9, 1.3) and re-run the whole signal set; still < 0.30 → **STOP-A**. 3, rate > 0.90 → switch once to λ ~ U(0.3, 0.9) and re-run; still > 0.90 → plan-reviser at the gate (task saturated). 3, cause mix or easy failures > 10% → shorten the hold to 0.15 s once and re-run; still fails → plan-reviser. After any switch, only data collected under the frozen revised task enters criteria 3–5 or later phases; earlier rows are kept as a record and marked invalidated. 4 fails → **STOP-B**: "this 12-candidate screen under the chosen executor setting found no braced action that beats the timing-matched control" (not a statement that the ceiling is never bracing; reported to the user with the reaction table; changing the policy scope — an active or feed-forward brace — is a user decision). 5 fails, 4 passes → **STOP-H**: "on this screen the LP heuristic was within 0.10 of the best screened action"; the user decides between handing back the heuristic and continuing; the other executor setting is not tried as a rescue.

### Phase R1: learn (≈ 1.5 agent-days)

Segments. **S1.1** `fit` and `decide` in `tasks/brace_bandit.py`; the policy file holds ensemble weights, the frozen normaliser, the executor setting, the feature and action contract hash and the manifest hash, on CPU tensors; `train.jsonl`. **S1.2** rounds 1–4: 1,600 new episodes. **S1.3** on-policy validation: screened validation contexts (≈ 36) × 3 fresh draws × {none, learned, LP-shared and LP-committed at SF 1.0 and 3.0}, de-duplicated; selection of the tuned LP heuristic; offline ablations. **S1.4** E1 size: from the validation data, estimate discordance and the design effect of the structure-clustered paired difference; set the number of test contexts for 80% power at D1 = 0.20 (planning effect [A]), between 60 and 120, with 4 draws each.

**Gate G-R1:** (a) data: ≥ 1,600 new episodes in rounds 1–4 with features, action, draw and outcome; a job with no row is re-run once; divergence counts as a failure; lost episodes ≤ 5%. (b) model: AUC of P̂(s) on validation-structure rows ≥ 0.75 [A]; calibration reported. (c) on-policy validation, point estimates, screening only: learned − none ≥ +0.10 and learned − tuned heuristic ≥ −0.05. (d) purity and contract tests pass; median decision time ≤ 2 s per step. (e) E1 size fixed and recorded with its predicted power. Branch: (b) or (c) fails → one extra round (+400), refit, re-validate on fresh draws; still failing → **STOP-C** (R2 then consists of E1 only).

### Phase R2: evaluate and hand back (≈ 1.5 agent-days plus the report)

| exp | design | metric |
|---|---|---|
| E1 primary | screened test contexts (60–120, fixed at G-R1) × 4 stratified draws matched across arms. Full arms: none, tuned LP heuristic, learned. learned-sham only where the learned policy braces. random and LP-committed (if it is not the tuned variant) on 2 of the 4 draws, descriptive. | step success; break, arm_arm and gate-miss rates; brace rate; u2_peak; cycle |
| E1-screen | 20 test contexts (the first 20 by structure order): the 12-candidate screen, pick re-run on the E1 draws | signed difference of each arm against screened-best |
| E2 reference benchmarks | the previously inspected contexts of §2.6 × 4 draws × {none, tuned heuristic, learned, random} | per-context table, descriptive |
| E3 full builds | 6 test structures whose unbraced, fixture-off full build completes, plus arch; fixture on at every step; {none, tuned heuristic, learned} × 3 | built == n, first failure and cause; shown beside the per-step results as descriptive transfer evidence; no independence or inferential full-build claim |
| E4 hand-back | `dual_arm_sim.py --strategy learned --press-scale 1.0` on arch, one test structure and a 2-deep user drawing; `decide` loads the frozen policy file and refuses to run on a contract or manifest mismatch | 0 arm_arm events; tests green |

**Gate G-R2** passes when what was run is complete and valid, whatever the verdict. Always: E1 arms on matched draws; exclusions applied before arm assignment; tables generated from row files; verdict computed by §1. Unless STOP-C: E1-screen, E2 and E3 reported; E4 run if the hand-back rule allows it. Preserved and checked: the pipeline default `--strategy none`; the frozen twin (`tests/test_planner.py` leaves the committed twin plans byte-identical); `env.lock`; capacity and break constants and weld attributes; `tests/test_arm_protocol.py` and the monitor self-check green. Phase 2's native acceptance (G2, G3, G4, G6, G7) is not replaced or re-evidenced by any fixture result.

## 5. Budget and order

Step-episodes, after job de-duplication.

| phase | item | nominal | contingency |
|---|---|---|---|
| R0 | harness 49; throughput 160; placer screen 200; signal 200; reaction 8; screen 224; validate 256; harm 96 | 1,193 | +200 (one signal re-run) |
| R1 | rounds 1,600; validation 432 | 2,032 | +724 (extra round 400, re-validation 324) |
| R2 | placer screen 85; E1 1,104 at 60 contexts (4.6 arm-equivalents × 4 draws); E1-screen 320; E2 153 | 1,662 | +1,189 (E1 at 120 contexts: 2,208; screen 170) |
| total | | ≈ 4,900 | ceiling ≈ 7,000 |

Plus ≈ 76 full builds (E3 and its pre-screen) and the static fixture checks (seconds each).

| | nominal | ceiling |
|---|---|---|
| wall-hours at the gate floor, 150/h [A] | ≈ 33 h | ≈ 47 h |
| wall-hours at 400/h with 4 workers [A] | ≈ 12 h | ≈ 18 h |
| worker-hours (wall × workers, 4 workers) | ≈ 50–130 | ≈ 70–190 |
| full builds: ≈ 2.2 min each [A: 110–150 s sim at RTF 1.0–1.45 [M] plus start-up] | ≈ 2.8 h serial, ≈ 0.8 h at 4 workers | same |

Per-episode start-up is not in any saved RTF (fact 5); the throughput batch is what turns these [A] figures into measurements. Order: S0.1 is CPU-only and can start now; S0.2 and S0.3 touch `dual_arm_sim.py` and follow Phase 2's commits on the same branch; GPU batches run when Phase 2 is not doing acceptance runs or a live demo; warm the Warp kernel cache with one run before starting workers; R1 after G-R0; R2 after G-R1.

## 6. Statistics

- **Estimand.** The mean over test contexts of the per-context difference in success rate between two arms, each context weighted equally (with 4 draws per context this equals the trial-weighted mean; structures with more critical contexts weigh more). A structure-equal-weight version is reported as a sensitivity row.
- **Resampling unit.** The structure. A resampled structure brings all its contexts, all draws and all arms as one matched block.
- **Interval.** Percentile cluster bootstrap, 10,000 resamples, unadjusted 95%, descriptive.
- **Test.** Structure-level sign-flip permutation test (10,000 flips of each structure's summed paired difference) on the same statistic. One confirmatory test (D1). D2 is tested only if D1 is significant and positive. No other test is confirmatory.
- **Power (planning arithmetic, not established).** Under independence, discordance 0.5: ≈ 172 pairs give 80% power for a difference of 0.15 [A]. With 8 trials per structure and ICC 0.2 (design effect 2.4): 240 pairs give ≈ 84% for 0.20 and ≈ 58% for 0.15; 480 pairs give ≈ 87% for 0.15 [A]. The planning effect for D1 is 0.20 [A], the screen's minimum benefit. E1's size is set at G-R1 from measured discordance and design effect, capped at 120 contexts. If 80% power for 0.20 is not reached at the cap, the report says so and gives the achieved minimum detectable effect. D2 has no planning effect size: an estimate with a gated test; its minimum detectable effect is reported.
- **R0** numbers are a feasibility screen; no CI-based claim is made from them.
- **Recorded-prefix threshold sensitivity.** From the per-frame two-sample u record, for a threshold of 0.9: a trial is a recorded-prefix failure if the record crosses 0.9 before the run's actual end, or if the run failed for any observed reason. It only converts successes to failures; it never erases an observed break, collision or gate failure. Reported in its own table, apart from simulated outcomes. It says nothing about thresholds above 1 (those runs ended at the break).
- **LP-model discrepancy.** LP u0 against simulated unbraced breaks; the bridge-type early break. Reported separately as model mismatch, not as readout noise.
- **Break margins.** The u at each break, by arm, descriptive.

## 7. Report

T-R0a unbraced failure by λ stratum and cause, fixture on/off; native-only u2_peak; LP u0 vs simulated break ("under the design-press fixture, x% of critical steps fail unbraced"; LP-vs-sim agreement as model discrepancy). T-R0b fixture wrench checks; reaction check for E0 and E1 (measured A reaction vs the LP's); the executor choice ("the fixture applies the stated wrench"; "A carried x% of the LP's expected reaction"). T-R0c screen and validation: none, sham, LP-shared, screened-best; excess harm; arm_arm by lean ("this screen found a benefit of Δ over the timing-matched control", or the STOP-B / STOP-H wording). T-R0d episodes/h vs workers; start-up; RTF. T-R1 samples per round; validation AUC and calibration; on-policy validation; heuristic variants; ablations; E1 size and predicted power. T-R2a arms × metrics; D1 and D2 with CI and p; magnitude labels; learned vs learned-sham; signed differences vs screened-best; sensitivity rows (the §1 verdict; "contact, not staging time, accounts for x of the gain", descriptive). T-R2b reference benchmarks (descriptive, previously inspected). T-R2c full builds beside per-step results (descriptive transfer only). T-R2d recorded-prefix sensitivity; break margins (robustness statements, labelled offline). F: u2_peak vs LP u0; outcome maps for 3 contexts; where and when the policy braces.

**Not claimable:** anything about the native regime (acceptance evidence there is Phase 2's, and S3 has a native placer failure); real hardware; vision-driven execution; structures not 2 deep or other grasp families; break physics beyond axial and prying load; equivalence or non-inferiority to any arm; an oracle or ceiling; inferential full-build success; step independence.
**Limitations to state:** the press fixture (additive to B's native load, LP load model); the nominal start; the weld-only idealisation and end-at-first-failure; the accepted J-b failure and readout validated on two patch geometries; the LP-vs-sim load-distribution mismatch; the corbel family as the distribution; the executor setting used and how it was chosen; a ground-truth-driven executor.

## 8. Risks

The passive hold delivers nothing: high / ends the work → reaction check first; STOP-B with qualified wording. Fixture mis-applied (convention, clearing, lever arm): medium / high → criterion 0 wrench checks before any task run. Pulse artifacts at 70–90 N: medium → criterion 3; one pre-registered pulse change, then a full re-run. B cannot place on generated shapes: medium → placer screen and exclusion; STOP-P. E1 unstable: medium / low → fall back to E0, disclosed. Underpowered E1: medium → size set at G-R1, cap 120 contexts, "not demonstrated" wording. Start-up dominates episode time: medium → measured early; plan cache; persistent workers. Overfit on ≈ 2,000 samples: medium → ensemble, structure split, frozen normaliser, logistic comparator. Nominal start differs from a build: medium → reported comparison; E3 descriptive. Policy invalid once WP5b gives real insertion contact: medium / later → contract hash; re-validate then.

## 9. Decisions for the user (recommended default first)

- **U-RL-1 Formulation.** Horizon-1 contextual bandit (recommended), or full-episode PPO (not feasible at the expected throughput).
- **U-RL-2 Learning signal.** The design-press fixture, RL experiments only (recommended), or a capacity curriculum, or wait for WP5b.
- **U-RL-3 Executor latitude.** Allow E1, i.e. A's arm and finger gains only, frozen values, chosen by the reaction check (recommended). Does not extend to B, weld stiffness, capacity, break thresholds or an active or feed-forward brace.
- **U-RL-4 Action space.** Three leans plus position over the relaxed candidate set (recommended), or position only.
- **U-RL-5 Stop rules.** Accept STOP-A, B, H, T, P and C with their qualified wording (recommended).
- **U-RL-6 Hand-back regime.** Learned bracing is enabled only with the design-press fixture; the native pipeline stays unbraced (recommended).
- **U-RL-7 Compute (corrected; approved by the user 2026-09-30, run overnight).** Nominal ≈ 4,900 step-episodes and 76 builds; ceiling ≈ 7,000: ≈ 33–47 GPU wall-hours at the 150/h gate floor, ≈ 12–18 at the expected 400/h. Work beyond the ceiling needs a new decision.

## 10. Ledger entries

At adoption: `decision v4_rl_brace_plan` (this revision, the U-RL answers, the review history); `deviation v4_press_fixture`; `deviation v4_step_episode_nominal_start`; `deviation v4_brace_executor_e1` (if U-RL-3 is yes); `probe v4_rl_planner_lp_probe` (the [P] and [R] numbers, flagged to be reproduced by S0.1). In R0: `decision v4_rl_task_manifest` (the frozen band, pulse, action set, split, de-duplication, seed map and executor setting, with the hash of `results/v4/rl/manifest.json`); one `deviation` per branch fired, naming the rows it invalidates. After the runs: results `v4_rl_r0_harness`, `v4_rl_r0_throughput`, `v4_rl_r0_signal`, `v4_rl_r0_reaction`, `v4_rl_r0_screen`; gate `v4_rl_R0`; results `v4_rl_r1_data`, `v4_rl_r1_model`, `v4_rl_r1_e1_size`; gate `v4_rl_R1`; results `v4_rl_r2_heldout`, `v4_rl_r2_benchmarks`, `v4_rl_r2_builds`; gate `v4_rl_R2`; decision `v4_rl_handback`.

## Files

To build: `tasks/brace_bandit.py`, `experiments/v4_brace.py`, `tests/test_brace_bandit.py`. To extend: `dual_arm_sim.py` (`Example.__init__`, `simulate` — fixture kernel after `clear_forces` —, `make_plan`, the CLI, `p0_row`, `JointBreaker.update` — two-sample records); `bracing.py` (`_record(..., tilt=None)`, `brace_at`, the `learned` dispatch before line 150); `scripts/run.sh` (test routing). Reused unchanged: `stability.py`, `blueprint.py`, `cell/contacts.py`, `tasks/ppo.py` (`mlp`, `RunningNorm`). Not built, by choice: multi-world batching and in-process reset (fallbacks under criterion 1); a kinematic hands-fit check; a PPO head; LP-label pre-training; worst-case relabelling; an on-policy no-LP arm; a `nearest` arm.

## 11. Adoption amendments for R0 (2026-09-30; supersede the text above where they conflict)

Adopted now: §0, §2 (task definition), §2.8 arms as used in R0, §4 Phase R0 with gate G-R0, the R0 rows of §5, the R0 items of §7 and §10 — with these changes from Codex round 2. Not adopted (draft until the G-R0 gate): §1's verdict tests, §3, §4 Phases R1 and R2, §6, the R1/R2 budget. Open items for that one revision: a cluster-aware test valid for the mean estimand (the structure-level sign-flip test needs sign symmetry and can over-reject), a frozen reproducible E1 sizing rule that selects whole structure blocks, and an executable job manifest that makes the ≈ 7,000-episode ceiling enforceable (worst-case counts reach ≈ 7,500).

- **A1 Benefit gate (criterion 4).** screened-best − **none** ≥ 0.20 (context-weighted point estimate), better than none in ≥ 5 of 16 contexts and worse in ≤ 1, and paired fixture-off excess failure of screened-best over none ≤ 0.10. screened-best − sham is reported as the attribution contrast, not gated. **STOP-B wording:** "the 12-candidate screen under the chosen executor setting did not meet the benefit gate" (with the measured differences against none and sham and the reaction table).
- **A2 Sham timing.** The sham run for a (context, draw) replays the matched braced run's measured staging duration: B's `stage` lasts as long as it did in the screened-best run of that context (recorded per run), not a fixed 5.6 s. Every row records B's transport onset and insert onset; the sham is valid for a pair only if its insert onset is within 0.1 s of the braced run's. No "contact, not staging time, explains the gain" statement is made unless that check passes on the pairs used.
- **A3 Throughput branch (criterion 1).** ≥ 150 episodes/h: pass. 60–149/h after the in-phase fixes: R0 continues (its ≈ 1,200–1,400 runs fit the approved budget), criterion 1 is recorded as failed, and the R1/R2 budget is re-derived at the gate and goes to the user. < 60/h: STOP-T.
- **A4 Distinct structures.** The signal set needs 40 distinct eligible train structures. With 24–39 after the placer screen: use all, report the reduced n. With < 24: extend the train split once with further unique structures from the same generator (test and validation keys untouched) and re-screen; still < 24: STOP-P.
- **A5 Reaction check contract (§2.7), frozen before any signal outcome.** Contexts: the highest-LP-u0 critical context of each of the first 4 train structures in split order that pass the placer screen. Scalar: the component of A's mean plateau force on the structure along the LP's expected reaction force for the SAME load case (λ = 1, zero lateral — recomputed for that case, not `expected_reaction_wrench`'s worst-of-5), divided by the LP force magnitude (denominator floor 0.5 N); moments about the LP brace point are reported. A run with an incomplete plateau is invalid and re-run once. The readout stores the signed plateau wrench (the existing per-brick maximum norms are not enough). E0 if the scalar is ≥ 0.5 in ≥ 3 of 4; otherwise E1; any E1 divergence → E0.
- **A6 Seed contract.** `hidden()` seeds numpy's PCG64 from the SHA-256 of a canonical JSON tuple (master seed, stage label, canonical structure key, original step index); replicate r is random-access (no dependence on job order); stage and round labels are fixed strings listed in the manifest; a test checks identical draws under reordered jobs and different worker counts.
- **A7 Scope of the R0 verdict.** R0's numbers are a feasibility screen with no CI-based claim. G-R0 is reviewed once by Codex; its outcome (pass, or STOP-A/B/H/T/P) goes to the user together with the one revision of R1/R2 if the work continues.
