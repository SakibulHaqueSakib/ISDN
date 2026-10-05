# Work log

Generated from `ledger.jsonl` by `scripts/worklog.py` — do not edit by hand.

201 entries. Later entries supersede earlier ones; retractions are marked.

## WP0 — Environment & platform verification

- **deviated** `docker`  ⚑ *needs a decision*
    - decision: native pinned venvs, no container
    - rationale: The Isaac stack is installed and working in ~/Codes/CAIRSS/Issac (a venv, not system python). Containerising now costs days and re-runs the risk R9 install that already broke this machine once.

- **problem** `two_interpreters`  ⚑ *needs a decision*
    - implication: D5 assumes cuRobo is callable for coarse motion. It cannot be imported in the interpreter that runs Isaac Lab, so M3 cannot run in-process with the sim.

- **result** `factory_forge_read`
    - implication: FORGE ships in this Isaac Lab tree, not just as a paper. D10 (force-budget conditioning) is forge_env_cfg.obs_order entry 'force_threshold' with contact_penalty_threshold_range randomisation; D11 (learned success predictor) is action[:,6] rescaled to [0,1] with a success_pred_error reward term and precision/recall logging. Both are to be reused, not written.

- **problem** `asset_root_404`
    - implication: Isaac Lab 3.0-beta2 asset URLs point at an Isaac 6.0 tree that does not carry the robot assets. Any script spawning a stock robot must rewrite 6.0 -> 5.1.

- **problem** `osc_task_frame`
    - implication: Isaac Lab's OperationSpaceController is unstable when given a pose_abs command against the default (identity) task frame, even with correct gains. The task frame MUST be placed at the target and the command sent relative to it, as scripts/tutorials/05_controllers/run_osc.py does. The docstring's '(x,y,z,w)' quaternion note is also stale -- the code is wxyz throughout.

- **problem** `bricksim_cli_rtx_404`

- **deviated** `v31_cpu_twin`

- **decided** `two_interpreters`
    - outcome: superseded in v3.1: coarse motion in the twin is IK + joint-space min-jerk in the same interpreter as the physics (runtime.goto_joint); cuRobo is not on the v3.1 path

- **decided** `docker`
    - outcome: v3.1: the twin runs from a pinned venv (requirements-twin.txt, env.lock [mujoco]); no container in either path

- **decided** `local_gpu_twin`
    - outcome: the v3.1 twin re-resolved on the GPU workstation (env.lock [mujoco]): same pins, torch 2.14.0+cu130. The PPO update runs on CUDA (tasks/ppo.py --device, checkpoints saved as CPU tensors); physics stays CPU MuJoCo 3.3.7, one process per core: runner --workers defaults to physical cores (16), PPO to 8 workers x 1 env (the batch of 8 R1-R4 used), eval_all runs its (policy, stage) jobs in parallel. Re…
    - finding: MuJoCo Warp 3.8.0.3 (Issac env) loads and steps the insertion model on the GPU, but a batched twin is a port, not a flag: noslip (runtime.Sim's press/brace phases) is not implemented, and tendon_limited and geom_conaffinity -- which the clutch model switches per env (static-friction lock, mated walls) -- are shared across worlds; curriculum stages 1-3 also change the model per episode

- **decided** `v4_plan_of_record`
    - outcome: plan of record v4 adopted: vision-driven, collision-free dual-arm assembly in standalone Newton (Docs/plan_v4.md). User's answers: U1 standalone Newton 1.2.1 in the Isaac 6.0 interpreter with ViewerGL; U2 measure, then decide 1x vs 2x at D1 from P1/P2 measurements; U4 decide after M1; every other U at its recommended default

- **deviated** `v4_platform_newton`
    - decision: U0 (CLAUDE.md rule amended), U1
    - rationale: user: collision-free arms (the twin's Panda links do not collide with each other), preferred Newton visuals, and arms driven by vision

- **decided** `v4_vision_only_definition`
    - outcome: U6: brick, plate and placement poses come from cameras only

- **decided** `v4_joint_scope`
    - outcome: U4: joint breaking (WP5 -> M2, E5) is decided after M1; until then the M1 clutch is a non-breaking weld pool. Alternatives then: yes (1-day contact spike + 3 days) or no (a perception-and-motion phase only)
    - correction: r2's rationale ('seated bricks carry 71-100 N under a speed-limited or ramped load', so the spike 'is expected to pass') was corrected on adoption: the ramp runs start an unseated brick (logic review round 3, verified against r_1x_fixed_ramp.txt)
    - rationale: bricks carry 71-100 N when seated first (then stepped), or pressed by a speed-limited stand-in; at 1x an unseated brick under a 0.5 s force ramp passes through at 71 N (ends -9.2 mm; -11.6 mm at 100 N), while at 2x the same ramp holds 100 N (-0.50 mm). WP5's 1-day spike is expected to pass at 2x and is uncertain at 1x

- **result** `newton_brick_press_tunnelling`
    - result: the plan-v4 r1 probe's 'a >= 2 N press pushes a brick through another' is a tunnelling artefact of a step body force on a free brick, not the Newton brick contacts' strength. Work-reviewer's reproduction (press.py, 29 result files; prototype brick model and solver settings, 60 fps x 16 substeps), stated as the corrected finding
    - finding: a step body force on a free 1.7 g brick starting 2 mm above seated tunnels from ~2-5 N at 1x (fixed lower brick: 2 N ends -2.4 mm, 3 N -8.9 mm, 5 N -18.3 mm, i.e. through); seated bricks (0.5 N, then a step to F) carry 71-100 N with 0.9-2.7 mm sink at 1x (fixed lower 0.91 / 1.17 mm; free lower or plate proxies 2.24 / 2.5-2.7 mm) and 0.45-0.6 mm at 2x; a 0.5 s force ramp on an unseated brick seats …
    - implication: every v4 press is arm-held and speed-limited, never a raw body force on a free brick; P1 measures F_seat, F_pt, sink under load and c_held with the arm's own press profile and the M1 clutch active; the scale decision (U2) has no planning-time lean

- **result** `curobo_08_runtime_compiled`
    - result: cuRobo 0.8 (checkout curobo/ @ 78fd485) compiles its CUDA kernels at runtime through cuda.core: curobo/setup.py builds pybind extensions only if CUROBO_USE_PYBIND == '1' (default '0', marked deprecated), and pyproject.toml depends on cuda-core[cu12]/[cu13]
    - implication: two_interpreters' option 'install cuRobo into the Issac env (torch 2.10/cu128 build must exist)' no longer applies: no torch-matched build is needed. Under the Isaac interpreter import fails only on missing modules (setuptools_scm, yourdfpy, numpy-quaternion, cuda.core), which the .vendor overlay supplies (U5). Still unverified until P3: warp 1.13 against code developed on 1.17, and cuda-core cu12…

- **probe** `v4_p1_smoke_finger_contact`
    - finding: the first P1 runs under plan v4 r3's protocol (scripts/10_newton_scale_probe.py smoke rows) showed four protocol faults and six probe departures; the reviewer reproduced each with independent runs. (1) Finger contacts: the FR3 finger shapes kept Newton's default ShapeConfig (ke 2500, kd 100 -> solref 20 ms, damping ratio 1) and have geom_priority 1, so MuJoCo uses the finger's solref alone at ever…
    - implication: the r3-protocol P1 rows that involve the fingers, (b), (d), (e), (f), are void; the (a), (c) and (g) rows involve no fingers and stay valid, so newton_brick_press_tunnelling and section 0 fact 6 stand. Plan v4 r4 (v4_plan_r4) changes finger contacts, the clutch gate, the P1 protocol and the contact monitor

- **result** `v4_p0_baseline`
    - result: P0, the prototype at HEAD (dual_arm_sim.py --viewer null --test, 1x, fixed feeder slots and plate, 3 repeats each of cube, arch, hollow_box, S3 under grasp_lp and lever_press, 24 runs): completed bricks per run cube 8/8 in all 6 runs; arch 10/11 in all 6; hollow_box 9/12, 10/12, 9/12 (grasp_lp) and 10/12 x3 (lever_press); S3 13/16 x3 (grasp_lp) and 14/16 x3 (lever_press; 2 of the 3 diverged, see v…
    - implication: the r3 contingency applies: WP1 fixes the brace executor and M0 uses lever_press until then. The brace-driven motion, the brace-independent failures and the S3 lever_press divergence are WP1 inputs, with an effort caveat on WP1's 1.75 days (plan section 5); M0(a) and P3 step 8 move their reference to P0' (v4_plan_r4, after the finger change)

- **problem** `v4_p0_s3_lever_press_diverged`
    - finding: in P0, 2 of 3 S3 lever_press runs stopped on RuntimeError: r0 'simulation diverged at t=145.58 s (step 14, B release, A park)' (8735 frames, 14 of 16 bricks placed) and r2 'simulation diverged at t=145.53 s (step 14, B insert, A park)' (8732 frames, 14/16); r1 did not diverge and ended on an AssertionError, 14 / 16 snapped (157.07 s, 9424 frames). Both diverged runs failed at S3 step 14, at t of a…
    - implication: the lever_press fallback brace is not clean on S3 either; WP1 must find the cause (plan section 3 P0 outcome, section 5 WP1 caveat)

- **decided** `v4_plan_r4`
    - outcome: plan of record v4 amended to r4 (Docs/plan_v4.md, revision block 'Revision r4'): finger contacts stiffened (section 2.1), clutch tested at rest in windows with dz in [-0.5s, +0.3s] mm (2.2), P1 protocol and gate rewritten (section 3 P1), contact monitor windows and 0.5 s merging (2.4), P0 recorded and P0' added (section 3 P0), R14 and U14 added. Effort: P1 +0.5 agent-day, P0' about 45 min GPU, tot…

- **deviated** `v4_finger_contact_stiffness`
    - decision: D3, D4
    - rationale: with the default, realised grip saturated at 1.2-2.0 N per finger whatever FINGER_KE, pads sank 0.9-1.6 mm and carry creep was 1.60 mm; with the r4 values at 1x grip is 6.9/13.2/24.9 N for 7.1/14.3/28.6 N nominal, pad penetration 0.09-0.24 mm, press slip 0.07-0.19 mm, carry creep 0.095 mm (v4_p1_smoke_finger_contact). Side effect: k_os 2.5-4.4 at 20 mm/s and F_cap 5 N, so v_press 2 mm/s is a pre-r…

- **deviated** `v4_clutch_gate_at_rest`
    - decision: D1, D2, D5 (band sanity gate)
    - rationale: +1.0s sat 0.1-0.37 mm below the 1x stud-top rest and admitted the 2x one, and transit passes scored; seated rest is about 0 to +0.18 mm, stud-top rest +1.1 to +1.37 mm (1x) and +1.84 to +2.69 mm (2x), so +0.3s clears both by >= 0.2s and >= 0.6s mm. Creep is bounded (<= 0.03 mm within a window). The raw jitter figure is linear creep (0.037-0.048 mm raw, <= 0.0022 mm detrended at 1x): under r3's P1(…

- **deviated** `v4_p1_protocol_r4`
    - decision: D2, D4, D5, D7, D8
    - rationale: the first probe runs showed the finger, clutch-gate and scoring faults in v4_p1_smoke_finger_contact; the smoke departures are blessed and one wording corrected. Effort P1 2 -> 2.5 agent-days, about 3.5 h GPU

- **deviated** `v4_contact_monitor_r4`
    - decision: D6
    - rationale: P0: the picked brick rests on the table during descend, grasp and lift (the implementer reported 172 spurious unintended events on one cube run without the window; that run was not saved); contiguous-only merging fragmented flickering contacts: 26-37 neighbour_rub events per cube run against 18 on every cube run with the merge, and arch grasp_lp's 453-507 displaced records against 35-40 events; th…

- **decided** `v4_agent_routing`
    - outcome: plan revisions go to a new plan-reviser agent (opus, high effort); the planner agent (opus, xhigh) is used only for start-of-phase long-horizon plans of record. Recorded because r4 was the first revision routed this way; the plan text says plan-reviser wherever a revision goes to the planner

- **probe** `v4_p4_brace_weld`
    - finding: P4 brace/weld probe (scripts/13_brace_weld_probe.py, parts A, B, C; outputs results/v4/p4_brace_weld/*). Part A (arch and S3 builds, current brace executor): arch step 10 arm-arm contact A-hand <-> B-arm mean 161 N during transport, 220-231 N from pre-insert to release, peak 442 N, correlation with welded-brick displacement 0.938; S3 steps 13/14 arm-arm peaks 173/243 N, 1026 N with a stiff weld; S…
    - implication: a brace can change an outcome essentially only at S3 step 13 above about 9 N (through a break); the r4 weld is too soft (F3), nothing breaks (F4), and the brace executor pushes far harder than the brace model assumes (F1, F2); motivates plan v4 r5

- **problem** `v4_brace_overforce`
    - finding: user observed braced builds bending and springing back; P4 traced it to three faults. F1: arm-arm contact between A's hand and B's arm during the brace is the dominant load (arch step 10 mean 161 N transport, 220-231 N pre-insert to release, peak 442 N, correlation 0.938 with welded-brick displacement; S3 13/14 peaks 173/243 N, 1026 N with a stiff weld). F2: the brace position target is 2 mm insid…
    - implication: the r4 statement that a brace cannot change an outcome once a brick is welded is contradicted (arch step 10 brace window: 51.4 mm peak, 21.9 mm remaining 1 s after retract); fixed by plan v4 r5 (r5-1 weld stiffness, r5-4 brace executor, r5-5 hand-off protocol and hands-fit)

- **decided** `v4_joint_scope_r5`
    - outcome: U4 = yes now: rigid, breakable joints in v4. WP5 splits: WP5a (stiff, breakable, grouped weld pool and its break model) is done now, before WP1, in the r5 J stages; WP5b (detent n*f_insert, v3.1 forces, contact spike, E5) still decides after M1
    - rationale: the user wants the joint model rigid and breakable now; WP5a needs no contact spike, since loads among welded bricks go through welds and only the new brick (3-6 N) and the arms touch the structure

- **decided** `v4_pipeline_gt_now`
    - outcome: image -> structure on ground truth now: a GT-first pipeline (stage PL): the user gives 1-2 drawings, the sim rebuilds the drawn shape from simulator poses (not vision); vision-only M1 unchanged

- **decided** `v4_plan_r5`
    - outcome: plan of record v4 amended to r5 (2026-09-30): weld stiffened (r5-1), structure collision group (r5-2), breakable clutch (r5-3), brace executor to grasp_lp semantics commanding no push (r5-4), acyclic hand-off protocol and executed-path hands-fit (r5-5), new stages J-pre, J-e, J-d (M0-J) and PL before WP1 (r5-6); gates J-a, J-b, J-c, J-pre, J-e, J-d, PL. M0, M0.5 and M1 slip about 4 agent-days (+ <…

- **deviated** `v4_clutch_weld_stiffness`
    - rationale: P4 part B: the r4 weld gives a column lateral stiffness of 0.70-1.35 N/mm and a 15 mm set after 50 N; ungrouped with the new solimp: column lateral 23-24 N/mm, vertical about 560 N/mm, cantilever 34 / 65-70 N/mm, set <= 0.03 mm, stable at 16 and 32 substeps. Gate J-a (grouped): column lateral secant >= 20 N/mm and vertical >= 300 N/mm at 1-50 N, cantilever >= 30 N/mm, set <= 0.05 mm after 50 N

- **deviated** `v4_structure_collision_group`
    - rationale: drops (i) bearing/friction between same-course welded neighbours, (ii) non-interpenetration among welded bricks (bounded by gate G3, M2 <= 1.0 mm), (iii) load sharing through contact; external loads on welded bricks (the pressed brick, B's and A's fingers) remain contacts. Readout with collisions off <= 0.12% (measured / reviewer-reported); runtime shape_collision_group.assign without re-capture (…

- **deviated** `v4_clutch_breakable`
    - rationale: U4 = yes now (WP5a). Restricted model accepted by the user (U-r5-5); J-c quantifies what frame sampling misses (pulses at u = 1.5 for 1-60 substeps, substep vs frame readout) and reports shear/torsion

- **deviated** `v4_build_ends_at_first_failure`
    - rationale: the grouped weld-only idealisation (v4_structure_collision_group) is valid only before the first failure; a brick is welded only onto grouped welded supports and post-break states are never scored (Codex round 1 finding 2). U-r5-2

- **deviated** `v4_brace_executor_r5`
    - rationale: F2 and F1 (v4_brace_overforce). The hold is measured, not assumed: gate G5 (max per-brick |F_A| <= 10 N [assumed threshold], realised grip >= 0.8x nominal over the closed-hold interval, contacted set within gripped_bricks); no 'cannot overload' claim. Newton plans change (brace_force_N 14.3, mu, expected_reaction_wrench, predicted_util_braced; feed-forward fields emitted but unused)

- **deviated** `v4_handoff_protocol_r5`
    - rationale: Codex round 1 found the zone deadlock/park; round 2 confirmed the acyclic protocol with the real Arm class and ideal tracking (arch 5,940, S3 9,104, all-braced S3 13,355 updates, no cycle). Margin: 12 mm provisional (P3 step5.json padding_needed_mm 12.08 at time scale 1.0 is cuRobo evidence), then max(5 mm, p99 + 2 mm) of the waypoint executor's measured deviation from J-pre (empirical padding); r…

- **deviated** `v4_milestone_order_r5`
    - rationale: U-r5-4 (J before WP1); M0, M0.5 and M1 slip about 4 agent-days (+ <= 1 day if the WP1 seating pull fires; pre-registered branch of J-pre). v3.1 frozen and not comparable; P0/P0' historical; P3 step 8 used P0' cube inserts and is re-run on P0'' before use; P3 steps 5/7 tracking/padding transfer only to cuRobo paths, steps 1-4 and 6 unaffected; P1(f) rows void, re-run under r5; P2 unaffected

- **result** `v4_j_s0`
    - result: plan v4 r5 step S0 (diagnostic: is there a crown-failure path that needs no brace?), three unbraced runs of scripts/13_brace_weld_probe.py part A with --strategy none, n = 1 per config (descriptive). (1) arch, none, default weld eq_solimp (0.95, 0.99, 0.001, 0.5, 2.0): placed 11/11, completed, error null, no divergence, 6134 frames, sim 102.23 s; crown b_010 seated: final_error 0.29 mm, dz -0.24 m…
    - implication: the arch crown miss does not occur without the brace, so it is caused by the brace execution (consistent with F1, the B-arm to A-hand contact, v4_brace_overforce; F1 is not separately proven as the cause by S0). S3's unbraced b_015 loss (dz -72.06 mm, tilt 90.03 deg, matching P0' S3 grasp_lp r0's b_015 loss exactly and r1's dz; r2 lost it differently (-68.68 mm, 180 deg), per v4_p0_baseline_stiff_…

- **decided** `v4_r5_phase_gated_execution`
    - outcome: plan v4 r5 is executed in phases > segments > subtasks; reviews and plan revisions only at phase gates, in-phase check failures logged and carried to the gate. No criterion changed. Phase 1 Joint model: S1 stiff weld + grouping (done, commit a890b68), S2 break model, S4a recording; gate J-a + J-b + J-c together. Phase 2 Brace and build: J-pre one-pass build + margin measurement, S3 brace / hand-of…

- **result** `v4_j_a_weld_grouped`
    - result: J-a PASS (plan v4 r5, segment S1; probe part B --group, weld solimp 0.9999, 16 substeps, plate + bricks in STRUCT_GROUP, torquescale null). Secant stiffness min over F = 1, 5, 10, 20, 50 N: column lateral (x) 23.128 N/mm (range 23.128-23.252; gate >= 20); column vertical up 560.408 and down 560.408 N/mm (ranges 560.408-561.346 up, 560.408-561.58 down; gate >= 300); cantilever down 32.303 N/mm (ran…

- **result** `v4_j_b_readout_carried`
    - result: J-b (grouped readout, solimp 0.9999, 16 substeps; tolerance 1 % or 1 mN (F) / 0.1 mN.m (M_patch) absolute, u 1 %): statics_pass true on the pre-registered single end-of-frame reading (statics_failing empty; jacobian_pass_local and jacobian_pass_world true; bridge_pass true), but statics_pass_all_30_samples false: 22 of 2250 sample readings over 30 frames fail (worst sample F error 19.5841 mN, M er…

- **result** `v4_j_b_chatter_diagnosis`
    - result: Diagnosis of the per-frame weld-force error in J-b (question: A micro-vibration or B readout defect, pre- vs post-integration pose). Not physical vibration: static and dynamic force balance residuals are equal (max per brick, mN: column 10.974 static / 10.9739 dyn on brick2; cantilever 22.3988 / 22.3972 on brick2; bridge 24.6456 / 24.6493 on brick2), m*a_com is at most 0.0106 mN; brick motion is a…

- **result** `v4_j_c_break_rule`
    - result: J-c PASS on its pre-registered items (i), (ii), (iii), (v) (plan v4 r5, segment S2; test fixture: body forces on weld-held bricks in grouped static scenes, solimp 0.9999, 16 substeps, break rule dual_arm_sim.JointBreaker with U_BREAK 1.0, BREAK_SAMPLES 2, BREAK_SETTLE_S 0.1, raw end-of-frame u from mjw_data.efc.force at the solve-time pose). (i) 2x4 pull-off, ramp 60 N/s: first break at 92.0 N (fr…

- **GATE** `v4_phase1_joint_model` → **changes_requested (J-b); J-a pass, J-c pass**
    - result: changes_requested (J-b); J-a pass, J-c pass
    - status: Phase 1 'Joint model' gate (plan v4 r5; phase commits a890b68, 4ad91a7, 0dd4f8b; base f227fd4), reviewed once by Codex (adversarial review of f227fd4..HEAD). Code approved: JointBreaker signs, moment shift, rotation, settle and the 2-consecutive rule; in-place grouping; build ends at first failure; record.py read-only. J-a PASS (v4_j_a_weld_grouped). J-c PASS on its pre-registered items (v4_j_c_br…

- **deviated** `v4_jb_accepted_with_known_cause`
    - rationale: J-c shows the break decision lands within 2 % of the model near threshold (2x4 pull-off +1.77 %, cantilever -1.11 %) and no false break at analytic u 0.98 on the full 2x4 patch (0 of 120 frames >= 1; the cantilever, with its own +0.025 bias, reads 1.005 there); the cause is solver-side weld-force error of tens of mN (GPU solve exits after 1-2 iterations), not a readout defect (v4_j_b_chatter_diagn…

- **result** `v4_j_pre`
    - result: J-pre (plan v4 r5 Phase 2, segment 1; commit 245697b; scripts/13_brace_weld_probe.py part A, the r5 breakable joints, the legacy brace executor, grasp_lp, strategy weakest_joint, n = 1 per structure, descriptive). cube: placed 8/8, built 8, brace_required 0, no breaks, no failure, u peak by step at most 0.0339 (step 4), sim 74.88 s, 4493 frames. hollow_box: placed 12/12, built 12, brace_required 0…

- **result** `v4_brace_executor_seg2`
    - result: Phase 2 segment 2 (S3.1-S3.4): the grasp-and-hold brace with contact guard and the acyclic hand-off protocol (commit 73a4bd5), plus brace grip and mu plumbed into the LP (commit de853f3), run once on arch (new executor, and --legacy-brace as the control) and once on S3 (new executor), n = 1 each, descriptive. arch, new executor: placed 10/11, built 9, joint_break at step 10 (failure joint_break, c…

- **problem** `v4_s3_step1_loss_divergence`
    - finding: S3 with the r5 joints loses b_001 at step 1 (failure gate_miss, failed_at_step 1, built 1; snap lateral 240.87 mm, tilt 90.03 deg at 18.583 s in J-pre) and the simulation then diverges at t=37.42 s (step 3, B release, A park) in J-pre (results/v4/j/pre/j_pre.jsonl). The seg2 S3 run with the new brace executor (results/v4/j/seg2/S3_new.jsonl) shows the same: b_001 lost at step 1 (lateral 206.86 mm,…

- **decided** `v4_brace_to_rl`
    - outcome: hand-planned bracing stops; bracing goes to an RL workflow. Dropped from r5: S3.5 dry-run hands-fit, J-e, the G1/G5 brace gate items, and the fixed / none / legacy brace controls. Stays in the code, unused by default: the committed grasp-and-hold brace executor and the acyclic hand-off protocol (commit 73a4bd5), the grip and mu plumbing (de853f3), and --legacy-brace. Bracing becomes a separately p…
    - rationale: J-pre and seg2 (v4_j_pre, v4_brace_executor_seg2): the hand-planned brace does not save the arch (legacy u 2.0862 in J-pre; new grasp-hold brace and hand-off protocol u 1.1365, still joint_break at step 10, 2 arm_arm events, realised grip 25.5 / 7.52 N against 14.3 N nominal, contact with 5 bricks while 2 were gripped), and the LP brace helps little at the executor's grip (S3 step 13 braced 1.9688…

- **decided** `v4_pipeline_unbraced_now`
    - outcome: the image -> structure pipeline runs unbraced now: no stabilizer (strategy none) and the r5 breakable joints; an overloaded joint breaks and ends the build (v4_build_ends_at_first_failure). This is the pipeline default until the RL bracing phase delivers a policy
    - rationale: J-pre: cube 8/8 and hollow_box 12/12 need no brace and record no break (v4_j_pre); ledger v4_j_s0 (before the breakable joints) built the arch 11/11 with strategy none under both weld settings. Not shown by any file: an unbraced arch or S3 build under the r5 breakable joints; the planner's unbraced utilisation is 0.9458 for the arch (step 10) and 3.4311 for S3 (step 13), so an unbraced break is po…

- **deviated** `v4_r5_phases_revised`
    - rationale: v4_brace_to_rl and v4_pipeline_unbraced_now (user direction 2026-09-30): hand-planned bracing is dropped from v4's critical path, so J-e, S3.5, the brace gate items and the brace controls leave Phase 2; the image pipeline moves up into Phase 2

- **decided** `v4_rl_brace_plan`
    - outcome: Phase R0 (feasibility) of the RL bracing plan (Docs/plan_v4_rl_brace.md, Phase 3) is ADOPTED with amendments A1-A7 (plan §11); Phases R1 and R2 remain a DRAFT, to be revised once at the G-R0 gate and only if R0 passes. Adopted: plan §0, §2 (task definition), §2.8 arms as used in R0, §4 Phase R0 with gate G-R0, the R0 rows of §5, the R0 items of §7 and §10. Not adopted (draft): §1 verdict tests, §3…
    - rationale: Review history: the planner wrote the plan 2026-09-30; Codex round 1 CHANGES REQUESTED (12 findings; the planner revised once); the user asked for a second Codex pass and approved the compute budget (U-RL-7); Codex round 2 CHANGES REQUESTED (6 blocking, 1 for R0); escalated to the user after two rounds

- **deviated** `v4_press_fixture`
    - rationale: on the tested unbraced builds the native regime gives a brace policy almost no break signal (cube 8/8, arch 11/11, hollow_box 12/12, no breaks; u_peak cube 0.033-0.036, arch 0.057-0.059, hollow_box 0.318-0.424; plan §0 fact 1), so training needs a load source; the LP predicts a where problem under a design press (v4_rl_planner_lp_probe)

- **deviated** `v4_step_episode_nominal_start`
    - rationale: a build reaching every scored state costs full-build time per sample; the step-episode makes per-context sampling affordable within the approved budget

- **deviated** `v4_brace_executor_e1`
    - rationale: at about 0.5 m reach A holds with about 1.6 N/mm at the hand against 23-24 N/mm (column lateral) and 34-70 N/mm (cantilever) for the welded structure, so the committed hold is expected to carry a few percent of the load (plan §0 fact 3); R0 measures the reaction before anything else is spent

- **probe** `v4_rl_planner_lp_probe`
    - result: Planner [P] and Codex [R] LP numbers, exactly as in the plan §0 facts 2, 7 and 8. S3 step 13: unbraced u0 3.43; best braced 1.97 with the committed mask [P], 1.56 with the relaxed mask [R]; at half press 0.49 / 0.08. S3 step 14: 1.96 -> 0.44 / 0.00. Arch step 10: 0.95 -> 0.95. The LP costs 2-4 ms per solve. Candidates per critical step: 40-178 on S3, S5 and the arch; median 28 on a corbel family (…
    - finding: unsaved CPU probe (planner [P] scripts not in the repo; [R] is Codex's recomputation), to be reproduced by segment S0.1 (--self-check writes results/v4/rl/r0/lp_probe.json) before any report cites it. The LP lets the brace supply any bounded wrench, so the braced values are optimistic, not executed benefits

- **result** `v4_phase2_acceptance`
    - result: Phase 2 segment (b), acceptance runs (commit 8ec883b): unbraced, ground-truth poses, breakable joints, 3 repeats each. cube 8/8 x3, arch 11/11 x3, hollow_box 12/12 x3; 0 breaks in all 9; failure null. Per shape [u_peak_max range; M2 peak excursion range mm (worst brick); M3 residual max mm; rtf range]: cube [0.0329-0.0362; 0.0234-0.0241 (b_004); 0.00010 (8.5e-5 to 9.8e-5); 1.442-1.446]; arch [0.05…

- **problem** `v4_pipeline_pyramid_inaccessible_grasp`
    - finding: The user's drawing (blueprints/user/pyramid_front.png + pyramid_side.png, --width 5; layers 5x5, 3x3, 1x1; 11 bricks; plans/sim_pyramid_none.json) run live by the user: steps 0 (b_000, 1x1) and 1 (b_001, 2x4) placed (final_error 0.23 / 0.19 mm, dz -0.37 / -0.31 mm, tilt 0.08 / 0.01 deg); step 2 (b_002, 1x4) gate miss (failure_mode gate: lateral 0.12 mm, dz +7.84 mm, tilt 0.9 deg, sim_time 26.98 s)…
    - implication: Phase 2's gate is met on the acceptance shapes (free grasp faces at every step, v4_phase2_acceptance) and is NOT met on the user's drawing (G7: built 2 of 11). Filled layers cannot be built with the side-pinch grasp. Carried to the Phase 2 gate; addressed by the placer track (v4_placer_rl_primitives).

- **decided** `v4_placer_rl_primitives`
    - outcome: the placer arm is trained with RL for placement in tight spaces between several bricks, and the pyramid (with other filled-layer shapes) goes into the RL training set. RL controls placement primitives: a new drop-and-press primitive (release above the neighbours, then press the brick down from the top) is added, and RL chooses per brick how to place: grasp-insert vs drop-and-press, release height,…
    - status: a plan of record for the placer track is being written by the planner, not yet adopted; the Phase 2 gate review by Codex is pending. The pyramid becomes a training and reference structure of the placer track.

- **decided** `v4_session_2026_09_30_handoff`
    - outcome: End-of-day state on branch v4-newton-vision (HEAD after this entry; work commits fd000b7..3ce7d90). DONE: plan v4 r5 adopted (fd000b7); S0 (f227fd4); workflow changed to phases > segments > subtasks with reviews and revisions only at phase gates (4c8cea7); Phase 1 Joint model closed: stiff grouped clutch welds (a890b68), breakable joints with build-ends-at-first-failure (4ad91a7), --record-all (0d…
    - rationale: The user ended the session ('pack up the current progress and log everything and thats all for today'); two background agents were stopped mid-work (the placer planner, by the user; the S0.1 implementer, by the orchestrator).

- **result** `v4_rl_r0_s01_done`
    - result: RL bracing R0 segment S0.1 (task library) finished and verified, commit cf2b27c. Bug fixed: the no-brace feature vector had 38 numbers vs NF 37 (hand-counted zero padding in tasks/brace_bandit.py _candidate_features); it is now derived from the FEATURES list. Added test_hidden_job_order_and_workers (A6: identical hidden() draws under shuffled job order and a 3-worker pool). tests/test_brace_bandit…

- **result** `v4_phase2_preflight_pinch_blocked`
    - result: Phase 2 close-out (commit 3fe7a44), the pending small fix of v4_session_2026_09_30_handoff item 4. planner.pinch_blocked(brick, placed, grasp) counts same-layer placed cells flush against the two pinched faces under the finger window (FINGER_W 17.5 mm about the grasp centre, tcp_offset_m included); dual_arm_sim.make_plan prints a preflight WARNING listing those steps and stores them in preflight.i…

- **decided** `v4_place_plan_draft`
    - outcome: The placer RL plan (user decision v4_placer_rl_primitives) is drafted by the orchestrator as the user asked ("dont do planner just write yourself"): Docs/plan_v4_rl_place.md. Feasibility phase P0 first: a drop_press primitive (release above the flush neighbours' studs, then press the brick down with closed fingertips) behind a --place-json flag; a 16-setting screen of grasp height, release margin,…
    - status: draft, not reviewed, not adopted; shown to the user before review.

## pre-WP0 — Kinematic slice (pre-gate exploration)

- **result** `kinematic_slice`
    - implication: §3.4 accessibility (cite BricksToBots) is already binding at 7 bricks with a stock Franka hand. Feeds the WP3 on-failure branch: grasp selection (grasp.face_pair is emitted but ignored) or the clutch tool.

## WP1 — Joint mechanics

- **result** `bricksim_cloned`
    - implication: The stability/force query the bracing contribution needs is available out-of-sim as a subprocess; the dynamic joint mechanics are not separable from Isaac Sim.

- **problem** `bricksim_stack_mismatch`  ⚑ *needs a decision*
    - implication: BrickSim cannot be imported into this project's Isaac env. It installs its own uv venv with a second copy of Isaac Sim (5.1). No Newton support anywhere in the repo -- 'newton' appears only as the force unit. Adopting BrickSim's dynamic path therefore means reverting D2 to PhysX, or running two simulators.

- **decided** `omniverse_eula`
    - outcome: OMNI_KIT_ACCEPT_EULA=1, scoped to the probe invocation
    - rationale: BrickSim's own CI (.github/workflows/ci.yml) uses the same variable; the workstation already runs Isaac Sim 6.0 under the same agreement.

- **branch** ``  ⚑ *needs a decision*
    - decision: deferred to human -- see open_question wp1_platform
    - implication: The report frames the branch as 3a-if-it-batches / 3b-otherwise. The measurement says 3a on throughput. Contribution (2) -- 'GPU-batched snap mechanics at RL scale' -- is separately weakened: a Warp kernel would optimise 0.2% of the step. The remaining question is platform, not performance.

- **problem** `probe_harness_bug`
    - implication: An in-process sweep would have reported 'BrickSim cannot build 4 envs' -- the exact wrong conclusion, and one that would have sent the project down the Warp-kernel path on a harness bug.
    - fix: one Kit process per env count (scripts/01_bricksim_sweep.sh); re-measured 4 envs at 309 env-steps/s with no hang.

- **decided** `wp1_platform`
    - outcome: branch 3a -- adopt BrickSim as the sim core

- **result** `static_solve_built`
    - implication: clutch_utilizations is the per-connection load fraction -- the §3.6 'internal force distribution' query the bracing contribution needs, available from a 0.4 ms subprocess with no simulator. The highest-utilisation connection IS the predicted failure joint. This replaces planner.py's lever-arm model.
    - next: convert our grid structures to BrickSim's topology JSON (bricksim.topology.legolization / bricksim-convert-topology) and check that the highest-utilisation joint on S3 is the cantilever joint the lever model predicts.

- **result** `threshold_comparison`
    - implication: The report's §2.3.3 numbers are tighter than BrickSim's defaults on every condition except yaw. WP2 must decide per condition whether to tighten BrickSim to the spec or adopt BrickSim's value with a recorded reason. PreloadedForce 3.5 N also has to be reconciled with the report's 8-15 N pull-off per stud -- they are different quantities and must not be conflated.

- **deviated** `vendored_patch`
    - decision: patched BrickSim's static_solve to accept applied loads
    - rationale: BreakageInput already carries per-part linear (J) and angular (H) impulses; only the CLI hard-coded gravity. Exposing them is ~100 lines and is the difference between answering 'most loaded by self-weight' and 'most loaded by the insertion press', which is the §3.6 query the contribution rests on.

- **problem** `g1_harness_false_pass`
    - implication: a motion test that reads USD while stepping without render silently passes no matter what the simulation does. Every future dynamic test in this project reads poses from the physics view (SingleRigidPrim) and ships a liveness control alongside.
    - fix: tests/test_joint.py pose_of() now reads SingleRigidPrim; control_free_brick_falls is asserted first and gates the meaning of everything after it.

- **result** `per_connection_wrench_available`

- **result** `clutch_forces_exposed`

- **problem** `joint_model_uncalibrated`
    - implication: every force-dependent result computed against BrickSim's defaults is meaningless until WP2 calibrates PreloadedForce. This includes the S3 single-arm claim below.

- **problem** `g1_pulloff_contaminated_in_session`
    - implication: a dynamic pull-off run after other tests in the same process reads 56% low, because released bricks from earlier articles land on the one under test and end its ramp early. Any force measurement in this project must run one-per-process.
    - fix: tests/test_joint.py no longer measures the pull-off itself; it reads the isolated sweep (scripts/02_dynamic_calibration.jsonl) and checks self-consistency plus the recorded repeatability.

- **CUT** ``

- **decided** `determinism_criterion_na`
    - outcome: NOT APPLICABLE under branch 3a
    - rationale: the criterion was written for branch 3b, where this project writes its own Warp kernel and must prove it batches identically. We took 3a: BrickSim owns the solver, so 1-vs-N determinism is an upstream property we neither control nor ship. Recording it as N/A rather than leaving it open indefinitely.

- **decided** `double_mate_rescoped`
    - outcome: guarded on our side; upstream behaviour recorded as a limitation

- **decided** `v31_clutch_model`
    - outcome: weld pool + stud-interference tendon (Coulomb static lock + kinetic friction n*f_insert) + 2.3.3 gate; break when the closed-form plastic utilisation >= 1 for 5 ms

- **decided** `bricksim_stack_mismatch`
    - outcome: superseded in v3.1: BrickSim's joint is re-implemented in the twin (sim/joint_model/clutch.py + capacity.py), matching static_solve's per-connection LP to 1e-15; BrickSim remains the GPU-path reference

- **decided** `bricksim_batching_scaling`
    - outcome: v3.1: the batching branch is moot on CPU; contribution (2) (GPU-batched snap mechanics) is not claimed. Contributions reported: (1) the bracing/insertion relationship (A6), (3) joint-model fidelity (A10)

- **problem** `v31_clutch_symmetry_bug`
    - finding: the twin's joint model expressed each connection in the LOWER brick's actual frame; a support mated 180 deg from nominal (the wrist yaw is chosen by IK margin) mirrored every off-centre patch: the gate saw 32 mm of error on a seated bridging 2x4 (never mated), the weld target sat 32 mm away, and the interference tendon was anchored under the wrong end
    - fix: symmetry-reduced frames (theta per brick) for gate, weld, tendon sites and wrenches; step trials never saw it because pre-placed bricks are never flipped

## WP2 — Joint calibration — static solver

- **result** `calibration_transfer_function`

- **problem** `static_calibration_does_not_transfer`
    - implication: the static QP and the live PhysX path do NOT agree on release force. WP5 trains in the dynamic path, so the dynamic number is the one that must be right; the static fit of 18.832 N would give the RL a joint roughly 2.4x weaker than intended.

- **result** `dynamic_repeatability`
    - finding: the dynamic release force is highly reproducible (0.5% spread) when the scene is clean, so the earlier scatter was harness contamination, not physics. Released bricks leave at speed -- tens of newtons on a few grams -- and land on later articles.

## WP2b — Joint calibration — live sim

- **result** `clean_dynamic_sweep`

- **problem** `total_force_ceiling`  ⚑ *needs a decision*
    - finding: dynamic release force saturates at roughly 88-90 N per CONNECTION regardless of how many studs are mated. Per-stud capacity therefore falls as bricks get larger rather than staying constant -- the opposite of how a real clutch behaves, where capacity scales with stud count.

- **decided** `two_path_calibration`
    - outcome: joint_calibration.json now carries SEPARATE preloads per path: static_solve 18.832 N, live_sim 120 N
    - rationale: both produce ~11.3 N/stud on a 2x4, but in different solvers. Applying the dynamic value to static_solve would model a joint 6.4x too strong (72 N/stud) and would corrupt every bracing and stability query.

- **decided** `total_force_ceiling`
    - outcome: not carried into v3.1: the twin's capacity is per stud by construction (utilisation = plastic LP of the patch); the ceiling was a property of BrickSim's live-sim joint, recorded as a fidelity difference

## WP3 — Build planner & bracing

- **deviated** `target_structures`
    - decision: hand-authored S1/S2/S3 instead of StableLego samples
    - rationale: dataset not local; the planner reads a plain brick list and is agnostic to the source.

- **deviated** `repo_layout`
    - decision: flat planner.py rather than planner/{voxelize,tiler,stability,accessibility,sequencer,bracing}.py
    - rationale: ~300 lines total; splitting into six modules before any of them has a second caller is scaffolding.

- **result** `weakest_joint_crosscheck`

- **problem** `static_solve_gravity_only`  ⚑ *needs a decision*
    - implication: §3.6 needs the force distribution under the INSERTION WRENCH, not under self-weight. As shipped, the tool cannot answer that question.

- **result** `bricksim_crosscheck_wired`

- **result** `s3_fails_single_arm`
    - finding: the unbraced S3 cantilever reaches clutch capacity at about 2 N and is at 12.7x capacity under a real 35.6 N insertion press -- it fails single-arm at roughly 6% of the force the task requires. S3 is a valid subject for ablation A6.

- **deviated** `s3_not_failing_when_calibrated`  ⟵ *retracts ledger entry s3_fails_single_arm (2026-09-20T19:25:00Z)*  ⚑ *needs a decision*
    - correction: the earlier finding that S3 fails single-arm at ~6% of the required force was an artifact of BrickSim's uncalibrated default preload of 3.5 N, not a property of the structure. The §WP3 preflight requirement is NOT met.

- **deviated** `s3_failure_restored`  ⟵ *retracts ledger entry s3_not_failing_when_calibrated (2026-09-20T21:18:00Z), which itself…*
    - status: S3 and S3L are both valid A6 subjects; S3L carries more margin.
    - correction: the previous retraction used preload 100 N, derived from the CANTILEVER prying ratio (capacity ~ preload/10.2) rather than the pure-tension pull-off ratio (0.6 x preload). Preload 100 N is 60 N/stud, four times above the top of the literature envelope. At the correctly measured calibration S3 fails single-arm across the whole 8-15 N/stud envelope, so the §WP3 preflight requirement IS met and the o…

- **deviated** `s3_does_not_fail_single_arm_in_sim`  ⟵ *retracts the suggestion in first_trial_all_structures that S3's tip-press failure was con…*  ⚑ *needs a decision*
    - implication: ABLATION A6 HAS NO SUBJECT until a structure demonstrably fails single-arm under the press the inserter really applies. Either the inserter must press with the specified force, or the structures must be made weaker, or the A6 premise has to be restated in terms of the achievable press.
    - correction: the earlier tip-press failure was an accessibility/grasp artifact, not physics. With the grasp fixed, S3 assembles unbraced without difficulty, so the live sim does NOT reproduce the static prediction.
    - next: measure the actual press force the scripted inserter develops, then re-examine which structures fail under it

- **deviated** `image_blueprint_reinstated`
    - decision: blueprint.py: silhouette images (front, optional side/top) -> visual hull on the 8 x 8 x 9.6 mm grid -> per-layer tiling that bridges overhangs first and staggers seams
    - rationale: the user wants the structure blueprint prepared from images, simple shapes first (cube, hollow box, arch). D15 deleted image/text->structure as a RESEARCH CONTRIBUTION; here it is only the pipeline's input stage and is not claimed.

- **problem** `brace_pose_on_load_side`
    - finding: brace_for pressed on the braced brick's CENTRE. On S3 that is y=-16 mm while the pier edge (the pivot) is at -24 mm: the brace sat on the load side and added to the prying.
    - fix: weakest_joint now returns the stud row of the joint's patch farthest from the load (S3: -36 mm, over the pier); brace_for presses there, on whatever brick is topmost. The 'nearest' baseline presses its brick's free cell farthest from the placement.

- **result** `grasp_accessibility_and_order`
    - result: grasp_for picks the closing axis whose fingers touch the fewest placed bricks (FR3 fingertip 17.5 mm wide: on a 16 mm face it grazes diagonal and in-line neighbours). Order stays layer-raster; S1/S2/S3/S3L plans keep their order.
    - finding: a 2-stud ring around a 2x2 hole cannot be built by a stock parallel gripper in any order (a fingertip does not fit a 16 mm hole); the hollow-box sample uses 1-stud walls. Sliding the grip off-centre to dodge a graze was tried and removed: on a 16 mm end face it lets the brick swing ~24 deg about the pad axis.

- **deviated** `v31_benchmark_structures`

- **CUT** ``

- **decided** `v31_stability_lp`
    - outcome: stability.py: StableLego-style global force balance as an LP (scipy HiGHS): minimax joint utilisation, then least total utilisation + brace effort; weakest joint ties go to the joint nearest the load

- **decided** `v31_brace_grasp`
    - outcome: the stabilizer GRASPS the structure (parallel jaw across x faces, 45 deg lean away from the placer); brace = friction-bounded wrench per gripped brick (shares by pad area, >= 4 mm pad height to count), pad couple mu*2*grip*5 mm, 60 N arm limit, as friction octagons

- **deviated** `v31_accessibility`

- **decided** `static_solve_gravity_only`
    - outcome: resolved: stability.py solves the global force balance UNDER THE INSERTION WRENCH (n x f_insert per patch + +-3 N lateral, 3.6 step 1) with braces as bounded wrenches

- **decided** `s3_does_not_fail_single_arm_in_sim`
    - outcome: resolved: the v3.1 S3 (corbelled bridge) fails single-arm in the twin under the press the inserter applies (step 12: 0/20 unbraced, joint-break cascades) -- see results/a6

- **decided** `s3_not_failing_when_calibrated`
    - outcome: resolved by the v3.1 S3 redesign (single-arm util 3.43; fails in the twin)

- **decided** `v31_sequence_press_point`
    - outcome: 2x2 bricks first within a layer; force-aware press point (stability.best_press_point)

- **deviated** `v31_safety_factor`

## WP4 — Motion stack & scripted baseline

- **result** `pipeline_runs_end_to_end`

- **problem** `placements_do_not_stick`
    - diagnosis_so_far: ["the brick IS grasped and lifted (179-183 mm), so this is not a grasp failure -- the first hypothesis, and it was wrong", "the gate reports essentially ZERO press force (-0.01 N against a required 1.0 N default), so the press is not loading the joint", "the first placement failing poisons the rest: connections 2-5 target bricks that should already be in the structure but are still lying in the wo…
    - next_step: instrument the assembly target pose against the achieved pose during the press -- the demo's press is a 0.5 mm position offset driven by RmpFlow, which is reactive and may simply not generate force if the commanded pose is not actually inside the surface

- **problem** `usd_poses_stale_headless`

- **decided** `rendering_path_not_taken`  ⚑ *needs a decision*
    - outcome: NOT TAKEN without the human
    - rationale: isaacsim_rtx_compat pins SHA-256 for both the package (_PKG) and the extracted shared object (_SO) and fails closed. Pointing it at the newer package means replacing a supply-chain integrity check on a binary that is then loaded into the process as a Vulkan layer. That is the user's call, not mine.

- **result** `scripted_baseline_works`
    - significance: §WP4 is the schedule's load-bearing wall and the scripted end-to-end pipeline now runs: our planner's structure and order, BrickSim's grasp and assemble skills, our episode log. This is also the residual base controller WP5 builds on.
    - fix: orchestration/run_plan.py injects a SingleXFormPrim subclass whose get_world_pose() reads the PHYSICS view, and wraps rmpflow.get_end_effector_as_prim the same way. That redirects every pose read BrickSim's skills make, at once, without editing BrickSim's source.

- **result** `first_structure_runs`
    - next: grasp reliability is the gap between 75% and the 90% G3 needs; the demo's grasp picks a top-down antipodal pose on the shorter edge and does not retry or re-plan when the brick does not leave the table

- **result** `first_trial_all_structures`
    - gap_to_G3: S2 at 75% against a 90% bar; the deficit is entirely grasp reliability

- **result** `grasp_reliability_fixed`

- **result** `failures_moved_to_assembly`
    - finding: with grasps reliable, every remaining loss is an assembly that does not connect after a successful grasp and press.

- **result** `alternate_grasp_face`

- **problem** `verification_measured_too_early`
    - implication: every WP4 success rate quoted before this point may understate the truth, because placements were verified the instant the skill returned.
    - fix: verify after 90 physics steps (~1.5 s), matching §WP7's 'verify 200 ms after release' in spirit but with margin

- **result** `scripted_baseline_current`
    - gap_to_G3: S2 at 71% against a 90% bar. S1 and S3 meet their bars on a single trial.

- **result** `press_force_measured`

- **problem** `assembly_thresholds_never_applied`
    - fix: apply the calibrated AssemblyThresholds alongside the BreakageThresholds

- **problem** `inserter_cannot_reach_spec_press`

- **problem** `positional_press_saturates`  ⚑ *needs a decision*
    - status: the scripted baseline assembles reliably at BrickSim's default 1 N gate (S1 6/6, S3 5/5, S2 5/7) but CANNOT assemble at the specified 8.9 N gate (S1 0/6).
    - finding: pressing five times deeper buys barely more force. The RmpFlow positional press SATURATES around 3 N, so it cannot deliver the §2.3.3 insertion force at any commanded depth. Deeper presses also start missing contact entirely (one placement sampled 1399 steps at 0.00 N).
    - implication: the demo's scripted inserter is fundamentally the wrong instrument for a force-specified insertion. §WP4.4 already specifies the right one: descend at 5 mm/s, and on contact run a 2 mm Archimedean spiral search while HOLDING 8 N down -- force control, not a position overshoot. §2.5 specifies the task-space impedance controller to do it with.

- **problem** `no_visual_for_physics_sim`
    - finding: Isaac Sim's renderer cannot start on this machine in ANY mode. The physics sim can only be observed through logs and episode records.

- **result** `rtx_crash_cause_confirmed`
    - implication: downgrading to 590.48.01 makes the driver-version test false, so the compat layer becomes a no-op ('older or already-safe drivers are left untouched') and the RTX renderer should initialise normally.

- **problem** `wrong_interpreter_is_easy_to_hit`
    - fix: ["envcheck.py -- every physics script verifies its interpreter BEFORE launching the app and prints the exact command to use; wrong env now costs about a second", "scripts/run.sh -- a launcher that routes each script to the right interpreter, sets OMNI_KIT_ACCEPT_EULA and the working directory", "the environment table is in PROGRESS.md and the viewing instructions now use the launcher"]

- **problem** `launcher_did_not_default_to_headless`
    - fix: run.sh now adds --headless for physics scripts unless --window is given explicitly, prints that it did so, and warns that --window will segfault until the driver is downgraded

- **decided** `gate_is_selectable`
    - outcome: run_plan.py takes --gate {spec,default}

- **problem** `driver_downgrade_instructions_were_wrong`
    - recommendation: take the RTX compat repin instead -- it needs no driver change

- **result** `rtx_crash_is_isaac_sim_51_not_the_driver`  ⟵ *retracts the claim that downgrading the driver to 590.48.01 would very likely fix the cra…*
    - implication: this machine CAN render Isaac Sim; it cannot render Isaac Sim 5.1, which is what BrickSim pins. So the physics runs cannot be rendered in place.
    - correction: I inferred the driver was at fault because the machine met the RTX compat layer's trigger condition. That inference was wrong. With the compat layer fully active -- package verified from two mirrors, layer extracted, all five VK_*/XDG_DATA_DIRS variables set, 'RTX Vulkan compatibility profile enabled' printed -- Isaac Sim 5.1 still segfaults. The allocation-limit theory is falsified by direct test…

- **deviated** `rtx_compat_repinned`

- **result** `run_playback_built`

- **?** `arm_in_both_renderers`
    - status: done

- **?** `viewers_did_not_launch`
    - status: done

- **problem** `viewer_drew_bricks_low`
    - finding: the physics grasp was fine (fingers 15.6 mm apart on a 15.8 mm brick, s1_arm recording). viz/play_viser.py and viz/to_usd.py drew each box CENTRED on BrickSim's bottom-centre origin, i.e. 4.8 mm low; run_assembly.py drew the fingers 80 mm open while carrying.
    - fix: boxes hang half a height above the recorded pose in both players; the kinematic viewer closes the fingers on the carried brick.

- **decided** `live_dual_arm_prototype`
    - outcome: dual_arm_sim.py: Newton 1.2.1 standalone (Isaac 6.0 env, MuJoCo-Warp solver), two FR3 arms -- B placer at +x, A stabilizer at -x -- studded SDF bricks reused from Newton's example_brick_stacking, a pre-allocated weld pool as the clutch (plan v2 §3.3 'loop joints'), live OpenGL or browser viewer; images -> blueprint -> plan -> sim in one command
    - rationale: the user found record-then-replay inefficient and asked for dual-arm and more detailed bricks. Isaac Sim 5.1 (BrickSim) cannot render here and takes 5-10 min to boot; Newton builds in ~10 s, runs ~real time and renders. Newton is D2's primary backend anyway.

- **result** `prototype_end_to_end`

- **problem** `v31_passive_brace_fails`
    - finding: a stiff passive hold (15 kN/m) does not take the LP's brace share: the welded joints are far stiffer than the arm, carry the load and break first (S3 step 12: brace z -12 N measured vs -35 N needed, 35 breaks)
    - fix: coordinated stabilizer: feed forward the least-effort LP wrench keeping joints <= 0.5 at 1.3 n f_insert, scaled by the placer's commanded press; the hold is re-latched before the press

- **problem** `v31_brace_collisions`
    - finding: the placer carried the brick at structure height + 60 mm straight into the stabilizer's leaning hand (25 joint breaks before the press); a fixed 28 mm brace/placer clearance both admitted colliding grasps and excluded feasible low ones
    - fix: travel height clears arm A's hand; brace candidates are checked in the twin (checks.BraceClearance: IK both arms, narrowphase A vs B + carried brick, 1.5 mm margin) -- mj_geomDistance returned 0.0 for box-box pairs cm apart and is not used

- **deviated** `v31_grip_force`

- **decided** `v31_scripted_inserter`
    - outcome: fast descent then 5 mm/s; touchdown on stud tops -> spiral search (2 mm, 3 mm/s, 8 N, 4000 N/m lateral, +-1.5 deg wiggle); else laterally free press to 1.3 n f_insert; stuck -> search; in-cavity needs a level wrist

- **decided** `v31_controller_fixes`
    - outcome: OSC gains -Lambda Jdot qdot (null-space leak 0.29 -> 0.013 mm), posture reference slewed at 0.5 rad/s, F/T low-pass 5 ms, noslip only while pressing/bracing, brick rotational inertia x10 (conditioning)

- **decided** `positional_press_saturates`
    - outcome: resolved: the twin's scripted inserter is force-controlled (press ramped to 1.3 n f_insert through the impedance's force feed-forward) per 2.5 / WP4.4

- **decided** `rendering_path_not_taken`
    - outcome: unchanged in v3.1: the twin runs headless; no rendering is needed for any gate

- **problem** `v31_feeder_yaw`
    - decision: feeder yaw jitter off (a fixtured feeder); position jitter kept
    - finding: with 3 deg feeder yaw jitter the grasp flips to the other wrist configuration; on touchdown the brick was shoved 0.3 mm along the grip axis, loaded the studs on one edge, turned ~1 deg in the pads during the press, and the engaging weld broke the support's base joint (S1: 3/3 failures with feeder jitter alone, 0/3 without)

- **problem** `v31_snap_backoff_reverted`
    - decision: reverted; the placer keeps pressing until MATED
    - finding: backing the press off when the joint engages made every structure fail: the weld then seats the brick by pulling the SUPPORT up, which breaks its base joint

- **result** `v31_executor_validation`
    - result: frozen executor (commit 5a432ed): S1 full assembly 5/5, S2 3/3 in validation; S3 0/4 -- step 11 (second corbel onto the pier) breaks the pier even braced, and 2x2 column bases break on some seeds at steps 3-4

- **problem** `v31_search_regression`
    - finding: G1's slow check (scripted insertion from <= 2 mm initial error, 10 trials) passed 9/10 at the first twin commit (one search each, 58-72 N peaks) and fails 2/10 on the frozen code: every trial with more than ~1.1 mm error needs 2-5 spiral searches and times out. On three of those trials the pre-freeze commit e76377c seats 2/3 and the frozen code 0/3; restoring the stiff search wiggle (150 N*m/rad) …
    - implication: the frozen scripted inserter does not recover from 1-2 mm lateral errors; G3, A6 and G6 hand it 0.3 mm (and re-centre on ground truth after a failed attempt), where it rarely searches, so their results stand, but the §WP4.4 claim 'snaps reliably from <= 2 mm' does not hold at the final code. Not fixed: changing the inserter now would invalidate the G3 and A6 batches. The test stays failing

## WP5 — RL insertion

- **deviated** `v31_force_targets`

- **deviated** `v31_rl_scale`

- **problem** `v31_env_ft_sensor`
    - finding: a MuJoCo force sensor between hand and held brick read 0.6 N while the stud interference held 7 N: tendon constraint forces are joint-space and absent from cfrc_ext
    - fix: wrist F/T = the hand free joint's constraint force (contacts + tendons + welds), low-passed 5 ms

- **problem** `v31_env_yaw_jam`
    - finding: the scripted base jammed a 2x4 at diagonal studs: > 0.4 deg of yaw wedges it (0.1 mm clearance over 16 mm); the rigid env grasp cannot twist as a real pad grasp does
    - fix: yaw compliance 10 N*m/rad about the tool axis (roll/pitch 150)

- **decided** `v31_a10_budget_confound`
    - decision: keep R1ht as run (it is what a builder who trusted the hand-tuned model would get) and add R1hc: hand-tuned joint with the budget and base press held at the calibrated seating force, so that only the joint's mechanics differ; evaluate both on the calibrated joint and on their own
    - finding: R1ht (A10, hand-tuned joint) collapsed in training: success on its own joint fell from 0.42 to 0.00 and the final policy never touches (every episode a 0 N timeout). The env scales the force budget and the scripted base's press by the joint model's insertion force, so on the 3 N/stud joint a 2x4's budget is 37-48 N, below the touchdown transient; the sparse -1 for a force violation made not pressi…

- **result** `a10_result`
    - result: positive in distribution: with the force budget held at the calibrated seating force, a policy trained on the hand-tuned joint seats 61% on it and 23% on the calibrated joint (R1: 66%, p = 1e-9), with 35 force violations per 100 against R1's 2; with the budget derived from the hand-tuned model, training collapses (R1ht, 0% on its own joint). Stage 3: no difference (19/12/18%)

- **probe** `mjwarp_insertion_probe`
    - finding: MuJoCo Warp 3.8.0.3 on the RTX 5090 D, insertion scene mid-press, physics only: stage 0 (2x4) 1.6M / 5.1M / 7.4M / 11.5M physics steps/s at 1k / 4k / 8k / 32k worlds; stage 3 plateaus at 5.9M from 8k worlds. CPU MuJoCo on the same states: 55k (stage 0) / 78k (stage 3) per core, ~0.9M / 1.25M on 16 cores. BUT the GPU numbers are for different physics: over every policy step of scripted insertions, …
    - implication: a GPU insertion env is blocked on contact fidelity before anything else (unit rescaling or different stud/wall primitives, gated by per-state contact parity against the CPU twin); the physics-only speedup over 16 CPU cores is ~5-13x, before the clutch model, impedance and scripted base are ported

## WP6 — Perception

- **decided** `vision_decision`
    - outcome: vision group not needed; WP6 skipped (cut M10); G5 not triggered

## WP7 — Integration

- **problem** `v31_rl_inserter_frame`
    - decision: observations in a task frame turned by the seated hand's yaw (orientation, position, velocity, wrench); on exit the arm holds the pose it is in. After the fix the head reads 0.07-0.31 and R1 seats 8 of 9 first attempts on S2's 2x4s; the first run's data were discarded
    - finding: the first G6 run (S1 x3, S2 x2, S3 x2, all failed) handed the policy world-frame poses of a hand at the plan's yaw (90 deg on S1); the training cell's hand is always near yaw 0, so the observation was out of distribution: the success head read 0.002 on every attempt and 7 of 12 first attempts failed. The inserter also left its accumulated orientation target in place on exit, so the restored stiff …

- **decided** `v31_rl_fallback_half_seated`
    - decision: with an RL inserter active, a half-seated brick goes straight to the scripted fallback, which presses it home; the scripted-only tree (G3, A6) is unchanged
    - finding: when the RL policy stops on its force budget mid-press the brick is half-seated; LiftAndRegrasp refused to lift it, so the tree retried the policy three times and never reached the scripted fallback of 7.1

- **problem** `v31_rl_press_stop`
    - decision: the adapter stops the press within 2 ms of MATED or of the budget being crossed; the policy still acts at 20 Hz. G6 was restarted from scratch on this code (the second run mixed adapter versions and was discarded). Remaining G6 failures include G3's base-joint break of a 2x2 at mating (scripted fallback after a failed RL attempt; G3 S1 seeds 5, 11, 14 fail the same way)
    - finding: the RL inserter checked for a seated joint and the force budget once per 50 ms policy step; on S3 step 2 the brick snapped home early in a step, the press ran on to 68 N (budget 57 N) and the 2x2 column's base broke 6 ms after the skill exited. The scripted skill stops within 2 ms

- **decided** `v31_g6_extra_seeds`
    - decision: add seeds 1-4 of the full system (weakest-joint bracing) on S1-S3; G6's gate verdict still reads the planned 15
    - finding: G6 as planned (S1-S5 x 3 strategies, one seed) is 15 assemblies, but S1, S2 and S4 trigger no brace, so their three strategies are the same deterministic run; the per-placement statistics of the RL-first inserter rest on few placements

- **probe** ``
    - result: holding the press 0.1 s after MATED and ramping it out over 0.2 s (RL path only, not committed): S1 seed 1 completes 6/6 (fails without the ramp), seed 2 still breaks. Inconclusive at n = 2; a real test changes the frozen scripted inserter and re-runs G3
    - status: open hypothesis, reported as such

## WP8 — Experiments & analysis

- **decided** `v31_a6_protocol`
    - outcome: A6 on critical-step trials: structure pre-placed and mated up to a step any strategy braces; each strategy on matched seeds; success = seated with no joint breaking anywhere during the step; n = 20 per cell on S3

- **result** `a6_result`
    - result: bracing is necessary: unbraced, S3's critical steps succeed 10/60 and S5's 0/50. Weakest-joint placement beats nearest at S3 step 14 (100% vs 65%, p = 0.016), S5 steps 24-25 (100% vs 60% and 0%), and cuts the placer's peak force 31% at S3 step 13 (p = 6e-6); it loses at S3 step 11 (0% vs 90%) and five S5 steps. Pooled, nearest beats weakest-joint on both structures: S3 85% vs 67% (p = 0.04), S5 58…

- **decided** `a6_verdict`
    - outcome: A6 is negative for the proposed method: placement matters (single steps differ by up to 100 points between strategies), but choosing it with the plastic force model is worse overall than the nearest-brick heuristic. Neither of 4.4's anticipated outcomes (positive / null); the G7 rule for a non-positive A6 applies: the results report is retitled around what was found (bracing necessary; the force m…

- **result** `a6_s5_20seeds`
    - result: A6 S5 extended from 5 to 20 seeds per cell (450 new trials, seeds 5-19, on the workstation: 16 workers, 18 min; no crashed trials). S5 pooled: none 0/200, nearest 114/200 = 57% [50, 64], weakest_joint 59/200 = 30% [24, 36]; nearest vs weakest_joint paired McNemar p = 2.9e-7 (0.008 at 5 seeds), none vs weakest_joint +29.5 points (p = 3e-18). The 5-seed per-step pattern holds and is now individually…

- **decided** `a6_verdict_20seeds`
    - outcome: a6_verdict stands and is strengthened: with 20 seeds per cell on both structures, pooled nearest beats weakest_joint on S5 at p = 3e-7 (S3 unchanged, p = 0.04). The results report's title and conclusions are unchanged; its S5 numbers, G7 row and limitations were updated

## Unfiled

- **decided** `project_start` — Isaac Lab v3.0.0-beta2, Isaac Sim 6.0.0.1, Newton 1.2.1 and cuRobo 0.8.0 are all already installed on this workstation; WP0 is runnable without a new install.
- **GATE** `G0` — 2048 envs cost 21% of the single-env step rate, i.e. batching is near-free at the scale WP5 needs, and per-env body state is writable from a Warp kernel, so the joint model can run as a kernel (D2, D3 fallback path stays open).
- **decided** `curobo_isaac_split` — cuRobo emits joint trajectories to disk from isdnenv; the Isaac task replays them. No rebuild of the Isaac env (risk R9 avoided), and WP5 only consumes the pre-insertion handoff pose, so the policy never calls cuRobo mid-episode.
- **GATE** `G0` — 
- **GATE** `G1` — 
- **GATE** `G1` — 
- **GATE** `WP2` — 
- **GATE** `G1` — 
- **GATE** `G1` — 
- **GATE** `G1` — 
- **CUT** `` — 
- **CUT** `` — 
- **CUT** `` — 
- **CUT** `` — 
- **CUT** `` — 
- **GATE** `G0` — 
- **GATE** `G1` — 
- **GATE** `G2` — 
- **GATE** `G3` — 
- **GATE** `G4` — 
- **GATE** `G5` — 
- **GATE** `G6` — 
- **GATE** `G7` — 
- **GATE** `G7` — 
- **deviated** `v4_triangle_pair_buffer` — Newton silently drops triangle pairs past the buffer (newton/_src/geometry/narrow_phase.py:923-924 breaks past it); the only sign is the printed warning 'Triangle pair buffer overflowed N > M'. The P0 batch log printed it 1274 times, all on the 6 cube runs (212-213 per run); N ran from 1,000,326 (first warning) to a peak of 1,163,533 (cube lever_press r0; per-run peaks 1,160,359-1,163,533); arch, …
- **result** `v4_p0_baseline_stiff_fingers` — 
- **problem** `v4_p0_stiff_s3_lever_press_diverged` — 
- **result** `v4_p0_baseline_stiff_fingers_r2` — 

## Gate history (latest wins)

- **G0**: pass — v3.1 twin: OSC impedance with -Lambda*Jdot*qdot and dynamically consistent null space, 500 Hz over 1 kHz physics
- **G1**: pass_with_recorded_exceptions — v3.1 twin joint model: 10/10 fast checks pass; the slow scripted-insertion check fails at the frozen code (2/10 from <= 2 mm, was 9/10) -- a scripted-search reg…
- **WP2**: pass_with_recorded_gap — 
- **G2**: pass_with_recorded_exceptions — drop test, plans for all strategies, single-arm failure of S3/S5 in the twin pass; accessibility fails for S4 and S5 (flanked 2x2 grasps, strict xfail)
- **G3**: fail — S1 17/20, S2 5/20, S3 0/20 (criterion S1, S2 >= 90%, S3 >= 60%); 37 of 38 failures are joint breaks and one a disconnected structure -- none is a failed inserti…
- **G4**: fail — best stage-3 success 24% (R4; criterion 85%); success-head AUC 0.41-0.78 on held-out episodes (criterion 0.85); comparisons R1 vs R3, R4 recorded; test_env 4/4
- **G5**: not_applicable — WP6 skipped on A3 (decision vision_decision)
- **G6**: fail — 0/15 planned full-loop assemblies (RL-first inserter R1, scripted fallback); full system on S1-S3 x 5 seeds: 1/15; RL seats 54% of its attempts, the fallback 10…
- **G7**: pass_with_recorded_exceptions — S5's A6 exception removed (20 seeds per cell, a6_s5_20seeds); remaining exception: G6 has one planned seed plus 4 extra on S1-S3
- **v4_phase1_joint_model**: changes_requested (J-b); J-a pass, J-c pass — Phase 1 'Joint model' gate (plan v4 r5; phase commits a890b68, 4ad91a7, 0dd4f8b; base f227fd4), reviewed once by Codex (adversarial review of f227fd4..HEAD). Co…

## Open — needs a decision

- `docker` (WP0): The Isaac stack is installed and working in ~/Codes/CAIRSS/Issac (a venv, not system python). Containerising now costs days and re-runs the risk R9 install that already broke this machine once.
- `?` (WP1): The report frames the branch as 3a-if-it-batches / 3b-otherwise. The measurement says 3a on throughput. Contribution (2) -- 'GPU-batched snap mechanics at RL scale' -- is separately weakened: a Warp kernel would optimise…
