# Work log

Generated from `ledger.jsonl` by `scripts/worklog.py` — do not edit by hand.

82 entries. Later entries supersede earlier ones; retractions are marked.

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

## Gate history (latest wins)

- **G0**: pass — 
- **G1**: pass_with_recorded_exceptions — 5 criteria pass, 1 guarded on our side, 1 N/A for branch 3a, 2 cut with reasons
- **WP2**: pass_with_recorded_gap — 

## Open — needs a decision

- `docker` (WP0): The Isaac stack is installed and working in ~/Codes/CAIRSS/Issac (a venv, not system python). Containerising now costs days and re-runs the risk R9 install that already broke this machine once.
- `two_interpreters` (WP0): D5 assumes cuRobo is callable for coarse motion. It cannot be imported in the interpreter that runs Isaac Lab, so M3 cannot run in-process with the sim.
- `bricksim_stack_mismatch` (WP1): BrickSim cannot be imported into this project's Isaac env. It installs its own uv venv with a second copy of Isaac Sim (5.1). No Newton support anywhere in the repo -- 'newton' appears only as the force unit. Adopting Br…
- `?` (WP1): The report frames the branch as 3a-if-it-batches / 3b-otherwise. The measurement says 3a on throughput. Contribution (2) -- 'GPU-batched snap mechanics at RL scale' -- is separately weakened: a Warp kernel would optimise…
- `static_solve_gravity_only` (WP3): §3.6 needs the force distribution under the INSERTION WRENCH, not under self-weight. As shipped, the tool cannot answer that question.
- `total_force_ceiling` (WP2b): dynamic release force saturates at roughly 88-90 N per CONNECTION regardless of how many studs are mated. Per-stud capacity therefore falls as bricks get larger rather than staying constant -- the opposite of how a real …
- `s3_does_not_fail_single_arm_in_sim` (WP3): ABLATION A6 HAS NO SUBJECT until a structure demonstrably fails single-arm under the press the inserter really applies. Either the inserter must press with the specified force, or the structures must be made weaker, or t…
- `positional_press_saturates` (WP4): the demo's scripted inserter is fundamentally the wrong instrument for a force-specified insertion. §WP4.4 already specifies the right one: descend at 5 mm/s, and on contact run a 2 mm Archimedean spiral search while HOL…
