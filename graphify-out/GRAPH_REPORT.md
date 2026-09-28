# Graph Report - ISDN_Robofab  (2026-09-29)

## Corpus Check
- 264 files · ~338,901 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2905 nodes · 5403 edges · 167 communities (143 shown, 13 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 201 edges (avg confidence: 0.86)
- Token cost: 498,307 input · 0 output

## Community Hubs (Navigation)
- PhysicsGraph Assembly Closure Tests
- BrickSim Connection Events
- Constraint Scheduler Tests
- Dynamic Graph Tests
- Twin Executor
- LegoGraph Hook Tests
- Breakage Debug Dumps
- Joint Calibration Adapter
- Topology IO / pybind
- USD LegoGraph Tests
- Connection Overlay UI
- Assembly Rewards & Commands
- BrickSim ExecPlans & Agents
- LEGO Units & Geometry
- In-hand Assembly Demo
- Clutch Model Tests
- Connection Segment Types
- LegoGraph Tests
- Assemble-Brick Env Config
- Isaac Type Configs CLI
- Isaac RTX Compat Shim
- Isaac Lab Event Shims
- Assembly Planner
- Artifact Preparer Build
- BrickSim Dataset Browser
- Topology Schema & Ordering
- PPO Actor-Critic
- Stability Evaluation
- Artifact Prep Config
- Expert Assembly CLI
- Topology Conversion CLI
- v3.1 Master Report Concepts
- Brick Part Markers
- Kit Runner Entry
- Clutch Joint Model
- BrickSim Main UI
- Contact Monitor
- Planner Tests (Gate G2)
- Motion Skills
- Patch Parsing
- Conversion Tests
- brickassembly Setup & Pins
- Work Packages & Tasks
- Assemble-Brick Command
- Gym Env Wrappers
- Bracing Assignment
- Joint Capacity Model
- Assembly Demo
- Scheduling Policy Tests
- Assembly Env Terminations
- Legolization Import
- v3.1 Results & Ablations
- MuJoCo Scene & Assets
- Blueprint Carving
- Dual-Arm Viewer Sim
- Static Solve Tool
- MDP Test Utils
- Multi-Agent Workflow
- Stability Analysis
- Static Analysis Heatmaps
- Literature Alternatives
- Dual-Arm Sim Example
- MuJoCo IK & Trajectories
- MuJoCo Arm Controller
- Insertion RL Env
- Teleop Demo
- PhysicsGraph Fixture
- Assemble-Brick Expert
- Hysteresis Gripper Action
- Viewing & Rendering Notes
- Assembly Run Script
- Controller Tests (G3)
- PhysX Brick Actors
- PhysX Foundation Setup
- Assembly Observations
- BrickSim Extension
- RL Inserter Policy
- BrickSim Plan Runner
- Progress & Interpreters
- Environment Check Probe
- MuJoCo Twin Checks
- MuJoCo Twin Runtime
- MultiKeySet Tests
- v4 Cell & Cameras
- Results Analysis
- RL Inserter Cell View
- Plan Load & Brace Clearance
- Build Lock Files
- PhysicsGraph Connection Types
- v2 Clutch & Collision Design
- v2 RL & Perception Design
- MaxRect Packing Tests
- MuJoCo Brick Geometry
- Contact Payload Types
- Selection Sync UI
- Toolbar Patches
- Novel Gaps & Related Work
- v3.1 Locked Decisions
- Structure Blueprints
- Interpreter Env Check
- Clutch Weld Internals
- PolyStore Tests
- v2 Plan & Ablations
- v3.1 Geometry & Ledger
- cuRobo Dual-Arm Planning
- Result Figures
- Env Tests (G4)
- Env Registration
- Frame-Time HUD
- Step-One-Frame Patch
- v4 Press & Capture Rules
- v4 Plan Questions & Phases
- Brick Carry Skills
- Impedance Check (G0)
- Newton Compat (Superseded)
- Joint Tests (G1)
- Keyboard Teleop
- StableLego Import
- BrickSim PID Finder
- Assembly Sequencing Literature
- Breakage Solver Tool
- Viewer Arm Model
- Scripted Insertion Base
- Build Artifacts
- Connection Thresholds
- Experiment Runner
- BrickSim Batching Probe
- Vectorized PPO Env
- Assembly Gate Tests
- FR3 Camera Assets
- BrickSim Build Script
- Course Robot Game
- v4 Executor & Firewall
- Multi-Seg Edge Test
- Self-Connect Test
- Brick Part Spawners
- Course Proposal
- Results Tables
- Dynamic Calibration
- Worklog Renderer
- Destroy Protocol
- BrickSim Launcher
- Vendor Overlay
- Brick Spec Tests
- MultiKeyMap Tests
- Planner Brick Numbering
- BrickSim Sweep Script
- Dynamic Sweep Script
- FR3 Asset Paths
- RealSense Asset Paths
- Stage Asset Paths
- Goal Layout
- Clangd Wrapper
- cuRobo Collision Table
- Impedance & IndustReal
- BrickSim Packages

## God Nodes (most connected - your core abstractions)
1. `ArtifactPreparer` - 34 edges
2. `Hooks` - 32 edges
3. `Hooks` - 31 edges
4. `PhysicsGraphScenarioFixture` - 30 edges
5. `make_stage()` - 29 edges
6. `make_brick()` - 28 edges
7. `main()` - 27 edges
8. `Results Report v3.1 (Bracing Interlocking Brick Structures)` - 26 edges
9. `UserError` - 25 edges
10. `build_three_parts()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Watch and Build final project brief (robotic LEGO assembly)` --conceptually_related_to--> `Master Report v3.1 (plan of record)`  [INFERRED]
  README.md → Docs/results_report.md
- `A6 bracing strategy success chart` --conceptually_related_to--> `Multi-robot LEGO assembly (humanoids, arms, hands)`  [AMBIGUOUS]
  brickassembly/results/figures/a6_success.png → BrickSim/docs/assets/bricksim_teaser.png
- `main()` --calls--> `assemble()`  [INFERRED]
  BrickSim/demos/demo_assembly.py → brickassembly/orchestration/twin_executor.py
- `probe()` --uses--> `AssembleBrickEnvCfg`  [INFERRED]
  brickassembly/scripts/01_bricksim_probe.py → BrickSim/python/bricksim/envs/assemble_brick/env.py
- `probe()` --uses--> `AssembleBrickExpert`  [INFERRED]
  brickassembly/scripts/01_bricksim_probe.py → BrickSim/python/bricksim/envs/assemble_brick/expert.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **P1-P3 probes feed decision point D1** — docs_plan_v4_p1_contact_probe, docs_plan_v4_p2_camera_probe, docs_plan_v4_p3_curobo_probe, docs_plan_v4_d1_decision_point [EXTRACTED 1.00]
- **Vision-only perception chain V3-V5-V7 with firewall** — docs_plan_v4_vision_only_rule, docs_plan_v4_v3_lookfeeder, docs_plan_v4_v4_plate_survey, docs_plan_v4_v5_relative_alignment, docs_plan_v4_v7_verification, docs_plan_v4_firewall_test [EXTRACTED 1.00]
- **Published components that must not be rebuilt** — docs_master_report_bricksim, docs_master_report_stablelego, docs_master_report_physasp, docs_master_report_brickgpt, docs_master_report_brickstobots [EXTRACTED 1.00]
- **Weak-joint bracing contribution: literature gap -> force model -> A6 test** — docs_literature_validation_quantified_bracing_gap, docs_literature_validation_stablelego, docs_results_report_weakest_joint_bracing, docs_results_report_a6_bracing_experiment, docs_results_report_plastic_force_model_unconservative [INFERRED 0.85]
- **CMU Intelligent Control Lab published brick stack** — docs_literature_validation_bricksim, docs_literature_validation_stablelego, docs_literature_validation_physics_aware_asp [EXTRACTED 1.00]
- **Plan v2 M1 build-plan pipeline** — docs_lego_dual_arm_plan_v2_beam_search_tiling, docs_lego_dual_arm_plan_v2_graph_mincut_stability_repair, docs_lego_dual_arm_plan_v2_greedy_sequencing, docs_lego_dual_arm_plan_v2_bracing_schedule, docs_lego_dual_arm_plan_v2_accessibility_check [EXTRACTED 1.00]
- **Images -> blueprint -> plan -> dual-arm Newton sim** — brickassembly_blueprint, brickassembly_planner, brickassembly_dual_arm_sim, brickassembly_viewing_dual_arm_prototype, brickassembly_readme_newton [EXTRACTED 1.00]
- **Per-task agent loop (scout, implementer/doc-writer, reviewer, orchestrator)** — _claude_agents_orchestrator, _claude_agents_scout, _claude_agents_implementer, _claude_agents_doc_writer, _claude_agents_reviewer [EXTRACTED 1.00]
- **Ledger -> worklog.py -> WORKLOG record keeping** — brickassembly_ledger, brickassembly_scripts_worklog, brickassembly_worklog, _claude_agents_doc_writer, brickassembly_scripts_status [EXTRACTED 1.00]
- **BrickSim native build & CI pipeline** — bricksim_native_cmakelists, bricksim__github_workflows_ci, bricksim__github_workflows_build_wheel, bricksim__github_actions_cache_bricksim_artifacts_action, bricksim__github_actions_free_disk_space_action, bricksim_agents_uv_pixi_toolchain [INFERRED 0.85]
- **C++26 modules architecture (vendor module + single module library)** — bricksim_agents_vendor_module, bricksim_docs_cpp26_modules_cheatsheet_vendor_wrapper_module, bricksim_docs_cpp26_modules_cheatsheet_one_library_target, bricksim_native_cmakelists_bricksim_modules, bricksim_docs_cpp26_modules_cheatsheet [INFERRED 0.85]
- **RealSense D435 sensor/mount USD asset conversion** — bricksim_python_bricksim_assets_sensors_realsense_d435_readme_d435_usd, bricksim_python_bricksim_assets_robots_fr3_camera_mount_readme_realsense_d435_camera_mount_usd, bricksim_python_bricksim_assets_robots_fr3_camera_mount_readme_omni_kit_asset_converter, bricksim_agents_isaac_sim_5_1 [INFERRED 0.75]
- **LEGO target structures described by blueprints** — brickassembly_blueprints_arch_front_arch_structure, brickassembly_blueprints_cube_front_cube_structure, brickassembly_blueprints_hollow_box_front_hollow_box_structure [INFERRED 0.85]
- **Multi-view silhouette blueprints** — brickassembly_blueprints_cube_front_cube_front_view, brickassembly_blueprints_cube_side_cube_side_view, brickassembly_blueprints_hollow_box_front_hollow_box_front_view, brickassembly_blueprints_hollow_box_top_hollow_box_top_view, brickassembly_blueprints_silhouette_blueprint_format [INFERRED 0.85]
- **A6 bracing strategy comparison** — brickassembly_results_figures_a6_success_bracing_strategy_chart, brickassembly_results_figures_a6_success_weakest_joint_bracing, brickassembly_results_figures_a6_success_nearest_bracing, brickassembly_results_figures_a6_success_wilson_ci [EXTRACTED 1.00]

## Communities (167 total, 13 thin omitted)

### Community 0 - "PhysicsGraph Assembly Closure Tests"
Cohesion: 0.05
Nodes (87): AssembledEvent, conn_seg, csid, csref, BrickHandle, actor, color, H (+79 more)

### Community 1 - "BrickSim Connection Events"
Cohesion: 0.05
Nodes (66): Return the configured stud/hole interface-pair query. Returns: Query describing…, ManagerBasedEnv, RigidObject, Resolve a BrickSim brick scene entity to its runtime Isaac Lab asset. Args:…, resolve_brick_rigid_object(), _has_reset_generation_tracking(), ManagerBasedRLEnv, Protocol (+58 more)

### Community 2 - "Constraint Scheduler Tests"
Cohesion: 0.11
Nodes (56): build_three_parts(), ConstraintEvents, created, destroyed, live_by_edge, live_by_handle, next_handle, BrickUnit (+48 more)

### Community 3 - "Dynamic Graph Tests"
Cohesion: 0.07
Nodes (58): apply_ops(), assert_components_equal(), BenchRes, checksum, impl, mops, N, n_add (+50 more)

### Community 4 - "Twin Executor"
Cohesion: 0.06
Nodes (37): assemble(), Brace, _brace_rel(), _clearance(), Ctx, DR, _episode_end(), _episode_start() (+29 more)

### Community 5 - "LegoGraph Hook Tests"
Cohesion: 0.06
Nodes (48): build_three_parts(), BrickUnit, ConnBundleEntry, ConnectionEndpoint, ConnSegEntry, ConnSegId, ConnSegRef, G (+40 more)

### Community 6 - "Breakage Debug Dumps"
Cohesion: 0.07
Nodes (52): BreakageDebugPath, BreakageDebugDump, BreakageDebugJson, BreakageInputDump, BreakageInputJson, BreakageSolutionDump, BreakageSolutionJson, BreakageStateDump (+44 more)

### Community 7 - "Joint Calibration Adapter"
Cohesion: 0.06
Nodes (49): joint_calibration.json, One force measurement per process rule, Static calibration does not transfer to live sim, available(), axial_forces(), binary(), calibrated_thresholds(), joint_of() (+41 more)

### Community 8 - "Topology IO / pybind"
Cohesion: 0.09
Nodes (44): m, from_json(), to_json(), to_ordered_json(), type_caster<nlohmann::json>, type_caster<nlohmann::ordered_json>, PYBIND11_MODULE(), handle (+36 more)

### Community 9 - "USD LegoGraph Tests"
Cohesion: 0.15
Nodes (47): almost_equal(), assert_brick_colliders(), contains_path(), BrickColor, BrickPart, BrickUnit, int64_t, PlateUnit (+39 more)

### Community 10 - "Connection Overlay UI"
Cohesion: 0.06
Nodes (28): AbstractGesture, _ConnectionOverlayManipulator, ConnectionOverlayScene, IEvent, TypedDict, UsdContext, Viewport overlay for visualizing BrickSim connection utilization., Viewport scene registered for BrickSim connection overlays. (+20 more)

### Community 11 - "Assembly Rewards & Commands"
Cohesion: 0.10
Nodes (43): assembly_check_connection_formed(), assembly_goal_offsets(), assembly_goal_target_pose(), assembly_goal_yaws(), assembly_moving_brick_name(), assembly_query_connection_state(), ManagerBasedRLEnv, Tensor (+35 more)

### Community 12 - "BrickSim ExecPlans & Agents"
Cohesion: 0.08
Nodes (42): PLANS.md (Codex ExecPlans spec), ExecPlan, Skeleton of a Good ExecPlan, Living plan and decision log, Prototyping milestones, ExecPlan self-containment requirement, Cache BrickSim artifacts action, Free disk space action (+34 more)

### Community 13 - "LEGO Units & Geometry"
Cohesion: 0.08
Nodes (35): Shared LEGO unit constants in SI units., c4_rotate_2d(), Tensor, Vectorized C4 grid-rotation helpers., Rotate 2D vectors by discrete C4 yaw indices. Args: vectors: Tensor with shape…, connection_brick_transforms(), connection_interface_transforms(), connection_overlap_areas() (+27 more)

### Community 14 - "In-hand Assembly Demo"
Cohesion: 0.09
Nodes (34): assemble_lego_part(), compose_transforms(), grasp_lego_part(), inverse_transform(), main(), move_ee_to(), ArticulationMotionPolicy, ndarray (+26 more)

### Community 15 - "Clutch Model Tests"
Cohesion: 0.11
Nodes (36): _det_worker(), _energy(), free_dofs(), _gate_fires(), jtf_on(), make(), _pull_until_break(), slow (+28 more)

### Community 16 - "Connection Segment Types"
Cohesion: 0.07
Nodes (29): ConnBundleEntry, ConnectionEndpoint, ConnSegEntry, ConnSegId, ConnSegRef, InterfaceSpec, PartEntry, PartId (+21 more)

### Community 17 - "LegoGraph Tests"
Cohesion: 0.16
Nodes (33): assert_connect_error(), build_three_parts(), collect_components(), BrickUnit, ConnSegId, G, InterfaceId, InterfaceRef (+25 more)

### Community 18 - "Assemble-Brick Env Config"
Cohesion: 0.08
Nodes (31): BinaryJointPositionActionCfg, ActionsCfg, CommandsCfg, EventCfg, ImagesCfg, ObservationsCfg, PolicyCfg, PrivilegedCfg (+23 more)

### Community 19 - "Isaac Type Configs CLI"
Cohesion: 0.15
Nodes (30): collect_isaaclab_source_roots(), collect_isaacsim_extension_paths(), _confirm_overwrite(), _dedupe_resolved_paths(), _ensure_overlay_package(), generate(), _iter_extension_python_modules(), _iter_parent_packages() (+22 more)

### Community 20 - "Isaac RTX Compat Shim"
Cohesion: 0.10
Nodes (27): _App, _Create, enable(), _hash(), _layer(), _limit(), _Maintenance3, _MemoryHeap (+19 more)

### Community 21 - "Isaac Lab Event Shims"
Cohesion: 0.14
Nodes (28): Articulation, ManagerBasedEnv, RigidObject, Tensor, Event terms for BrickSim-managed Isaac Lab environments., Deallocate all BrickSim-managed runtime objects for the selected environments.…, Reset the scene to defaults. Avoid writing PhysX velocities for kinematic rigid…, reset_bricksim_managed() (+20 more)

### Community 22 - "Assembly Planner"
Cohesion: 0.14
Nodes (29): brace_for(), brick_pose(), _bricksim_crosscheck(), build_plan(), cells(), _extent(), footprint(), grasp_for() (+21 more)

### Community 23 - "Artifact Preparer Build"
Cohesion: 0.22
Nodes (6): ArtifactPreparer, log(), Path, Downloads, extracts, patches, and installs build artifacts., Print one script-owned log line., Hardlink untouched entries; copy files that patches will modify.

### Community 24 - "BrickSim Dataset Browser"
Cohesion: 0.12
Nodes (18): BricksimDatasetItem, download_bricksim_dataset(), get_bricksim_dataset(), is_bricksim_dataset_available(), load_bricksim_dataset(), Path, Download and load the BrickSim structures dataset., Return the previously loaded BrickSim structures dataset. Returns: Dataset… (+10 more)

### Community 25 - "Topology Schema & Ordering"
Cohesion: 0.13
Nodes (25): bfs_sort_connections(), Topology post-processing helpers., r"""Return a copy of ``topology`` with its connections sorted in a BFS order.…, JsonBrickPayload, JsonConnection, JsonPart, JsonPoseHint, JsonTopology (+17 more)

### Community 26 - "PPO Actor-Critic"
Cohesion: 0.10
Nodes (17): FORGE (Isaac Lab force-budget env), ActorCritic, _auc(), eval_all(), _eval_job(), evaluate(), load(), mlp() (+9 more)

### Community 27 - "Stability Evaluation"
Cohesion: 0.17
Nodes (19): _annotate_json_text(), _as_optional_bool(), EvalEntry, _extract_solver_result(), _iter_eval_entries(), _json_scalar(), JsonMemberSpan, JsonObjectSpan (+11 more)

### Community 28 - "Artifact Prep Config"
Cohesion: 0.12
Nodes (20): Config, main(), Run the command-line interface. Returns: Process exit status., Read one patch entry from the TOML config. Returns: A validated patch…, Prepare archive artifacts for a build. Reads a TOML file listing archives by…, Read one artifact entry from the TOML config. Returns: A validated artifact…, Displayed without a traceback., Validated TOML configuration for artifact preparation. (+12 more)

### Community 29 - "Expert Assembly CLI"
Cohesion: 0.11
Nodes (25): build_argument_parser(), main(), parse_goal_rows(), ArgumentParser, Path, Tensor, CLI for running the assemble-brick expert policy., Save camera observation images for one rollout step. Args: images_obs: Non-… (+17 more)

### Community 30 - "Topology Conversion CLI"
Cohesion: 0.11
Nodes (23): _build_argument_parser(), _detect_input_format(), main(), _parse_baseplate(), ArgumentParser, CLI for converting supported LEGO structure formats to topology JSON., Run the topology conversion command. Returns: Process exit code., is_legolization_json() (+15 more)

### Community 31 - "v3.1 Master Report Concepts"
Cohesion: 0.10
Nodes (28): A3D adaptive affordance assembly, Ablation A6 (main experiment), Benchmark structures S1-S5, Weakest-joint bracing assignment (WP3.6), Bracing baselines: none / nearest / weakest_joint, BrickGPT, BrickSim, BricksToBots (+20 more)

### Community 32 - "Brick Part Markers"
Cohesion: 0.12
Nodes (23): parse_color(), Parse a named or hex RGB color. Args: name: Color name from ``colors.json`` or…, Create and toggle command-owned target marker visualization. Args: debug_vis:…, Write current command target poses to marker prims., _build_marker_wireframe_points(), _configure_marker_curves(), Prim, Stage (+15 more)

### Community 33 - "Kit Runner Entry"
Cohesion: 0.12
Nodes (25): main(), _open_default_stage(), current_target(), has_target(), _is_awaitable(), _is_none_task(), _is_path(), _is_target_function() (+17 more)

### Community 34 - "Clutch Joint Model"
Cohesion: 0.14
Nodes (8): ClutchModel, The multiple of the brick's symmetry period nearest its current yaw relative to…, A mated brick keeps the symmetry it was mated at; a free one takes the nearest…, Put a brick straight into MATED at full stiffness (pre-placed bricks)., Interference off (the brick left the zone, or mated)., Coulomb stud/cavity interference on each connection tendon., Force pushing brick `name` onto its supports along the stud axis: insertion-…, Wrench on U at each mated connection's patch centre, patch frame. A weld's…

### Community 35 - "BrickSim Main UI"
Cohesion: 0.11
Nodes (10): LegoUI, Destroy the BrickSim settings window., Return the current env_id from the main UI., Return the currently selected color as an RGB tuple., BrickSim settings and import/export UI., Create the BrickSim settings window and controls., parse_brick_prim_dimensions(), Prim (+2 more)

### Community 36 - "Contact Monitor"
Cohesion: 0.12
Nodes (10): ContactMonitor, _Ex, _Phase, Sec 2.4 contact monitor -- read-only classification of the cell's contacts.…, Bricks A's fingers may touch this frame (grasp_lp's gripped_bricks, or…, Event class for a candidate (d<0) pair, or None if permitted., Call once after the run ends, to close out any still-open events., {brick_id: {other_id, ...}}, same layer (k) and touching/overlapping footprints. (+2 more)

### Community 37 - "Planner Tests (Gate G2)"
Cohesion: 0.13
Nodes (23): _brick(), parametrize, slow, Gate G2 -- master_report §WP3 acceptance, in the MuJoCo twin's frame. cd…, requires_brace <=> predicted utilisation x SF >= 1 (§3.6 step 4), the same…, All three strategies give a §2.6-complete brace wherever one is needed, on…, S3 and S5 need a second arm by the force model; S1, S2, S4 do not., §3.3: all joints mated, 2 s under gravity, 0.5 N lateral push at the top -> max… (+15 more)

### Community 38 - "Motion Skills"
Cohesion: 0.11
Nodes (22): brace(), brace_feedforward(), brace_pose(), brick_pose(), close(), grip_for(), grip_yaw(), hand_tilt_deg() (+14 more)

### Community 39 - "Patch Parsing"
Cohesion: 0.11
Nodes (14): Patch, PatchPathTree, One patch applied to an artifact after output preparation., Parse the unified diff and return paths it modifies. Walks ``---``/``+++``…, Extract a file path from a ``---`` or ``+++`` diff header. Returns: The…, Return the path field from a unified-diff file header., Parse the C-style quoted path syntax used by Git and GNU diff. Returns: The…, Return old/new line counts from a unified-diff hunk header. (+6 more)

### Community 40 - "Conversion Tests"
Cohesion: 0.22
Nodes (21): A, B, approx_equal(), approx_mat(), approx_quat(), approx_xf(), M, main() (+13 more)

### Community 41 - "brickassembly Setup & Pins"
Cohesion: 0.12
Nodes (21): brickassembly README, static_solve applied-load patch (--load/--torque), Branch 3a: adopt BrickSim, BrickSim (Isaac Sim 5.1 LEGO clutch sim), env.lock pinned versions, Kinematic slice (cuRobo + Viser, no contact physics), Lever-arm force model (weakest_joint), S3 fails single-arm at ~6% of task press force (+13 more)

### Community 42 - "Work Packages & Tasks"
Cohesion: 0.17
Nodes (22): Three bracing strategies: none / nearest / weakest_joint, tasks.yaml task graph, vision_decision (A3: R4 vs R1), WP0 environment_verification, WP1 joint_mechanics, WP2 joint_calibration, WP3 planner_and_bracing, WP4 motion_and_scripted_baseline (+14 more)

### Community 43 - "Assemble-Brick Command"
Cohesion: 0.11
Nodes (13): AssembleBrickCommand, slice, Return whether this command provides debug visualization. Returns: Always…, Return the sampled command tensor. Returns: Tensor with shape ``(num_envs,…, Return the resolved discrete goal table. Returns: Long tensor with shape…, Refresh target poses without timer-based resampling. Args: dt: Elapsed…, Leave command metrics empty for this command term., Sample goal rows for reset environments. Args: env_ids: Environment ids whose… (+5 more)

### Community 44 - "Gym Env Wrappers"
Cohesion: 0.13
Nodes (14): _ActT, device, Gymnasium wrappers used by BrickSim environments., Move observations, rewards, and info structures to one torch device., Initialize the wrapper with the destination device., Step the wrapped environment and move the result to the target device. Returns:…, Reset the wrapped environment and move the result to the target device.…, Render the wrapped environment and move the result to the target device.… (+6 more)

### Community 45 - "Bracing Assignment"
Cohesion: 0.16
Nodes (20): assign(), _brace_obj(), candidates(), feasible(), gripped(), lean_for(), pad_contact(), predicted() (+12 more)

### Community 46 - "Joint Capacity Model"
Cohesion: 0.13
Nodes (12): grid_patch(), Patch, Clutch capacity of one stud connection under a general axial/prying load. The…, Overlap patch of one connection, in its own frame (origin at the patch centre,…, u >= 1 means the clutch lets go. Scalar or broadcast arrays., Index of the edge the patch pries open about (0:-x 1:+x 2:-y 3:+y)., Concentric pull-off force (N)., Patch of a full nx x ny overlap centred on the origin. (+4 more)

### Community 47 - "Assembly Demo"
Cohesion: 0.20
Nodes (20): assemble_lego_part(), compose_transforms(), grasp_lego_part(), inverse_transform(), main(), move_ee_to(), ArticulationMotionPolicy, ndarray (+12 more)

### Community 48 - "Scheduling Policy Tests"
Cohesion: 0.18
Nodes (18): build_adjacency(), ComponentView, verts, generator, PartId, size_t, unordered_map, UnorderedPair (+10 more)

### Community 49 - "Assembly Env Terminations"
Cohesion: 0.15
Nodes (18): Isaac Lab manager-based environment for the assemble-brick task., gripper_is_open(), ManagerBasedRLEnv, Tensor, Shared MDP helpers for assemble-brick observations, rewards, and terms., Return a mask indicating whether the gripper is open. Args: env: Manager-based…, brick_height_below_threshold(), non_target_connection_formed() (+10 more)

### Community 50 - "Legolization Import"
Cohesion: 0.14
Nodes (20): DefaultLegoLibraryBrick, _extract_bricks_from_lego_json(), LegoLibraryBrick, legolization_json_to_topology_json(), LegoStructureBrick, load_default_lego_library(), _load_default_lego_library(), Brick (+12 more)

### Community 51 - "v3.1 Results & Ablations"
Cohesion: 0.14
Nodes (21): Gate 2 Scripted End-to-End (critical), Novel Gap 3: Does Snap-Model Fidelity Change Learned Policy, Results Report v3.1 (Bracing Interlocking Brick Structures), A10 Joint-Model Fidelity Ablation, A1 RL vs Scripted Inserter, A3 Vision Ablation (vision not needed), Base-Joint Break of 2x2 Columns (failure mode), G1 Joint Model Gate (+13 more)

### Community 52 - "MuJoCo Scene & Assets"
Cohesion: 0.14
Nodes (17): fetch(), _get(), Franka Panda model from MuJoCo Menagerie, fetched at a pinned commit. python -m…, Download panda.xml and its meshes; returns the path to panda.xml., build_cell(), Cell, _cells_world(), centre_plan_origin() (+9 more)

### Community 53 - "Blueprint Carving"
Cohesion: 0.16
Nodes (18): ascii(), carve(), components(), load(), mask(), Images -> brick blueprint: silhouettes -> voxel grid -> tiled, bonded bricks.…, Number of stud-connected groups, ignoring the baseplate., Top-down text view of each layer, one letter per brick. (+10 more)

### Community 54 - "Dual-Arm Viewer Sim"
Cohesion: 0.16
Nodes (15): add_proxies(), build_arm(), feeder_slots(), make_plan(), qmul(), qrot(), qz(), Dual-arm LEGO assembly, live: images -> blueprint -> plan -> two Frankas. Arm B… (+7 more)

### Community 55 - "Static Solve Tool"
Cohesion: 0.13
Nodes (18): AppliedLoad, force, has_point, json_part_id, point, torque, Graph, int64_t (+10 more)

### Community 56 - "MDP Test Utils"
Cohesion: 0.14
Nodes (18): enumerate_env_ids(), device, slice, Tensor, Return explicit environment ids as a long tensor. Args: num_envs: Total number…, Tests for MDP utility helpers., Slices should follow Python slice semantics over ``range(num_envs)``., Negative-step slices should produce reversed environment ids. (+10 more)

### Community 57 - "Multi-Agent Workflow"
Cohesion: 0.22
Nodes (18): doc-writer agent, implementer agent, orchestrator agent, planner agent, reviewer agent, scout agent, ledger.jsonl, Newton 1.2.1 physics (+10 more)

### Community 58 - "Stability Analysis"
Cohesion: 0.16
Nodes (16): analyze(), best_press_point(), Brace, brick_centre(), Connection, connections(), insertion_loads(), insertion_utilisation() (+8 more)

### Community 59 - "Static Analysis Heatmaps"
Cohesion: 0.22
Nodes (17): _apply_viewpoint(), _build_fixed_heatmap(), _derive_heatmap_output_path(), _load_and_convert(), main(), _parse_baseplate(), _parse_viewpoint(), ndarray (+9 more)

### Community 60 - "Literature Alternatives"
Cohesion: 0.15
Nodes (18): Allegro Hand Stabilizer / Parallel-Jaw Placer, Clutch Tool Placer Alternative, Run C Vision Student Distillation, Stabilizer + Placer Dual-Arm Role Split, Text-to-Structure LLM Stretch, Literature Validation of the Dual-Arm LEGO Assembly Plan, BrickGPT / LegoGPT (Pun et al. 2025), Stabilize to Act / BUDS (Grannen et al. 2023) (+10 more)

### Community 61 - "Dual-Arm Sim Example"
Cohesion: 0.15
Nodes (7): Example, Joint angles that put the hand at PARK, pointing down., Hand the next plan step to B; send A to brace when that step needs it., Press closed fingertips on the brace spot, leaning away from the placer. Two…, Put the held BRICK on target, not the hand (ground-truth poses, §WP4). The in-…, §2.3.3 gate on the pressed brick; if it passes, engage its clutch welds., yaw_of()

### Community 62 - "MuJoCo IK & Trajectories"
Cohesion: 0.15
Nodes (13): down_rot(), ik(), mat_to_quat(), PLAI, quat_to_mat(), Task-space impedance control for the Pandas -- master_report §2.5. RL policy /…, Policy-level action integrator ([IndustReal]): accumulate clipped pose…, action: 6-vector in [-1, 1] (dx dy dz droll dpitch dyaw), EE frame. (+5 more)

### Community 63 - "MuJoCo Arm Controller"
Cohesion: 0.17
Nodes (6): Arm, Per-finger opening (m); the pad gap is about 2x this., Contact wrench ON the hand from the environment, world frame, about the F/T…, Latch the current TCP pose as the target (no motion)., Force on the split tendon (half of it reaches each finger). Closing (grip_open…, Index bookkeeping and the impedance law for one Panda in a Cell.

### Community 64 - "Insertion RL Env"
Cohesion: 0.25
Nodes (5): InsertionEnv, joint: "calibrated" (§2.3.3 / WP2: 8.9 N insertion, 11.3 N break per stud, 1.2…, Environment wrench on the brick, as the wrist feels it (world frame, force N,…, The environment's wrench on hand + brick: the hand free joint's constraint…, _rot6()

### Community 65 - "Teleop Demo"
Cohesion: 0.22
Nodes (13): connect_leader(), disconnect_leader(), main(), open_record_writer(), open_replay_reader(), Any, ndarray, read_leader_action() (+5 more)

### Community 66 - "PhysicsGraph Fixture"
Cohesion: 0.12
Nodes (16): MetricSystem, PartId, PhysicsGraphFixture, actorA, actorB, csid, env, graph (+8 more)

### Community 67 - "Assemble-Brick Expert"
Cohesion: 0.28
Nodes (5): AssembleBrickExpert, Tensor, Official memory-less scripted policy for assemble_brick task. The policy uses…, Initialize the expert with static action configuration., Return a batched action tensor from the current privileged observation. Args:…

### Community 68 - "Hysteresis Gripper Action"
Cohesion: 0.14
Nodes (10): BinaryJointPositionAction, HysteresisBinaryJointAction, ManagerBasedEnv, Tensor, Binary joint action terms with hysteresis., Binary joint position action with open/close hysteresis., Initialize the action term and action latch. Args: cfg: Binary joint position…, Initial command based on the configuration's initial state. (+2 more)

### Community 69 - "Viewing & Rendering Notes"
Cohesion: 0.16
Nodes (13): Physics poses never reach USD headless, show(), NVIDIA driver downgrade to 590.48.01, Record headless, play back (Viser / USD), isaacsim_rtx_compat SHA-256 pin, Isaac Sim 5.1 RTX renderer crash, load(), main() (+5 more)

### Community 70 - "Assembly Run Script"
Cohesion: 0.20
Nodes (14): brick_dims(), check_gate(), goal(), hand_poses(), hand_target(), main(), Execute an assembly_plan.json with a cuRobo-planned Franka, live in Viser. A…, Hand-frame goal for holding a brick whose bottom face is at brick_xyz. (+6 more)

### Community 71 - "Controller Tests (G3)"
Cohesion: 0.17
Nodes (14): dump(), fixture, parametrize, Gate G3's controller checks -- master_report §WP4 acceptance, in the twin. cd…, Closing on a 16 mm brick: fingers close at <= 0.1 m/s and then squeeze the…, A raw set-point step (no trajectory, no feed-forward) settles with < 5%…, Swinging the posture target (elbow and wrist, 0.4 rad) moves the TCP < 0.1 mm:…, A known static force on the hand (at the TCP) reads back within 5% once the arm… (+6 more)

### Community 72 - "PhysX Brick Actors"
Cohesion: 0.26
Nodes (12): BrickPart, PxRigidDynamic, PxShape, PxTransform, create_brick_actor(), main(), make_contact_payload(), test_bind_unbind_scene() (+4 more)

### Community 73 - "PhysX Foundation Setup"
Cohesion: 0.13
Nodes (15): PxDefaultAllocator, PxDefaultCpuDispatcher, PxDefaultErrorCallback, PxFoundation, PxMaterial, PxPhysics, PxScene, PhysxEnv (+7 more)

### Community 74 - "Assembly Observations"
Cohesion: 0.22
Nodes (15): franka_gripper_speed(), franka_gripper_width(), obs_command_connection_created(), obs_command_target_pose(), obs_moving_brick_grasped(), obs_moving_brick_pose(), ManagerBasedRLEnv, SceneEntityCfg (+7 more)

### Community 75 - "BrickSim Extension"
Cohesion: 0.18
Nodes (10): BrickSimExtension, Omniverse extension entry point for BrickSim., Register and tear down BrickSim UI components inside Omniverse., Initialize BrickSim extension UI at startup., Destroy BrickSim extension UI components at shutdown., ConnectionOverlayController, Register the connection overlay and menubar display toggle., Remove the viewport overlay and menu integration. (+2 more)

### Community 76 - "RL Inserter Policy"
Cohesion: 0.25
Nodes (5): Hand pose in the task frame (world turned by the seated yaw)., Wrench about the TCP, task frame., RLInserter, Policy, Deterministic policy (mean action) + success probability, for eval and for the…

### Community 77 - "BrickSim Plan Runner"
Cohesion: 0.20
Nodes (13): assemble_structure(), load_demo(), main(), patch_demo_pose_reads(), patch_grasp_face(), physics_pose_prim_class(), WP4 -- scripted end-to-end assembly baseline (master_report §WP4). §WP4 asks…, A drop-in SingleXFormPrim whose get_world_pose() reads PHYSICS, not USD.… (+5 more)

### Community 78 - "Progress & Interpreters"
Cohesion: 0.18
Nodes (11): PROGRESS.md (how to check progress), Three interpreters (BrickSim/.venv, isdnenv, Isaac), cuRobo motion planner, run.sh script, artifact(), latest_gate(), Project status at a glance -- reads the ledger and the artifacts on disk.…, The most recent verdict for a gate -- the ledger is append-only, so a later… (+3 more)

### Community 79 - "Environment Check Probe"
Cohesion: 0.23
Nodes (13): _brick_world(), main(), probe_curobo(), probe_newton(), probe_warp_writable(), WP0 / Gate G0 -- environment and platform verification (master_report §WP0).…, Newton throughput on the 2-brick scene at 1 / 512 / 2048 envs (G0)., G0: can a Warp kernel read and write per-env body state? (+5 more)

### Community 80 - "MuJoCo Twin Checks"
Cohesion: 0.14
Nodes (10): drop_test(), Physical plan checks in the twin -- master_report §WP3.3. drop_test(plan) ->…, MuJoCo (CPU) twin of the assembly cell -- see sim/mj/README section in…, Stepping the twin: controllers at 500 Hz, joint model every physics step. sim =…, Show `sim` in MuJoCo's viewer, paced to `speed` x real time (0.25 = slow…, Block until every window watch() opened has been closed., wait_viewers(), watch() (+2 more)

### Community 81 - "MuJoCo Twin Runtime"
Cohesion: 0.20
Nodes (6): Wrist wrench on the hand from the environment (world frame), low- passed at…, Min-jerk move of the TCP to pos (and orientation R). Blocking. posture: steer…, Large move: IK the goal (seeded from `seed`, default the park configuration, so…, Step for `seconds` of sim time, or until until(sim) is true. Returns True if…, MuJoCo's no-slip post-solver, on while forces are high (a press, a brace hold):…, Sim

### Community 82 - "MultiKeySet Tests"
Cohesion: 0.16
Nodes (10): copies_sum(), Key, assigns, copies, move_assigns, moves, v, main() (+2 more)

### Community 83 - "v4 Cell & Cameras"
Cohesion: 0.20
Nodes (14): Cameras: top, wrist_B (ring light), up fallback, Contact monitor (cell/contacts.py), Section 2.5.4 error budget (RSS per axis, radial p95 = 2.45 sigma), Live ViewerGL viewer with HUD and image panels, P0 re-establish baseline, P2 camera probe and look-pose sweep, V3 LookFeeder (mandatory wrist look before grasp), V4 plate coarse/fine wrist survey (+6 more)

### Community 84 - "Results Analysis"
Cohesion: 0.32
Nodes (12): a6(), figures(), g3(), g6(), main(), mcnemar(), Statistics and figures -- master_report §4.2 / WP8. python -m…, WP7 full loop: every structure x strategy with the RL-first inserter. (+4 more)

### Community 85 - "RL Inserter Cell View"
Cohesion: 0.15
Nodes (3): _CellView, WP7: the WP5 policy as the behaviour tree's insertion skill -- master_report…, The subset of InsertionEnv's interface that ScriptedBase reads.

### Community 86 - "Plan Load & Brace Clearance"
Cohesion: 0.18
Nodes (9): load_plan(), The §2.6 assembly_plan for the twin (planner.py --frame twin writes them;…, BraceClearance, Pose the arm by IK; returns the pose error (position m, or 1.0 if the…, Any of A's geoms within MARGIN of B's hand or of brick `bid`?, Do the two hands fit? -- §3.4 accessibility for the brace, in the twin. ok =…, plans(), fixture (+1 more)

### Community 87 - "Build Lock Files"
Cohesion: 0.15
Nodes (8): LockFile, log_tool(), Advisory flock on one lock file., Remember the lock file path., Acquire the lock. Returns: This lock object., Release the cache lock., Prepare every configured artifact., Print one external-tool log line.

### Community 88 - "PhysicsGraph Connection Types"
Cohesion: 0.24
Nodes (11): AssembledEvent, conn_seg, csid, csref, ConnectionSegment, ConnSegId, ConnSegRef, vector (+3 more)

### Community 89 - "v2 Clutch & Collision Design"
Cohesion: 0.18
Nodes (13): Clutch State Machine Warp Kernel, Isaac Lab Factory/FORGE/AutoMate Envs, Hollow-Shell Brick Collision Geometry, Pre-allocated Loop-Joint Pool, newton_compat.py Adapter, Newton (MJWarp) Primary, PhysX Fallback, Clutch Calibration Against VBD Thin-Shell Reference, BrickSim (Wen et al. 2026) (+5 more)

### Community 90 - "v2 RL & Perception Design"
Cohesion: 0.19
Nodes (13): Pose-Error Curriculum (Stages 0-4), Six-Term Dense Reward, Task-Space Impedance Control Stack, Two-Stage Perception (YOLOv8s-seg + ICP), Policy-Level Action Integrator, RL Precision Insertion MDP (182-D obs), Run B Residual Policy, IndustReal (Tang et al. 2023) (+5 more)

### Community 91 - "MaxRect Packing Tests"
Cohesion: 0.30
Nodes (11): Bin, check_no_overlap(), intersects(), main(), test_basic_2x2_in_4x4(), test_bottom_left_style(), test_impossible_case(), test_rotation_fit() (+3 more)

### Community 92 - "MuJoCo Brick Geometry"
Cohesion: 0.21
Nodes (11): add_baseplate(), add_brick(), outer(), Brick geometry for the MuJoCo twin -- master_report §2.3 / §2.3.1. Each brick…, Studded baseplate covering grid cells [-margin, ni+margin) x [-margin,…, Outer footprint (Lx, Ly) in metres., Stud centres in the brick frame, row-major over (i, j)., Vertices of a wall prism centred on the wall's mid-plane. Cross-section in (u,… (+3 more)

### Community 93 - "Contact Payload Types"
Cohesion: 0.17
Nodes (12): ContactPayload, extra, header, impulses, pair, patch, array, PxContactPair (+4 more)

### Community 94 - "Selection Sync UI"
Cohesion: 0.20
Nodes (6): AssemblySelectionSync, IEvent, Selection synchronization for BrickSim connected-component picking., Expands a single-part pick into its full connected component. This applies when…, Subscribe to USD selection changes., Unsubscribe from USD selection changes.

### Community 95 - "Toolbar Patches"
Cohesion: 0.23
Nodes (11): _install_bridges(), install_toolbar_patches(), _patch_select_button_group(), Toolbar patches for BrickSim connected-component selection., Patch SelectButtonGroup. - add a 'Connected Component' entry in the Select Mode…, Redirect SelectModeModel to watch and write a UI-specific setting key. Use this…, Rebuild the SelectButtonGroup instance on the main toolbar. Make it pick up our…, Entry point invoked from our extension startup. Patch the Isaac toolbar at… (+3 more)

### Community 96 - "Novel Gaps & Related Work"
Cohesion: 0.21
Nodes (12): Static-Analysis Bracing Schedule (nearest connected brick), Domain Randomization incl. Brace Slip, A3D Adaptive Affordance Assembly (Liang et al. 2026), Kumar et al. 2025 Dual-Arm Snap-Fit (SnapNet), Novel Gap 1: Quantified Bracing at Predicted Weak Joint, A6 Main Experiment: Principled Bracing at Critical Steps, Brace Wrench Prediction (expected_reaction_wrench), Nearest-Brick Bracing Heuristic (+4 more)

### Community 97 - "v3.1 Locked Decisions"
Cohesion: 0.21
Nodes (12): Ablation A3 (no vision), py_trees behaviour tree executor (WP4), Contribution 4: force-only vs multimodal under self-occlusion, FORGE, Informed Asymmetric Actor-Critic, Locked decisions D1-D16, Marougkas residual RL (ICRA 2025), RL insertion WP5 (residual + sparse primary) (+4 more)

### Community 98 - "Structure Blueprints"
Cohesion: 0.22
Nodes (11): Arch Front-View Blueprint, Arch Structure (two pillars + spanning lintel with rounded opening), Cube Front-View Blueprint, Cube Structure (solid stacked block, taller than wide), Cube Side-View Blueprint, Hollow Box Front-View Blueprint, Hollow Box Structure (four walls around an empty core), Hollow Box Top-View Blueprint (square ring) (+3 more)

### Community 99 - "Interpreter Env Check"
Cohesion: 0.18
Nodes (8): banner(), Fail fast, and helpfully, when a script is run with the wrong interpreter. This…, One line naming the interpreter, so a log says which env produced it., Exit with an actionable message unless this interpreter can serve `env`.…, Enable BrickSim's Vulkan workaround before anything initialises Vulkan. Isaac…, require(), rtx_compat(), Open a recorded run's USD in Isaac Sim 6.0, which renders on this machine. Must…

### Community 100 - "Clutch Weld Internals"
Cohesion: 0.29
Nodes (8): _quat_conj(), _quat_mul(), _qz(), Pose of U in L's frame (L = -1 means world). theta_l: L's symmetry rotation…, Switch a connection's weld on at its seated pose, with the brick's own…, Relative-pose errors of one connection vs its seated pose., _rel_pose(), _rz()

### Community 101 - "PolyStore Tests"
Cohesion: 0.20
Nodes (10): Chest, gold, string, Enemy, x, y, main(), Player (+2 more)

### Community 102 - "v2 Plan & Ablations"
Cohesion: 0.20
Nodes (11): Dual-Arm LEGO Assembly 7-Week Research Plan v2, Ablations A1-A10, cuRobo Gripper Accessibility Check, Beam-Search Brick Tiling, py_trees Behaviour Tree Orchestration, cuRobo MotionGen, insertion_episode Data Contract, Prioritized A-then-B Dual-Arm Planning (+3 more)

### Community 103 - "v3.1 Geometry & Ledger"
Cohesion: 0.20
Nodes (11): AI orchestration pipeline: agent roles, task cards, guardrails, Hollow-shell collision geometry, LEGO System geometry (pitch 8 mm, height 9.6 mm), Project ledger (ledger.jsonl), SnapFitFEA1/2 force envelope, v3.1 force targets normalised by seating force (M6), Fact 6: brick contacts under a press (tunnelling finding), Grip rule N >= F_press/mu (+3 more)

### Community 104 - "cuRobo Dual-Arm Planning"
Cohesion: 0.24
Nodes (11): cuRobo, Dual-arm prioritized planning, stabilizer first, Stavridis bimanual relative motion (counter-position), Per-link contact-segment collision check, CUDA-graph discipline (pre-loop warm-up, capture guard), cuRobo 0.8 MotionPlanner (one per arm), FR3 collision sphere config (motion/fr3.yml via sphere_fit), Motion routes: in-process .vendor overlay / socket server / Newton-native RRT (+3 more)

### Community 105 - "Result Figures"
Cohesion: 0.20
Nodes (10): A6 bracing strategy success chart, nearest bracing strategy, weakest_joint bracing strategy, 95% Wilson confidence interval, curriculum stage resets success window, R1ht success collapse to 0, WP5 training success curves, Multi-robot LEGO assembly (humanoids, arms, hands) (+2 more)

### Community 106 - "Env Tests (G4)"
Cohesion: 0.20
Nodes (9): dump(), fixture, Gate G4's environment checks -- master_report §WP5 acceptance. cd brickassembly…, 10k random-action policy steps across stages: finite obs, bounded reward., success (scripted base, no DR), force violation (a tiny budget), joint break (a…, test_deterministic(), test_obs_dims(), test_random_rollout() (+1 more)

### Community 107 - "Env Registration"
Cohesion: 0.20
Nodes (5): Assemble-brick task environment package., Isaac Lab environment registrations for BrickSim tasks., _ensure_bricksim_extension_enabled(), BrickSim Isaac Sim extension package., Shared pytest support.

### Community 108 - "Frame-Time HUD"
Cohesion: 0.22
Nodes (6): _install_viewport_fps_patch(), Viewport HUD integration for BrickSim frame-time profiling., Unregister the menu item and restore the viewport FPS patch., Install the viewport FPS patch and menu item., _restore_viewport_fps_patch(), _ViewportFpsPatch

### Community 109 - "Step-One-Frame Patch"
Cohesion: 0.24
Nodes (9): install_step_one_frame_patch(), _patch_play_button_group(), _play_then_pause_immediately(), Toolbar patch that adds a one-frame step button., Advance the timeline by one app update. Issue PLAY, then pause on the next app…, Entry point invoked from extension startup. Patch the Isaac toolbar with an…, Patch PlayButtonGroup to add a dedicated "Step one frame" button. Place it…, Rebuild the PlayButtonGroup instance. Make it pick up our monkey patch that… (+1 more)

### Community 110 - "v4 Press & Capture Rules"
Cohesion: 0.27
Nodes (10): v3.1 scripted inserter fixes (M7, 71 N press), Arm-held, speed-limited guarded press (Insert), c_held(s): arm-held capture offset, D1 decision point (scale, cameras, motion route), D1 rule: alignment, press window, sink, grip criteria, F_seat, F_pt, k_os press metrics, P1 Newton contact physics, arm-held press and capture vs scale, P1(h) contact-spike arm: designed 0.6s mm tube-stud clearance (+2 more)

### Community 111 - "v4 Plan Questions & Phases"
Cohesion: 0.31
Nodes (10): End-of-phase experiments E1-E6, Plan of record v4 (r3): vision-driven collision-free dual-arm assembly in Newton, Prioritised dual-arm planning with shared zone, Q1 precision: chained alignment error vs scale, Q2 coordination: prioritised planning removes contacts, Q3 cross-simulator bracing check, v4 risk register R1-R13, Brick scale s in {1, 2} (U2) (+2 more)

### Community 112 - "Brick Carry Skills"
Cohesion: 0.25
Nodes (9): in_hand_offset(), put_down(), Brick pose in the TCP frame (measured once it is held)., TCP position that puts the brick's frame origin at brick_target., Carry the held brick to `dz` above target via a travel height., Set the held brick down at `spot` (world xy on the table) and let go -- the…, release(), tcp_for_brick() (+1 more)

### Community 113 - "Impedance Check (G0)"
Cohesion: 0.28
Nodes (7): ee_state(), main(), configclass, InteractiveSceneCfg, WP0 / G0 -- impedance controller tracks a 50 mm free-space circle. Acceptance…, (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, q, qd) -- run_osc.py., SceneCfg

### Community 114 - "Newton Compat (Superseded)"
Cohesion: 0.31
Nodes (8): body_array(), make_state(), SUPERSEDED 2026-09-20 -- WP1 adopted BrickSim, which is PhysX-only, and D2 was…, The (state_0, state_1, control, contacts) tuple a step needs., One control step. Returns the state that holds the result. Newton double-…, The live Warp array for a state field, by the name pinned above., _self_check(), step()

### Community 115 - "Joint Tests (G1)"
Cohesion: 0.33
Nodes (8): main(), pose_of(), Gate G1 -- joint mechanics acceptance tests (master_report §WP1). WP1 took…, Kinetic + gravitational potential energy of the listed rigid bodies (J)., Pose from the PHYSICS view, not from USD. world.step(render=False) does not…, step_n(), system_energy(), two_brick_topology()

### Community 116 - "Keyboard Teleop"
Cohesion: 0.28
Nodes (6): getch(), main(), Async version of your keyboard-action parser., read_keyboard_action(), Utility helpers shared by BrickSim MDP terms., Typing helpers for third-party APIs with incomplete stubs.

### Community 117 - "StableLego Import"
Cohesion: 0.31
Nodes (8): _construct_world_grid(), _gen_key(), _out_boundary(), LegoLibrary, LegoStructure, StableLego-based static stability analysis., Run StableLego static analysis for one LEGO structure. Returns: Stability…, run_stable_lego()

### Community 118 - "BrickSim PID Finder"
Cohesion: 0.33
Nodes (8): find_matches(), _is_bricksim_cmd(), _is_debugpy_wrapper(), main(), Find the PID of the running BrickSim process., Return BrickSim process matches as PID and command line pairs., Print the BrickSim PID once exactly one matching process is found. Returns:…, _read_cmdline()

### Community 119 - "Assembly Sequencing Literature"
Cohesion: 0.28
Nodes (9): Graph Min-Cut Stability Repair, Greedy Topological Sequencing with Lookahead, CMU Intelligent Control Lab Brick Stack, Legolization (Luo et al. 2015), Physics-Aware Assembly Sequence Planning via Action Masking (Liu et al. 2024), StableLego (Liu et al. 2024), A6 Negative Verdict (naive heuristic beats weakest-joint pooled), Elastic Joint Model / Simulate-Candidate Bracing (recommended next step) (+1 more)

### Community 120 - "Breakage Solver Tool"
Cohesion: 0.39
Nodes (7): BreakageSolution, string, elapsed_ms(), load_dump(), main(), now(), print_solution_info()

### Community 121 - "Viewer Arm Model"
Cohesion: 0.25
Nodes (4): Arm, a folded into (-period/2, period/2]., A queue of waypoints, eased between with smoothstep. A move finishes when its…, wrap()

### Community 122 - "Scripted Insertion Base"
Cohesion: 0.25
Nodes (5): WP5 insertion environment in the MuJoCo twin -- master_report §5.1-§5.5. env =…, §WP4.4 as a per-policy-step controller (the residual's base). The same phases…, Increments act on the impedance TARGET x_d (as skills.insert sets x_d…, ScriptedBase, _yaw_rot()

### Community 123 - "Build Artifacts"
Cohesion: 0.25
Nodes (4): Artifact, One archive artifact from the config file., Return the SHA256 digest for a file., sha256_file()

### Community 124 - "Connection Thresholds"
Cohesion: 0.29
Nodes (6): Finalize simulation, viewer, and gripper task settings., configure_assembly_thresholds(), configure_breakage_thresholds(), Configuration helpers for native BrickSim connection thresholds., Configure global native assembly-detection thresholds for the active process.…, Configure global native breakage-detection thresholds for the active process.…

### Community 125 - "Experiment Runner"
Cohesion: 0.43
Nodes (6): critical_steps(), _key(), main(), Batch trials in the twin, in parallel: G3's benchmark and ablation A6. python…, Steps where any bracing strategy braces (the A6 unit)., _run()

### Community 126 - "BrickSim Batching Probe"
Cohesion: 0.33
Nodes (5): main(), probe(), WP1 batching probe (master_report §WP1 step 2) -- does BrickSim batch? Steps…, Time `steps` expert-driven steps at n_envs. Returns a result dict., Official expert policy for the assemble-brick task.

### Community 128 - "Assembly Gate Tests"
Cohesion: 0.43
Nodes (6): base_topology(), main(), G1 -- assembly gate boundaries and scripted insertion (master_report §2.3.3).…, Drop an offset brick onto a mated one, press, and report whether it snapped., run_case(), yaw_quat()

### Community 129 - "FR3 Camera Assets"
Cohesion: 0.43
Nodes (7): FR3 RealSense D435 Camera Mount USD README, Franka RealSense D435 camera mount CAD (STEP/glTF), omni.kit.asset_converter, realsense_d435_camera_mount.usd, Intel RealSense D435 USD README, d435.usd, realsense-ros realsense2_description (URDF/Xacro)

### Community 130 - "BrickSim Build Script"
Cohesion: 0.67
Nodes (6): die(), ensure_pixi_toolchain(), read_prebuilt_manifest(), build.sh script, use_prebuilt_native_if_configured(), validate_prebuilt_manifest()

### Community 131 - "Course Robot Game"
Cohesion: 0.43
Nodes (6): new_game(), show_success(), update_errors(), update_robot(), ISDN Robofab README, Watch and Build final project brief (robotic LEGO assembly)

### Community 132 - "v4 Executor & Firewall"
Cohesion: 0.33
Nodes (7): Episode log results/v4/episodes.jsonl, Executor behaviour tree (cell/executor.py, py_trees), Firewall test (test_cell.py::test_firewall) and Robot facade, Read-only ground-truth mirror, Tray as kitting dispenser, refilled at step start, Vision-only rule (U6), WP4 integration -> M1

### Community 133 - "Multi-Seg Edge Test"
Cohesion: 0.60
Nodes (5): InterfaceId, InterfaceSpec, hole(), main(), stud()

### Community 134 - "Self-Connect Test"
Cohesion: 0.60
Nodes (5): InterfaceId, InterfaceSpec, hole(), main(), stud()

### Community 135 - "Brick Part Spawners"
Cohesion: 0.40
Nodes (6): BrickPartCfg, MarkerBrickPartCfg, configclass, Spawner config for a non-physical marker brick wireframe., Spawner config for a physical BrickSim brick part., SpawnerCfg

### Community 136 - "Course Proposal"
Cohesion: 0.33
Nodes (6): ISDN5240 RoboFab Preliminary Proposal Template, Non-planar 3D Printing Based on Coral Reef Morphology (sample), Fluid Forms (ETH Zurich DBT), MX3D 3D-Printed Steel Bridge (WAAM), Why Robotics Justification Section, ISDN Project Robotic LEGO Assembly README

### Community 137 - "Results Tables"
Cohesion: 0.60
Nodes (4): ci(), main(), pct(), Markdown tables for Docs/results_report.md, from results/analysis.json. python…

### Community 138 - "Dynamic Calibration"
Cohesion: 0.50
Nodes (4): main(), pair_topology(), WP2b -- calibrate against the LIVE sim. ONE measurement per process. WP2 fitted…, Baseplate + a brick + an identical brick lapped squarely on top.

### Community 139 - "Worklog Renderer"
Cohesion: 0.50
Nodes (3): one_line(), Render WORKLOG.md from ledger.jsonl. python scripts/worklog.py # rewrite…, render()

### Community 140 - "Destroy Protocol"
Cohesion: 0.40
Nodes (4): Protocol, Release resources held by the object., Object with an explicit teardown hook., _SupportsDestroy

### Community 141 - "BrickSim Launcher"
Cohesion: 0.50
Nodes (4): main(), Command-line launcher for running BrickSim inside Isaac Sim., Launch Isaac Sim with BrickSim and optional forwarded script arguments., _uncache_bricksim_package()

### Community 142 - "Vendor Overlay"
Cohesion: 0.50
Nodes (3): add_overlay(), Route 1 overlay -- plan v4 §2.4 rule 1, P3 step 1. cuRobo 0.8 is pure Python;…, Append the .vendor overlay and the curobo checkout to sys.path (idempotent).

### Community 144 - "MultiKeyMap Tests"
Cohesion: 0.83
Nodes (3): main(), piecewise_and_move_only(), smoke_basic_two_keys()

## Ambiguous Edges - Review These
- `ISDN5240 RoboFab Preliminary Proposal Template` → `ISDN Project Robotic LEGO Assembly README`  [AMBIGUOUS]
  Docs/ISDN_Project-Robotic_LEGO_Assembly-main/README.md · relation: conceptually_related_to
- `Brace Wrench Prediction (expected_reaction_wrench)` → `Domain Randomization incl. Brace Slip`  [AMBIGUOUS]
  Docs/results_report.md · relation: conceptually_related_to
- `A6 bracing strategy success chart` → `Multi-robot LEGO assembly (humanoids, arms, hands)`  [AMBIGUOUS]
  brickassembly/results/figures/a6_success.png · relation: conceptually_related_to

## Knowledge Gaps
- **218 isolated node(s):** `T_a_b`, `a`, `b`, `created`, `destroyed` (+213 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1120 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `ISDN5240 RoboFab Preliminary Proposal Template` and `ISDN Project Robotic LEGO Assembly README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Brace Wrench Prediction (expected_reaction_wrench)` and `Domain Randomization incl. Brace Slip`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `A6 bracing strategy success chart` and `Multi-robot LEGO assembly (humanoids, arms, hands)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PhysicsGraphScenarioFixture` connect `PhysicsGraph Assembly Closure Tests` to `Scheduling Policy Tests`, `PPO Actor-Critic`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `train()` connect `PPO Actor-Critic` to `Vectorized PPO Env`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `TestGraph` connect `Scheduling Policy Tests` to `PhysX Brick Actors`, `PhysicsGraph Assembly Closure Tests`, `PhysicsGraph Fixture`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **What connects `T_a_b`, `a`, `b` to the rest of the system?**
  _218 weakly-connected nodes found - possible documentation gaps or missing edges._