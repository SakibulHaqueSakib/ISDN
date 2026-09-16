# Literature Validation of the Dual-Arm LEGO Assembly Plan

**Prepared for:** Sakib
**Date:** 15 September 2026
**Scope:** Every load-bearing methodological claim in plan v1/v2, checked against published work.
**Verdict in one line:** the architecture is sound and well-supported, but **four of the modules I told you to build already exist as published, open-source work — one of them on your exact platform.** The plan needs restructuring from "build a system" to "build on top of an existing stack and answer a narrower question."

---

## 0. Executive Summary

| Decision | Verdict | Key evidence |
|---|---|---|
| Explicit snap-constraint layer instead of emergent contact | **Validated — and already implemented** | BrickSim (2026); Lämmle et al. (2022) |
| Hybrid architecture (engine does rigid body, separate module does clutch) | **Validated — exact match to published design** | BrickSim (2026) |
| Two-phase coarse → RL-precision pipeline | **Validated** | IndustReal (2023); Marougkas et al. (2025) |
| Policy-level action integrator | **Validated** | IndustReal (2023) |
| Residual RL over a scripted base controller | **Validated, and possibly should be the *primary*, not fallback** | Marougkas et al. (2025); Zhang et al. (2023) |
| Curriculum over initial pose error | **Validated, incl. for snap-fits specifically** | IndustReal (2023); Monnet et al. (2025) |
| Task-space impedance / variable compliance action space | **Strongly validated** | Beltran-Hernandez et al. (2020); Chen et al. (2023) |
| Force/torque sensing in the observation | **Validated, but necessity is contested** | FORGE (2024) vs. Markert et al. (2023), Nguyen et al. (2024) |
| Asymmetric actor-critic + teacher–student distillation | **Validated, with a documented failure mode I did not warn you about** | Messikommer et al. (2024); Kim et al. (2025); Cai et al. (2024) |
| Stabilizer + placer dual-arm role split | **Validated — but it's an established paradigm, not a novelty** | BUDS "Stabilize to Act" (2023); ManiGaussian++ (2025); A3D (2026) |
| Dense multi-term reward shaping | **Partially contradicted** | Marougkas et al. (2025) — sparse reward + model-based prior outperformed |
| Beam-search brick tiling + graph stability repair | **Superseded** | StableLego (2024); Kollsker et al. (2021) |
| Greedy topological assembly sequencing | **Superseded** | Liu et al. (2024) physics-aware action masking |
| Gripper-accessibility coupling "under-explored" | **False — I was wrong** | Barghi et al. (2024) |
| LLM text→structure with validator/repair loop (stretch) | **Already published, twice, and superseded** | BrickGPT/LegoGPT (2025); Rollback-Free (2026) |
| VBD deformable calibration of the clutch | **Novel-ish, but the analytical theory already exists** | Yoshida et al. (2020), *Phys. Rev. Lett.* |
| Single-GPU vision distillation budget | **Contradicted — likely under-resourced** | VIRAL (2025) |

---

## 1. The Finding That Changes Everything: BrickSim

**Wen et al. (2026), "BrickSim: A Physics-Based Simulator for Manipulating Interlocking Brick Assemblies."** arXiv `10.48550/arxiv.2603.16853`. Open source: `github.com/intelligent-control-lab/BrickSim`

What it does, stated in their own framing: existing rigid-body simulators do not faithfully capture snap-fit mechanics. They introduce a compact force-based mechanics model for snap-fit connections, solve the internal force distribution with a structured convex quadratic program, and use a hybrid architecture that delegates rigid-body dynamics to the underlying physics engine while handling snap-fit mechanics separately. On 150 real-world assemblies they report 100% accuracy in static stability prediction at ~5 ms average solve time, and in dynamic drop tests they reproduce both the occurrence and the location of real structural collapse. **It is built on Isaac Sim.**

### What this means for you

1. **My Section 3.3 was right in architecture and wrong in economics.** The hybrid design — physics engine for rigid bodies, separate module for the clutch — is exactly what they built. That's a strong validation of the reasoning. It is also a strong argument that you should not spend Week 1 writing a Warp kernel to reproduce it.

2. **Their formulation is better than mine.** I proposed a per-connection PD spring with a break threshold. They solve the *internal force distribution across the whole assembly* as a convex QP. That distinction matters enormously for your use case: when arm B presses a brick into a structure, the reaction propagates through every clutched joint below it. My per-connection spring model handles each joint independently and cannot tell you *which* joint fails first. Theirs can — which is precisely what you need to decide where arm A should brace.

3. **It validates against real hardware.** 150 real assemblies, breakage-location accuracy. Your VBD calibration idea (plan v2 §3.4) was aiming at exactly this fidelity question, but against a simulated reference rather than physical ground truth. Their validation is stronger.

**Revised recommendation:** Week 1 becomes *evaluate and integrate BrickSim*, not *build a clutch kernel*. Budget 3 days. Two outcomes, both fine:
- It works on your Isaac Sim 6.0 / Isaac Lab 3.0 setup → you have saved a week and inherited a validated snap model. Your contribution moves up the stack to the RL policy.
- It doesn't (version mismatch, PhysX-only, not batched for 1000+ envs) → **the batched-GPU-parallel version becomes a legitimate contribution**, because a QP-per-environment at 5 ms does not obviously scale to thousands of parallel RL environments. That's a real gap and a defensible engineering result. Either way you should implement against their interface so results are comparable.

Note also: BrickSim, StableLego, and the physics-aware sequence planner below all come from the same group (Intelligent Control Lab, CMU — Ruixuan Liu and colleagues). **There is a coherent published stack here.** Read it as one body of work, not three papers.

---

## 2. Module M1 (Build Planner) — Substantially Superseded

I had you writing beam-search tiling, a graph-articulation stability heuristic, and a greedy topological sequencer. All three have better published versions.

### 2.1 Stability analysis

**Liu et al. (2024), "StableLego: Stability Analysis of Block Stacking Assembly,"** *IEEE RA-L*, `10.1109/lra.2024.3414281`. They formulate stability as an optimization over force-balancing equations, report that it correctly predicts stability and — critically — **accurately locates the weakest parts of a design**, outperforming existing methods. They release **StableLego: 50k+ 3D objects with LEGO layouts**.

Two consequences:
- My graph min-cut heuristic (v2 §4.2 Step 3) is the weaker approach. Luo et al.'s *Legolization* (`10.1145/2816795.2818091`) already moved past pure heuristics to force-based analysis in 2015; StableLego is the current form.
- **You no longer need to hand-author S1–S5.** Sample five structures of graded difficulty from StableLego with known stability labels. This removes an afternoon of work *and* gives your evaluation set external provenance, which reviewers will prefer to "we made up five shapes."

For completeness on the optimization side: Kollsker et al. solved LEGO construction as MIP + adaptive large neighbourhood search with a QP static-equilibrium check, scaling to ~77,000 brick positions (`10.1007/s43069-021-00062-3`, `10.1016/j.ejor.2020.07.004`). My "don't reach for an ILP" advice was about speed and is still defensible for a beam search inside an RL data pipeline — but it should be stated as a deliberate trade-off against this literature, not as if optimization were impractical.

### 2.2 Assembly sequencing

**Liu et al. (2024), "Physics-Aware Combinatorial Assembly Sequence Planning Using Data-Free Action Masking,"** *IEEE RA-L*, `10.1109/lra.2025.3555074`. They learn a construction policy with an **online physics-aware action mask** that filters invalid actions, applied to LEGO assembly across **more than 250 3D structures with a 100% success rate**, where the best comparable baseline failed on more than 40 structures.

My greedy topological sort with lookahead-3 is the baseline this paper beats. Use their action-masking formulation, or at minimum cite it and justify the simpler method on runtime grounds.

Related: Ma et al. (`10.1109/icra48891.2023.10160424`) provide a graph-transformer ASP framework with a LEGO dataset; Ahn et al. (`10.48550/arxiv.2210.01021`) handle budget-constrained sequential brick assembly with a sparse 3D CNN and a one-initialized convolution filter for constraint validation — an elegant trick worth knowing.

### 2.3 Gripper-accessibility coupling — I was wrong

In v1 §4.3 and v2 §4.3 I wrote that the coupling between gripper geometry and the build plan is "genuinely under-explored" and told you to call it out in the writeup. **That is false.**

**Barghi et al. (2024), "From Bricks to Bots: Automated Collision-Aware Sequence Planning for LEGO Reconstruction with a Two-Finger Gripper,"** `10.1109/iccia65044.2024.10768173`. They integrate exactly this constraint — avoiding collisions between the gripper and the partially constructed structure — into LEGO sequence planning, with an assembly-by-disassembly algorithm, instance segmentation combined with stud detection, and named placement techniques ("Safe Approach," "Wiggling Technique").

Delete that claim from the plan. Cite Barghi instead, and note that they used a 3-DoF delta robot with a two-finger gripper, so the dual-arm and dexterous-hand extension is where your differentiation has to come from.

### 2.4 The image→structure and text→structure stretch goals

Both are done and already iterated on.

- **BrickGPT / LegoGPT (Pun et al., 2025, ICCV** `10.1109/iccv51701.2025.01373`, arXiv `10.48550/arxiv.2505.05469`**)** generate physically stable LEGO from text via an autoregressive LLM with **an efficient validity check and physics-aware rollback during inference that prunes infeasible token predictions using physics laws and assembly constraints.** They release StableText2Lego (47k+ structures) and show the designs assembled by robotic arms.

  My proposed §14 stretch — LLM emits JSON, validator checks overlap/connectivity/stability, feed violations back, cap at 5 iterations — **is their published method.** Do not present it as novel.

- **Rollback-Free Stable Brick Structures (2026,** `10.48550/arxiv.2605.06947`**)** has already superseded that: they move physical-validity enforcement from test-time rejection sampling to training-time RL with assembly-level rewards, achieving rollback-free generation orders of magnitude faster. BrickAnything (`10.48550/arxiv.2605.26182`) and LegoACE (`10.1145/3757377.3763881`) cover geometry-conditioned and multi-view-conditioned generation respectively; BrickNet (`10.48550/arxiv.2604.22984`) covers graph-based build sequences over thousands of real part types.

**Revised recommendation:** drop both stretch goals entirely. Use BrickGPT or StableLego as an *input source* for target structures and spend the freed time on evaluation. The generation problem is crowded and moving fast; the manipulation problem is where your contribution is.

---

## 3. The RL Insertion Core — Well Validated, With Two Corrections

This is the part of the plan that holds up best.

### 3.1 What is confirmed

**IndustReal (Tang et al., 2023,** `10.48550/arxiv.2305.17110`**)** proposes simulation-aware policy updates, signed-distance-field rewards, sampling-based curricula, and a policy-level action integrator to minimize error at deployment. Every one of those appears in your plan and each is correctly attributed.

**FORGE (Noseworthy et al., 2024,** *IEEE RA-L* `10.1109/lra.2025.3551637`**)** is the closest published system to what you're building. It combines a force-threshold mechanism with dynamics randomization for robust transfer under significant pose uncertainty; policies are **conditioned on a maximum allowable force** and adaptively perform contact-rich tasks while avoiding aggressive behaviour regardless of controller gains; they predict task success for efficient termination. Most relevantly: **they demonstrate forceful insertion of snap-fit connectors.**

Two things to steal from FORGE that my plan lacks:
1. **Condition the policy on a force budget** rather than only penalizing force in the reward. This is a cleaner mechanism and it directly addresses my "slam-and-snap" failure mode (v2 §7.5) by construction instead of by reward tuning.
2. **Learn a success predictor** as an auxiliary head. My `VerifySnap` (v2 §12.2) is a hand-coded threshold check; a learned predictor enables early termination and autonomous threshold tuning.

**Impedance / variable compliance action spaces** are strongly supported: Beltran-Hernandez et al. (`10.3390/app10196923`, 193 citations) for variable compliance control on position-controlled manipulators with hole-position uncertainty; Chen et al. (`10.1109/tcyb.2023.3310505`) for multimodal vision + proprioception + F/T compact representations with impedance control and domain randomization embedded in the policy to narrow the sim-real gap. Hou et al. (`10.1109/tase.2020.3024725`) work in an impedance action space with a fuzzy-logic baseline. My A8 ablation (6-D vs 9-D with learned stiffness) is well-motivated by this literature.

**Curriculum learning for snap-fits specifically** is validated by Monnet et al. (`10.1515/auto-2024-0177`), who combine RL and curriculum learning for robot snap-fit assembly with adapting part geometries and friction coefficients during training, reaching **over 95% success after iterative adjustment.** That is close to your Gate 3 target of 85%, which suggests the target is achievable but not trivially so.

### 3.2 Correction 1: my reward may be over-engineered

**Marougkas et al. (2025), ICRA,** `10.1109/icra55743.2025.11128860`, address insertion under sub-millimetre tolerance. They note that recent RL efforts *often depend on careful definition of dense reward functions*, and instead use a potential-field-based model-based controller to acquire a policy given full observability in simulation, then integrate it with **residual RL trained on only a sparse, goal-reaching reward**, with a curriculum over observation noise and action magnitude. Trained purely in simulation, zero-shot transferred, and it **outperforms recent RL-based methods in this domain and prior hybrid-policy efforts.**

This is a direct challenge to plan v2 §7.1, where I gave you a six-term hand-weighted reward. Their result suggests the model-based prior carries the shaping burden and the dense reward is unnecessary.

**Revised recommendation:** reorder the training schedule. Make **Run B (residual) the primary** and Run A (end-to-end dense reward) the ablation, not the other way round. Add a sparse-reward variant of Run B. This also de-risks the schedule — residual converges faster, which matters in a 7-week budget.

### 3.3 Correction 2: whether you need vision at all is genuinely contested

I asserted a vision + F/T multimodal observation. The literature is split, and it splits in a way that should make you take ablations A2/A3 seriously rather than treating them as box-ticking.

Evidence that **force alone suffices** at close range:
- Markert et al. (`10.1109/icmla58977.2023.00155`): a state vector consisting **only of F/T signals from the tooltip, with no position information**, reaching **100% success at 0.2 mm clearance** via offline-trained DQN.
- Nguyen et al. (ICRA 2024, `10.1109/icra57147.2024.10610103`): a partially observable formulation with a memory-based agent acting **purely on haptic and proprioceptive signals**, comparable to or outperforming a state-based agent.
- PolyFit (`10.1109/iros58592.2024.10802554`): abandons RL entirely for **F/T-based supervised learning** of extrinsic pose, reporting 96.7% and 91.3% real-world success on unseen polygon shapes.
- Cao et al. (`10.1109/lra.2024.3404749`): F/T-based insertion succeeding when **pose uncertainty exceeds task clearance by more than 10×**.

Evidence that **multimodal helps**: Chen et al. (`10.1109/tcyb.2023.3310505`), and the general multimodal representation line.

There's a physical reason to expect force to dominate here: at the final 3 mm of a LEGO insertion, the brick and the gripper occlude the socket. A wrist camera is looking at the back of the brick it is holding.

**Revised recommendation:** run **A3 (no vision) first, not last.** If a force-only policy matches the multimodal one, you have simplified your system, saved the entire vision phase (Week 6), and produced a clean negative result worth reporting. If it doesn't, you've earned the vision module. Either way, this is a better use of Week 4 than assuming the answer.

There is also precedent for a much more direct approach to snap detection than my force-threshold gate. **Kumar et al. (2025)**, below, detect snap engagement from **joint-velocity transients using proprioception alone**, with over 96% recall.

---

## 4. The Dual-Arm Decision — Validated, But Not Novel, And Contested

### 4.1 The role split is an established paradigm

**Grannen et al. (2023), "Stabilize to Act: Learning to Coordinate for Bimanual Manipulation,"** `10.48550/arxiv.2309.01087` (70 citations). Their framing, which is almost word-for-word the justification I gave you: a **stabilizing arm holds an object in place to simplify the environment while an acting arm executes the task**, motivated by the high dimensionality of the bimanual action space. BUDS achieves 76.9% success across four tasks from 20 demonstrations and is **56.0% more successful than an unstructured baseline** — because of the precision these tasks require.

That last clause is the strongest available support for your architecture: the structured role assignment wins specifically *because the task demands precision*. LEGO insertion is a precision task.

Reinforced by **ManiGaussian++ (IROS 2025,** `10.1109/iros60139.2025.11246564`**)**, which builds a leader-follower hierarchical world model that explicitly differentiates acting and stabilizing arms, reporting 20.2% improvement over prior bimanual SOTA.

**So:** cite BUDS as the paradigm you're instantiating. Do not present stabilizer+placer as your idea.

### 4.2 Where to brace is already a learned problem

**A3D (2026),** `10.48550/arxiv.2601.11076`: dual-arm furniture assembly where one arm manipulates parts while the other provides collaborative support and stabilization. They **learn adaptive affordances to identify optimal support and stabilization locations**, using dense point-level geometric representations, with an adaptive module that adjusts support strategy from interaction feedback as the assembly state evolves.

My v2 §4.2 bracing rule — static analysis, brace on the nearest structurally-connected placed brick — is the naive baseline for this. Combined with StableLego's ability to locate the weakest part of a structure and BrickSim's internal force distribution, you have a much better principled option available: **brace at the joint the QP says will fail first.** That is a concrete, defensible improvement and it's an integration of existing tools rather than new theory.

### 4.3 Someone has already done dual-arm snap-fit assembly

**Kumar et al. (2025), "A Coordinated Dual-Arm Framework for Delicate Snap-Fit Assemblies"** (arXiv `10.48550/arxiv.2511.18153`; Humanoids version `10.1109/humanoids65713.2025.11203142`). They introduce SnapNet, which detects snap-fit engagement from joint-velocity transients in real time using proprioception only, and a dynamical-systems dual-arm coordination framework with event-triggered impedance modulation, reporting **over 96% detection recall and up to 30% reduction in peak impact forces** versus standard impedance control.

This is uncomfortably close to your system. The differences you can defend:
- They use a coupled dynamical system, not a learned policy. You're doing RL.
- They do one snap-fit at a time; you're doing **long-horizon, multi-brick assembly where every insertion changes the structure being braced.**
- Their arms *coordinate with selective decoupling during insertion*; they do not have one arm actively resisting reaction force at a computed weak joint.

The 30% peak-force reduction figure is also your benchmark. If your bracing doesn't beat that, say so.

### 4.4 A genuine counter-argument to the whole approach

**Stavridis et al. (2018), IROS,** `10.1109/iros.2018.8593928`. They argue explicitly **against** using one arm as a fixture: rather than utilizing one arm as a fixture for holding one of the parts while the other performs the assembly, they propose motion generation in the relative end-effector frame involving both arms, with a task-priority strategy optimizing motion and force capabilities.

This is a real position in the literature and your plan should address it rather than ignore it. The counter-argument for your case: in LEGO assembly the base is fixed to a baseplate, so relative-motion generation has nothing to gain — the sub-assembly cannot be repositioned to a more manipulable configuration. State that explicitly; it's a good justification and it shows you read the opposing view.

### 4.5 My anti-dexterous-hand argument was too strong

I told you a multi-fingered hand is the wrong tool for the placer. **Lee et al. (2022), *IEEE RA-L*,** `10.1109/lra.2022.3187497`, implement peg-in-hole using dual arms *and* dexterous robotic hands, with "advanced blind grasping," in-hand manipulation for workpiece reorientation, feed-forward task-space force control, and a four-stage assembly strategy built from "perturbation pattern" unit motions, demonstrated on a 50-DoF upper-body robot with a keyhole-like shape.

So it demonstrably works. My argument should be narrowed from "wrong tool" to "**adds a sample-complexity and contact-count burden disproportionate to a 7-week simulation budget**," which is still true and is now a scoping argument rather than a technical claim. Keep the hand on the stabilizer (where A3D's affordance framing supports it), and cite Lee et al. when you note that placer-side hands are viable but out of scope.

Relatedly, on fingertip design: Hartisch et al. (`10.1109/lra.2026.3655310`) optimize compliant finger geometry in simulation against task-level success, increasing tolerable workpiece variation by 2.29×. Notably for you, they report that **failure during insertion has a small sim-to-real gap, whereas failure during search and in-hand slip has a much larger one** — which is a caution about how much to trust simulated grasp-stability results.

---

## 5. Teacher–Student Distillation — A Failure Mode I Didn't Warn You About

My plan (v2 §7.4) has you train a privileged state teacher, then distill a vision student. The paradigm is well-supported — asymmetric actor-critic and expert distillation are standard, and Cai et al. (`10.48550/arxiv.2412.00985`) provide theoretical grounding, including polynomial sample and computational complexity for expert distillation under a "deterministic filter condition."

But the same paper identifies **a pitfall of expert distillation in finding near-optimal policies**, and two recent papers name the practical version:

- **Messikommer et al. (2024), "Student-Informed Teacher Training,"** `10.48550/arxiv.2412.09149`: the student may be **unable to imitate the teacher because of partial observability**, since the teacher is trained without considering whether the student can imitate the learned behaviour. Their fix is joint training — penalizing the teacher for the approximated teacher–student action difference, plus a supervised alignment step.
- **Kim et al. (2025), IROS,** `10.1109/iros60139.2025.11247406`: existing approaches either produce realizable-but-suboptimal teachers or force the student to explore missing information alone; both are inefficient. They have the student query the teacher adaptively and reset from recovery states.

**This is a live risk for you specifically.** Your privileged critic sees true brick-to-socket relative pose and per-stud engagement depth. A vision+F/T student can observe neither during the final millimetres, when the brick occludes the socket. A teacher that learns "move exactly 0.4 mm in +x because I can see the offset" is teaching a behaviour the student cannot reproduce.

**Revised recommendation:**
1. Use the **informed asymmetric actor-critic** framing (Ebi et al., `10.48550/arxiv.2509.26000`): privileged signals need not be full state, any state-dependent privileged signal yields unbiased policy gradients, and carefully selected signals can match or outperform full-state baselines while using strictly less information. Choose critic privileges the student can plausibly *infer* from force history, rather than the full pose.
2. Budget for the failure explicitly. If A3 (§3.3) shows force-only works, this entire risk evaporates — another reason to run A3 first.
3. Consider ACGD's dual IL/RL objective with an expert critic providing both action labels and value estimates (`10.1109/iros60139.2025.11247025`) rather than plain behaviour cloning.

### 5.1 The compute warning

**VIRAL (2025),** `10.48550/arxiv.2511.15200`, is the most directly comparable vision-distillation recipe: privileged RL teacher on full state with a delta action space, vision student distilled via large-scale simulation with tiled rendering, trained with a mixture of online DAgger and behaviour cloning. Their finding you need to read: **compute scale is critical — scaling simulation to tens of GPUs (up to 64) makes both teacher and student training reliable, while low-compute regimes often fail.**

You have one RTX 5090D. My v2 estimate of 512–1024 vision environments and ~6 hours for Run C is probably optimistic by a wide margin. This is the strongest single argument for running A3 first and possibly skipping the vision phase altogether.

---

## 6. The Clutch Physics — Theory Already Exists

My v2 §3.4 proposed calibrating the analytic clutch against a VBD deformable thin-shell reference, framed as the novel contribution. Two corrections.

**The analytical theory is published.** Yoshida et al. (2020), "Mechanics of a Snap Fit," *Physical Review Letters* `10.1103/physrevlett.125.194301`. They analyze precisely the configuration you care about — **a rigid cylinder and a thin elastic shell** — combining theory, simulation and experiment, constructing a phase diagram over geometric parameters, identifying four distinct mechanical phases, and deriving analytical predictions from linear elasticity theory combined with static friction. They explain the **operational asymmetry** of snap fits: easy to assemble, difficult to disassemble, emerging from the interaction of geometry, elasticity and friction.

A LEGO stud is a rigid cylinder; a LEGO cavity wall is a thin elastic shell. This paper gives you a closed-form model, the insertion/extraction asymmetry your clutch model should reproduce, and a phase diagram to check against — **without running a VBD simulation at all.** Use it as the primary calibration reference; use VBD only to check the analytic model in the regime where its assumptions break.

**The "rigid sim + joining model" pattern is established.** Lämmle et al. (2022), *Procedia CIRP* `10.1016/j.procir.2022.05.151`, extend rigid multi-body simulation with physical joining models specifically because established engines cannot simulate deformation during snap-fit assembly, and validate that the extension generates valid RL training data from standard rigid-body simulations. Same architecture, same motivation, published four years ago. Yoon et al. (`10.48550/arxiv.2202.13098`) take the data-driven route for tight-tolerance contact simulation with real experimental validation.

For empirical snap-fit force magnitudes, the FEA literature gives you ranges to sanity-check against: cantilever snap-fit studies report insertion forces spanning roughly 3–35 N and retention forces 1.7–41 N depending on beam geometry, with simulation-to-experiment errors under ~2.4% for well-designed models (`10.15282/ijame.22.3.2025.3.0960`, `10.48084/etasr.6715`). My 8–15 N per stud estimate sits inside that envelope, which is reassuring but should be cited rather than asserted.

---

## 7. What Is Actually Left That's Novel

Being blunt: the system as specified is an integration of published components. That is a perfectly good engineering project and a weak paper. Here are the four gaps that survived this review, ranked by how defensible they are.

**(1) Quantified bracing for interlocking-brick assembly.** BUDS established the role split qualitatively. A3D learns *where* to support for furniture. StableLego locates weak joints. BrickSim computes internal force distribution. **Nobody has closed the loop: brace at the joint the force model predicts will fail, and measure the effect on insertion success and peak force in a multi-brick assembly.** Kumar et al.'s 30% peak-force reduction is your benchmark. This is the strongest remaining contribution and it is essentially your ablation A6 promoted to the main result.

**(2) GPU-batched snap mechanics for RL scale.** If BrickSim's convex QP at ~5 ms per solve doesn't batch to thousands of parallel environments, a Warp-kernel formulation that does is a real systems contribution — with a fidelity comparison against BrickSim as the validation. Determine this in Week 1; it decides whether you have one contribution or two.

**(3) Does snap-model fidelity change what the policy learns?** (Ablation A10.) BrickSim validates stability *prediction* against reality. Nobody has asked whether a higher-fidelity mating model produces measurably different learned insertion behaviour — different force profiles, different search strategies. If it does, that's a result with implications well beyond LEGO. If it doesn't, that's a useful negative result about how much simulation fidelity contact-rich RL actually requires.

**(4) Force-only vs multimodal at sub-millimetre clearance under self-occlusion.** The literature genuinely disagrees (§3.3), and the LEGO case has a clean physical argument for occlusion. A careful ablation here is publishable on its own.

Note that **none of these require the build planner, the perception module, or the image/text input.** That tells you where to cut.

---

## 8. Revised Plan — Concrete Deltas

| Plan section | Change |
|---|---|
| v2 §0 Scope | Add: target structures sampled from **StableLego**, not hand-authored. |
| v2 §3.3 Clutch kernel | **Replace "build" with "evaluate BrickSim, then extend if it doesn't batch."** 3-day Week-1 gate. |
| v2 §3.4 VBD calibration | Demote. Calibrate against **Yoshida et al.'s analytic model** first; VBD only for regime-of-validity checking. |
| v2 §4.2 Tiling/stability | Replace graph min-cut with StableLego's force-balance formulation. |
| v2 §4.2 Sequencing | Replace greedy sort with **physics-aware action masking** (Liu et al. 2024), or cite it and justify the simpler method. |
| v2 §4.2 Bracing | Replace static-analysis heuristic with **brace at the QP-predicted weakest joint**. This is now a contribution, not a utility. |
| v2 §4.3 | Delete the "under-explored" claim. Cite Barghi et al. |
| v2 §7.1 Reward | Add a **sparse-reward variant**; add **force-budget conditioning** (FORGE) instead of relying on the force penalty term. |
| v2 §7.4 Run order | **Residual (Run B) becomes primary.** Add SnapNet-style joint-velocity engagement detection as a cheap alternative to the force gate. |
| v2 §7.4 Distillation | Use **informed asymmetric AC** — choose critic privileges the student can infer. Budget for teacher-unrealizability. |
| v2 §12.4 Ablations | **Run A3 (no vision) in Week 4, before building the vision module.** Promote A6 from ablation to main experiment. |
| v2 §11 Phase 4 | Vision phase becomes **conditional** on A3's outcome. VIRAL's compute finding says a single GPU may not be enough. |
| v2 §14 Stretch | **Delete both.** Already published (BrickGPT) and already superseded (Rollback-Free). |

Net effect: you lose roughly two weeks of build work and gain two weeks of experiments. That's the right trade for a 7-week project that needs a result rather than a demo.

---

## 9. Bibliography

**Brick-specific simulation and planning**
- Wen et al. (2026). *BrickSim: A Physics-Based Simulator for Manipulating Interlocking Brick Assemblies.* `10.48550/arxiv.2603.16853`
- Liu et al. (2024). *StableLego: Stability Analysis of Block Stacking Assembly.* IEEE RA-L. `10.1109/lra.2024.3414281`
- Liu et al. (2024). *Physics-Aware Combinatorial Assembly Sequence Planning Using Data-Free Action Masking.* IEEE RA-L. `10.1109/lra.2025.3555074`
- Barghi et al. (2024). *From Bricks to Bots: Automated Collision-Aware Sequence Planning for LEGO Reconstruction with a Two-Finger Gripper.* `10.1109/iccia65044.2024.10768173`
- Luo et al. (2015). *Legolization: Optimizing LEGO Designs.* ACM TOG. `10.1145/2816795.2818091`
- Testuz et al. (2013). *Automatic Generation of Constructable Brick Sculptures.* `10.2312/conf/eg2013/short/081-084`
- Kollsker et al. (2021). *Optimisation and Static Equilibrium of Three-Dimensional LEGO Constructions.* `10.1007/s43069-021-00062-3`
- Kollsker et al. (2021). *Models and Algorithms for Optimising Two-Dimensional LEGO Constructions.* EJOR. `10.1016/j.ejor.2020.07.004`
- Ma et al. (2022). *Planning Assembly Sequence with Graph Transformer.* ICRA. `10.1109/icra48891.2023.10160424`
- Ahn et al. (2022). *Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction.* TMLR. `10.48550/arxiv.2210.01021`
- Xu et al. (2019). *Computational LEGO Technic Design.* ACM TOG. `10.1145/3355089.3356504`

**Brick generation (all superseding the stretch goals)**
- Pun et al. (2025). *Generating Physically Stable and Buildable Brick Structures from Text* (BrickGPT). ICCV. `10.1109/iccv51701.2025.01373` / `10.48550/arxiv.2505.05469`
- Xu et al. (2026). *Rollback-Free Stable Brick Structures Generation.* `10.48550/arxiv.2605.06947`
- Ni et al. (2026). *BrickAnything: Geometry-Conditioned Buildable Brick Generation.* `10.48550/arxiv.2605.26182`
- Kulits et al. (2026). *BrickNet: Graph-Backed Generative Brick Assembly.* `10.48550/arxiv.2604.22984`
- Xu et al. (2025). *LegoACE: Autoregressive Construction Engine for Expressive LEGO Assemblies.* SIGGRAPH Asia. `10.1145/3757377.3763881`

**Contact-rich RL assembly**
- Tang et al. (2023). *IndustReal.* `10.48550/arxiv.2305.17110`
- Noseworthy et al. (2024). *FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation Under Uncertainty.* IEEE RA-L. `10.1109/lra.2025.3551637`
- Marougkas et al. (2025). *Integrating Model-Based Control and RL for Sim2Real Transfer of Tight Insertion Policies.* ICRA. `10.1109/icra55743.2025.11128860`
- Monnet et al. (2025). *Leveraging Reinforcement and Curriculum Learning for Flexible Robot-Based Snap-Fit Assembly Automation.* at–Automatisierungstechnik. `10.1515/auto-2024-0177`
- Beltran-Hernandez et al. (2020). *Variable Compliance Control for Robotic Peg-in-Hole Assembly.* `10.3390/app10196923`
- Chen et al. (2023). *Multimodality Driven Impedance-Based Sim2Real Transfer Learning for Robotic Multiple Peg-in-Hole Assembly.* IEEE T-Cyb. `10.1109/tcyb.2023.3310505`
- Zhang et al. (2023). *Efficient Sim-to-Real Transfer with Online Admittance Residual Learning.* `10.48550/arxiv.2310.10509`
- Hou et al. (2022). *Fuzzy Logic-Driven Variable Time-Scale Prediction-Based RL for Multiple Peg-in-Hole.* IEEE T-ASE. `10.1109/tase.2020.3024725`

**Force vs. vision (the contested question)**
- Markert et al. (2023). *Robotic Peg-in-Hole Insertion with Tight Clearances: A Force-Based Deep Q-Learning Approach.* `10.1109/icmla58977.2023.00155`
- Nguyen et al. (2024). *Symmetry-Aware RL for Robotic Assembly under Partial Observability with a Soft Wrist.* ICRA. `10.1109/icra57147.2024.10610103`
- Lee et al. (2023). *PolyFit.* IROS. `10.1109/iros58592.2024.10802554`
- Cao et al. (2024). *On Efficient and Flexible Autonomous Robotic Insertion Assembly in the Presence of Uncertainty.* IEEE RA-L. `10.1109/lra.2024.3404749`

**Bimanual coordination**
- Grannen et al. (2023). *Stabilize to Act: Learning to Coordinate for Bimanual Manipulation.* `10.48550/arxiv.2309.01087`
- Kumar et al. (2025). *A Coordinated Dual-Arm Framework for Delicate Snap-Fit Assemblies.* `10.48550/arxiv.2511.18153` / Humanoids `10.1109/humanoids65713.2025.11203142`
- Liang et al. (2026). *A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation.* `10.48550/arxiv.2601.11076`
- Yu et al. (2025). *ManiGaussian++.* IROS. `10.1109/iros60139.2025.11246564`
- Stavridis et al. (2018). *Bimanual Assembly of Two Parts with Relative Motion Generation.* IROS. `10.1109/iros.2018.8593928` — **the counter-position**
- Lee et al. (2022). *Peg-in-Hole Assembly With Dual-Arm Robot and Dexterous Robot Hands.* IEEE RA-L. `10.1109/lra.2022.3187497`
- Krüger et al. (2011). *Dual Arm Robot for Flexible and Cooperative Assembly.* CIRP Annals. `10.1016/j.cirp.2011.03.017`

**Privileged learning / distillation**
- Messikommer et al. (2024). *Student-Informed Teacher Training.* `10.48550/arxiv.2412.09149`
- Kim et al. (2025). *Distilling Realizable Students from Unrealizable Teachers.* IROS. `10.1109/iros60139.2025.11247406`
- Cai et al. (2024). *Provable Partially Observable RL with Privileged Information.* `10.48550/arxiv.2412.00985`
- Ebi et al. (2025). *Informed Asymmetric Actor-Critic.* `10.48550/arxiv.2509.26000`
- Srinivasan et al. (2025). *ACGD: Visual Multitask Policy Learning with Asymmetric Critic Guided Distillation.* IROS. `10.1109/iros60139.2025.11247025`
- He et al. (2025). *VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation.* `10.48550/arxiv.2511.15200` — **the compute warning**

**Snap-fit / interference-fit mechanics**
- Yoshida et al. (2020). *Mechanics of a Snap Fit.* Phys. Rev. Lett. `10.1103/physrevlett.125.194301`
- Lämmle et al. (2022). *Extension of Established Modern Physics Simulation for the Training of Robotic Electrical Cabinet Assembly.* Procedia CIRP. `10.1016/j.procir.2022.05.151`
- Yoon et al. (2022). *Fast and Accurate Data-Driven Simulation Framework for Contact-Intensive Tight-Tolerance Robotic Assembly.* `10.48550/arxiv.2202.13098`
- Abdul Manan et al. (2025). *Analyzing Insertion and Retention Forces in Cantilever Snap-Fits.* IJAME. `10.15282/ijame.22.3.2025.3.0960`
- Stefanoaea et al. (2024). *A Simulation Method for the One-Time Snap-Fit Assembly Process of PA6 GF60 Components.* ETASR. `10.48084/etasr.6715`
- Hartisch et al. (2025). *Toward Simulation-Based Optimization of Compliant Fingers for High-Speed Connector Assembly.* IEEE RA-L. `10.1109/lra.2026.3655310`
- Müller et al. (2020). *Detailed Rigid Body Simulation with Extended Position Based Dynamics.* CGF. `10.1111/cgf.14105`
