> **UNADOPTED. Renamed from `plan_v4_r5_draft.md` on 2026-09-30 when r5 was adopted with different content (`Docs/plan_v4.md`, "Revision r5").** Row 1 was adopted in r5 (pre-failure, weld-only idealisation; the "WP5 ungrouped" claim is void). Row 3 is subsumed by r5's corner metrics (M1-M4) and returns in r6. Rows 2 and 4-7 need rebasing on r5; row 2's RTF numbers were measured on the old weld. Ledger ids become `*_r6`. The text below still says r5.

# Plan v4 — revision r5 (plan-reviser draft, 2026-09-29, for review)

Line 1 becomes: `# Plan of record v4 (r5) — vision-driven, collision-free dual-arm assembly in Newton`

Append to the status paragraph (line 3):
> **Amended 2026-09-29 (r5, from P1(f)'s real-time measurements, P3 step 8 and P2's look-pose re-run; user answers in ledger `v4_plan_r5`). Milestone dates unchanged; WP6 compute is re-based on P1(f)'s measured throughput.**

Insert above "## Revision r4":

---

## Revision r5 (real time vs structure size, finger end-face graze, P2 protocol)

P1(f) showed that the real-time factor falls as the structure grows, because every seated brick keeps generating contacts with its neighbours and the plate. P3 step 8 showed that at 1× a 2x4 gripped on its 16 mm end faces beside a flush neighbour cannot be separated from a real contact by any finger padding. The P2 implementer tightened gate (c) after review. The reviewer's real-time numbers come from `results/v4/p1_rtf_filter/` (`fexp.py`, `ftime.py`). **Their outputs were not saved as result files**, so r5's (f) re-run saves them (`results/v4/p1_f/`), and that re-run is the gate evidence. P3 step 8's evidence is `results/v4/p3/step8.json`.

**Precedence:** r4's rule stands. An item that fails at one scale routes through the existing branch. r5 adds no per-scale gate items.

| # | change | evidence | what changes |
|---|---|---|---|
| 1 | Seated bricks and the plate share one negative collision group (F1) | Measured on an idle GPU at s = 1, collision once per frame, buffers 32,768 contacts / njmax 65,536. **State:** S3's 16 bricks seated, 14 spares, B sweeping. **Ungrouped:** RTF 0.117 / 0.186 / 0.351 at 16 / 8 / 4 substeps (nacon ≈ 20.8k). The collide step alone takes ≈ 2 ms; the cost is the MuJoCo solver per substep. Weld pool size 0 / 17 / 465 makes no difference. 30 loose bricks: 0.58–0.70 at 16. A timestamped prototype cube build spent 6.5 / 8.9 / 9.6 / 9.1 / 11.7 / 17.3 / 22.2 s of wall time per ≈ 9 s simulated cycle (≈ 1.4× falling to ≈ 0.4×); the earlier "≈ 1×" came from the first 30 s only. **Newton:** shapes in the same negative group never collide with each other and collide with everything else (`test_group_pair`, `newton/_src/geometry/broad_phase_common.py:133–151`, verified). The group is read from `model.shape_collision_group` on every `collide()`, so `assign()` works after graph capture (nacon 20,785 → 2,440, no re-capture). **Grouped (seated bricks + plate):** RTF 0.68 (0.68–0.81 across runs) at 16 substeps and 1.285 at 8; nacon ≈ 2.4k. A 16-brick stack with 17 welds drifts 0.17 / 0.10 mm (max / mean) at 16, the same as unfiltered. **Alternatives measured:** static per-welded-pair filters 0.337 / 0.59 at 16 / 8. Brick and plate gap 10 → 2 mm, ungrouped: 0.12 → 0.24 (nacon 6.3k); irrelevant once grouped. Arm gap 100 → 10 mm (109 of 486 shapes): 0.81, grouped at 16. | §2.2: the plate starts in the structure group G_s = −2. When a brick's first weld fires, the scene writes G_s into that brick's shapes (no re-capture). The brick keeps colliding with arms, fingers, the held brick, tray bricks and every unwelded brick. It stops colliding with other welded bricks and the plate, including partners it did not weld to, so the welds carry the structure. Bricks that are unwelded, `seated_unwelded` or in the tray stay in group 1. P1(d)'s welded-lower rig uses the same grouping, so its sink under load includes the weld's compliance. Under WP5 (U4 = yes) contacts must carry compression again, so WP5 runs ungrouped (§3 WP5). Gaps unchanged. |
| 2 | P1(f) measures two states with grouping; n_sub rule unchanged; WP6 compute re-based (F1) | Row 1. r4's (f) (30 bricks, full pool) fails at 16 and at 8 ungrouped, so r4 sends it to U14. r4's WP6 figures (E4 ≈ 220 builds × 3–5 min, 3–5 h at 4 workers) assumed ≈ 1× per process and linear scaling over 4 workers. Neither was measured. | §3 P1(f): two states, **start** (30 loose bricks, r4's configuration) and **end** (S3 seated, grouped, plus 14 spares). Each is run 3× per n_sub, and the median is gated. U14's rule (D7) is unchanged: 16 if both states reach ≥ 0.8×, else 8, else 16 below real time. **Expected: n_sub = 8** (grouped end state: 0.68 at 16, 1.285 at 8), with kd_f 450; carry creep at 8 measured 0.28 mm, under 0.5. **n_sub guard:** at 8, a D1(ii)/(iii) failure at a scale is re-run at 16 before it is declared. (f) also measures aggregate throughput at 1 / 2 / 4 concurrent workers, which gives WP6's worker count and compute. §2.1's "about real time" is corrected. |
| 3 | Contact monitor under grouping (F1) | Row 1. Welded bricks produce no contact rows with each other, so the contact-based penetration class cannot see structure pairs. P0′'s arch and S3 `grasp_lp` runs show 10–13 and 62–63 penetration events, brace-driven, which is exactly where grouping would hide them. Displaced is pose-based and unaffected. | §2.4: penetration between two welded bricks (or a welded brick and the plate) is computed from ground-truth poses: body-box overlap depth > 1s mm (studs excluded), each frame, scoring side only. Pairs with an unwelded, held or tray brick keep the contact-based rule. Displaced is unchanged. **M0(a) runs ungrouped at P0′'s exact config (16 substeps).** Grouping on vs off on that config is reported in M0. |
| 4 | 1× end-face grasps beside a flush neighbour: a declared, permitted, bounded finger graze; finger contact-segment check on exact boxes (F2, F4(f)) | `results/v4/p3/step8.json`, P0′ cube, 48 insert segments. There are 6 real monitor events, all at b_step 5: both B fingers touch the flush neighbour b_004 in all 6 runs, at 0.45–0.87 N and max penetration −0.015 to −0.027 mm. **Sphere model (fr3.yml):** clean segments reach −0.110 mm (b_step 7) while event segments are at most −0.027 mm, so no padding separates them (pad 0: 12 false positives, pad 1 mm: 36). **Exact URDF finger boxes:** events at most +0.140 mm, clean segments at least +0.181 mm, a 0.04 mm margin, which is below P0′'s 0.04–0.19 mm lateral placement error. **Mechanism:** the fingertip is 17.5 mm wide and a 2x4's end face is 16 mm, so ≈ 0.75 mm of each pad overhangs, coplanar with the neighbour's end face. The gap is 0 ± pad sink (0.09–0.24 mm) ± placement error. `planner.grasp_for` already marks exactly these grasps `accessible = false` (cube steps 1, 3, 5, 7, all 6 runs' flagged segments). There is no alternative grasp: the long faces sit against the neighbour. At 2× (32 mm face) there is no overhang. Declared steps at 1× in `plans/sim_*_weakest_joint.json`: cube 4/8, arch 1/11, hollow_box 0/12, S1 0/6, S3 0/16. **Clean non-declared steps (0/2/4/6) clear by ≥ 0.887 mm (boxes) and ≥ 0.443 mm (spheres).** | `grasp_for` adds `graze_bricks` (the placed bricks its `touched()` count found) to the grasp record. The grasp choice does not change. §2.4: new permitted class **`finger_graze`**: B finger ↔ a `graze_bricks` member, from the final Insert segment through Release and a V7 re-press. It is logged and reported like `neighbour_rub`. It counts as unintended if max penetration > 0.3s mm (r4's pad bound), and a displaced neighbour still counts as displaced. §2.3/P3: fingers are checked as exact URDF boxes in the held brick's numpy OBB test, with the declared pair exempt. P3 step 8 gates 0 false positives on non-declared pairs. D1 reports declared-graze steps per shape and scale as an input, not a criterion. Q2 and the allowed claims name the class. |
| 5 | P2 gate (c) and noise model tightened by the implementer (F3; reviewer: sensible, a gate change after review) | At camera offset 75 mm the held brick's near stud row leaves the frame. V5(b) fitted 2x2 held bricks in 30/50 renders at s = 2 (20 of 23 2x2 cases failed) while gate (c) passed. Edge dropout used a 3 mm depth-jump threshold at both scales. | Gate (c) adds **held-brick studs visible ≥ min(4, n_held_studs)**. The look-pose sweep adds a 2x2 held "small" target. Edge dropout applies at depth jumps > 3s mm. P2's pick after the change is h_look 45s mm and offset 60 mm (full re-run in progress; P2's result, not a plan change). The rotation × offset term at h_look 45s is 0.079s mm, inside the budget's 0.04–0.10 mm. |
| 6 | Implementation definitions recorded (F4(a)–(e)) | (a) Instantaneous `body_qd` on a proxy lower shows a steady false ≈ 1.8 mm/s relative velocity (substep chatter aliased at 60 Hz), so the clutch never fired. (b)–(d) P1 probe protocol as implemented. (e) P3: a sphere-mesh world for the other arm cost 4 s per plan. Without `JointTargetMode.POSITION_VELOCITY` the velocity feed-forward is ignored (p99 deviation 77 mm vs 12.5 mm at time scale 1.0). | (a) §2.2 "at rest" = \|Δrel\|·fps < 1s mm/s for ≥ 6 consecutive frames, where Δrel is the frame-to-frame change of the brick origin's position in the partner's frame. (b) P1(d) F_pt presses from the Insert start height, with a seat-first check before any failure. (c) The FK false-positive rate is conditional on FK-seated. (d) (e) runs right after (d) (`--gm auto`). (e) The planner uses coarse `fr3_plan.yml`; the other arm is one PCA cuboid per link; `fr3_link0` is dropped from planner links; POSITION_VELOCITY is set in `build_arm`; time scale 1.0 is defined; (time scale, padding) smoke pairs are recorded. |
| 7 | Effort | — | WP1 +0.25 (grouping at weld, `graze_bricks`/`finger_graze`, pose-based structure penetration, POSITION_VELOCITY, tests). P1 +0.25 and ≈ +1 h GPU ((f) re-run, throughput, n_sub guard). P3 step 8 re-run: minutes. **Total 22.25–23.25 agent-days without WP5.** M1 dates unchanged, because WP3's path is the critical one. WP6 compute is re-based (§5). |

---

## r5 replacement texts, by section

### §2.1 Simulator

Replace the bullet "It is what the user liked: live ViewerGL, about 10 s to start, about real time." with:
> - It is what the user liked: live ViewerGL, about 10 s to start, and about real time early in a build. The prototype then slows as bricks accumulate: ≈ 1.4× falling to ≈ 0.4× over a cube build (r5), because every seated brick keeps generating contacts. The cell restores real time with the structure collision group (§2.2) and P1(f)'s n_sub.

Replace the **Physics** sentence's ending "n_sub = 16 unless P1(f) picks 8 for real time (U14)." with:
> n_sub = 16 unless P1(f), measured with the structure collision group (§2.2, r5), picks 8 for real time (U14). The expectation is 8.

Add after the finger-contact bullets:
> - Structure collision group (r5): see §2.2. It changes which pairs collide, not any contact parameter. The shape gaps (bricks and plate 10 mm, arm shapes 100 mm as imported) are unchanged.

### §2.2 Cell and brick model

Replace "At s = 2 the 17.5 mm finger pads stop overhanging a 2x2 face, which removes v3.1's flanked-2x2 inaccessibility (so S5 becomes feasible)." with:
> At s = 2 the 17.5 mm finger pads stop overhanging a 16s mm face. That removes v3.1's flanked-2x2 inaccessibility (so S5 becomes feasible) and the 1× end-face graze of a 2x4 placed flush beside a neighbour (r5, §2.4 `finger_graze`).

Replace the "At rest" bullet with:
> - "At rest" and the pose are taken relative to the partner (the lower brick, or the plate). Let Δrel be the frame-to-frame change of the brick origin's position in the partner's frame (r5). On every frame where \|Δrel\|·fps < 1s mm/s has held for ≥ 6 consecutive frames (0.1 s), the scene tests the brick against each unwelded partner whose footprint overlaps. Instantaneous `body_qd` is not used: substep chatter aliases into a steady false ≈ 1.8 mm/s on a proxy lower, and the clutch never fired with it.

Replace "On pass: the weld's `relpose` is set to that lattice pose and it is enabled. The snap is independent of the plan, so any tray brick of the right type can be used for any step." with:
> - On pass: the weld's `relpose` is set to that lattice pose and it is enabled. The snap is independent of the plan, so any tray brick of the right type can be used for any step.
> - **Structure collision group (r5).** The plate's shapes are in group G_s = −2 from the start. When a brick's first weld fires, its shapes are set to G_s: one write to `model.shape_collision_group`, read on every `collide()`, so no graph re-capture is needed.
>   - Same-negative-group shapes never collide; a negative group collides with every other group (`test_group_pair`). A welded brick therefore still collides with the arms, the fingers, the held brick, tray bricks and every unwelded brick. It no longer collides with other welded bricks or the plate, including partners it did not weld to. The welds carry the structure; M1's welds never release.
>   - Unwelded, `seated_unwelded` and tray bricks stay in group 1.
>   - Why: S3's seated state generates ≈ 20.8k contacts (RTF 0.117 at 16 substeps). Grouped: ≈ 2.4k, RTF 0.68 at 16 and 1.285 at 8. A 16-brick stack with 17 welds drifts 0.17 / 0.10 mm (max / mean), the same as ungrouped.
>   - Consequences: contacts among welded bricks carry no information under M1, so §2.4 computes their penetration from poses. WP5 (breakable joints) needs compressive contacts among seated bricks, so it runs ungrouped (§3 WP5).

In "Joint model, M2", append:
> WP5 runs without the structure collision group (r5): the detent-and-break model needs brick–brick contacts to carry compression, and grouping would route compression through the weld wrench that feeds `utilization()`.

### §2.3 Arm control

Replace the first bullet's "Written to `control.joint_target_pos` **and `joint_target_vel`** (velocity feed-forward cuts PD lag)." with:
> Written to `control.joint_target_pos` **and `joint_target_vel`**; the arm dofs use `JointTargetMode.POSITION_VELOCITY` (set in `dual_arm_sim.build_arm`, r5). Without it the feed-forward is ignored: P3 measured a p99 deviation of 77 mm against 12.5 mm at time scale 1.0. Time scale 1.0 is cuRobo's timing at the URDF velocity limits and 10 rad/s² acceleration.

Replace the contact-segment sub-bullets (from "**arm links and finger spheres**" to "P3 measures the false-positive rate on the cube's insert segments.") with:
> - **arm links** vs the world model minus only the brick being grasped (B) or the braced `gripped_bricks` (A), using cuRobo's `RobotCollisionChecker` with `motion/fr3.yml`'s spheres (r5: the fingers are no longer checked as spheres).
> - **fingers and the held brick** as exact boxes, in one numpy OBB test against the world's cuboids (r5).
>   - The fingers use the URDF finger collision boxes. They are checked against the world minus the grasped or braced bricks and minus the step's declared `graze_bricks` (§2.4).
>   - The held brick is checked against the world minus its supports and same-course neighbours.
>   - Why boxes: the sphere fit over-approximates the pad corners. On P0′'s cube, clean segments reached −0.110 mm while real events were at most −0.027 mm, so no padding separated them. The boxes are what Newton collides.
>   - Finger padding is P3 step 8's choice from {0, 0.25, 0.5} mm.
> - A failure triggers a replan with the next grasp yaw or approach. P3 measures the false-positive rate on the cube's insert segments.

### §2.4 Collision-free motion and coordination

Replace the **Robot model** paragraph with:
> **Robot model:** the same `fr3_franka_hand.urdf` Newton loads. P3 generated two sphere sets with `curobo.sphere_fit`:
> - `motion/fr3_plan.yml`, coarse (382 spheres; arm links ≤ 2 mm; fingers 2 mm), for planning;
> - `motion/fr3.yml` (fingers ≤ 0.5 mm), for the arm links in the contact-segment check (§2.3).
>
> `fr3_link0` (static) is not a planner collision link. **Padding comes from P3's measured executed deviation** (p99 + 2 mm, at least 5 mm).

In **World model**, replace "**the other arm** as its link spheres at its current or hold configuration." with:
> **the other arm** as one cuboid per link (a PCA box containing that link's planning spheres), at its current or hold configuration (r5: a sphere-mesh world cost 4 s per plan in P3).

In **Contact monitor**, "Permitted, with their phase windows", add after the held-brick ↔ supports bullet:
> - **B fingers ↔ the step's declared `graze_bricks`** (r5; `planner.grasp_for`'s `accessible = false` grasps), from the start of the final Insert segment through Release, including a V7 re-press. At 1× a 2x4 gripped on its 16 mm end faces overhangs each 17.5 mm pad by ≈ 0.75 mm, coplanar with a flush neighbour's end face. The gap is 0 ± pad sink ± placement error, and no grasp avoids it (cube steps 1, 3, 5, 7; arch 1 step; none at 2×).
>   - Logged as `finger_graze` (max penetration, peak force, duration) and reported with `neighbour_rub`, not counted as unintended.
>   - It **counts as unintended** if its max penetration exceeds 0.3s mm (the pad-penetration bound).
>   - A graze that moves the neighbour is still counted in the displaced class.

Replace the **Penetration event** bullet with:
> - **Penetration event:** any brick–brick pair with d < −1·s mm (catches pushed-through bricks). D1(iii) keeps the legitimate sink under the press ≤ 0.5s mm, so the threshold stays meaningful.
>   - Pairs of two welded bricks, or a welded brick and the plate, generate no contacts under the structure group (§2.2, r5). For those pairs the monitor computes the overlap depth of the brick body boxes (studs excluded) from ground-truth poses each frame and uses the same 1s mm threshold (scoring side only).
>   - Pairs with an unwelded, held or tray brick keep the contact rule.

(The displaced class is pose-based and unchanged.)

### §1.1 Q2

Replace the *Positive* line with:
> *Positive:* zero arm–arm contacts (single-frame sensitivity), including on braced placements and in the zone stress test; ≤ 1 unintended arm/held-brick event per 50 placements; cycle time ≤ 1.5× the prototype's. Declared 1× end-face finger grazes (`finger_graze`, §2.4, r5) are reported separately and count as unintended only above 0.3s mm penetration.

### §2.5.2 Cameras

Replace the noise-model bullet with:
> - **Noise model:** depth σ = 2e-3·r² m on the ray distance (20 µm at 0.1 m, 2.4 mm at 1.1 m), plus 1 % dropout at depth edges. A depth edge is a depth jump > 3s mm between neighbouring pixels (r5; it was 3 mm at both scales). RGB σ = 2/255. Lighting is randomised per seed (intensity 0.5–2.0, direction).

### §3 P1

Replace the "Order (r4)" bullet with:
> - Order (r5): (f) first, with no other GPU job running, to fix n_sub. Then (a), (d), (e), (b), (c) and (h) at n_sub; (g) any time. (e) follows (d) directly, so the grip multiple m is chosen (`--gm auto`) before (b) and (c). Rows run at another m are flagged and re-run.

In "Common setup", replace "The lower brick is either a first-course brick on the plate's proxy studs, or a brick welded to the plate (M1's state)." with:
> The lower brick is either a first-course brick on the plate's proxy studs (unwelded, group 1), or a brick welded to the plate (M1's state). In the welded case, the lower brick and the plate are in the structure group, as in the cell (r5), so the weld alone carries the press and the sink under load includes its compliance.

In (b), replace "and the FK-seated false-positive rate: FK seated at the end of the hold, but no weld after release." with:
> and the FK-seated false-positive rate: of the trials FK calls seated at the end of the hold, the fraction with no weld after release (conditional on FK-seated, r5; stricter than the joint rate).

In (d), replace the F_pt bullet with:
> - F_pt(s): the smallest F_cap at which, in any trial, the brick sinks > 1s mm below seated or a brick–brick contact reaches d < −1s mm. The press starts from the Insert's start height, and the cap is held for 1 s after the stop.
>   - r5: r4 said "seated first". Pressing from the start height is conservative, because the impact can only lower F_pt and raise the sink.
>   - If D1(ii) or (iii) fails at a scale and not every trial at that level was seated when the cap triggered (`n_seated_at_trigger`), the level is re-run seated first before the failure is declared.

Replace (f) with:
> - (f) Real-time factor, run first on an idle GPU (r5):
>   - **Two states.** Both have 2 arms and the plate, with the full weld pool allocated (465; pool size does not change the cost) and B sweeping between the tray and the zone:
>     - **start:** 30 loose bricks (r4's configuration);
>     - **end:** S3's 16 bricks seated and welded, in the structure group with the plate (§2.2), plus 14 loose spares.
>   - Each state runs at n_sub ∈ {16, 8, 4}, collision once per frame and every substep, 3 runs of ≥ 30 s simulated. RTF = simulated / wall time after warm-up. The median of the 3 runs is gated. The end state also runs once ungrouped per n_sub, for the record. Output goes to `results/v4/p1_f/`.
>   - n_sub = 16 if both states reach 0.8× at 16; else 8 if both reach 0.8× at 8; else U14.
>   - 4 is excluded: kd_f ≤ 240 gives an 8.3 ms finger time constant, and carry creep scales with it (≈ 0.6 mm predicted, over the gate).
>   - **Throughput for WP6:** 1, 2 and 4 concurrent headless processes of the end state at the chosen n_sub give the aggregate simulated seconds per wall second, A(k). WP6 uses the k with the largest A(k) within P3's memory bound.
> - **n_sub guard (r5):** if n_sub = 8 and D1(ii) or (iii) fails at a scale, (d) is repeated at 16 substeps (kd_f 900) at that scale before the failure is declared. If 16 passes, D1 presents the choice: that scale at 16 substeps below real time (U14's third branch), or the other scale. Why: §0 fact 6 found the pass-through threshold depends on substeps, and a viewing criterion should not silently decide the scale.

In "Gate P1", replace the last item with:
> - real time (r5): both (f) states reach ≥ 0.8× at n_sub, medians of 3 runs (else U14).

### §3 P2

In the look-pose sweep, replace "for each grasp orientation (14.8 mm grasp, long axis pointing at the camera; 4-stud-axis grasp) and target type" with:
> for each grasp orientation (14.8 mm grasp, long axis pointing at the camera; 4-stud-axis grasp), held-brick size (2x4, and a 2x2 "small" held brick, r5) and target type

In Gate P2 (c), add after "the held brick's camera-side face ≥ 500 px.":
> - **held-brick studs visible ≥ min(4, n_held_studs)** (r5). Without this item gate (c) passed at the 75 mm offset while V5(b) fitted 2x2 held bricks in only 30/50 renders at s = 2 (20 of 23 2x2 cases failed), because the near stud row left the frame.

### §3 P3

Replace deliverable step 2 with:
> 2. **generate the FR3 sphere sets** with `sphere_fit` on the FR3 link meshes: `motion/fr3.yml` (every vertex within 2 mm of the sphere union, within 0.5 mm on the finger links) and the coarse planning set `motion/fr3_plan.yml` (382 spheres; arm ≤ 2 mm, fingers 2 mm) (r5);

Replace deliverable step 5 with:
> 5. **execute 20 plans in Newton at time scales {1.0, 0.5, 0.3}** (1.0 = cuRobo's timing at URDF velocity limits and 10 rad/s²) with velocity feed-forward under `JointTargetMode.POSITION_VELOCITY`, at P1(f)'s n_sub. Report the max executed-vs-planned link deviation (p99), the executed minimum clearance, and events from the monitor. Smoke (time scale, padding) pairs: 1.0 / 15 mm, 0.5 / 7 mm, 0.3 / 5 mm (r5; full run in progress);

Replace deliverable step 8 with:
> 8. **contact-segment false positives:** the §2.3 per-link check (arm spheres, exact finger boxes, held-brick OBB) runs on every insert segment of P0′'s cube runs (`results/v4/p0_stiff_inserts/`) at finger padding {0, 0.25, 0.5} mm.
>    - The declared finger ↔ `graze_bricks` pairs are exempt. Their clearances are reported: on P0′, events reach +0.140 mm and clean segments +0.181 mm with boxes.
>    - A false positive is a segment flagged on a non-declared pair although the monitor saw no event on it. A miss is a monitor event outside the `finger_graze` class on a segment the check passed.
>    - The finger padding is the largest level with 0 false positives. From `step8.json`, non-declared segments clear by ≥ 0.887 mm with boxes, so 0.5 mm is expected.

In Gate P3, replace "**0 contact-segment false positives** on the cube's inserts." with:
> - **0 contact-segment false positives on non-declared pairs** on the cube's inserts, at the chosen finger padding; misses reported (r5).

In "On failure", replace "False positives → finger padding 0 mm, or a tighter finger fit." with:
> - False positives on non-declared pairs → finger padding 0 mm. If any remain, the pairs go to the plan-reviser (exact boxes leave no tighter fit).

### §3 D1

Add a bullet:
> - declared finger-graze steps per shape at each scale (r5; 1×: cube 4/8, arch 1/11, hollow_box, S1 and S3 0; 2×: none). This is an input the user weighs, not a qualifying criterion.

### §3 WP1

Add to the deliverable:
> - (r5) `cell/scene.py`: the structure collision group (the plate in G_s from the start; each brick's shapes set to G_s when its first weld fires). `dual_arm_sim.build_arm`: arm dofs in `JointTargetMode.POSITION_VELOCITY`. `planner.grasp_for`: `graze_bricks` in the grasp record. `cell/contacts.py`: `finger_graze`, and pose-based penetration for welded pairs. `tests/test_cell.py` adds: a welded pair produces no contact row while the held brick against a welded brick still does; the group is set on weld fire without re-capture; a declared graze is logged as `finger_graze`, and as unintended above 0.3s mm.

Replace Gate M0 (a) with:
> - **(a)** on **P0's exact config** (1×, fixed layout, waypoint IK, 16 substeps, **no structure grouping**) with r4's finger contacts:
>   - the r5 monitor's per-class event counts (merged as §2.4 defines) fall **within P0′'s run-to-run spread over its 3 runs, or ±1 event per class, whichever is larger**;
>   - P0′ is re-scored offline with the r5 classes. Its cube finger–b_004 events become `finger_graze`.

Add to Gate M0:
> - **(d) reported, not gated (r5):**
>   - the (a) configuration once with grouping on: per-class counts and completion, and which penetration events move to the pose-based rule;
>   - the per-cycle real-time factor of the (b) S3 builds at n_sub. If any cycle is below 0.8×, U14's third branch applies to live runs (`--record` videos).

### §3 WP5

Add to the spike:
> - **Grouping (r5):** the spike and WP5 run **without** the structure collision group, because compression among seated bricks must go through contacts, not through the weld wrench that feeds `utilization()`. The spike reports RTF ungrouped at n_sub. E5 is headless and may run below real time.

### §4

E6 row, metrics: replace "startup, × real time, …" with:
> startup, × real time per cycle over each build (r5), per-stage perception latency, plan latency

Limitations: add after the §0 fact 6 item:
> - the structure collision group (r5): welded bricks do not contact each other or the plate, so the welds carry the structure; penetration among them is computed from poses;
> - at 1×, declared end-face finger grazes (`finger_graze`) on 2x4s placed flush beside a neighbour.

Allowed claims: replace "“collision-free” only as measured by the §2.4 monitor;" with:
> - "collision-free" only as measured by the §2.4 monitor. At 1×, declared end-face finger grazes are reported next to it, not hidden (r5);

### §5

Replace these table rows:
> | P1 | 2.75 | ~4.5 h GPU ((f) first on an idle GPU: two states × n_sub × 3 runs, the 1/2/4-worker throughput, and the n_sub guard if it fires; release windows; collision-mode cross-check) | P0, P2, P3 |
> | P3 | 1.75 | ~1.5 h (soak, false-positive check; step 8 re-run under r5: minutes) | P0–P2 |
> | WP1 (M0) | 2 | small | P1–P3 (after P0) |
> | WP6 | 3 | ≈ 23–30 h simulated at 1× (r4's own figures: E4 11–18 h, E2 ≈ 4 h, E3 ≈ 8 h). Wall time = simulated time / A(k), from P1(f)'s measured throughput. r4's "3–5 h at 4 workers" assumed A = 4. | — |

Add under the WP6 row (calendar notes):
> - WP6 compute (r5): 2 days holds if A(k) ≥ 1.5 (≤ 20 h wall). Below that, the orchestrator re-estimates at M0 and tells the user. Ungrouped, the RTF falls to ≈ 0.4 late in a build, and 0.12 in S3's seated state at 16 substeps, which is why r5 groups the structure.

Replace the calendar's last line with:
> - whole phase: **the table sums to 22.25 agent-days (WP4 at 2) to 23.25 (WP4 at 3) without WP5**, and 26.25–27.25 with WP5 (1 + 3), before the WP1 caveat. r5 adds 0.25 to WP1 and 0.25 to P1. M1 dates are unchanged, because WP3's path (days 3.5–7.5) stays critical.

### §6.1 Risks

R14 mitigation: append
> ; end-face grazes at 1× are declared and bounded (`finger_graze`, r5)

Add:
> | R15 | the structure collision group hides contact effects among welded bricks: interpenetration without resistance under a brace, lost load sharing between non-welded neighbours, welds carrying everything | medium / medium | M1's welds never release, so these contacts carry no success information. Pose-based penetration for welded pairs; the displaced class (pose-based); M0(d) on/off comparison; WP5 ungrouped; stated limitation |
> | R16 | n_sub = 8 (for real time) changes press physics: pass-through thresholds depend on substeps (§0 fact 6), and kd_f 450 is underdamped (0.45) | medium / medium | P1 runs at n_sub; the n_sub guard re-runs a failing D1(ii)/(iii) at 16 before a scale fails; carry creep at 8 measured 0.28 mm (< 0.5) |

### §6.2 U14

Replace U14 with:
> - U14 Real time vs substeps (r4; r5 adds the structure collision group and two states):
>   - n_sub = 16 if both P1(f) states (30 loose bricks; S3 seated and grouped plus 14 spares) reach 0.8× at 16 (medians of 3 runs); else 8 if both reach 0.8× at 8 (kd_f 450, P1 measured at 8, with the n_sub guard); else 16, with live runs below real time and `--record` videos for real-time playback.
>   - **Expected: 8** (grouped end state 0.68 at 16, 1.285 at 8).
>   - Headless batches have no real-time requirement, but their compute scales with 1/RTF (§5 WP6).
>   - Alternatives: 16 below real time (≈ 0.68×) for 16-substep press physics; 16 with the arm shapes' gap cut from 100 to 10 mm (0.81 grouped, marginal against run spread 0.68–0.81, and it narrows once-per-frame contact look-ahead for fast arm links). r4's "check at the shown shape's pool" is moot: pool size does not change the cost.

### §6.3 Ledger

Add:
> - On r5 adoption: `probe v4_p1_rtf_structure_growth`, `probe v4_p3_finger_end_face_graze`, `decision v4_plan_r5`, `deviation v4_structure_collision_group`, `deviation v4_p1_protocol_r5`, `deviation v4_contact_monitor_r5`, `deviation v4_p3_protocol_r5`, `deviation v4_p2_protocol_r5`.
> - After P1(f): `result v4_p1_rtf` (both states, n_sub, A(k)). n_sub and kd_f then go into `decision v4_scale` as r4 says. If the n_sub guard fires, that is recorded there too.

---

## What r5 invalidates

- **r4's (f) gate text and any ungrouped (f) verdict.** The reviewer's ungrouped and grouped numbers are recorded as a probe, not as the gate. The (f) re-run is the evidence; the reviewer's outputs were not saved as result files.
- **P1 rows (b)–(e), (h) produced at 16 substeps before (f) is re-read.** They are not gate evidence if n_sub = 8 and must be re-run at 8 (`results/v4/` holds none yet). Rows at another grip multiple are already flagged (F4(d)).
- **P3 step 8 (`results/v4/p3/step8.json`, ok false):** re-run under the r5 check. **P3 step 5:** read at P1(f)'s n_sub; the full run in progress should use it.
- **P2 gate (c) verdicts computed before the held-stud item** (the 75 mm offset pick): replaced by the full re-run in progress.
- **M0(a) reference:** P0′ is re-scored offline with the r5 classes. P0′'s recorded numbers (`v4_p0_baseline_stiff_fingers`, 2 unintended per cube run under r4's classes) stay as recorded; the re-scored counts go in a new entry.
- **§5 WP6 compute and §2.1's "about real time":** restated as above.
- Unchanged: every r4 physics number, the clutch band, D1's rule (i)–(iv), §0 fact 6, the P0 and P0′ records, and all v3.1 results.

---

## Items for the user (recommended default in bold). Items that change a protocol or a conclusion are marked ●

- **r5-1 ● Structure collision group** (modelling assumption; changes what the monitor can see).
  - **Default: adopt.** The plate starts in G_s = −2, and each brick joins G_s when its first weld fires.
  - Alternatives: static per-welded-pair filters (0.337× / 0.59× at 16 / 8: too slow); no filter, with 8 substeps and live runs below real time (0.186×).
- **r5-2 ● P1(f) and n_sub.**
  - **Default:** measure two states (start: 30 loose; end: S3 seated and grouped plus 14 spares), medians of 3, ≥ 0.8×, under the unchanged D7 rule (expected n_sub = 8). Add the n_sub guard, and base WP6 compute on the measured A(k).
  - Alternatives: fix 16 below real time (≈ 0.68×; about 1.9× the compute of 8); 16 with the arm gap cut to 10 mm (marginal).
- **r5-3 ● Monitor under grouping.**
  - **Default:** penetration between welded bricks from pose overlap at the same 1s mm threshold; displaced unchanged; M0(a) ungrouped at P0′'s config, plus the reported on/off comparison.
  - Alternative: restrict the penetration class to pairs with a non-welded brick and state it. This makes M0.5 and M1's "0 penetration" blind exactly where P0′ saw 62–63 per S3 `grasp_lp` run.
- **r5-4 WP5 without grouping** (applies only if U4 = yes): **Default: adopt.** It is decided with U4 after M1.
- **r5-5 ● 1× end-face finger graze** (changes Q2's positive-outcome definition, the P3 gate and the allowed claims).
  - **Default:**
    - a declared, permitted `finger_graze` class, bounded at 0.3s mm penetration, with displacement still counted and reported next to the unintended count;
    - exact finger boxes in the contact-segment check;
    - P3 step 8 gated on non-declared pairs;
    - `grasp_for`'s rule unchanged (it already flags exactly these grasps), plus a `graze_bricks` list;
    - declared-graze counts shown at D1 as an input, not a criterion.
  - Alternatives:
    - treat these grasps as blocked. The cube (the M0/M1 gate shape) becomes infeasible at 1×, which forces s = 2 before P1 has measured anything;
    - keep counting them as unintended. P0′ shows 2 events per 8 cube placements (≈ 12 per 50), so M0.5 and M1's "≤ 1 per 50" fails at 1× by construction.
- **r5-6 ● P2 gate (c) and noise** (a gate change after review): **Default: adopt** the held-stud item min(4, n_held_studs), the 2x2 held target and edge dropout at > 3s mm.
- **r5-7 Recorded, no decision needed:** F4(a)–(e) (at-rest definition, the F_pt start height with a seat-first check, the conditional FK rate, the (d)→(e) order, P3's planner model, cuboid other arm, link0, POSITION_VELOCITY, time-scale definition). (b) cannot make a scale fail spuriously, because of the seat-first check, and cannot make one pass spuriously, because it is conservative.

---

## Ledger entries (doc-writer, after the user's answers)

- `probe v4_p1_rtf_structure_growth`
  - finding: row 1's numbers (ungrouped 0.117 / 0.186 / 0.351; loose 30 at 0.58–0.70; the prototype cube's per-cycle wall times; grouped 0.68 (0.68–0.81) / 1.285; static filters 0.337 / 0.59; gaps; drift 0.17 / 0.10 mm; nacon 20,785 → 2,440 with no re-capture; Newton `test_group_pair` semantics);
  - evidence: `results/v4/p1_rtf_filter/fexp.py`, `ftime.py` (scripts only; figures are the reviewer's, to be confirmed by `result v4_p1_rtf`);
  - implication: r4's (f) fails ungrouped at 16 and 8; WP6 compute was under-estimated.
- `probe v4_p3_finger_end_face_graze`
  - finding: `step8.json` (48 segments; 6 events at b_step 5 vs b_004, 0.45–0.87 N; spheres: clean min −0.110 mm vs events max −0.027 mm, 12 to 36 false positives at 0 to 1 mm padding; boxes: events max +0.140 mm vs clean min +0.181 mm; the 17.5 mm pad vs 16 mm face mechanism; `grasp_for` already flags cube steps 1, 3, 5, 7);
  - implication: r4's P3 step 8 gate cannot pass at 1× as written.
- `decision v4_plan_r5`: this revision, with the answers to r5-1…r5-7; `authorised_by: user`.
- `deviation v4_structure_collision_group`
  - from: every brick and the plate in collision group 1 (all pairs collide);
  - to: the plate in G_s = −2 from the start, and each brick's shapes set to G_s when its first weld fires, so welded bricks do not collide with each other or the plate; WP5 runs without it;
  - rationale: row 1; decision r5-1, r5-4.
- `deviation v4_p1_protocol_r5`
  - from: r4's (f) (30 bricks, full pool, one state), F_pt "seated first", the joint FK false-positive rate, order (a), (d), (b), (c), (e);
  - to: (f) two states with grouping, medians of 3, A(k) throughput, the n_sub guard; F_pt from the Insert start height with a seat-first check; the conditional FK rate; order (a), (d), (e), (b), (c) with `--gm auto`; the welded-lower rig grouped; the §2.2 at-rest definition by Δrel;
  - decision r5-2, r5-7.
- `deviation v4_contact_monitor_r5`
  - from: r4's monitor, with penetration from contacts only and B fingers ↔ placed neighbours counted as unintended;
  - to: pose-based penetration for welded pairs; the `finger_graze` class for declared `graze_bricks` (unintended above 0.3s mm); M0(a) ungrouped at P0′'s config, with P0′ re-scored offline;
  - decision r5-3, r5-5.
- `deviation v4_p3_protocol_r5`
  - from: finger spheres in the contact-segment check, and the gate "0 false positives on the cube's inserts";
  - to: exact finger boxes in the OBB test, declared pairs exempt, gate on non-declared pairs, padding from {0, 0.25, 0.5} mm; also `fr3_plan.yml` for planning, the other arm as PCA cuboids, `fr3_link0` dropped, POSITION_VELOCITY, the time-scale definition and the smoke pairs;
  - decision r5-5, r5-7.
- `deviation v4_p2_protocol_r5`
  - from: gate (c) without a held-stud item, a sweep without a 2x2 held target, edge dropout at a 3 mm jump;
  - to: held-brick studs visible ≥ min(4, n_held_studs), the 2x2 held target, dropout at > 3s mm;
  - rationale: V5(b) fitted 2x2 held bricks in 30/50 renders at s = 2 at the 75 mm offset while (c) passed;
  - decision r5-6.
- Later: `result v4_p1_rtf` (after the (f) re-run); a P0′ re-score entry, `result v4_p0_stiff_rescored_r5`, for M0(a).
