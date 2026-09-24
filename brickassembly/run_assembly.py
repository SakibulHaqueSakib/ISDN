"""Execute an assembly_plan.json with a cuRobo-planned Franka, live in Viser.

A small-scale slice of master_report.md WP4: plan -> transport -> descend ->
snap gate -> release, logging insertion_episode.jsonl per §2.6.

    isdn                                   # curobo + viser env
    python planner.py
    python run_assembly.py --plan plans/S3_weakest_joint.json
    # open http://localhost:8080

ponytail: KINEMATIC ONLY.  There is no contact physics here -- the snap gate
(§2.3.3) is evaluated against the pose cuRobo actually achieves, so IK/trajopt
error is the only error source.  Force, compliance and the real joint model
arrive with Isaac Lab + Newton at WP1/WP5; the gate and the episode log are
written so they can be swapped in without touching the rest of the loop.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import viser
from viser.extras import ViserUrdf

from curobo.content import get_assets_path
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState

import planner as P

# panda_hand origin to the point between the fingertips. Calibration knob:
# the fingertip offset is a real geometric quantity, measure it, do not assume.
TCP_OFFSET = 0.1034
# The fingers grip the brick's upper half, so the hand rides above the brick
# centre -- gripping at the base would drive the fingertips into the baseplate.
GRASP_DZ = P.BRICK_H / 2
FEEDER_POSE = [0.35, 0.30, 0.05]     # where bricks are picked from
GATE = {"lateral_mm": 1.2, "dz_mm": (-0.3, 1.0), "yaw_deg": 5.0}
BRICK_COLORS = [(220, 60, 50), (60, 130, 220), (240, 200, 60), (90, 190, 100)]


def xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]


def hand_target(brick_xyz):
    """Hand-frame goal for holding a brick whose bottom face is at brick_xyz."""
    return [brick_xyz[0], brick_xyz[1], brick_xyz[2] + GRASP_DZ + TCP_OFFSET]


def goal(pos, quat_wxyz, planner):
    """A single-pose GoalToolPose for panda_hand."""
    return GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[pos]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[quat_wxyz]]]], device="cuda", dtype=torch.float32),
    )


def brick_dims(btype, yaw_index):
    nx, ny = P.footprint(btype, yaw_index)
    return (nx * P.PITCH - 0.0002, ny * P.PITCH - 0.0002, P.BRICK_H)


def hand_poses(planner, q):
    """Batched FK: hand position (n, 3) and quaternion (n, 4) for joint rows (n, dof)."""
    dof = planner.kinematics.dof   # trajectories carry the finger joints too
    state = planner.compute_kinematics(JointState.from_position(
        q[..., :dof].contiguous(), joint_names=planner.joint_names[:dof]))
    pose = state.tool_poses.get_link_pose(planner.tool_frames[0])
    return pose.position.cpu().numpy(), pose.quaternion.cpu().numpy()


def yaw_of(quat_wxyz):
    """Yaw about the world z axis, for a gripper-down orientation."""
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def check_gate(target_pos, target_quat, got_pos, got_quat):
    """§2.3.3 gate conditions that a kinematic run can actually evaluate."""
    want = hand_target(target_pos)
    dxy = math.hypot(*(got_pos[:2] - np.array(want[:2]))) * 1000.0
    dz = (got_pos[2] - want[2]) * 1000.0
    dyaw = math.degrees(abs(yaw_of(got_quat) - yaw_of(target_quat)))
    dyaw = min(dyaw % 90.0, 90.0 - dyaw % 90.0)   # bricks are 90-deg symmetric
    ok = (dxy < GATE["lateral_mm"]
          and GATE["dz_mm"][0] <= dz <= GATE["dz_mm"][1]
          and dyaw < GATE["yaw_deg"])
    return ok, {"lateral_mm": round(float(dxy), 3), "dz_mm": round(float(dz), 3),
                "yaw_deg": round(float(dyaw), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="plans/S3_weakest_joint.json")
    ap.add_argument("--speed", type=float, default=2.0, help="playback multiplier")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    bricks = {b["id"]: b for b in plan["bricks"]}
    print("plan %s: %d steps, %d braced, strategy %s" % (
        plan["structure_id"], len(plan["sequence"]),
        plan["validation"]["brace_required_count"],
        plan["validation"]["bracing_strategy"]))

    # --- planner -------------------------------------------------------
    # Tolerances are tighter than cuRobo's defaults because the snap gate is
    # 1.2 mm wide; the default 5 mm would pass IK and fail every gate.
    table = Cuboid(name="table", pose=[0.45, 0.0, -0.029, 1, 0, 0, 0],
                   dims=[0.9, 0.9, 0.05])   # top at z = -0.004, under the baseplate
    mp = MotionPlanner(MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model={"cuboid": {"table": {"dims": table.dims, "pose": table.pose}}},
        collision_cache={"obb": 64},
        position_tolerance=0.0005,
        orientation_tolerance=0.01,
    ))
    mp.warmup(enable_graph=True, num_warmup_iterations=5)
    dt = mp.trajopt_solver.config.interpolation_dt
    # The arm has 7 actuated joints; the two finger joints are appended for
    # visualization only.
    dof = mp.kinematics.dof
    q = JointState.from_position(
        mp.default_joint_state.position[:dof].unsqueeze(0).contiguous(),
        joint_names=mp.joint_names[:dof])

    # --- viser ---------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    urdf = ViserUrdf(server, urdf_or_path=get_assets_path()
                     / "robot/franka_description/franka_panda.urdf", load_meshes=True)
    server.scene.add_box("/table", dimensions=tuple(table.dims),
                         position=(0.45, 0.0, -0.029), color=(120, 120, 120))
    ox, oy, oz = plan["voxel_origin"]
    server.scene.add_box("/baseplate", dimensions=(0.16, 0.16, 0.004),
                         position=(ox + 0.04, oy + 0.04, oz - 0.002),  # 4 mm thick
                         color=(70, 70, 70))
    carried = server.scene.add_frame("/carried", show_axes=False)
    brace_marker = server.scene.add_frame("/brace", axes_length=0.05, axes_radius=0.003,
                                          visible=False)

    finger = [0.04]   # per-finger opening drawn; closed on the brick while carrying

    def play(traj):
        """Animate an interpolated trajectory and return its final joint state."""
        pos = traj.position.reshape(-1, traj.position.shape[-1])
        tcp, _ = hand_poses(mp, pos)
        tcp = tcp - np.array([0.0, 0.0, TCP_OFFSET + GRASP_DZ])
        for row, p in zip(pos.cpu().numpy(), tcp):
            urdf.update_cfg(np.concatenate([row[:dof], [finger[0], finger[0]]]))
            carried.position = tuple(p)
            time.sleep(dt / args.speed)
        return JointState.from_position(pos[-1, :dof].unsqueeze(0).contiguous(),
                                        joint_names=mp.joint_names[:dof])

    def move(q_now, position, quat_wxyz, what):
        result = mp.plan_pose(goal(position, quat_wxyz, mp), q_now)
        if result is None or not result.success.any():
            print("  plan failed: %s" % what)
            return q_now, False
        return play(result.get_interpolated_plan()), True

    # --- execute -------------------------------------------------------
    log = open(Path(args.plan).with_suffix(".episodes.jsonl"), "w")
    scene_cuboids = {"table": {"dims": table.dims, "pose": table.pose}}
    placed = 0
    for step in plan["sequence"]:
        b = bricks[step["brick_id"]]
        dims = brick_dims(b["type"], b["yaw_index"])
        pre, tgt = step["pre_insertion_pose"], step["target_pose"]
        quat = xyzw_to_wxyz(tgt[3:])
        t0 = time.time()
        print("step %d: place %s (%s)%s" % (
            step["step"], b["id"], b["type"],
            "  [brace %s @ %.1f N]" % (step["brace"]["target_brick_id"],
                                       step["brace"]["brace_force_N"])
            if step["requires_brace"] else ""))

        if step["requires_brace"]:
            bp = step["brace"]["brace_pose"]
            brace_marker.position = tuple(bp[:3])
            brace_marker.visible = True

        # pick from the feeder, transport to pre-insertion, descend, release
        q, ok = move(q, hand_target(FEEDER_POSE), quat, "reach feeder")
        if ok:
            carried_box = server.scene.add_box(
                "/carried/brick", dimensions=dims,
                position=(0, 0, dims[2] / 2),
                color=BRICK_COLORS[placed % len(BRICK_COLORS)])
            # fingers close along the hand's y axis, which the plan's yaw puts
            # along world y at yaw_index 0 and along x at 1
            finger[0] = dims[1 - b["yaw_index"]] / 2
            q, ok = move(q, hand_target(pre[:3]), quat, "transport to pre-insertion")
        if ok:
            q, ok = move(q, hand_target(tgt[:3]), quat, "descend to target")

        snapped, err = False, {}
        if ok:
            got_pos, got_quat = hand_poses(mp, q.position)
            snapped, err = check_gate(tgt[:3], quat, got_pos[0], got_quat[0])
            carried_box.remove()
            finger[0] = 0.04
            server.scene.add_box(
                "/placed/%s" % b["id"], dimensions=dims,
                position=(tgt[0], tgt[1], tgt[2] + dims[2] / 2),
                wxyz=tuple(quat),
                color=BRICK_COLORS[placed % len(BRICK_COLORS)])
            placed += 1
            # the structure is an obstacle for every later placement (§2.5)
            scene_cuboids[b["id"]] = {
                "dims": list(dims),
                "pose": [tgt[0], tgt[1], tgt[2] + dims[2] / 2] + list(quat)}
            mp.update_world(Scene.create({"cuboid": scene_cuboids}))
            q, _ = move(q, hand_target(pre[:3]), quat, "retract")

        print("   %s  %s" % ("SNAP" if snapped else "MISS", err))
        log.write(json.dumps({
            "run_id": "%s_%s" % (plan["structure_id"],
                                 plan["validation"]["bracing_strategy"]),
            "brick_id": b["id"], "policy": "scripted_kinematic", "backend": "curobo_fk",
            "joint_model": "gate_only", "success": bool(snapped),
            "duration_s": round(time.time() - t0, 2),
            "final_error_mm": err.get("lateral_mm"), "final_error_deg": err.get("yaw_deg"),
            "dz_mm": err.get("dz_mm"),
            "brace_active": step["requires_brace"],
            "n_studs_engaged": sum(m[1] for m in step["mating_studs"]),
            "failure_mode": None if snapped else ("plan_failed" if not ok else "gate"),
        }) + "\n")
        log.flush()
        brace_marker.visible = False

    log.close()
    print("done: %d/%d placed. Episodes -> %s"
          % (placed, len(plan["sequence"]),
             Path(args.plan).with_suffix(".episodes.jsonl")))
    print("viewer stays up; ctrl-c to exit")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
