"""WP0 / G0 -- impedance controller tracks a 50 mm free-space circle.

Acceptance (§WP0): < 0.5 mm RMS position error. Gains are the §2.5 spec:
K_p translational diag(1200, 1200, 600) N/m, rotational 40 N·m/rad, damping 1.0.

Adapted from IsaacLab scripts/tutorials/05_controllers/run_osc.py -- the state
extraction below is that tutorial's, kept verbatim so a tutorial update is easy
to diff against.

    ~/Codes/CAIRSS/Issac/bin/python scripts/00b_impedance_check.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--radius_m", type=float, default=0.05)
parser.add_argument("--period_s", type=float, default=4.0)
parser.add_argument("--laps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip

# §2.5 gains.
KP_TASK = [1200.0, 1200.0, 600.0, 40.0, 40.0, 40.0]
CIRCLE_CENTRE_B = (0.5, 0.0, 0.4)      # free space, well inside the workspace
SETTLE_STEPS = 480                     # 2 s at 240 Hz before error is recorded


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        # Isaac Lab 3.0-beta2 points the Franka at the Isaac 6.0 asset tree,
        # where panda_instanceable.usd 404s. 5.1 has it. Risk R1, in the ledger.
        self.robot.spawn.usd_path = self.robot.spawn.usd_path.replace(
            "/Assets/Isaac/6.0/", "/Assets/Isaac/5.1/")
        # The OSC supplies the joint torques, so the position actuators must not
        # fight it. This is the tutorial's setup.
        self.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.robot.actuators["panda_shoulder"].damping = 0.0
        self.robot.actuators["panda_forearm"].stiffness = 0.0
        self.robot.actuators["panda_forearm"].damping = 0.0
        # run_osc.py disables gravity on the arm and leaves the OSC's gravity
        # compensation off. Tracking with gravity on is a separate question and
        # a separate check -- do not conflate the two in one gate.
        self.robot.spawn.rigid_props.disable_gravity = True


def ee_state(robot, ee_idx, arm_ids):
    """(jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, q, qd) -- run_osc.py."""
    jacobi_ids = [j + robot.num_base_dofs for j in arm_ids]
    jacobian_w = robot.data.body_link_jacobian_w.torch[:, ee_idx - 1, :, jacobi_ids]
    mass_matrix = robot.data.mass_matrix.torch[:, jacobi_ids, :][:, :, jacobi_ids]
    gravity = robot.data.gravity_compensation_forces.torch[:, jacobi_ids]

    jacobian_b = jacobian_w.clone()
    root_rot = matrix_from_quat(quat_inv(robot.data.root_quat_w.torch))
    jacobian_b[:, :3, :] = torch.bmm(root_rot, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot, jacobian_b[:, 3:, :])

    root_pos_w, root_quat_w = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w,
        robot.data.body_pos_w.torch[:, ee_idx], robot.data.body_quat_w.torch[:, ee_idx])
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    rel_vel_w = robot.data.body_vel_w.torch[:, ee_idx, :] - robot.data.root_vel_w.torch
    ee_vel_b = torch.cat([quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3]),
                          quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])], dim=-1)
    return (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
            robot.data.joint_pos.torch[:, arm_ids], robot.data.joint_vel.torch[:, arm_ids])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args_cli.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    # run_osc.py's frame. The Jacobian is indexed ee_idx - 1, which only holds
    # for the frames that tutorial uses -- do not substitute another link.
    ee_idx = robot.find_bodies("panda_leftfinger")[0][0]
    arm_ids = robot.find_joints(["panda_joint.*"])[0]

    osc = OperationalSpaceController(OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        motion_stiffness_task=KP_TASK,
        motion_damping_ratio_task=1.0,
        inertial_dynamics_decoupling=True,
        gravity_compensation=False,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="none",
    ), num_envs=1, device=sim.device)

    dt = sim.get_physics_dt()
    robot.update(dt)
    joint_centres = torch.mean(robot.data.soft_joint_pos_limits.torch[:, arm_ids, :], dim=-1)
    hold_quat = ee_state(robot, ee_idx, arm_ids)[3][:, 3:7].clone()

    n_steps = SETTLE_STEPS + int(args_cli.laps * args_cli.period_s / dt)
    sq_err, samples, peak_mm = 0.0, 0, 0.0
    for step in range(n_steps):
        jac, mass, grav, ee_pose_b, ee_vel_b, q, qd = ee_state(robot, ee_idx, arm_ids)

        # Hold the centre while settling, then walk the circle.
        theta = 0.0 if step < SETTLE_STEPS else \
            2 * math.pi * (step - SETTLE_STEPS) * dt / args_cli.period_s
        target = torch.tensor([[CIRCLE_CENTRE_B[0] + args_cli.radius_m * math.cos(theta),
                                CIRCLE_CENTRE_B[1] + args_cli.radius_m * math.sin(theta),
                                CIRCLE_CENTRE_B[2]]], device=sim.device)
        # run_osc.py places the task frame ON the target and sends the command
        # relative to it, so the command is identity and the task-space gains
        # are rotated into the target frame. Sending an absolute pose against an
        # identity task frame is NOT equivalent -- it was unstable here.
        task_frame = torch.cat([target, hold_quat], dim=-1)
        rel_pos, rel_quat = subtract_frame_transforms(
            task_frame[:, :3], task_frame[:, 3:], task_frame[:, :3], task_frame[:, 3:])
        command = torch.cat([rel_pos, rel_quat], dim=-1)

        osc.set_command(command=command, current_ee_pose_b=ee_pose_b,
                        current_task_frame_pose_b=task_frame)
        robot.set_joint_effort_target_index(
            target=osc.compute(jacobian_b=jac, current_ee_pose_b=ee_pose_b,
                               current_ee_vel_b=ee_vel_b, mass_matrix=mass, gravity=grav,
                               current_joint_pos=q, current_joint_vel=qd,
                               nullspace_joint_pos_target=joint_centres
                               if osc.cfg.nullspace_control != "none" else None),
            joint_ids=arm_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        scene.update(dt)

        if step % 240 == 0 or step == n_steps - 1:
            print("  step %5d  ee %s  target %s" % (
                step, ee_pose_b[0, 0:3].tolist(), target[0].tolist()), flush=True)

        if step >= SETTLE_STEPS:
            err_mm = float(torch.norm(ee_pose_b[:, 0:3] - target)) * 1000.0
            sq_err += err_mm ** 2
            peak_mm = max(peak_mm, err_mm)
            samples += 1

    rms_mm = (sq_err / samples) ** 0.5
    result = {"rms_mm": round(rms_mm, 4), "peak_mm": round(peak_mm, 4),
              "samples": samples, "radius_m": args_cli.radius_m,
              "period_s": args_cli.period_s, "kp_task": KP_TASK,
              "ee_frame": "panda_leftfinger", "backend": str(sim.device),
              "pass": rms_mm < 0.5}
    out = Path(__file__).parent / "00b_impedance_check.json"
    out.write_text(json.dumps(result, indent=1))
    print("\n-- G0 impedance --")
    print("  RMS %.3f mm, peak %.3f mm over %d samples -> %s"
          % (rms_mm, peak_mm, samples, "PASS" if result["pass"] else "FAIL"))
    print("  report -> %s" % out)
    simulation_app.close()


if __name__ == "__main__":
    main()
