# Prompt: Make a minimal Panda cube-picking game. Use joint sliders to move
# the robot, randomly place and rotate a cube on the ground, and show a success
# popup with a New Game button when ee_link reaches any valid parallel-jaw
# grasp pose. Visualize the closest grasp and live pose errors in Viser.

import time

import numpy as np
import torch
from curobo.config_io import load_yaml
from curobo.content import get_robot_configs_path
from curobo.types import JointState
from curobo.viewer import ViserVisualizer


CUBE_SIZE = 0.05
POSITION_TOLERANCE = 0.06
ROTATION_TOLERANCE = np.deg2rad(15.0)
rng = np.random.default_rng()

# Request ee_link so cuRobo reads and computes this frame from the Panda URDF.
robot_config = load_yaml(str(get_robot_configs_path() / "franka.yml"))
robot_config["robot_cfg"]["kinematics"]["tool_frames"].append("ee_link")
viz = ViserVisualizer(
    content_path=robot_config,
    connect_ip="0.0.0.0",
    connect_port=8080,
    add_robot_to_scene=True,
    add_control_frames=False,
    visualize_robot_spheres=False,
)

# Add the current end-effector frame.
ee_pose = viz._kinematics.compute_kinematics(
    viz._kinematics.default_joint_state
).tool_poses.get_link_pose("ee_link")
ee_frame = viz._server.scene.add_frame(
    "/ee_link",
    position=ee_pose.position[0].cpu().numpy(),
    wxyz=ee_pose.quaternion[0].cpu().numpy(),
    axes_length=0.08,
    axes_radius=0.003,
)

# The cube is a child of its target frame, and its bottom rests at z=0.
cube_frame = viz._server.scene.add_frame(
    "/cube_target",
    axes_length=0.08,
    axes_radius=0.003,
)
viz._server.scene.add_box(
    "/cube_target/cube",
    dimensions=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
    color=(255, 165, 0),
)
closest_grasp_frame = viz._server.scene.add_frame(
    "/closest_grasp",
    axes_length=0.1,
    axes_radius=0.004,
)

# Show live errors and the positional gap in the Viser scene.
viz._server.gui.add_markdown(
    f"Goal: position < **{POSITION_TOLERANCE:.2f} m**, "
    f"rotation < **{np.rad2deg(ROTATION_TOLERANCE):.0f}°**  \n"
    "Angle uses the closest of four top-down cube grasps."
)
position_display = viz._server.gui.add_number(
    "Position error (m)", 0.0, step=0.001, disabled=True
)
rotation_display = viz._server.gui.add_number(
    "Rotation error (deg)", 0.0, step=0.1, disabled=True
)
distance_line = viz._server.scene.add_line_segments(
    "/distance_to_cube",
    points=np.zeros((1, 2, 3)),
    colors=(255, 0, 0),
    thickness=0.005,
)

game_won = False
grasp_wxyzs = np.zeros((4, 4))


def update_errors(ee_position, ee_wxyz):
    position_error = np.linalg.norm(ee_position - cube_frame.position)

    # Choose the closest of four equivalent top-down parallel-jaw grasps.
    quaternion_dots = np.clip(np.abs(grasp_wxyzs @ ee_wxyz), 0.0, 1.0)
    closest_index = int(np.argmax(quaternion_dots))
    rotation_error = 2.0 * np.arccos(quaternion_dots[closest_index])
    closest_grasp_frame.position = cube_frame.position
    closest_grasp_frame.wxyz = grasp_wxyzs[closest_index]

    position_display.value = float(position_error)
    rotation_display.value = float(np.rad2deg(rotation_error))
    distance_line.points = np.array([[ee_position, cube_frame.position]])
    return position_error, rotation_error


def new_game():
    global game_won, grasp_wxyzs
    game_won = False
    yaw = rng.uniform(-np.pi, np.pi)
    cube_frame.position = np.array(
        [rng.uniform(0.3, 0.6), rng.uniform(-0.3, 0.3), CUBE_SIZE / 2]
    )
    cube_frame.wxyz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    # Point ee_link downward and allow 90-degree rotations about the cube Z-axis.
    grasp_yaws = yaw + np.arange(4) * np.pi / 2
    grasp_wxyzs = np.column_stack(
        (
            np.zeros(4),
            np.cos(grasp_yaws / 2),
            np.sin(grasp_yaws / 2),
            np.zeros(4),
        )
    )
    update_errors(np.asarray(ee_frame.position), np.asarray(ee_frame.wxyz))


def show_success():
    global game_won
    game_won = True
    modal = viz._server.gui.add_modal("Success!")
    with modal:
        viz._server.gui.add_markdown("The **ee_link** reached a valid cube grasp.")
        button = viz._server.gui.add_button("New game")

    @button.on_click
    def _(_):
        modal.close()
        new_game()


new_game()

# Add one slider for every URDF joint.
joint_names = viz.joint_names
joint_limits = viz._viser_urdf.get_actuated_joint_limits()
sliders = []


async def update_robot(_):
    positions = torch.tensor(
        [slider.value for slider in sliders],
        device=viz._kinematics.device_cfg.device,
        dtype=viz._kinematics.device_cfg.dtype,
    )
    joint_state = JointState.from_position(positions, joint_names)
    viz.set_joint_state(joint_state)

    active_state = viz._kinematics.get_active_js(joint_state)
    ee_pose = viz._kinematics.compute_kinematics(
        active_state
    ).tool_poses.get_link_pose("ee_link")
    ee_position = ee_pose.position[0].cpu().numpy()
    ee_wxyz = ee_pose.quaternion[0].cpu().numpy()
    ee_frame.position = ee_position
    ee_frame.wxyz = ee_wxyz

    position_error, rotation_error = update_errors(ee_position, ee_wxyz)
    if not game_won and position_error < POSITION_TOLERANCE and rotation_error < ROTATION_TOLERANCE:
        show_success()


for joint_name, initial_value in zip(joint_names, viz._urdf.cfg):
    minimum, maximum = joint_limits[joint_name]
    slider = viz._server.gui.add_slider(
        joint_name,
        min=float(minimum),
        max=float(maximum),
        step=float((maximum - minimum) / 200),
        initial_value=float(initial_value),
    )
    slider.on_update(update_robot)
    sliders.append(slider)

while True:
    time.sleep(1)
