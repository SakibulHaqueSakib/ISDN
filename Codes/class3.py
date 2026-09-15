import time
import numpy as np
import viser
from viser.extras import ViserUrdf
from curobo.content import get_assets_path
server = viser.ViserServer()
urdf_path = get_assets_path() / "/home/bakasakib/Documents/ISDN_Robofab/curobo/curobo/content/assets/robot/g1/g1_29dof_rev_1_0.urdf"
robot = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True)
# joint order per g1_29dof URDF: left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)
# one-knee "knighting" pose: left leg planted forward, right knee down, torso upright
KNEEL_POSE = np.array([
    -1.4, 0.0, 0.0, 1.6, -0.3, 0.0,   # left leg: hip bent forward, knee up, foot flat
    0.6, 0.0, 0.0, 1.6, -0.9, 0.0,    # right leg: hip back, knee folded under, on the ground
    0.0, 0.0, 0.1,                    # waist: slight forward lean
    0.3, 0.1, 0.0, 1.2, 0.0, 0.0, 0.0,  # left arm: resting on raised knee
    0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0,  # right arm: relaxed at side
])
robot.update_cfg(KNEEL_POSE)

# One slider per actuated joint, same pattern as robot_game.py.
sliders = []


def update_robot(_):
    robot.update_cfg(np.array([slider.value for slider in sliders]))


reset_button = server.gui.add_button("Reset to kneel")


@reset_button.on_click
def _(_):
    for slider, value in zip(sliders, KNEEL_POSE):
        slider.value = float(value)


joint_limits = robot.get_actuated_joint_limits()
for (joint_name, (minimum, maximum)), initial_value in zip(joint_limits.items(), KNEEL_POSE):
    slider = server.gui.add_slider(
        joint_name,
        min=float(minimum),
        max=float(maximum),
        step=float((maximum - minimum) / 200),
        initial_value=float(np.clip(initial_value, minimum, maximum)),
    )
    slider.on_update(update_robot)
    sliders.append(slider)

update_robot(None)

while True:
    time.sleep(1.0)
