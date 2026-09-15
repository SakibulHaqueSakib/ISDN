import time
import numpy as np
import viser
from pathlib import Path
from viser.extras import ViserUrdf
server = viser.ViserServer()
urdf_path = Path(__file__).resolve().parent / "urdf" / "planar_3dof_arm.urdf"
robot = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True)

joint_limits = robot.get_actuated_joint_limits()
sliders = []

def update_robot(_):
    robot.update_cfg(np.array([slider.value for slider in sliders]))

for joint_name, (lower, upper) in joint_limits.items():
    minimum = -np.pi if lower is None else float(lower)
    maximum = np.pi if upper is None else float(upper)
    initial_value = min(max(0.0, minimum), maximum)

    slider = server.gui.add_slider(
        joint_name,
        min=minimum,
        max=maximum,
        step=max((maximum - minimum) / 200.0, 0.001),
        initial_value=initial_value,
    )
    sliders.append(slider)
    slider.on_update(update_robot)

update_robot(None)

while True:
    time.sleep(1.0)
