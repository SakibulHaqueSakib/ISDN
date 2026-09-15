import time
import numpy as np
import viser
from viser.extras import ViserUrdf
from curobo.content import get_assets_path
server = viser.ViserServer()
urdf_path = get_assets_path() / "robot/franka_description/franka_panda.urdf"
server.scene.add_frame("/left_franka", position=[0.0, 0.0, 0.0], wxyz = [1, 0, 0, 0])
server.scene.add_frame("/right_franka", position=[1.1, 0.0, 0.0], wxyz = [np.cos(np.pi/2), 0, 0, np.sin(np.pi/2)])
left_franka = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True, root_node_name = "/left_franka")
right_franka = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True, root_node_name = "/right_franka")
left_franka.update_cfg(np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8, 0.04, 0.04]))
right_franka.update_cfg(np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8, 0.04, 0.04]))
while True:
    time.sleep(1.0)
