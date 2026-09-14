import time
from curobo.types import ContentPath
from curobo.viewer import ViserVisualizer

viz = ViserVisualizer(
   content_path=ContentPath(robot_config_file="franka.yml"),
   connect_ip="0.0.0.0",
   connect_port=8080,
   add_robot_to_scene=True,
   add_control_frames=False,
   visualize_robot_spheres=False,
)

while True:
   time.sleep(1)
