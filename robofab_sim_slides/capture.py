"""Capture screenshots and a clip of robofab_sim's dual_arm_sim.py (headless GL viewer).

Run from robofab_sim/ with the Isaac/Newton interpreter:
  python ../robofab_sim_slides/capture.py OUT_PREFIX EVERY SHOT_FRAMES [dual_arm_sim args...]
  e.g. ... capture.py ../robofab_sim_slides/media/cube 8 1500,4000,7700 --shape cube --num-frames 7800 --headless
CAM="x,y,z,pitch,yaw" sets the viewer camera. With --show-cameras also OUT_PREFIX_feeds_f<N>.png (camera tiles). Writes OUT_PREFIX.mp4 (every EVERY-th frame at 30 fps) and OUT_PREFIX_f<N>.png at each SHOT_FRAMES frame.
"""
import os, runpy, sys
import imageio.v2 as iio
import numpy as np
import newton.examples
import warp as wp

out, every, shots = sys.argv[1], int(sys.argv[2]), {int(s) for s in sys.argv[3].split(",")}
sys.argv = ["dual_arm_sim.py"] + sys.argv[4:]


def run(example, args):
    v, n = example.viewer, 0
    if os.environ.get("CAM"):          # CAM="x,y,z,pitch,yaw" overrides the sim's viewer camera
        x, y, z, pitch, yaw = map(float, os.environ["CAM"].split(","))
        v.set_camera(pos=wp.vec3(x, y, z), pitch=pitch, yaw=yaw)
    w = iio.get_writer(out + ".mp4", fps=30, codec="libx264", quality=8, macro_block_size=8)
    while v.is_running() and (not args.num_frames or n < args.num_frames):
        example.step()
        example.render()
        n += 1
        if n % every == 0 or n in shots:
            img = v.get_frame().numpy()
            if n % every == 0:
                w.append_data(img)
            if n in shots:
                iio.imwrite("%s_f%d.png" % (out, n), img)
                if getattr(example, "feed", None) is not None:      # --show-cameras: top | wrist_A | wrist_B, RGB over depth
                    f = example.feed
                    iio.imwrite("%s_feeds_f%d.png" % (out, n), np.concatenate([np.concatenate(f[:3], 1), np.concatenate(f[3:6], 1)], 0))
    w.close()
    print("captured %d frames" % n)


newton.examples.run = run
runpy.run_path("dual_arm_sim.py", run_name="__main__")
