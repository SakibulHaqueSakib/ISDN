"""Replay a recorded physics run in the browser (Viser).

Isaac Sim 5.1 -- the version BrickSim pins -- segfaults on this machine's RTX
stack, while Isaac Sim 6.0 renders fine (ledger: rtx_crash_is_isaac_sim_51_not_
the_driver). So physics runs are recorded headless and watched here instead.
This needs no Isaac rendering at all.

    source ~/Documents/ISDN_Robofab/isdnenv/bin/activate
    python viz/play_viser.py results/s1_run.poses.jsonl
    # open http://localhost:8080

Controls in the browser: play/pause, a time slider, and playback speed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

BRICK_COLOURS = [(220, 60, 50), (60, 130, 220), (240, 200, 60), (90, 190, 100),
                 (200, 120, 220), (250, 160, 60)]
BASEPLATE_COLOUR = (150, 150, 150)


def load(path):
    """Frames plus the sidecar metadata written alongside the recording."""
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"parts": {}}
    frames = [json.loads(line) for line in path.read_text().splitlines()
              if line.strip() and not line.startswith('{"meta"')]
    if not frames:
        raise SystemExit("no frames in %s -- was the run recorded with "
                         "--record?" % path)
    return meta, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    meta, frames = load(args.recording)
    parts = meta.get("parts", {})
    duration = frames[-1]["t"] - frames[0]["t"]
    print("%d frames, %.1f s of simulated time, %d parts"
          % (len(frames), duration, len(parts)))

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    # The arm. Driven by the RECORDED JOINT ANGLES through the URDF, mapped by
    # name -- Isaac's articulation order is not the URDF's, so positional
    # mapping would silently bend the wrong joints.
    urdf, urdf_joint_index = None, None
    base_pos, base_quat = meta.get("robot_base", [[0, 0, 0], [1, 0, 0, 0]])
    joint_names = meta.get("joint_names") or []
    try:
        from curobo.content import get_assets_path
        urdf_path = get_assets_path() / "robot/franka_description/franka_panda.urdf"
        server.scene.add_frame("/robot", position=tuple(base_pos),
                               wxyz=tuple(base_quat), show_axes=False)
        urdf = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True,
                         root_node_name="/robot")
        target = [j.name for j in urdf._urdf.actuated_joints]
        urdf_joint_index = [joint_names.index(n) if n in joint_names else None
                            for n in target]
        missing = [n for n, i in zip(target, urdf_joint_index) if i is None]
        print("arm: %d URDF joints, %d mapped%s"
              % (len(target), sum(i is not None for i in urdf_joint_index),
                 (", unmapped: %s" % missing) if missing else ""))
    except Exception as exc:
        print("arm not drawn (%s: %s)" % (type(exc).__name__, exc))

    # One box per brick, sized from the topology the run recorded. BrickSim's
    # part origin is the BOTTOM centre, and a Viser box is drawn about its
    # centre, so the box hangs half a height up from a frame at the recorded
    # pose. Centring it on the pose drew every brick 4.8 mm low, which made
    # the closed fingers look like they were gripping air above it.
    handles = {}
    for i, (pid, info) in enumerate(sorted(parts.items(), key=lambda kv: int(kv[0]))):
        dims = info.get("dims") or [0.016, 0.032, 0.0096]
        baseplate = int(pid) == 0
        handles[pid] = server.scene.add_frame("/part_%s" % pid, show_axes=False)
        server.scene.add_box(
            "/part_%s/box" % pid,
            dimensions=tuple(dims),
            position=(0.0, 0.0, dims[2] / 2),
            color=BASEPLATE_COLOUR if baseplate
            else BRICK_COLOURS[i % len(BRICK_COLOURS)])

    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("play", True)
        slider = server.gui.add_slider("frame", min=0, max=len(frames) - 1,
                                       step=1, initial_value=0)
        speed = server.gui.add_slider("speed", min=0.1, max=10.0, step=0.1,
                                      initial_value=args.speed)
        clock = server.gui.add_text("sim time", initial_value="0.00 s")

    def show(index):
        frame = frames[index]
        for pid, pose in frame.get("parts", {}).items():
            handle = handles.get(pid)
            if handle is None:
                continue
            pos, quat = pose
            handle.position = tuple(pos)
            handle.wxyz = tuple(quat)
        if urdf is not None and frame.get("joints"):
            q = frame["joints"]
            cfg = [q[i] if (i is not None and i < len(q)) else 0.0
                   for i in urdf_joint_index]
            urdf.update_cfg(np.asarray(cfg, dtype=float))
        clock.value = "%.2f s" % frame["t"]

    show(0)
    print("open http://localhost:%d" % args.port)

    index = 0
    while True:
        if playing.value:
            index = (index + 1) % len(frames)
            slider.value = index
        else:
            index = int(slider.value)
        show(index)
        dt = (frames[min(index + 1, len(frames) - 1)]["t"]
              - frames[index]["t"]) or 0.05
        time.sleep(max(0.005, dt / max(0.1, speed.value)))


if __name__ == "__main__":
    main()
