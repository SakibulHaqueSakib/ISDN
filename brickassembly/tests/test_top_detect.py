"""top_detect.py against the real top camera of the cell (cell/cameras.py), ground truth read here only. Needs the GPU (one Example,
nominal start at step 5: the feeder bricks lie on the table, the first five stand on the plate, as in a build under way).

  - every feeder brick is found: colour, size (studs), xy and yaw (mod pi), on the noised full-size capture and the noise-free
    preview, full size and at stride 2 (the matching rays);
  - with the camera's TRUE pose the xy error is under 1.5 mm; with the nominal pose (what an executor has) under 4 mm: the hidden
    extrinsic error of the top camera (1 mm, 0.2 deg per axis, up to 4 mm at the table) is in it;
  - the plate: studs (NI + 4) x (NJ + 4) (the visible slab is two studs wider each side than the build), centre within 4 mm;
  - the topmost brick of the build: on the plate, its layer and its min-corner cell;
  - the firewall (top_detect imports without newton / warp / torch) is in tests/test_vision.py, with vision.py.

    bash scripts/run.sh tests/test_top_detect.py          # or: python -m pytest tests/test_top_detect.py
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import top_detect as TD  # noqa: E402

START = 5
NAMES = ["red", "blue", "yellow", "green", "white", "orange"]            # dual_arm_sim.COLORS, in order (brick n gets COLORS[n % 6])


@pytest.fixture(scope="module")
def rig():
    try:
        import newton
        import dual_arm_sim as D
        import blueprint as B
    except Exception as e:                                               # no newton / GPU
        pytest.skip("cell unavailable: %r" % (e,))
    bricks = B.load("cube")
    args = argparse.Namespace(front=None, side=None, top=None, width=None, depth=None, name="cube", shape=None, structure=None,
                              bricks=bricks, strategy="none", brace_model="grasp_lp", plan=None, spec="top_detect_test",
                              start_step=START, only_step=False, brace_json=None, press_scale=None, press_lateral=None,
                              fixture_hold=False, exe="E0", viewer="null", legacy_brace=False, show_cameras=True)
    ex = D.Example(newton.viewer.ViewerNull(num_frames=10 ** 9), args)
    for _ in range(3):
        ex.step()
    return D, ex


def truth(D, ex):
    """(per brick: n, id, centre xyz, yaw, colour name, footprint) from the state, and the plate's centre / studs."""
    bq = ex.state_0.body_q.numpy()
    out = []
    for n, (bid, body) in enumerate(ex.body.items()):
        L, W = ex.dims[bid]
        out.append(dict(n=n, id=bid, p=bq[body][:3].copy(), yaw=Rot.from_quat(bq[body][3:7]).as_euler("xyz")[2], colour=NAMES[n % 6],
                        studs="%dx%d" % (round(max(L, W) / 0.008), round(min(L, W) / 0.008)), L=L, W=W))
    return out


def check_feeder(det, tr, tol_mm, tag):
    feeder = [t for t in tr if t["n"] >= START]
    assert len(feeder) == 3
    for t in feeder:
        b = min(det["bricks"], key=lambda b: math.hypot(b["x"] - t["p"][0], b["y"] - t["p"][1]))
        e = math.hypot(b["x"] - t["p"][0], b["y"] - t["p"][1]) * 1e3
        dy = abs(math.degrees((b["yaw"] - t["yaw"] + math.pi / 2) % math.pi - math.pi / 2))
        assert e < tol_mm and dy < 3.0 and b["colour"] == t["colour"] and b["studs"] == t["studs"] and not b["on_plate"], (tag, t["id"], e, dy, b["colour"], b["studs"])


def check_plate(D, ex, det, tol_mm):
    p, (NI, NJ) = det["plate"], ex.NIJ
    assert p["studs"] == (NI + 4, NJ + 4), p["studs"]
    assert math.hypot(p["x"], p["y"]) * 1e3 < tol_mm, (p["x"], p["y"])      # the plate is centred on the world origin (voxel origin -NI*pitch/2)


def test_capture_and_preview(rig):
    D, ex = rig
    tr = truth(D, ex)
    I = ex.cams.cams["top"]
    tpos, tR, npos, nR = ex.cams.poses("top")
    cap = ex.cams.capture("top", state=ex.state_0)
    pv = {st: ex.cams.preview("top", state=ex.state_0, stride=st) for st in (1, 2)}
    for tag, rgb, dep, st in (("capture", cap["rgb"], cap["depth"], 1), ("preview", pv[1]["rgb"], pv[1]["depth"], 1), ("preview/2", pv[2]["rgb"], pv[2]["depth"], 2)):
        rays = I["rays_np"][::st, ::st]
        det_true, det_nom = TD.detect(rgb, dep, rays, tpos, tR), TD.detect(rgb, dep, rays, npos, nR)
        check_feeder(det_true, tr, 1.5 if st == 1 else 2.0, tag + " true pose")
        check_feeder(det_nom, tr, 4.0, tag + " nominal pose")
        check_plate(D, ex, det_nom, 4.0)
    det = TD.detect(cap["rgb"], cap["depth"], cap["intrinsics"], npos, nR)      # the intrinsics dict gives the same rays
    assert len(det["bricks"]) == len(TD.detect(cap["rgb"], cap["depth"], I["rays_np"], npos, nR)["bricks"])
    # the build: the topmost brick (step START - 1) sees nothing above it
    top = tr[START - 1]
    b = min(det["bricks"], key=lambda b: math.hypot(b["x"] - top["p"][0], b["y"] - top["p"][1]))
    pl = ex.plan["voxel_origin"]
    c0 = (pl[0] - 2 * 0.008, pl[1] - 2 * 0.008)                              # the plate's min corner
    hx, hy = (abs(math.cos(top["yaw"])) * top["L"] + abs(math.sin(top["yaw"])) * top["W"]) / 2, (abs(math.sin(top["yaw"])) * top["L"] + abs(math.cos(top["yaw"])) * top["W"]) / 2
    cell = (round((top["p"][0] - hx - c0[0]) / 0.008), round((top["p"][1] - hy - c0[1]) / 0.008))
    layer = round((top["p"][2] - 0.0032) / 0.0096)
    assert b["on_plate"] and b["cell"] == cell and b["layer"] == layer and b["colour"] == top["colour"], (b["cell"], cell, b["layer"], layer, b["colour"])
    assert math.hypot(b["x"] - top["p"][0], b["y"] - top["p"][1]) < 4e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
