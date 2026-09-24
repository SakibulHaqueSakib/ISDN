"""Turn a recorded physics run into a time-sampled USD you can scrub.

Isaac Sim 5.1 cannot render on this machine, but Isaac Sim 6.0 can (ledger:
rtx_crash_is_isaac_sim_51_not_the_driver). So the run is recorded headless
under 5.1 and the animation is rebuilt here as plain USD geometry, which any
USD viewer will open -- Isaac Sim 6.0, usdview, Blender, Omniverse.

    ~/Codes/CAIRSS/Issac/bin/python viz/to_usd.py results/s1_run.poses.jsonl
    # then, to watch it:
    ~/Codes/CAIRSS/Issac/bin/python viz/open_usd.py results/s1_run.usda

Bricks are drawn as boxes sized from the topology rather than referencing
BrickSim's own assets, so the file stands alone.
"""

import argparse
import json
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

FPS = 24.0
COLOURS = [(0.86, 0.24, 0.20), (0.24, 0.51, 0.86), (0.94, 0.78, 0.24),
           (0.35, 0.75, 0.39), (0.78, 0.47, 0.86), (0.98, 0.63, 0.24)]
BASEPLATE = (0.59, 0.59, 0.59)


def load(path):
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"parts": {}}
    frames = [json.loads(line) for line in path.read_text().splitlines()
              if line.strip() and not line.startswith('{"meta"')]
    if not frames:
        raise SystemExit("no frames in %s" % path)
    return meta, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.recording.with_suffix(".usda")

    meta, frames = load(args.recording)
    parts = meta.get("parts", {})

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    # Real seconds -> frames, so playback runs at wall-clock speed.
    t0 = frames[0]["t"]
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode((frames[-1]["t"] - t0) * FPS)

    xforms = {}
    for i, (pid, info) in enumerate(sorted(parts.items(), key=lambda kv: int(kv[0]))):
        dims = info.get("dims") or [0.016, 0.032, 0.0096]
        path = "/World/part_%s" % pid
        xform = UsdGeom.Xform.Define(stage, path)
        cube = UsdGeom.Cube.Define(stage, path + "/geom")
        cube.GetSizeAttr().Set(1.0)
        # A Cube is unit-sized and centred; lift it half a height, since the
        # recorded pose is BrickSim's bottom-centre origin, then scale it.
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, dims[2] / 2))
        UsdGeom.Xformable(cube).AddScaleOp().Set(
            Gf.Vec3f(dims[0], dims[1], dims[2]))
        colour = BASEPLATE if int(pid) == 0 else COLOURS[i % len(COLOURS)]
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(*colour)])
        xforms[pid] = (UsdGeom.Xformable(xform).AddTranslateOp(),
                       UsdGeom.Xformable(xform).AddOrientOp())

    # The arm. Each link is referenced from the Franka asset and given its own
    # time-sampled transform, taken from the RECORDED LINK POSES -- so the USD
    # needs no articulation, no joint drives and no forward kinematics.
    link_xforms = {}
    links = meta.get("links") or []
    robot_usd = meta.get("robot_usd")
    if links and frames[0].get("links"):
        asset_root = os.environ.get(
            "ISAAC_ASSET_ROOT",
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/6.0")
        robot_ref = (asset_root + robot_usd) if robot_usd else None
        for name in links:
            path = "/World/robot_%s" % name
            xf = UsdGeom.Xform.Define(stage, path)
            if robot_ref:
                # Pull in just this link's geometry from the Franka asset.
                geom = stage.OverridePrim(path + "/geom")
                geom.GetReferences().AddReference(robot_ref, "/panda/" + name)
                # The asset's link prims carry their own rest transform. Ours
                # are world poses, so keeping theirs would apply the pose twice.
                UsdGeom.Xformable(geom).SetXformOpOrder([])
            link_xforms[name] = (UsdGeom.Xformable(xf).AddTranslateOp(),
                                 UsdGeom.Xformable(xf).AddOrientOp())
        print("  arm: %d links referenced from %s"
              % (len(link_xforms), robot_ref or "<no asset>"))

    written = 0
    for frame in frames:
        tc = (frame["t"] - t0) * FPS
        for pid, pose in frame.get("parts", {}).items():
            ops = xforms.get(pid)
            if ops is None:
                continue
            pos, quat = pose
            ops[0].Set(Gf.Vec3d(*pos), time=tc)
            # recording stores wxyz; UsdGeom wants (real, imaginary)
            ops[1].Set(Gf.Quatf(float(quat[0]),
                                Gf.Vec3f(float(quat[1]), float(quat[2]),
                                         float(quat[3]))), time=tc)
        for name, pose in (frame.get("links") or {}).items():
            ops = link_xforms.get(name)
            if ops is None:
                continue
            pos, quat = pose
            ops[0].Set(Gf.Vec3d(*pos), time=tc)
            ops[1].Set(Gf.Quatf(float(quat[0]),
                                Gf.Vec3f(float(quat[1]), float(quat[2]),
                                         float(quat[3]))), time=tc)
        written += 1

    stage.GetRootLayer().Save()
    print("wrote %s" % out)
    print("  %d parts, %d arm links, %d time samples, %.1f s at %g fps"
          % (len(xforms), len(link_xforms), written, frames[-1]["t"] - t0, FPS))
    print("  open with: ~/Codes/CAIRSS/Issac/bin/python viz/open_usd.py %s" % out)


if __name__ == "__main__":
    main()
