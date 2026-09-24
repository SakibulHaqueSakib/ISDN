"""WP4 -- scripted end-to-end assembly baseline (master_report §WP4).

§WP4 asks for a complete pipeline with ground-truth poses and a scripted
inserter. BrickSim's demos/demo_assembly.py already implements exactly that:
top-down antipodal grasp, transport, and a press-to-assemble that relies on
the physical gate firing. §1.2/§0.3 say not to rebuild what exists, so this
drives that demo from OUR assembly plan and logs OUR episode records instead
of reimplementing the skills.

What this module owns:
  * the structure comes from planner.to_bricksim_topology(), not the demo's
    hard-coded STRUCTURE_PATH
  * the assembly ORDER is our planner's sequence, not the demo's BFS re-sort,
    because bracing is defined against our order
  * the calibrated clutch from WP2b is applied before anything is assembled
  * one insertion_episode.json record per placement (§2.6)

    cd ~/Documents/ISDN_Robofab/BrickSim
    OMNI_KIT_ACCEPT_EULA=1 .venv/bin/python \\
        ../brickassembly/orchestration/run_plan.py --headless --structure S1
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

# Fail now, not five minutes into Isaac Sim startup, if this is the wrong
# interpreter: bricksim lives only in BrickSim's venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envcheck  # noqa: E402

envcheck.require("bricksim", ["bricksim", "isaacsim"])
envcheck.banner("bricksim")
envcheck.rtx_compat()          # before AppLauncher touches Vulkan

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
BRICKSIM = ROOT.parent / "BrickSim"

parser = argparse.ArgumentParser()
parser.add_argument("--structure", default="S1", help="a planner.STRUCTURES key")
parser.add_argument("--demo-structure", action="store_true",
                    help="use BrickSim's own demo structure and BFS order "
                         "instead of ours -- bisects harness vs data")
parser.add_argument("--trial", type=int, default=0, help="trial index, for the log")
parser.add_argument("--out", type=Path, default=ROOT / "results" / "wp4_episodes.jsonl")
parser.add_argument("--timeout-s", type=float, default=900.0)
parser.add_argument("--grasp-attempts", type=int, default=3)
parser.add_argument("--record", type=Path, default=None,
                    help="sample poses to a JSONL for playback (viz/play_viser.py, "
                         "viz/to_usd.py). Isaac Sim 5.1 cannot render on this "
                         "machine, so this is how a run is watched.")
parser.add_argument("--record-hz", type=float, default=20.0)
parser.add_argument("--gate", choices=("spec", "default"), default="spec",
                    help="'spec' uses §2.3.3 / joint_calibration (8.9 N, "
                         "1.18 mm) -- correct physics, but the scripted "
                         "inserter cannot reach it and places nothing. "
                         "'default' uses BrickSim's own gate (1.0 N, 2.0 mm), "
                         "which the current inserter CAN satisfy.")
parser.add_argument("--press-depth-mm", type=float, default=0.5,
                    help="how far the inserter presses past the seated pose; "
                         "force is roughly stiffness x depth")
parser.add_argument("--measure-press", action="store_true",
                    help="sample contact force during each press (§WP4: the "
                         "static analysis assumes 35.6 N; what does the "
                         "scripted inserter actually develop?)")
parser.add_argument("--preload", type=float, default=None,
                    help="override PreloadedForce; default is WP2b's calibration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import math  # noqa: E402
import numpy as np  # noqa: E402

# Verify a placement only after the scene settles: the gate can fire while
# things come to rest, so an immediate check marks real successes as failures.
VERIFY_SETTLE_STEPS = 90        # ~1.5 s at 60 Hz

# Franka links worth drawing. Recorded as world poses so playback needs no
# forward kinematics and no joint-order guessing.
ROBOT_LINKS = ["panda_link0", "panda_link1", "panda_link2", "panda_link3",
               "panda_link4", "panda_link5", "panda_link6", "panda_link7",
               "panda_hand", "panda_leftfinger", "panda_rightfinger"]
ROBOT_BASE_POS = (-0.1, 0.0, 0.0)
ROBOT_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)
ROBOT_USD = ("/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")

# ORDER MATTERS. bricksim must be imported before any extension is enabled:
# enabling one re-runs module registration and evicts the bricksim namespace
# from sys.modules, so a later `from bricksim.core import ...` dies with
# KeyError: 'bricksim'.
import bricksim  # noqa: E402,F401  (self-enables the bricksim Kit extension)
from bricksim.core import (  # noqa: E402
    AssemblyThresholds,
    BreakageThresholds,
    are_parts_connected,
    get_assembly_debug_infos,
    get_sync_to_usd,
    set_breakage_thresholds,
    set_sync_to_usd,
)

# Then motion-generation (RmpFlow), which the bricksim CLI's experience
# enables but AppLauncher's headless config does not. Not debug_draw: it
# drags in omni.kit.widget.toolbar, absent headless, and the demo guards it.
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.motion_generation")

sys.path.insert(0, str(ROOT))
import planner as P  # noqa: E402


_PHYS_CACHE = {}


def physics_pose_prim_class():
    """A drop-in SingleXFormPrim whose get_world_pose() reads PHYSICS, not USD.

    BrickSim's skills read every pose they need through SingleXFormPrim, which
    returns the USD transform. Headless there is no render pass, so physics
    never writes transforms back to USD and those reads are stale -- measured
    at 181.7 mm on a brick that had just been lifted 181.7 mm. Swapping the
    class in the demo's namespace fixes every call site at once without
    editing BrickSim's source.
    """
    from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim

    class PhysicsPosePrim(SingleXFormPrim):
        def get_world_pose(self):
            path = self.prim_path
            if path not in _PHYS_CACHE:
                try:
                    _PHYS_CACHE[path] = SingleRigidPrim(prim_path=path)
                except Exception:
                    _PHYS_CACHE[path] = None          # not a rigid body
            body = _PHYS_CACHE[path]
            if body is not None:
                try:
                    return body.get_world_pose()
                except Exception:
                    _PHYS_CACHE[path] = None
            return super().get_world_pose()

    return PhysicsPosePrim


# The demo chooses its grasp face purely from the brick's dimensions:
#   if W <= L: close across W, else close across L
# Both branches grip the 2-stud dimension, so swapping L and W does not change
# the grip width -- it rotates the approach by 90 degrees. That is the second
# grasp candidate §WP4.1 asks for, obtained without touching BrickSim's source.
GRASP_FACE = {"swap": False}


def patch_grasp_face(demo):
    original = demo.parse_brick_prim_dimensions

    def dims(prim):
        got = original(prim)
        if got is None or not GRASP_FACE["swap"]:
            return got
        L, W, H = got
        return (W, L, H)

    demo.parse_brick_prim_dimensions = dims


def patch_demo_pose_reads(demo, rmpflow):
    """Point the demo's pose reads at the physics view."""
    cls = physics_pose_prim_class()
    demo.SingleXFormPrim = cls

    # The end-effector prim comes from RmpFlow, not from SingleXFormPrim, so
    # it has to be wrapped separately.
    original = rmpflow.get_end_effector_as_prim
    wrapped = {}

    def get_ee():
        prim = original()
        path = prim.prim_path
        if path not in wrapped:
            wrapped[path] = cls(prim_path=path, name="ee_physics_pose")
        return wrapped[path]

    rmpflow.get_end_effector_as_prim = get_ee
    return cls


def load_demo():
    """Import BrickSim's demo as a library. It defines functions and a main();
    nothing runs at import, so this is safe once the app is up."""
    path = BRICKSIM / "demos" / "demo_assembly.py"
    spec = importlib.util.spec_from_file_location("demo_assembly", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_assembly"] = module
    spec.loader.exec_module(module)
    return module


async def assemble_structure(demo, topo_path, bt, record_placement):
    """Scene setup + our sequence, driving the demo's grasp and assemble skills."""
    from isaacsim.core.api.world import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import (RigidPrim, SingleArticulation,
                                     SingleGeometryPrim, SingleRigidPrim,
                                     SingleXFormPrim)
    from isaacsim.core.utils.stage import (add_reference_to_stage,
                                           get_current_stage, open_stage_async)
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.robot_motion.motion_generation import (ArticulationMotionPolicy,
                                                         RmpFlow)
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_motion_policy_config)
    from pxr import Gf

    from bricksim.assets.stages import DEFAULT_STAGE_PATH
    from bricksim.core import (arrange_parts_in_workspace, import_lego,
                               set_assembly_thresholds)

    # Physics writes transforms back to USD only when that writeback is on.
    # Headless with no render pipeline it is effectively off, so every
    # SingleXFormPrim read the demo's skills make is stale -- measured as a
    # 180.92 mm z discrepancy on a brick that had just been lifted 180.9 mm.
    # Enabling the renderer instead segfaults on this driver (the RTX compat
    # layer cannot be downloaded), so turn the writeback on directly.
    import carb
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/updateToUsd", True)
    settings.set_bool("/physics/updateVelocitiesToUsd", True)
    settings.set_bool("/physics/fabricUpdateTransformations", False)

    if World._world_initialized:
        World.clear_instance()
    await open_stage_async(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    await world.initialize_simulation_context_async()

    robot_path = "/World/Robot"
    add_reference_to_stage(usd_path=get_assets_root_path() + ROBOT_USD,
                           prim_path=robot_path)
    stage = get_current_stage()
    stage.GetPrimAtPath("/World/Robot/panda_rightfinger/geometry").SetInstanceable(False)
    stage.GetPrimAtPath("/World/Robot/panda_leftfinger/geometry").SetInstanceable(False)

    # Grip has to hold a brick against the press, so the demo's tuned pad
    # friction and finger drive are kept verbatim.
    pad = PhysicsMaterial(prim_path="/World/PhysicsMaterials/FingerPad",
                          static_friction=2.5, dynamic_friction=2.0,
                          restitution=0.0)
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        SingleGeometryPrim(
            prim_path="/World/Robot/%s/geometry/%s" % (finger, finger)
        ).apply_physics_material(pad)
    joint = stage.GetPrimAtPath("/World/Robot/panda_hand/panda_finger_joint1")
    joint.GetAttribute("drive:linear:physics:maxForce").Set(15.0)
    joint.GetAttribute("drive:linear:physics:damping").Set(80.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(400.0)
    stage.GetPrimAtPath(robot_path).GetAttribute(
        "physxArticulation:solverPositionIterationCount").Set(64)

    table = PhysicsMaterial(prim_path="/World/PhysicsMaterials/Tabletop",
                            static_friction=1.0, dynamic_friction=0.8,
                            restitution=0.2)
    SingleGeometryPrim(
        prim_path="/World/scene/roomScene/colliders/table/tableTopActor"
    ).apply_physics_material(table)

    # The demo deactivates these before running. I skipped them as cosmetic --
    # they are not: floor, walls and windows are COLLIDERS, and /World/Cube is
    # a body in the scene. Leaving them active changes what the arm and the
    # bricks can collide with.
    for path in ("/World/Cube",
                 "/World/scene/roomScene/colliders/floor",
                 "/World/scene/roomScene/colliders/walls",
                 "/World/scene/roomScene/colliders/windows",
                 "/World/scene/roomScene/renderables"):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.SetActive(False)

    SingleXFormPrim(prim_path=robot_path, name="Robot").set_world_pose(
        position=ROBOT_BASE_POS, orientation=ROBOT_BASE_QUAT)
    robot = SingleArticulation(prim_path=robot_path, name="Robot")
    world.scene.add(robot)

    rmpflow = RmpFlow(**load_supported_motion_policy_config("Franka", "RMPflow"))
    motion_policy = ArticulationMotionPolicy(robot, rmpflow)
    patch_demo_pose_reads(demo, rmpflow)
    patch_grasp_face(demo)
    demo.PRESS_DEPTH = args_cli.press_depth_mm / 1000.0
    print("  press depth %.2f mm" % args_cli.press_depth_mm, flush=True)

    workspace = stage.GetPrimAtPath("/World/LegoWorkspace")
    workspace.GetAttribute("xformOp:scale").Set(Gf.Vec3d(0.4, 0.3, 0.2))
    workspace.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.35, 0.0, 0.1))
    workspace.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    workspace.GetRelationship("lego:workspace_obstacles").AddTarget(
        "/World/Robot/panda_link0")

    # The demo reads every pose it needs through USD (SingleXFormPrim), and
    # physics only flushes transforms to USD when rendering. Headless that
    # leaves reads stale, which is invisible during the grasp -- the brick has
    # not moved yet -- and silently wrong for assembly, which needs the CURRENT
    # pose of the carried brick and the end-effector.
    print("  sync_to_usd was %s; forcing on" % get_sync_to_usd(), flush=True)
    set_sync_to_usd(True)

    # The GATE thresholds, which were never applied before: the runner set the
    # clutch preload only, so assembly ran at BrickSim's defaults (1.0 N,
    # 2.0 mm). That is why the inserter developed ~1 N of press -- the gate
    # asked for 1 N. These are §2.3.3's values, carried in the calibration.
    cal_doc = json.loads((ROOT / "joint_calibration.json").read_text())
    at = AssemblyThresholds()
    if args_cli.gate == "spec":
        at.required_force = cal_doc["bricksim_settings"]["required_force"]
        at.position_tolerance = cal_doc["bricksim_settings"]["position_tolerance"]
        at.z_angle_tolerance = 4.0 * math.pi / 180.0
    else:
        # BrickSim's own defaults. Not the spec, but the only gate the current
        # positional-overshoot inserter can actually satisfy.
        at.required_force = 1.0
        at.position_tolerance = 0.002
        at.z_angle_tolerance = 5.0 * math.pi / 180.0
    at.distance_tolerance = 0.001
    at.max_penetration = 0.005
    at.yaw_tolerance = 5.0 * math.pi / 180.0
    set_assembly_thresholds(at)
    print("  assembly gate [%s]: required_force %.1f N, position_tolerance "
          "%.2f mm" % (args_cli.gate, at.required_force,
                       at.position_tolerance * 1000.0), flush=True)
    if args_cli.gate == "spec":
        print("  NOTE: the scripted inserter develops at most ~2.6 N, so at "
              "the spec gate it places nothing (ledger: "
              "inserter_cannot_reach_spec_press). Use --gate default to see "
              "the baseline work.", flush=True)

    GRASP_ATTEMPTS = args_cli.grasp_attempts
    grasp_failures = []
    placed_bricks = set()
    face_retry_wins = []

    topology = json.loads(Path(topo_path).read_text())
    placed_topology = dict(topology)
    placed_topology["parts"] = [p for p in topology["parts"] if p["id"] == 0]
    placed_topology["connections"] = []
    pre_placed, _ = import_lego(json=placed_topology, env_id=-1,
                                ref_pos=[0.3, 0.0, 0.0],
                                ref_rot=(1.0, 0.0, 0.0, 0.0))

    loose_topology = dict(topology)
    loose_topology["parts"] = [p for p in topology["parts"] if p["id"] != 0]
    loose_topology["connections"] = []
    loose_topology["pose_hints"] = []
    loose, _ = import_lego(json=loose_topology, env_id=-1)
    arrange_parts_in_workspace(workspace_path="/World/LegoWorkspace",
                               parts_to_arrange=list(loose.values()))

    paths = dict(pre_placed)
    paths.update(loose)
    # Brick footprints, so a player can draw boxes without the USD assets.
    part_dims = {p["id"]: [p["payload"]["L"] * 0.008,
                           p["payload"]["W"] * 0.008,
                           p["payload"]["H"] * 0.0032]
                 for p in topology["parts"]}
    await world.reset_async()
    # PLAY, not pause. The demo pauses here because it is interactive and the
    # user presses Play; headless, a paused world never steps physics and the
    # first skill hangs forever awaiting a physics callback.
    await world.play_async()

    # RmpFlow is a reactive policy, not a planner: it can settle into a
    # configuration from which the next target is unreachable, which shows up
    # as a grasp where the gripper never even touches the brick (measured:
    # three attempts, brick displaced 0.0 mm, nothing crowding it). Returning
    # to the home configuration before each attempt removes that history. The
    # gripper is empty at that moment, so setting joint positions is safe.
    home_q = np.asarray(robot.get_joint_positions(), dtype=float).copy()
    print("  home configuration captured (%d joints)" % home_q.size, flush=True)

    async def go_home():
        robot.set_joint_positions(home_q)
        robot.set_joint_velocities(np.zeros_like(home_q))
        for _ in range(30):
            world.step(render=False)

    press_samples = []

    async def record_poses(paths, robot, out_path, hz):
        """Sample every managed part and the arm's joints while the run proceeds.

        Poses come from the PHYSICS view -- USD is stale headless (that was the
        WP4 bug) -- so this is the only faithful record of what happened.
        """
        from bricksim.utils.physics_step import wait_for_physics_step
        from isaacsim.core.prims import RigidPrim

        views = {}
        for pid, prim in paths.items():
            try:
                v = RigidPrim(prim_paths_expr=prim)
                v.initialize()
                views[pid] = (prim, v)
            except Exception:
                pass

        # The arm's links, so a player can draw the robot without solving
        # forward kinematics. Joint values alone would need the URDF and the
        # right joint ordering; link poses are unambiguous.
        link_views = {}
        for name in ROBOT_LINKS:
            try:
                v = RigidPrim(prim_paths_expr="%s/%s" % (robot_path, name))
                v.initialize()
                link_views[name] = v
            except Exception:
                pass
        print("  recording %d parts and %d robot links"
              % (len(views), len(link_views)), flush=True)

        dt = float(world.get_physics_dt() or (1.0 / 60.0))
        every = max(1, int(round(1.0 / (hz * dt))))
        frames, step_i, t = [], 0, 0.0
        handle = out_path.open("w")
        try:
            while True:
                await wait_for_physics_step(world)
                step_i += 1
                t += dt
                if step_i % every:
                    continue
                frame = {"t": round(t, 4), "parts": {}}
                for pid, (prim, v) in views.items():
                    try:
                        pos, quat = v.get_world_poses()
                        frame["parts"][str(pid)] = [
                            [round(float(x), 6) for x in pos[0]],
                            [round(float(x), 6) for x in quat[0]]]
                    except Exception:
                        pass
                links = {}
                for name, v in link_views.items():
                    try:
                        pos, quat = v.get_world_poses()
                        links[name] = [[round(float(x), 6) for x in pos[0]],
                                       [round(float(x), 6) for x in quat[0]]]
                    except Exception:
                        pass
                if links:
                    frame["links"] = links
                try:
                    frame["joints"] = [round(float(q), 5)
                                       for q in robot.get_joint_positions()]
                except Exception:
                    pass
                handle.write(json.dumps(frame) + "\n")
                frames.append(frame)
        except asyncio.CancelledError:
            handle.close()
            print("  recorded %d frames -> %s" % (len(frames), out_path),
                  flush=True)
            raise

    async def sample_press_force(hole_path, stud_path, out):
        """Record contact force on the carried brick while the press runs.

        Two independent measures: the brick's net contact force from the
        physics view, and the gate's own projected_force along the stud axis,
        which is the quantity it compares against required_force.
        """
        from bricksim.utils.physics_step import wait_for_physics_step
        view = RigidPrim(prim_paths_expr=hole_path)
        view.initialize()
        net_peak, gate_peak, n = 0.0, 0.0, 0
        try:
            while True:
                await wait_for_physics_step(world)
                n += 1
                try:
                    f = np.asarray(view.get_net_contact_forces(), dtype=float)
                    net_peak = max(net_peak, float(np.linalg.norm(f)))
                except Exception:
                    pass
                for info in get_assembly_debug_infos():
                    if hole_path in (info.hole_path, info.stud_path):
                        gate_peak = max(gate_peak, abs(float(info.projected_force)))
        except asyncio.CancelledError:
            out.append({"brick": hole_path, "steps_sampled": n,
                        "peak_net_contact_n": round(net_peak, 3),
                        "peak_gate_projected_force_n": round(gate_peak, 3)})
            raise

    recorder = None
    if args_cli.record:
        # The launcher runs from BrickSim, so a relative path would land there.
        if not args_cli.record.is_absolute():
            args_cli.record = ROOT / args_cli.record
        args_cli.record.parent.mkdir(parents=True, exist_ok=True)
        try:
            joint_names = list(robot.dof_names)
        except Exception:
            joint_names = []
        meta = {"meta": True, "structure": args_cli.structure,
                "robot_prim": robot_path,
                "robot_usd": ROBOT_USD,
                "robot_base": [list(ROBOT_BASE_POS), list(ROBOT_BASE_QUAT)],
                "joint_names": joint_names,
                "links": ROBOT_LINKS,
                "parts": {str(pid): {"prim": prim,
                                     "dims": part_dims.get(pid)}
                          for pid, prim in paths.items()}}
        with args_cli.record.with_suffix(".meta.json").open("w") as f:
            f.write(json.dumps(meta, indent=1))
        recorder = asyncio.ensure_future(
            record_poses(paths, robot, args_cli.record, args_cli.record_hz))

    # Our planner's order, connection by connection.
    for step, conn in enumerate(topology["connections"]):
        print("=== Assembly Step %d / %d ==="
              % (step + 1, len(topology["connections"])), flush=True)
        stud, hole = paths[conn["stud_id"]], paths[conn["hole_id"]]

        # A brick resting on two supports produces TWO connections, so the
        # same brick comes round twice. The second visit would try to grasp a
        # brick already built into the structure and log a spurious failure.
        # The demo's own loop skips already-connected pairs; mine did not.
        if are_parts_connected(stud, hole):
            print("  already connected; skipping", flush=True)
            continue
        if hole in placed_bricks:
            print("  %s already placed via another support; skipping"
                  % hole, flush=True)
            continue

        # grasp_lego_part returns the success of its own motions, not whether
        # a brick is actually held (its docstring claims a tuple; it returns a
        # bool). Verify physically: a held brick must have risen. Retry on
        # failure -- every WP4 loss so far was a grasp, none was an assembly.
        held = SingleRigidPrim(prim_path=hole)
        lifted_mm, attempts = 0.0, 0
        for attempt in range(1, GRASP_ATTEMPTS + 1):
            attempts = attempt
            await go_home()
            before = held.get_world_pose()[0]
            ok = await demo.grasp_lego_part(world, robot, rmpflow,
                                            motion_policy, hole)
            after = held.get_world_pose()[0]
            lifted_mm = (float(after[2]) - float(before[2])) * 1000.0
            if lifted_mm >= 5.0:
                break
            # Diagnose before retrying: did the brick get knocked aside, and
            # how crowded is its neighbourhood?
            moved_mm = round(float(np.linalg.norm(
                np.asarray(after) - np.asarray(before))) * 1000.0, 2)
            nearest = None
            for other_id, other_path in paths.items():
                if other_path == hole:
                    continue
                try:
                    op = SingleRigidPrim(prim_path=other_path).get_world_pose()[0]
                except Exception:
                    continue
                d = float(np.linalg.norm(np.asarray(op) - np.asarray(after))) * 1000.0
                nearest = d if nearest is None else min(nearest, d)
            print("  grasp attempt %d failed: rose %.1f mm, brick shifted "
                  "%.1f mm, nearest neighbour %s mm"
                  % (attempt, lifted_mm, moved_mm,
                     round(nearest, 1) if nearest else "n/a"), flush=True)

            # Two different failures need two different responses.
            #   brick shifted  -> the gripper hit it; it is somewhere new now,
            #                     so simply retrying re-reads the pose and
            #                     usually works.
            #   brick unmoved  -> the gripper never reached it. Retrying the
            #                     identical pose is deterministic and will
            #                     fail again (observed: three attempts, 0.0 mm
            #                     every time). Move the brick instead: rotate
            #                     90 deg and bring it closer to the base, which
            #                     changes both the approach orientation and the
            #                     reach.
            if moved_mm < 5.0 and attempt < GRASP_ATTEMPTS:
                view = RigidPrim(prim_paths_expr=hole)
                view.initialize()
                pos, quat = view.get_world_poses()
                new_pos = np.array(pos, dtype=float)
                new_pos[0][0] -= 0.03 * attempt          # toward the robot base
                new_pos[0][2] += 0.01                    # lift clear, then drop
                half = math.pi / 4.0 * attempt           # 45 deg per attempt
                spin = np.array([[math.cos(half), 0.0, 0.0, math.sin(half)]])
                view.set_world_poses(positions=new_pos, orientations=spin)
                view.set_velocities(np.zeros((1, 6)))
                print("    repositioned for retry (-%.0f mm x, %.0f deg yaw)"
                      % (30.0 * attempt, math.degrees(2 * half)), flush=True)

            for _ in range(90):          # let the scene settle before retrying
                world.step(render=False)

        if lifted_mm < 5.0:
            print("  GRASP FAILED after %d attempts" % attempts, flush=True)
            grasp_failures.append({"step": step, "brick": hole,
                                   "attempts": attempts})
            continue
        if attempts > 1:
            print("  grasp succeeded on attempt %d" % attempts, flush=True)
        if args_cli.measure_press:
            sampler = asyncio.ensure_future(
                sample_press_force(hole, stud, press_samples))
            try:
                await demo.assemble_lego_part(world, robot, rmpflow,
                                              motion_policy, stud, hole,
                                              tuple(conn["offset"]), conn["yaw"])
            finally:
                sampler.cancel()
                try:
                    await sampler
                except asyncio.CancelledError:
                    pass
            if press_samples:
                print("    press force: net %.2f N, gate-projected %.2f N "
                      "(over %d steps)"
                      % (press_samples[-1]["peak_net_contact_n"],
                         press_samples[-1]["peak_gate_projected_force_n"],
                         press_samples[-1]["steps_sampled"]), flush=True)
        else:
            await demo.assemble_lego_part(world, robot, rmpflow, motion_policy,
                                          stud, hole, tuple(conn["offset"]),
                                          conn["yaw"])

        # Verify AFTER settling. The gate can fire while the scene comes to
        # rest, so checking the instant the skill returns marks real successes
        # as failures -- observed: an episode logged FAILED whose brick was
        # connected moments later.
        for _ in range(VERIFY_SETTLE_STEPS):
            world.step(render=False)

        # If it still has not connected, the usual cause is that the fingers
        # cannot reach beside a brick already in that layer (§3.4
        # accessibility). Re-place with the other grasp face, which rotates the
        # approach 90 degrees into the free direction.
        used_alternate_face = False
        if not are_parts_connected(stud, hole):
            print("  not connected; retrying with the other grasp face",
                  flush=True)
            used_alternate_face = True
            GRASP_FACE["swap"] = True
            try:
                await go_home()
                before = SingleRigidPrim(prim_path=hole).get_world_pose()[0]
                await demo.grasp_lego_part(world, robot, rmpflow,
                                           motion_policy, hole)
                after = SingleRigidPrim(prim_path=hole).get_world_pose()[0]
                if (float(after[2]) - float(before[2])) * 1000.0 >= 5.0:
                    await demo.assemble_lego_part(
                        world, robot, rmpflow, motion_policy, stud, hole,
                        tuple(conn["offset"]), conn["yaw"])
                    for _ in range(VERIFY_SETTLE_STEPS):
                        world.step(render=False)
                else:
                    print("  alternate-face grasp did not lift", flush=True)
            finally:
                GRASP_FACE["swap"] = False

        connected = bool(are_parts_connected(stud, hole))
        if connected:
            placed_bricks.add(hole)
            if used_alternate_face:
                face_retry_wins.append(hole)
                print("  alternate grasp face SUCCEEDED", flush=True)
        record_placement(step, conn, stud, hole, connected,
                         used_alternate_face, attempts)

    if recorder is not None:
        recorder.cancel()
        try:
            await recorder
        except asyncio.CancelledError:
            pass


def main():
    # --- our structure, our order ---------------------------------------
    if args_cli.demo_structure:
        # BrickSim's own demo structure, ordered the way its demo orders it.
        # If this also fails, the fault is in our headless harness, not in the
        # topology our planner emits.
        from bricksim.topology.ordering import bfs_sort_connections
        src = BRICKSIM / "demos" / "demo_assembly_structure1.json"
        topo = bfs_sort_connections(json.loads(src.read_text()))
        brick_ids = {}
        topo_path = ROOT / "plans" / "demo_structure1.topology.json"
        topo_path.write_text(json.dumps(topo))
        print("BISECT: running BrickSim's own structure through our harness",
              flush=True)
    else:
        topo = P.to_bricksim_topology(args_cli.structure)
        brick_ids = topo.pop("_brick_ids", {})
        topo_path = ROOT / "plans" / ("%s.topology.json" % args_cli.structure)
        topo_path.write_text(json.dumps(topo))

    demo = load_demo()

    # --- WP2b calibration, before anything is assembled ------------------
    cal = json.loads((ROOT / "joint_calibration.json").read_text())
    bt = BreakageThresholds()
    bt.preloaded_force = (args_cli.preload if args_cli.preload is not None
                          else cal["bricksim_settings"]["live_sim"]["preloaded_force"])
    set_breakage_thresholds(bt)

    # --- log one episode per placement -----------------------------------
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    original_assemble = demo.assemble_lego_part

    def record_placement(step, conn, stud, hole, connected, alt_face, attempts):
        row = {
            "run_id": "wp4_%s_trial%02d" % (args_cli.structure, args_cli.trial),
            "structure": args_cli.structure, "step": step,
            "stud_path": stud, "hole_path": hole,
            "offset": list(conn["offset"]), "yaw_index": conn["yaw"],
            "policy": "scripted_bricksim_demo", "backend": "physx",
            "joint_model": "bricksim",
            "preloaded_force_n": bt.preloaded_force,
            "grasp_attempts": attempts,
            "used_alternate_grasp_face": bool(alt_face),
            "success": bool(connected),
            "verified_after_settle_steps": VERIFY_SETTLE_STEPS,
            "failure_mode": None if connected else "not_connected",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        episodes.append(row)
        with args_cli.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("  PLACEMENT %s %s%s" % (hole, "OK" if connected else "FAILED",
                                       " (alt face)" if alt_face else ""),
              flush=True)

    # --- run OUR orchestrator under the Kit loop -------------------------
    # Not demo.main(): it ends by pointing the viewport camera at the scene,
    # and /OmniverseKit_Persp does not exist headless ("Accessed invalid null
    # prim"). §WP4 wants orchestration/ to be ours anyway; the scene setup
    # below is the demo's, minus the cosmetics.
    task = asyncio.ensure_future(
        assemble_structure(demo, topo_path, bt, record_placement))
    deadline = time.time() + args_cli.timeout_s
    while not task.done() and time.time() < deadline:
        simulation_app.update()
    timed_out = not task.done()
    if timed_out:
        task.cancel()
    elif task.exception() is not None:
        print("  demo.main() raised: %r" % task.exception(), flush=True)

    placed = len({e["hole_path"] for e in episodes if e["success"]})
    # Count BRICKS to place, not connections: a brick resting on two supports
    # has two connections but is placed once.
    total = len({c["hole_id"] for c in topo["connections"]})
    print("\n-- WP4 %s trial %d --" % (args_cli.structure, args_cli.trial), flush=True)
    print("  placed %d / %d   %s" % (placed, total,
                                     "TIMED OUT" if timed_out else ""), flush=True)
    print("  episodes -> %s" % args_cli.out, flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
