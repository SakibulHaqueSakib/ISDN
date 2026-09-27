"""Assembly cell for the MuJoCo twin: two Pandas, a baseplate, bricks, clutch pool.

    cell = build_cell(plan)          # plan: assembly_plan dict (planner.build_plan)
    cell.model, cell.data            # mujoco.MjModel / MjData

Layout mirrors dual_arm_sim.py (the Newton prototype) so plans transfer:
arm A, the stabilizer, at (-0.45, 0, 0) facing +x; arm B, the placer, at
(+0.45, 0, 0) facing -x; the build centred on the origin; loose bricks in a
feeder on the placer's -y side, already at their final yaw.

The clutch pool (master_report §2.3.3, WP1 step 4 "pre-allocated pool; runtime
joint creation is too expensive") holds, for every planned connection
(brick U on support L, or on the baseplate):
  * an INACTIVE weld U->L at the seated relative pose -- the MATED state;
  * a spatial tendon from U's anti-stud face (at the patch centre) to a
    point 50 mm below L's stud face on the same axis, whose limit/frictionloss
    the clutch model drives while U's studs are inside L's cavity -- the
    stud/cavity interference that must be pushed through to seat a brick.
    Its line of action passes through the interface, so it puts no moment on
    U (a site 50 mm ABOVE U tilted the gripped brick through a 45 mm lever);
    the 50 mm length keeps that line vertical for any lateral offset. Because it is a tendon, the resisting force acts
    equal-and-opposite on U and L (§2.3.3 invariant 1) and is solved
    implicitly by MuJoCo's constraint solver, so it cannot pump energy into
    a 1 g brick the way an explicit force would.
sim/joint_model/clutch.py drives both.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

import planner as P
from sim.mj import assets
from sim.mj import bricks as Bk

ARM_BASE = {"A": ((-0.45, 0.0, 0.0), 0.0), "B": ((0.45, 0.0, 0.0), math.pi)}
Z0 = 0.0032                      # baseplate top: a 3.2 mm plate on the table
TIMESTEP = 0.001                 # s; see §2.3.2 -- 1/500 s is the eval spec, 1 ms is finer
TENDON_SITE_DZ = 0.05            # L's tendon site BELOW its stud face (see below)
# Stiff contacts for the bricks and hands (measured: default solref lets a
# 35 N press sink a 1.25 g brick 21 mm; (0.002, 1) holds it to 0.1 mm).
SOLREF = [0.002, 1.0]
SOLIMP = [0.99, 0.9999, 0.0001, 0.5, 2.0]   # near-hard: thin walls vs 4.6 mm studs
WELD_SOLREF = [0.003, 1.0]
WELD_SOLIMP = [0.9999, 0.9999, 0.0001, 0.5, 2.0]   # d->1: a 1 g brick otherwise yields degrees
PAD_FRICTION = [1.0, 0.01, 0.0001]   # brick-finger U(0.5, 1.2), §5.5
TCP_OFFSET = 0.1034              # hand frame -> Franka TCP (pad centre)
ARM_MOTOR_LIMIT = [87, 87, 87, 87, 12, 12, 12]
GRIP_FORCE_LIMIT = 140.0         # N on the split tendon (70 N per finger)


def _load_panda():
    xml = assets.fetch()
    root = ET.parse(xml).getroot()
    asset = root.find("asset")
    keep = {"link0_c", "link1_c", "link2_c", "link3_c", "link4_c", "link5_c0", "link5_c1",
            "link5_c2", "link6_c", "link7_c", "hand_c"}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in keep:
            asset.remove(mesh)
    for body in root.iter("body"):
        for g in list(body.findall("geom")):
            # visual meshes are not fetched headless; the finger's collision
            # hull reaches ~3 mm below its rubber pads and lands on the
            # neighbouring stud before a brick seats, so the pads alone collide
            if g.get("class") == "visual" or g.get("mesh") == "finger_0":
                body.remove(g)
    root.remove(root.find("actuator"))
    root.remove(root.find("keyframe"))
    root.find("compiler").set("meshdir", str(xml.parent / "assets"))
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


@dataclass
class Connection:
    """One planned stud connection: brick U onto support L (None = baseplate)."""
    upper: str
    lower: object                 # brick id or None
    n_studs: int
    patch_center_u: np.ndarray    # patch centre in U's frame (z = 0)
    studs_u: np.ndarray           # (n, 2) stud xy in U's frame
    rect_u: tuple                 # (xmin, xmax, ymin, ymax) overlap rectangle, U frame
    rel_pos: np.ndarray           # seated pose of U in L's frame (world if baseplate)
    rel_quat: np.ndarray          # wxyz
    eq_id: int = -1
    tendon_id: int = -1


@dataclass
class Cell:
    model: mujoco.MjModel
    data: mujoco.MjData
    plan: dict
    targets: dict                 # brick id -> (pos(3), quat wxyz) seated world pose
    dims: dict                    # brick id -> (nx, ny)
    feeder: dict                  # brick id -> spawn pos
    connections: list
    arms: dict = field(default_factory=dict)

    def body(self, name):
        return self.model.body(name).id


def _cells_world(b):
    """Grid cells (i, j) and layer k of a planner brick dict."""
    nx, ny = P.footprint(b["type"], b["yaw_index"])
    i, j, k = b["grid_pos"]
    return {(i + a, j + c) for a in range(nx) for c in range(ny)}, k


def feeder_slots(n):
    """Pick-up spots on the placer's -y side, nearest first (dual_arm_sim layout)."""
    base = np.array(ARM_BASE["B"][0][:2])
    cand = [(x, -y) for x in (0.03, 0.10, 0.17, 0.24, 0.31, 0.38)
            for y in (0.20, 0.27, 0.34, 0.41, 0.48, 0.55)]
    cand = [g for g in cand if 0.36 <= np.linalg.norm(np.array(g) - base) <= 0.66]
    cand.sort(key=lambda g: np.linalg.norm(np.array(g) - base))
    if n > len(cand):
        raise ValueError("%d bricks but only %d feeder slots" % (n, len(cand)))
    return cand[:n]


def centre_plan_origin(bricks):
    """Set planner.VOXEL_ORIGIN so the build is centred between the arms."""
    NI = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    NJ = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    P.VOXEL_ORIGIN = (-NI * P.PITCH / 2, -NJ * P.PITCH / 2, Z0)
    return NI, NJ


def build_cell(plan, preplaced=(), colors=None, with_arms=True, spawn=None):
    """Compile the cell for an assembly_plan dict.

    preplaced: brick ids spawned already seated at their targets (their
    connections then start MATED -- the clutch model is told by the caller);
    everything else starts in the feeder, or at spawn[brick_id] if given.
    with_arms=False gives a bricks-only model for joint-model tests.
    """
    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.noslip_iterations = 0     # switched on per phase: runtime.Sim.noslip
    spec.compiler.autolimits = True
    spec.stat.meansize = 0.05
    w = spec.worldbody
    table = w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.1])
    table.contype, table.conaffinity = Bk.WORLD_TYPE, Bk.WORLD_AFFINITY
    table.rgba = [0.8, 0.78, 0.74, 1]
    w.add_light(pos=[0, 0, 2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    # --- arms ---------------------------------------------------------------
    for name, (pos, yaw) in (ARM_BASE.items() if with_arms else ()):
        arm = _load_panda()
        f = w.add_frame(pos=list(pos), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(arm, frame=f, prefix=name + "/")
        hand = spec.body(name + "/hand")
        hand.add_site(name=name + "/tcp", pos=[0, 0, TCP_OFFSET], size=[0.002, 0, 0])
        hand.add_site(name=name + "/ft", pos=[0, 0, 0], size=[0.002, 0, 0])
        spec.add_sensor(name=name + "/ft_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name + "/ft")
        spec.add_sensor(name=name + "/ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name + "/ft")
        for j, lim in enumerate(ARM_MOTOR_LIMIT):
            a = spec.add_actuator(name="%s/motor%d" % (name, j + 1), target="%s/joint%d" % (name, j + 1),
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gainprm[0] = 1.0
            a.ctrlrange = [-lim, lim]
            a.forcerange = [-lim, lim]
        g = spec.add_actuator(name=name + "/grip", target=name + "/split",
                              trntype=mujoco.mjtTrn.mjTRN_TENDON)
        g.gainprm[0] = 1.0
        g.ctrlrange = [-GRIP_FORCE_LIMIT, GRIP_FORCE_LIMIT]
        g.forcerange = [-GRIP_FORCE_LIMIT, GRIP_FORCE_LIMIT]

    # --- bricks -------------------------------------------------------------
    bricks = {b["id"]: b for b in plan["bricks"]}
    steps = plan["sequence"]
    ox, oy, oz = plan["voxel_origin"]
    NI = max(max(i for i, _ in _cells_world(b)[0]) for b in bricks.values()) + 1
    NJ = max(max(j for _, j in _cells_world(b)[0]) for b in bricks.values()) + 1
    Bk.add_baseplate(spec, NI, NJ, (ox, oy), Z0)

    targets, dims, feeder = {}, {}, {}
    slots = feeder_slots(len(steps))
    palette = colors or [(0.80, 0.10, 0.10, 1), (0.10, 0.25, 0.80, 1), (0.95, 0.75, 0.10, 1),
                         (0.10, 0.60, 0.20, 1), (0.90, 0.90, 0.90, 1), (0.95, 0.45, 0.10, 1)]
    for n, s in enumerate(steps):
        b = bricks[s["brick_id"]]
        nx, ny = P.footprint(b["type"], b["yaw_index"])
        i, j, k = b["grid_pos"]
        tgt = np.array([ox + (i + nx / 2) * P.PITCH, oy + (j + ny / 2) * P.PITCH, oz + k * P.BRICK_H])
        targets[b["id"]] = (tgt, np.array([1.0, 0, 0, 0]))
        dims[b["id"]] = (nx, ny)
        if b["id"] in preplaced:
            at = tgt.copy()
        elif spawn and b["id"] in spawn:
            at = np.asarray(spawn[b["id"]], float)
        else:
            at = np.array([slots[n][0], slots[n][1], 0.0])
        feeder[b["id"]] = np.array([slots[n][0], slots[n][1], 0.0])
        Bk.add_brick(spec, w, b["id"], nx, ny, at, rgba=palette[n % len(palette)])

    # --- clutch pool ----------------------------------------------------------
    conns = []
    placed = {}
    for s in steps:
        b = bricks[s["brick_id"]]
        cu, ku = _cells_world(b)
        supports = [m[0] for m in s["mating_studs"]] or [None]
        for sid in supports:
            if sid is None:
                patch = cu
            else:
                cl, kl = _cells_world(bricks[sid])
                assert kl == ku - 1, (sid, b["id"])
                patch = cu & cl
            pu, qu = targets[b["id"]]
            cells_xy = np.array([[ox + (i + 0.5) * P.PITCH, oy + (j + 0.5) * P.PITCH] for i, j in patch])
            studs_u = cells_xy - pu[:2]
            cen = studs_u.mean(axis=0)
            xs, ys = studs_u[:, 0], studs_u[:, 1]
            rect = (xs.min() - P.PITCH / 2, xs.max() + P.PITCH / 2,
                    ys.min() - P.PITCH / 2, ys.max() + P.PITCH / 2)
            if sid is None:
                rel_pos, rel_quat = pu.copy(), np.array([1.0, 0, 0, 0])
            else:
                pl, _ = targets[sid]
                rel_pos, rel_quat = pu - pl, np.array([1.0, 0, 0, 0])
            c = Connection(upper=b["id"], lower=sid, n_studs=len(patch),
                           patch_center_u=np.array([cen[0], cen[1], 0.0]), studs_u=studs_u,
                           rect_u=rect, rel_pos=rel_pos, rel_quat=rel_quat)
            tag = "%s_on_%s" % (b["id"], sid or "plate")
            # tendon sites: 50 mm below L's stud face, and on U's anti-stud face
            lower_body = w if sid is None else spec.body(sid)
            lower_site_pos = (np.array([pu[0] + cen[0], pu[1] + cen[1], Z0 - TENDON_SITE_DZ])
                              if sid is None else
                              np.array([pu[0] + cen[0] - pl[0], pu[1] + cen[1] - pl[1],
                                        P.BRICK_H - TENDON_SITE_DZ]))
            lower_body.add_site(name="clutch/%s/L" % tag, pos=lower_site_pos.tolist(), size=[0.0005, 0, 0])
            spec.body(b["id"]).add_site(name="clutch/%s/U" % tag, pos=[cen[0], cen[1], 0.0],
                                        size=[0.0005, 0, 0])
            t = spec.add_tendon(name="clutch/%s" % tag)
            t.wrap_site("clutch/%s/L" % tag)
            t.wrap_site("clutch/%s/U" % tag)
            t.frictionloss = 0.0
            t.solref_friction = [0.002, 1.0]
            t.solimp_friction = [0.9999, 0.9999, 0.0001, 0.5, 2.0]
            # static friction: a lower length limit the clutch model locks at
            # the current length while the brick is stuck (no creep)
            t.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            t.range = [0.0, 1.0]
            t.solref_limit = [0.002, 1.0]
            t.solimp_limit = [0.9999, 0.9999, 0.0001, 0.5, 2.0]
            eq = spec.add_equality(name="clutch/%s" % tag, type=mujoco.mjtEq.mjEQ_WELD,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1="world" if sid is None else sid, name2=b["id"])
            eq.active = False
            eq.solref = list(WELD_SOLREF)
            eq.solimp = list(WELD_SOLIMP)
            data = np.zeros(11)
            # anchor at U's origin: with relpose given, MuJoCo places body1's
            # weld point from relpose alone, so a non-zero anchor leaves a
            # residual equal to the anchor (16 mm on S3's cantilever, and a
            # 0.01 J kick at t=0). A weld is a rigid 6-D attachment wherever
            # its anchor is; the clutch model moves the read-back wrench to
            # the patch centre.
            data[0:3] = 0.0
            data[3:6] = rel_pos
            data[6:10] = rel_quat
            data[10] = 1.0                                 # torquescale
            eq.data = data.tolist()
            conns.append(c)
        placed[b["id"]] = b

    # --- contact parameters -------------------------------------------------------
    for geom in spec.geoms:
        geom.solref = list(SOLREF)
        geom.solimp = list(SOLIMP)
        pname = geom.name or ""
        if pname.startswith(("A/", "B/")) or geom.parent.name.startswith(("A/", "B/")):
            parent = geom.parent.name
            if parent.endswith(("hand", "left_finger", "right_finger")):
                geom.contype, geom.conaffinity = Bk.HAND_TYPE, Bk.HAND_AFFINITY
                geom.friction = list(PAD_FRICTION)
                geom.condim = 4
                # stiff pads: measured in-hand drift over 8 transports 0.58 deg
                # (vs 0.80 at the brick default); noslip during the grasp itself
                # made it worse (2.2-40 deg), so it is only used while pressing
                geom.solimp = [0.9999, 0.9999, 0.0001, 0.5, 2.0]
            else:
                # links 0-7 never touch anything: plans keep them clear, as
                # the Newton prototype did
                geom.contype, geom.conaffinity = 0, 0

    model = spec.compile()
    for c in conns:
        tag = "%s_on_%s" % (c.upper, c.lower or "plate")
        c.eq_id = model.equality("clutch/" + tag).id
        c.tendon_id = model.tendon("clutch/" + tag).id
        model.tendon_limited[c.tendon_id] = 0
        # MuJoCo regularises a constraint with invweight0, evaluated at qpos0 --
        # where U sits in the feeder and this tendon is long and horizontal, so
        # a huge rotational lever makes its friction 37x too soft. The tendon
        # only ever acts seated, vertical, on the two bricks' translation:
        inv = 1.0 / model.body_mass[model.body(c.upper).id]
        if c.lower is not None:
            inv += 1.0 / model.body_mass[model.body(c.lower).id]
        model.tendon_invweight0[c.tendon_id] = inv
    data = mujoco.MjData(model)
    cell = Cell(model=model, data=data, plan=plan, targets=targets, dims=dims, feeder=feeder,
                connections=conns)
    mujoco.mj_forward(model, data)
    return cell
