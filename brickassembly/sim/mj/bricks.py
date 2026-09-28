"""Brick geometry for the MuJoCo twin -- master_report §2.3 / §2.3.1.

Each brick is the hollow shell §2.3.1 prescribes (a solid box would sit
1.8 mm high on the studs): a 1.0 mm top plate, four 1.5 mm walls enclosing an
8.6 mm cavity, and one cylinder per stud. The body frame is the brick's
BOTTOM centre, x along its first footprint dimension -- the same convention as
planner.brick_pose and BrickSim.

Two departures from the bare §2.3.1 list, both geometric necessities:
  * stud colliders are 0.1 mm under nominal radius (CLEARANCE), so a seated
    stud does not interpenetrate the cavity wall it nominally touches -- the
    0.1 mm radial interference of a real clutch lives in the joint model
    (sim/joint_model/clutch.py), not in the contact geometry;
  * each wall's inner bottom edge carries a 0.6 mm x 45 deg lead-in chamfer
    (the wall is a convex pentagonal prism), so a brick within ~0.7 mm of
    alignment is guided onto the studs, as a real brick's rounded stud tops
    guide it. Without a lead-in the capture window is the 0.1 mm clearance.

Collision filtering (§2.3.2 "exclude stud-cavity pairs of mated bricks") is
done with contype/conaffinity bits rather than runtime pair exclusion, which
MuJoCo cannot change after compile:

    G  world: table, baseplate slab     T  tops: top plates and studs
    W  walls                            F  robot hands and fingers

A brick's walls collide with tops/world/fingers while it is loose and ONLY
with fingers once it is mated (MATED_WALL_AFFINITY), so a mated brick keeps
touching the hands that grasp it but stops fighting the joint weld.
"""

import numpy as np
import mujoco

import planner as P

# §2.3 geometry, metres
PITCH = P.PITCH                  # 8.0 mm
BRICK_H = P.BRICK_H              # 9.6 mm
STUD_H = P.STUD_H                # 1.8 mm
STUD_R = 0.0024                  # 4.8 mm diameter
WALL_T = 0.0015
TOP_T = 0.0010
CAVITY = BRICK_H - TOP_T         # 8.6 mm
FOOT_GAP = 0.0002                # footprint = 8n - 0.2 mm
CLEARANCE = 0.0001               # radial, stud collider vs nominal
CHAMFER = 0.0006                 # lead-in on the wall's inner bottom edge
MASS_PER_STUD = 0.0025 / 8       # §2.3: a 2x4 is ~2.5 g
INERTIA_SCALE = 10.0             # numerical conditioning, NOT physics: a 1 g brick's
                                 # 4e-8 kg m^2 rattles in the 0.1 mm stud clearance at
                                 # MuJoCo's stiff-contact stability edge (0.2 rad/s,
                                 # ~1e-9 J; tests/test_clutch.py energy test). Mass,
                                 # gravity and every contact/clutch force are unchanged;
                                 # only a free brick tumbles slower.

# collision bits
G, T, W, F = 1, 2, 4, 8
TOP_TYPE, TOP_AFFINITY = T, T | F | G
WALL_TYPE, WALL_AFFINITY = W, T | F | G
MATED_WALL_AFFINITY = F
WORLD_TYPE, WORLD_AFFINITY = G, F
HAND_TYPE, HAND_AFFINITY = F, T | W | G | F

BRICK_FRICTION = (0.6, 0.005, 0.0001)   # slide, torsion, roll; DR U(0.3, 0.9) §5.5


def outer(nx, ny):
    """Outer footprint (Lx, Ly) in metres."""
    return nx * PITCH - FOOT_GAP, ny * PITCH - FOOT_GAP


def stud_xy(nx, ny):
    """Stud centres in the brick frame, row-major over (i, j)."""
    return [((i - (nx - 1) / 2) * PITCH, (j - (ny - 1) / 2) * PITCH)
            for i in range(nx) for j in range(ny)]


def _wall_prism(length, inner_sign_axis):
    """Vertices of a wall prism centred on the wall's mid-plane.

    Cross-section in (u, z), u across the wall thickness with +u pointing
    INTO the cavity: the pentagon (-t/2,0) (t/2-c,0) (t/2,c) (t/2,H) (-t/2,H).
    Extruded +-length/2 along the wall. Returned in the brick frame's axes by
    the caller's axis mapping.
    """
    t, c, h = WALL_T, CHAMFER, CAVITY
    sec = [(-t / 2, 0.0), (t / 2 - c, 0.0), (t / 2, c), (t / 2, h), (-t / 2, h)]
    verts = []
    for s in (-length / 2, length / 2):
        for u, z in sec:
            verts.append((u, s, z) if inner_sign_axis == "x" else (s, u, z))
    return np.array(verts)


def add_brick(spec, parent, name, nx, ny, pos, quat=(1, 0, 0, 0), rgba=(0.8, 0.1, 0.1, 1),
              free=True):
    """Add an nx x ny brick body under `parent` (an MjsBody). Returns the body.

    Geom names: <name>/top, <name>/wall{0..3}, <name>/stud{k}. Sites:
    <name>/bottom (anti-stud frame, at the bottom centre) and <name>/top
    (stud frame, at the top centre) -- the two frames §2.3.3's gate compares.
    """
    Lx, Ly = outer(nx, ny)
    body = parent.add_body(name=name, pos=list(pos), quat=list(quat))
    if free:
        body.add_freejoint(name=name + "/free")
    mass = MASS_PER_STUD * nx * ny
    # inertia of a uniform box (x INERTIA_SCALE, see above)
    body.mass = mass
    body.ipos = [0, 0, BRICK_H / 2]
    body.inertia = [INERTIA_SCALE * mass / 12 * (Ly ** 2 + BRICK_H ** 2),
                    INERTIA_SCALE * mass / 12 * (Lx ** 2 + BRICK_H ** 2),
                    INERTIA_SCALE * mass / 12 * (Lx ** 2 + Ly ** 2)]
    body.explicitinertial = True

    def geom(**kw):
        g = body.add_geom(**kw)
        g.rgba = list(rgba)
        g.friction = list(BRICK_FRICTION)
        g.condim = 4
        g.mass = 0.0
        g.density = 0.0
        return g

    top = geom(name=name + "/top", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[Lx / 2, Ly / 2, TOP_T / 2], pos=[0, 0, BRICK_H - TOP_T / 2])
    top.contype, top.conaffinity = TOP_TYPE, TOP_AFFINITY

    # walls: +x, -x (full length along y), +y, -y (between the x walls)
    walls = [("x", +1, Ly, (Lx / 2 - WALL_T / 2, 0)),
             ("x", -1, Ly, (-(Lx / 2 - WALL_T / 2), 0)),
             ("y", +1, Lx - 2 * WALL_T, (0, Ly / 2 - WALL_T / 2)),
             ("y", -1, Lx - 2 * WALL_T, (0, -(Ly / 2 - WALL_T / 2)))]
    for k, (axis, sign, length, (cx, cy)) in enumerate(walls):
        v = _wall_prism(length, axis)
        # the pentagon's +u faces the cavity; a wall on the + side has its
        # cavity towards -axis, so mirror u
        if axis == "x":
            v[:, 0] *= -sign
        else:
            v[:, 1] *= -sign
        mesh_name = "%s/wall%d" % (name, k)
        spec.add_mesh(name=mesh_name, uservert=v.flatten().tolist())
        g = geom(name=mesh_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                 pos=[cx, cy, 0])
        g.contype, g.conaffinity = WALL_TYPE, WALL_AFFINITY

    for k, (sx, sy) in enumerate(stud_xy(nx, ny)):
        g = geom(name="%s/stud%d" % (name, k), type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[STUD_R - CLEARANCE, STUD_H / 2, 0], pos=[sx, sy, BRICK_H + STUD_H / 2])
        g.contype, g.conaffinity = TOP_TYPE, TOP_AFFINITY

    body.add_site(name=name + "/bottom", pos=[0, 0, 0], size=[0.001, 0, 0])
    body.add_site(name=name + "/top", pos=[0, 0, BRICK_H], size=[0.001, 0, 0])
    return body


def add_baseplate(spec, ni, nj, origin, z_top, margin=1, rgba=(0.45, 0.47, 0.45, 1)):
    """Studded baseplate covering grid cells [-margin, ni+margin) x [-margin, nj+margin).

    origin: world (x, y) of grid cell (0, 0)'s corner, as planner.VOXEL_ORIGIN.
    The slab is world geometry (bit G); its studs are tops (bit T).
    """
    w = spec.worldbody
    x0, y0 = origin[0] - margin * PITCH, origin[1] - margin * PITCH
    NX, NY = ni + 2 * margin, nj + 2 * margin
    slab = w.add_geom(name="baseplate", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[NX * PITCH / 2, NY * PITCH / 2, z_top / 2],
                      pos=[x0 + NX * PITCH / 2, y0 + NY * PITCH / 2, z_top / 2])
    slab.rgba = list(rgba)
    slab.contype, slab.conaffinity = WORLD_TYPE, WORLD_AFFINITY
    slab.friction = list(BRICK_FRICTION)
    for a in range(NX):
        for b in range(NY):
            g = w.add_geom(name="baseplate/stud_%d_%d" % (a - margin, b - margin),
                           type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[STUD_R - CLEARANCE, STUD_H / 2, 0],
                           pos=[x0 + (a + 0.5) * PITCH, y0 + (b + 0.5) * PITCH, z_top + STUD_H / 2])
            g.rgba = list(rgba)
            g.contype, g.conaffinity = TOP_TYPE, TOP_AFFINITY
            g.friction = list(BRICK_FRICTION)
    return slab
