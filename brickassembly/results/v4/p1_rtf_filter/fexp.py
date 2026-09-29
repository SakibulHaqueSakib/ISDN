"""args: mode(none|static|group|group_rt) sub gap_mm(-1 = keep) nfr. Seated S3 + 14 spares, 17 welds on, sweep on, collide per frame, s=1."""
import importlib.util, sys, time, math, types
import numpy as np, warp as wp
spec = importlib.util.spec_from_file_location("pr", "/home/bakasakib/Documents/ISDN_Robofab/brickassembly/scripts/10_newton_scale_probe.py")
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
das, newton, ik = pr.das, pr.newton, pr.ik
mode, sub, gap_mm, nfr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
s = 1
lay, (ni, nj) = pr.s3_layout(s)
geo = lambda t: pr.Geo.get(t, s)
tray_types = ["2x4"] * 8 + ["2x2"] * 6
slots = das.feeder_slots(len(tray_types))
scene = newton.ModelBuilder(); newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
arm = das.build_arm(); model_ik = arm.finalize()
park = das.Example.solve_park(types.SimpleNamespace(model_ik=model_ik))
arm.joint_q[:9] = park; arm.joint_target_pos[:9] = park
fing = [k * arm.shape_count + i for k in (0, 1) for i, bd in enumerate(arm.shape_body) if bd in (12, 13)]
for pos, yaw in (das.ARM_A, das.ARM_B):
    scene.add_builder(arm, xform=wp.transform(pos, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)))
plate_sh = pr.add_plate(scene, geo("2x4"), das.Z0, ni=ni + 4, nj=nj + 4, margin=2)
bodies, bsh = [], []
for n, b in enumerate(lay):
    bd = scene.add_body(xform=wp.transform(tuple(b["pos"]), das.qz(b["yaw"])))
    ids = pr.add_brick(scene, bd, wp.transform_identity(), geo(b["type"])); bodies.append(bd); bsh.append(ids["all"])
for n, ((x, y), t) in enumerate(zip(slots, tray_types)):
    bd = scene.add_body(xform=wp.transform((x, y, 0.0005), wp.quat_identity())); pr.add_brick(scene, bd, wp.transform_identity(), geo(t)); bodies.append(bd)
mate = {}
for n, b in enumerate(lay):
    for par in b["parents"]:
        p1, y1 = (np.zeros(3), 0.0) if par is None else (lay[par]["pos"], lay[par]["yaw"])
        mate[(par, n)] = wp.transform(das.rotate(das.qz(-y1), b["pos"] - p1), das.qz(b["yaw"] - y1))
for (par, n), r in mate.items():
    scene.add_equality_constraint_weld(body1=-1 if par is None else bodies[par], body2=bodies[n], relpose=r, enabled=True, custom_attributes=das.WELD_ATTRS)
npairs = 0
if mode == "static":
    for (par, n) in mate:
        other = plate_sh if par is None else bsh[par]
        for a in bsh[n]:
            for c in other:
                scene.add_shape_collision_filter_pair(a, c); npairs += 1
model = pr.Sim.finalize(scene)
if mode.startswith("group"):
    # one negative group for the whole built structure + plate: same negative group -> no collision (test_group_pair), negative vs positive -> collide
    g = model.shape_collision_group.numpy()
    if mode == "group":
        for sh in plate_sh + [x for l in bsh for x in l]: g[sh] = -2
        model.shape_collision_group.assign(g)
_=("GAPS", {float(v): int(c) for v, c in zip(*np.unique(model.shape_gap.numpy(), return_counts=True))})
if gap_mm >= 0:
    gp = model.shape_gap.numpy(); gp[np.abs(gp - 0.01) < 1e-4] = gap_mm * 1e-3; model.shape_gap.assign(gp)
import os
if os.environ.get('ARMGAP'):
    gp = model.shape_gap.numpy(); gp[np.abs(gp - 0.1) < 1e-4] = float(os.environ['ARMGAP']) * 1e-3; model.shape_gap.assign(gp)
ke, kd = model.shape_material_ke.numpy(), model.shape_material_kd.numpy()
ke[fing], kd[fing] = pr.FINGER_CONTACT_KE, pr.finger_kd(sub if sub in (16, 8) else 4)
model.shape_material_ke.assign(ke); model.shape_material_kd.assign(kd)
sim = pr.Sim(None, 1, sub, "frame", cmax=4096 * 8, model=model, tri=pr.TRI_F)
ik_pos = wp.array([das.PARK] * 2, dtype=wp.vec3)
ik_solver = ik.IKSolver(model=model_ik, n_problems=2, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
    objectives=[ik.IKObjectivePosition(link_index=das.EE, link_offset=wp.vec3(0, 0, 0), target_positions=ik_pos),
                ik.IKObjectiveRotation(link_index=das.EE, link_offset_rotation=wp.quat_identity(), target_rotations=wp.array([das.Q_DOWN] * 2, dtype=wp.vec4)),
                ik.IKObjectiveJointLimit(joint_limit_lower=model_ik.joint_limit_lower, joint_limit_upper=model_ik.joint_limit_upper)])
ikq = wp.array(np.tile(park, (2, 1)), dtype=wp.float32)
with wp.ScopedCapture() as c_:
    ik_solver.step(ikq, ikq, iterations=24)
sim.start()
mj = sim.solver.mjw_data
tgt = sim.control.joint_target_pos.numpy(); fr = [0]
def one():
    t = fr[0] / 60; fr[0] += 1
    sw = das.PARK + [0.0, 0.06 * math.sin(2 * math.pi * t / 4.0), 0.03 * math.sin(2 * math.pi * t / 2.0)]
    ik_pos.assign(np.array([das.PARK, sw], np.float32)); wp.capture_launch(c_.graph)
    q = ikq.numpy()
    for k in range(2): tgt[9 * k:9 * k + 7] = q[k, :7]
    sim.control.joint_target_pos.assign(tgt)
    sim.step()
    return sim.s0.body_q.numpy()
nb16 = bodies[:len(lay)]
p0 = sim.s0.body_q.numpy()[nb16, :3].copy()
def nacon(): return int(mj.nacon.numpy()[0])
if mode == "group_rt":       # toggle after capture: graph must pick it up from the array contents
    for _ in range(10): one()
    n_before = nacon()
    g = model.shape_collision_group.numpy()
    for sh in plate_sh + [x for l in bsh for x in l]: g[sh] = -2
    model.shape_collision_group.assign(g)
else:
    n_before = -1
for _ in range(30): one()
wp.synchronize(); t0 = time.time()
ncs = []
for k in range(nfr):
    bq = one()
    if k % 20 == 0: ncs.append(nacon())
wall = time.time() - t0
d = np.linalg.norm(bq[nb16, :3] - p0, axis=1) * 1e3
sh_tri = sim.tri_peak
print("RESULT mode=%s sub=%d gap=%s npairs=%d neq=%d RTF %.3f ms/frame %.1f nacon(mean of %d samples) %.0f (pre-toggle %d) drift16 max %.3f mm mean %.3f mm finite %s" % (
    mode, sub, gap_mm, npairs, model.equality_constraint_count, nfr / 60 / wall, 1e3 * wall / nfr, len(ncs), np.mean(ncs), n_before, d.max(), d.mean(), np.isfinite(bq).all()), flush=True)
