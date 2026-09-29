"""Timing variants of probe (f). args: pool(0|17|210|465) seated(0/1) sweep(0/1) sub collide nfr"""
import importlib.util, sys, time, math, types
import numpy as np, warp as wp
sys.argv_saved = sys.argv[:]
spec = importlib.util.spec_from_file_location("pr", "/home/bakasakib/Documents/ISDN_Robofab/brickassembly/scripts/10_newton_scale_probe.py")
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
das, newton, ik = pr.das, pr.newton, pr.ik
pool, seated, sweep_on, sub, collide, nfr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
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
pr.add_plate(scene, geo("2x4"), das.Z0, ni=ni + 4, nj=nj + 4, margin=2)
bodies = []
for n, b in enumerate(lay):
    xf = wp.transform(tuple(b["pos"]), das.qz(b["yaw"])) if seated else wp.transform((3.0 + 0.1 * n, 0.0, 0.0005), das.qz(0.0))
    bd = scene.add_body(xform=xf); pr.add_brick(scene, bd, wp.transform_identity(), geo(b["type"])); bodies.append(bd)
for n, ((x, y), t) in enumerate(zip(slots, tray_types)):
    bd = scene.add_body(xform=wp.transform((x, y, 0.0005), wp.quat_identity())); pr.add_brick(scene, bd, wp.transform_identity(), geo(t)); bodies.append(bd)
mate = {}
for n, b in enumerate(lay):
    for par in b["parents"]:
        p1, y1 = (np.zeros(3), 0.0) if par is None else (lay[par]["pos"], lay[par]["yaw"])
        mate[(par, n)] = wp.transform(das.rotate(das.qz(-y1), b["pos"] - p1), das.qz(b["yaw"] - y1))
nb = {0: 0, 17: 0, 210: 20, 465: 30}[pool]
n_on = 0
if pool == 17:   # only the mating welds (prototype-like pool), enabled if seated
    for (par, n), r in mate.items():
        scene.add_equality_constraint_weld(body1=-1 if par is None else bodies[par], body2=bodies[n], relpose=r, enabled=bool(seated), custom_attributes=das.WELD_ATTRS); n_on += 1
for i in range(nb):
    r = mate.get((None, i)) if seated else None
    scene.add_equality_constraint_weld(body1=-1, body2=bodies[i], relpose=r or wp.transform_identity(), enabled=r is not None, custom_attributes=das.WELD_ATTRS)
    n_on += r is not None
    for j in range(i + 1, nb):
        r = mate.get((i, j)) if seated else None
        n_on += r is not None
        scene.add_equality_constraint_weld(body1=bodies[i], body2=bodies[j], relpose=r or wp.transform_identity(), enabled=r is not None, custom_attributes=das.WELD_ATTRS)
model = pr.Sim.finalize(scene)
ke, kd = model.shape_material_ke.numpy(), model.shape_material_kd.numpy()
ke[fing], kd[fing] = pr.FINGER_CONTACT_KE, pr.finger_kd(sub if sub in (16, 8) else 4)
model.shape_material_ke.assign(ke); model.shape_material_kd.assign(kd)
sim = pr.Sim(None, 1, sub, collide, cmax=4096 * 8, model=model, tri=pr.TRI_F)
ik_pos = wp.array([das.PARK] * 2, dtype=wp.vec3)
ik_solver = ik.IKSolver(model=model_ik, n_problems=2, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
    objectives=[ik.IKObjectivePosition(link_index=das.EE, link_offset=wp.vec3(0, 0, 0), target_positions=ik_pos),
                ik.IKObjectiveRotation(link_index=das.EE, link_offset_rotation=wp.quat_identity(), target_rotations=wp.array([das.Q_DOWN] * 2, dtype=wp.vec4)),
                ik.IKObjectiveJointLimit(joint_limit_lower=model_ik.joint_limit_lower, joint_limit_upper=model_ik.joint_limit_upper)])
ikq = wp.array(np.tile(park, (2, 1)), dtype=wp.float32)
with wp.ScopedCapture() as c_:
    ik_solver.step(ikq, ikq, iterations=24)
sim.start()
tgt = sim.control.joint_target_pos.numpy(); fr = [0]
def one(host=True):
    t = fr[0] / 60; fr[0] += 1
    if host:
        sw = das.PARK + ([0.0, 0.06 * math.sin(2 * math.pi * t / 4.0), 0.03 * math.sin(2 * math.pi * t / 2.0)] if sweep_on else [0, 0, 0])
        ik_pos.assign(np.array([das.PARK, sw], np.float32)); wp.capture_launch(c_.graph)
        q = ikq.numpy()
        for k in range(2): tgt[9 * k:9 * k + 7] = q[k, :7]
        sim.control.joint_target_pos.assign(tgt)
    sim.step(); sim.check_tri()
    return sim.s0.body_q.numpy()
for _ in range(30): one()
wp.synchronize(); t0 = time.time()
for _ in range(nfr): bq = one()
wall = time.time() - t0
# pure graph only
wp.synchronize(); t1 = time.time()
for _ in range(nfr // 2): sim.step()
wp.synchronize(); w2 = (time.time() - t1) / (nfr // 2)
mj = sim.solver.mjw_data
nefc = int(mj.nefc.numpy().sum()) if hasattr(mj, "nefc") else -1
nacon = int(mj.nacon.numpy()[0])
print("RESULT pool=%d seated=%d sweep=%d sub=%d collide=%s neq=%d on=%d  RTF %.3f  ms/frame %.1f  graph-only ms/frame %.1f (RTF %.3f)  tri_peak %d nacon %d nefc %d finite %s" % (
    pool, seated, sweep_on, sub, collide, model.equality_constraint_count, n_on, nfr / 60 / wall, 1e3 * wall / nfr, 1e3 * w2, 1 / 60 / w2, sim.tri_peak, nacon, nefc, np.isfinite(bq).all()), flush=True)
wp.synchronize(); t2 = time.time()
for _ in range(20): sim.pipe.collide(sim.s0, sim.contacts)
wp.synchronize(); print("RESULT collide-only ms %.1f" % ((time.time() - t2) / 20 * 1e3))
