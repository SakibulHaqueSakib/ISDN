"""Static squeeze: free brick (gravity off) between two box pads on world prismatic joints, constant inward joint force F.
At rest the contact force = F; penetration read from the pad joint q. Variants isolate the finger-brick contact params."""
import importlib.util, sys, copy, math, numpy as np, warp as wp, newton
sys.argv = sys.argv[:1] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("p1", "/home/bakasakib/Documents/ISDN_Robofab/brickassembly/scripts/10_newton_scale_probe.py")
p1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1)
das, ex = p1.das, p1.ex
s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
FINGER_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
FORCES = [1, 2, 5, 10, 20, 50]
D = newton.ModelBuilder.ShapeConfig()   # finger shapes in the URDF: default ke 2500, kd 100, mu 1
V = {
    "das_finger":   (dict(ke=2500., kd=100., mu=1.0), 1, FINGER_SOLIMP, 1),
    "ke1e5_kd100":  (dict(ke=1e5, kd=100., mu=1.0), 1, FINGER_SOLIMP, 1),
    "ke1e5_kd200":  (dict(ke=1e5, kd=200., mu=1.0), 1, FINGER_SOLIMP, 1),
    "ke2.5e5_kd300": (dict(ke=2.5e5, kd=300., mu=1.0), 1, FINGER_SOLIMP, 1),
    "ke2.5e5_kd100": (dict(ke=2.5e5, kd=100., mu=1.0), 1, FINGER_SOLIMP, 1),
    "ke1e6_kd500":  (dict(ke=1e6, kd=500., mu=1.0), 1, FINGER_SOLIMP, 1),
}
g = p1.Geo.get("2x4", s)
half_w = g.W * g.pitch / 2
hy = 0.003
scene = newton.ModelBuilder(gravity=0.0); newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
cases = [(v, F) for v in V for F in FORCES]
for vname, F in cases:
    cfgd, prio, solimp, rho = V[vname]
    b = newton.ModelBuilder(gravity=0.0); newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    bb = b.add_body(xform=wp.transform((0, 0, 0.05), wp.quat_identity()), label="brick")
    ids = p1.add_brick(b, bb, wp.transform_identity(), g)
    if rho != 1:
        pass
    pc = newton.ModelBuilder.ShapeConfig(density=1000.0, **cfgd)
    zc = 0.05 + g.H - 2.75e-3 * s
    for sgn in (1, -1):
        pad = b.add_link(xform=wp.transform((0, sgn * (half_w + hy), zc), wp.quat_identity()), label="pad%+d" % sgn)
        sh = b.add_shape_box(pad, hx=0.005 * s, hy=hy, hz=2.5e-3 * s, cfg=pc)
        j = b.add_joint_prismatic(parent=-1, child=pad, parent_xform=wp.transform((0, sgn * (half_w + hy), zc), wp.quat_identity()),
                                  axis=(0, -sgn, 0), armature=0.15)
        b.add_articulation([j])
        pr = b.custom_attributes["mujoco:geom_priority"]; pr.values = pr.values or {}; pr.values[sh] = prio
        si = b.custom_attributes["mujoco:geom_solimp"]; si.values = si.values or {}; si.values[sh] = solimp
    scene.add_world(b)
m = scene.finalize()
nw = len(cases)
bpw = m.body_count // nw
# brick density factor: scale mass/inertia of the brick body directly
bm, bI, bim, bII = m.body_mass.numpy(), m.body_inertia.numpy(), m.body_inv_mass.numpy(), m.body_inv_inertia.numpy()
for w, (v, F) in enumerate(cases):
    r = V[v][3]
    i = w * bpw
    bm[i] *= r; bI[i] *= r; bim[i] /= r; bII[i] /= r
m.body_mass.assign(bm); m.body_inertia.assign(bI); m.body_inv_mass.assign(bim); m.body_inv_inertia.assign(bII)
sim = p1.Sim(None, nw, 16, "sub", model=m)
dpw = m.joint_dof_count // nw
jf = np.zeros(m.joint_dof_count, np.float32)
for w, (v, F) in enumerate(cases):
    jf[w * dpw + 6] = F; jf[w * dpw + 7] = F
sim.control.joint_f = wp.array(jf, dtype=float)
sim.start()
for k in range(90):
    sim.step()
q = sim.s0.joint_q.numpy(); qd = sim.s0.joint_qd.numpy()
jqs = m.joint_q_start.numpy(); qpw = m.joint_coord_count // nw
print("s=%g brick mass %.2f g (x rho)" % (s, bm[0] * 1e3))
print("%-18s %6s %9s %9s %9s %10s" % ("variant", "F[N]", "pen_L mm", "pen_R mm", "K kN/m", "|qd| max"))
for w, (v, F) in enumerate(cases):
    qL, qR = q[w * qpw + 7], q[w * qpw + 8]
    pen = 0.5 * (qL + qR)
    print("%-18s %6.1f %9.3f %9.3f %9.2f %10.2e" % (v, F, qL * 1e3, qR * 1e3, F / pen / 1e3 if pen > 0 else float('nan'),
                                                    np.abs(qd[w * dpw:(w + 1) * dpw]).max()))
