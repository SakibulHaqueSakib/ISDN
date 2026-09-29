"""Run the probe's arm-held (e) cases at s with the finger shapes' contact ke/kd patched (monkeypatch, no repo edit)."""
import importlib.util, sys, types, json, math, numpy as np
spec = importlib.util.spec_from_file_location("p1", "/home/bakasakib/Documents/ISDN_Robofab/brickassembly/scripts/10_newton_scale_probe.py")
p1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1)
das = p1.das
s, ke_c, kd_c = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["press", "carry"]
orig = p1.arm_builder
def arm_builder(ke, cache):
    if ke not in cache:
        arm = orig(ke, {})
        if ke_c > 0:
            for i, bd in enumerate(arm.shape_body):
                if bd in (12, 13):
                    arm.shape_material_ke[i] = ke_c; arm.shape_material_kd[i] = kd_c
        cache[ke] = arm
    return cache[ke]
p1.arm_builder = arm_builder
a = types.SimpleNamespace(substeps=16, collide="sub", cmax=4096, zbase=0.05, floor=0.06, tmax=20.0, trace=0)
rng = np.random.default_rng(0)
g = p1.Geo.get("2x4", s)
cs = [c for c in p1.cases_e(types.SimpleNamespace(trials=1), s, 20.0, rng, 5.0) if c["mode"] in modes]
cs += [p1._case("press", 10.0, 1.0, s, 20.0, rng, above_mm=30.0, level=1.0)] if "press" in modes else []
rows = p1.run_arm_batch(g, "welded", cs, a)
for r in rows:
    print("s=%g fke=%g %-5s F=%4.1f gm=%.0f gripN=%5.1f real=%6.2f sq=%.3f pen=%.3f slip=%.3f slip_z=%+.3f fired=%s dz_end=%s pkF=%.2f" % (
        s, ke_c, r["mode"], r["F_cap"], r["grip_mult"], r["grip_N"], r["grip_real_N"] or float('nan'), r["squeeze_mm"] or float('nan'),
        r["finger_pen_mm"], r["slip_mm"], r["slip_z_mm"], r["fired"], r["dz_end_mm"] and round(r["dz_end_mm"], 3), r["peak_fz_N"]))
