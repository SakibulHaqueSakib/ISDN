"""Stud-top rest vs seated dz (proto and clear, s=1,2) + jitter measures over the last 2 s and a 4 s run."""
import importlib.util, sys, inspect, types, json, numpy as np
spec = importlib.util.spec_from_file_location("p1", "/home/bakasakib/Documents/ISDN_Robofab/brickassembly/scripts/10_newton_scale_probe.py")
p1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1)
src = inspect.getsource(p1.run_free).replace("    return out\n", "    return out, traj, dzs\n").replace("def run_free", "def run_free2")
exec(src, p1.__dict__)
a = types.SimpleNamespace(substeps=16, collide="sub", cmax=4096)
T = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
res = {}
for clear in (False, True):
    for s in (1.0, 2.0):
        g = p1.Geo.get("2x4", s, clear=clear)
        cs = [dict(F=0.0, start_mm=2.0 * s, off_mm=0.0, state="studtop"),
              dict(F=0.0, start_mm=2.0, off_mm=0.0, state="studtop_2mm"),
              dict(F=0.5, start_mm=2.0, off_mm=0.0, t_rel=0.5, state="seated_0.5N"),
              dict(F=1.0, start_mm=2.0, off_mm=0.0, t_rel=0.5, state="seated_1N")]
        out, traj, dzs = p1.run_free2(g, cs, "fixed", T, a)
        for w, c in enumerate(cs):
            z = dzs[:, w] * 1e3
            t = np.arange(len(z)) / 60.0
            def win(t0, t1):
                m = (t >= t0) & (t < t1); zz = z[m]; tt = t[m]
                p = np.polyfit(tt, zz, 1); r = zz - np.polyval(p, tt)
                return dict(p2p=float(np.ptp(zz)), std=float(zz.std()), slope_um_s=float(p[0] * 1e3), detr_p2p=float(np.ptp(r)), detr_std=float(r.std()))
            xy = traj[:, w, :2] * 1e3
            res.setdefault("%s s=%g" % ("clear" if clear else "proto", s), {})[c["state"]] = dict(
                dz_1s=float(z[59]), dz_3s=float(z[min(179, len(z) - 1)]), dz_end=float(z[-1]),
                w1_3=win(1.0, 3.0), wlast2=win(T - 2.0, T), xy_p2p_last2=float(np.ptp(xy[t >= T - 2], 0).max()))
json.dump(res, open("rest_%g.json" % T, "w"), indent=1)
for k, v in res.items():
    print("==", k)
    for st, d in v.items():
        print("  %-12s dz@1s %+.3f dz@3s %+.3f dz_end %+.3f | 1-3s p2p %.4f std %.4f slope %+.2f um/s detr_p2p %.4f detr_std %.4f | last2 p2p %.4f detr_p2p %.4f slope %+.2f xy_p2p %.4f" % (
            st, d["dz_1s"], d["dz_3s"], d["dz_end"], d["w1_3"]["p2p"], d["w1_3"]["std"], d["w1_3"]["slope_um_s"], d["w1_3"]["detr_p2p"], d["w1_3"]["detr_std"],
            d["wlast2"]["p2p"], d["wlast2"]["detr_p2p"], d["wlast2"]["slope_um_s"], d["xy_p2p_last2"]))
