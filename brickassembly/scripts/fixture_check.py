"""G-R0 criterion 0, fixture wrench checks (plan_v4_rl_brace S0.2): the design-press fixture's effect on static
weld-held scenes against statics.

Each scene is a nominal start (steps < n welded at their targets, both arms parked, --fixture-hold: the fixture at full
amplitude from t = 0.5 s, JointBreaker off so a weld cannot let go). The weld wrench is JointBreaker.sample(), the
median over the last 60 frames of a 4 s hold; fixture on minus fixture off (the off run is the same scene without --press-scale) removes
gravity. Expected: the loads transferred down the determinate chain, F_ext on the support patch, moved to each weld's
patch centre c: dFp = R (-sum F_j), dMp = R (-sum (p_j - c) x F_j), in the weld's patch frame (JointBreaker.sample's: R and c
from the lower brick's measured pose -- a loaded column leans 1-2 degrees, which turns a vertical force into patch shear --
and p_j the load point at the same pose). Pass: within 5% with floors 25 mN and 0.25 mN.m. The off check: before 0.5 s
the fixture's wrench array is exactly 0 and the weld wrenches equal the no-fixture run's to float noise (1e-6; two identical
no-fixture runs differ by ~1e-8, the GPU solver is not bitwise repeatable).

    bash scripts/run.sh scripts/fixture_check.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import newton

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as D  # noqa: E402

PIER = [("p%d" % k, "2x2", 0, 0, k, 0) for k in range(5)]                        # p4 is the pressed brick: loads p3
CANT = [("c0", "2x4", 0, 0, 0, 0), ("c1", "2x4", 0, 2, 1, 0), ("c2", "2x2", 0, 4, 2, 0)]   # c1 hangs 2 studs over c0; c2 loads
                                                                                  # c1 at a patch centre 1 stud off its COM
FRAMES, TAIL, ON_FRAME = 240, 60, 29  # median over the last TAIL frames; frame 29 ends before the fixture starts (t = 0.5 s)
# Why a median: once settled (by t = 0.6 s) the poses are static and the weld wrench equals statics to 1e-3, but on the loaded
# cantilever the solver's end-of-frame efc.force of the c1/c0 weld intermittently jitters (~1 frame in 4: shear 0.04-0.70 N
# around 0.564, Fz +-0.5 N, Mx +-1%) -- 15-iteration solver noise, not motion; a 20-frame mean caught it (error 0.04-0.05 N
# against a 0.028 N tolerance). The off run is steady.
FLOOR_F, FLOOR_M, REL, NOISE = 0.025, 0.25e-3, 0.05, 1e-6


def run(name, bricks, lam=None, lat=None):
    """-> (samples at the last pre-fixture frame, the median over the last TAIL frames, the Example, the fixture wrench array
    at the pre-fixture frame, the solver's (xpos, xmat) of each of those frames, those frames' samples)."""
    args = argparse.Namespace(front=None, side=None, top=None, width=None, depth=None, name=name, shape=None, structure=None,
                              bricks=bricks, strategy="none", brace_model="grasp_lp", plan=None, spec="fixture_check",
                              start_step=len(bricks) - 1, only_step=False, brace_json=None, press_scale=lam,
                              press_lateral=lat, fixture_hold=True, exe="E0", viewer="null", legacy_brace=False)
    ex = D.Example(newton.viewer.ViewerNull(num_frames=10 ** 9), args)
    ex.breaker.enabled = False
    pre, tail, w0, poses = None, [], None, []
    for f in range(FRAMES):
        ex.step()
        if f == ON_FRAME:
            pre, w0 = ex.breaker.sample(), None if ex.fx is None else ex.fx_w.numpy()
        if f >= FRAMES - TAIL:
            d = ex.solver.mjw_data
            tail.append(ex.breaker.sample())
            poses.append((d.xpos.numpy()[0], d.xmat.numpy()[0]))
    mean = {k: {q: np.median([t[k][q] for t in tail], 0) for q in ("F", "M", "Fp", "Mp")} for k in tail[0]}
    return pre, mean, ex, w0, poses, tail


def statics(ex, weld, pose):
    """Expected (dFp, dMp) of `weld` = (upper, support) in its patch frame: minus the fixture loads on every brick standing
    on `upper`, moved to the patch centre, at the poses of `pose` = (xpos, xmat) as JointBreaker.sample reads them."""
    xpos, xmat = pose
    lower = {b: [w[1] for w in ws] for b, ws in ex.welds.items()}
    on = lambda b, up: b == up or any(l is not None and on(l, up) for l in lower[b])
    sid, conn = weld[1], ex.breaker.conn[weld]
    if sid is None:
        c, R = conn.centre, np.eye(3)
    else:
        lo, (p_nom, yaw, _) = ex.breaker.mj[sid], ex.target[sid]
        R_nom = D.rotmat(D.qz(yaw))
        c, R = xpos[lo] + xmat[lo] @ R_nom.T @ (conn.centre - p_nom), R_nom @ xmat[lo].T
    mine = [j for j, i in enumerate(ex.fx["ids"]) if on(i, weld[0])]
    F = ex.fx["F"][mine]
    p = np.array([xpos[ex.breaker.mj[ex.fx["ids"][j]]] + xmat[ex.breaker.mj[ex.fx["ids"][j]]] @ ex.fx["p_body"][j] for j in mine])
    return R @ -F.sum(0), R @ -np.cross(p - c, F).sum(0)


def check(title, name, bricks, off, lam, lat=None):
    pre0, mean0 = off[:2]
    pre1, mean1, ex, w0, poses, tail = run(name, bricks, lam, lat)
    print("\n%s: lambda %s, lateral %s, loads %s" % (title, lam, lat, [(i, np.round(f, 3).tolist(), np.round(p, 4).tolist())
                                                                    for i, f, p in zip(ex.fx["ids"], ex.fx["F"], ex.fx["p"])]))
    print("  %-12s %-6s %12s %12s %10s %10s  %s" % ("weld", "qty", "expected", "measured", "error", "tolerance", ""))
    ok = True
    for w in sorted(mean1, key=str):
        dF, dM = mean1[w]["Fp"] - mean0[w]["Fp"], mean1[w]["Mp"] - mean0[w]["Mp"]
        eF, eM = (np.median(x, 0) for x in zip(*(statics(ex, w, p) for p in poses)))
        rows = [("Fz N", eF[2], dF[2], FLOOR_F), ("Mx mN.m", eM[0] * 1e3, dM[0] * 1e3, FLOOR_M * 1e3),
                ("My mN.m", eM[1] * 1e3, dM[1] * 1e3, FLOOR_M * 1e3),
                ("shear N", float(np.hypot(*eF[:2])), float(np.hypot(*dF[:2])), FLOOR_F)]
        for q, e, m, floor in rows:
            tol = max(REL * abs(e), floor)
            good = abs(m - e) <= tol
            ok &= good
            print("  %-12s %-7s %12.4f %12.4f %10.4f %10.4f  %s" % ("%s/%s" % w, q, e, m, m - e, tol, "ok" if good else "FAIL"))
    off_diff = max(float(np.abs(pre1[w][q] - pre0[w][q]).max()) for w in pre0 for q in ("F", "M", "Fp", "Mp"))
    off_ok = off_diff <= NOISE and not w0.any()
    print("  fixture off (frame %d, before 0.5 s): wrench array max |w| = %.1f, max |fixture run - no-fixture run| over every "
          "weld wrench component = %.1e  %s" % (ON_FRAME, np.abs(w0).max(), off_diff, "ok" if off_ok else "FAIL"))
    print("  peak applied |F_i| %.3f N, total %.3f N; u of the welds under load: %s" % (
        ex.fx_peak[0], ex.fx_peak[1], {"%s/%s" % w: round(float(ex.breaker.conn[w].patch.utilization(
            mean1[w]["Fp"][2], mean1[w]["Mp"][0], mean1[w]["Mp"][1])), 3) for w in mean1}))
    noisy = sum(any(abs(t[w]["Fp"][2] - mean1[w]["Fp"][2]) > FLOOR_F for w in mean1) for t in tail)
    print("  (%d of the last %d frames have a weld Fz more than 25 mN off its median: solver noise, see above)" % (noisy, TAIL))
    return ok and off_ok


if __name__ == "__main__":
    res = {}
    pier_off, cant_off = run("fx_pier", PIER), run("fx_cant", CANT)
    res["(i) axial, column"] = check("(i) axial: 4-high 2x2 column, fixture on its top brick", "fx_pier", PIER, pier_off, 1.0)
    res["(ii) eccentric, cantilever"] = check("(ii) eccentric: 2x4 cantilever, loaded 1 stud off its COM", "fx_cant", CANT, cant_off, 1.0)
    res["(iii) lateral 3 N, column"] = check("(iii) lateral 3 N (1.8, 2.4) on the column", "fx_pier", PIER, pier_off, 1.0, (1.8, 2.4))
    print("\n" + "\n".join("%-30s %s" % (k, "PASS" if v else "FAIL") for k, v in res.items()))
    sys.exit(0 if all(res.values()) else 1)
