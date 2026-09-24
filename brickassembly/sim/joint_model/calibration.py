"""WP2 -- joint calibration (master_report §WP2).

BrickSim ships uncalibrated: its default PreloadedForce of 3.5 N gives a
pull-off far below the 8-15 N per stud the report's §2.3 cites. This module
measures BrickSim's preload -> capacity transfer function, inverts it onto the
literature target, and emits joint_calibration.json per §2.6.

    python sim/joint_model/calibration.py            # measure, fit, emit
    python sim/joint_model/calibration.py --check    # self-check only

WHAT THIS IS NOT. §WP2 steps 1-2 ask for the insertion and extraction peaks to
be *derived* from [MechanicsSnapFit]'s closed-form rigid-cylinder /
thin-elastic-shell relations, and for the LEGO geometry to be placed in one of
its four mechanical phases. That is not done here: the paper's relations are
not in hand, and inventing them would violate guardrail §5.4.5 (every number
traces to a source). This takes the fallback §WP2 explicitly allows -- "use
literature values directly, record 'literature-calibrated', and note the
calibration gap as a limitation". The phase regime is therefore recorded as
null, not guessed.

What IS measured here is BrickSim's own behaviour: how its knobs map to a
pull-off force, which is what makes the literature target reachable at all.
"""

import argparse
import json
from pathlib import Path

import bricksim_adapter as bs

# --- literature targets, all traceable to master_report -----------------------
# §2.3 table: pull-off force per stud, citing the [SnapFitFEA1]/[SnapFitFEA2]
# cantilever snap-fit envelope (3-35 N insertion, 1.7-41 N retention).
PULL_OFF_ENVELOPE_N = (8.0, 15.0)
# §2.6's worked joint_calibration.json example. These are the report's numbers,
# adopted rather than derived -- see the module docstring.
TARGET_F_BREAK_PER_STUD_N = 11.3
TARGET_INSERTION_PEAK_N = 8.9
TARGET_EXTRACTION_PEAK_N = 11.3
TARGET_LATERAL_STIFFNESS_N_PER_MM = 42.1
TARGET_GATE_LATERAL_MM = 1.18
TARGET_RAMP_TIME_S = 0.047

# Fit articles: concentric full laps only, because that is the geometry a
# pull-off test measures. (nx, ny, overlap_j); mated studs = nx * (ny - overlap).
ARTICLES_FIT = [(2, 2, 0), (2, 4, 0), (2, 6, 0), (1, 2, 0), (1, 4, 0)]
# Checked but NOT fitted: a half lap is eccentric, so pulling the top brick
# applies a prying moment and the joint reaches capacity at a much lower
# average force. Including it in the fit conflates two load modes.
ARTICLES_CHECK = [(2, 4, 2), (2, 6, 3)]
ARTICLES = ARTICLES_FIT + ARTICLES_CHECK
PRELOADS_N = [1.0, 2.0, 3.5, 6.0, 10.0, 20.0, 35.0, 60.0, 100.0]

OUT = Path(__file__).resolve().parents[2] / "joint_calibration.json"


def pair_topology(nx, ny, overlap_j=0):
    """Baseplate + a brick + a second brick lapped onto it by overlap_j studs."""
    return {
        "schema": "bricksim/lego_topology@2",
        "parts": [
            {"id": 0, "type": "brick",
             "payload": {"L": 20, "W": 20, "H": 1, "color": [155, 161, 157]}},
            {"id": 1, "type": "brick",
             "payload": {"L": nx, "W": ny, "H": 3, "color": [201, 26, 9]}},
            {"id": 2, "type": "brick",
             "payload": {"L": nx, "W": ny, "H": 3, "color": [0, 85, 191]}},
        ],
        "connections": [
            {"id": 0, "stud_id": 0, "stud_iface": 1, "hole_id": 1,
             "hole_iface": 0, "offset": [8, 8], "yaw": 0},
            {"id": 1, "stud_id": 1, "stud_iface": 1, "hole_id": 2,
             "hole_iface": 0, "offset": [0, overlap_j], "yaw": 0},
        ],
        "pose_hints": [{"part": 0, "pos": [0.0, 0.0, 0.0],
                        "rot": [1.0, 0.0, 0.0, 0.0]}],
    }


def pull_off_force(topology, connection=1, part=2, tol=0.002, **kw):
    """Straight upward force (N) at which `connection` first reaches capacity.

    A pure axial pull, which is what a pull-off test measures and what the
    literature envelope quotes. NOT the same as loading the joint with a
    prying moment: on a cantilever the far studs reach capacity at a much
    lower average force, so the two numbers are not interchangeable.
    """
    def over(f):
        util = bs.utilizations(topology, loads=[(part, (0.0, 0.0, +f))], **kw)
        # The QP stops converging well past capacity; that is "over", not
        # "unknown", but it must never be read as "under".
        return True if not util else util[connection] > 1.0

    lo, hi = 0.001, 1.0
    for _ in range(24):                       # bracket upward
        if over(hi):
            break
        lo, hi = hi, hi * 2.0
    else:
        return None
    if over(lo):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if over(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def characterise(preloads=PRELOADS_N, articles=ARTICLES_FIT):
    """Measure pull-off per stud for every (preload, article) combination."""
    rows = []
    for nx, ny, overlap in articles:
        topo = pair_topology(nx, ny, overlap)
        studs = nx * (ny - overlap)
        for preload in preloads:
            force = pull_off_force(topo, thresholds={"preloaded_force": preload})
            if force is None:
                continue
            rows.append({"brick": "%dx%d" % (nx, ny), "overlap_j": overlap,
                         "studs": studs, "preload_n": preload,
                         "pull_off_n": round(force, 4),
                         "per_stud_n": round(force / studs, 4)})
    return rows


def fit_per_stud(rows):
    """Least-squares slope of per-stud capacity against preload, through zero.

    Returns (slope, rmse, n). Fitting through the origin is a choice, not a
    measurement: it is checked against the free-intercept fit below and only
    used because the intercept comes out negligible.
    """
    xs = [r["preload_n"] for r in rows]
    ys = [r["per_stud_n"] for r in rows]
    slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    resid = [y - slope * x for x, y in zip(xs, ys)]
    rmse = (sum(r * r for r in resid) / len(resid)) ** 0.5
    return slope, rmse, len(rows)


def fit_per_stud_affine(rows):
    """Free-intercept fit, to check that forcing through zero is honest."""
    xs = [r["preload_n"] for r in rows]
    ys = [r["per_stud_n"] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rmse = (sum(r * r for r in resid) / len(resid)) ** 0.5
    return slope, intercept, rmse


def calibrate(target_per_stud=TARGET_F_BREAK_PER_STUD_N):
    """Measure, fit, invert onto the target, and verify the result."""
    rows = characterise()

    # 2xN and 1xN bricks do not share a per-stud capacity, so they are fitted
    # separately rather than averaged into a number that describes neither.
    families = {}
    for family in ("2x", "1x"):
        subset = [r for r in rows if r["brick"].startswith(family)]
        if not subset:
            continue
        slope, rmse, n = fit_per_stud(subset)
        aslope, intercept, armse = fit_per_stud_affine(subset)
        families[family + "N"] = {
            "n_points": n,
            "slope_n_per_stud_per_preload_n": round(slope, 6),
            "rmse_n_per_stud": round(rmse, 6),
            "free_intercept_check": {"slope": round(aslope, 6),
                                     "intercept_n": round(intercept, 6),
                                     "rmse": round(armse, 6)},
        }

    slope_2x = families["2xN"]["slope_n_per_stud_per_preload_n"]
    preload = target_per_stud / slope_2x

    # Verify by measuring at the fitted preload rather than trusting the fit.
    verify = []
    for nx, ny, overlap in ARTICLES_FIT + ARTICLES_CHECK:
        topo = pair_topology(nx, ny, overlap)
        studs = nx * (ny - overlap)
        force = pull_off_force(topo, thresholds={"preloaded_force": preload})
        verify.append({"brick": "%dx%d" % (nx, ny), "overlap_j": overlap,
                       "studs": studs,
                       "per_stud_n": round(force / studs, 4) if force else None,
                       "role": "fit" if (nx, ny, overlap) in ARTICLES_FIT
                               else "check_eccentric"})
    fitted_bricks = {"%dx%d" % (nx, ny) for nx, ny, ov in ARTICLES_FIT
                     if nx == 2 and ov == 0}
    achieved = [v["per_stud_n"] for v in verify
                if v["per_stud_n"] is not None and v["overlap_j"] == 0
                and v["brick"] in fitted_bricks]
    residual = [a - target_per_stud for a in achieved]
    residual_rmse = (sum(r * r for r in residual) / len(residual)) ** 0.5

    return {"rows": rows, "families": families, "preload_n": preload,
            "verify": verify, "residual_rmse_n": residual_rmse}


def emit(path=OUT):
    """Write joint_calibration.json per §2.6."""
    fit = calibrate()
    lo, hi = PULL_OFF_ENVELOPE_N
    in_envelope = lo <= TARGET_F_BREAK_PER_STUD_N <= hi
    asymmetry = TARGET_EXTRACTION_PEAK_N / TARGET_INSERTION_PEAK_N

    doc = {
        "reference": "literature_values_master_report_2.3_and_2.6",
        "fitted": {
            "f_break_per_stud_N": TARGET_F_BREAK_PER_STUD_N,
            "insertion_peak_N": TARGET_INSERTION_PEAK_N,
            "extraction_peak_N": TARGET_EXTRACTION_PEAK_N,
            "asymmetry_ratio": round(asymmetry, 3),
            "lateral_stiffness_N_per_mm": TARGET_LATERAL_STIFFNESS_N_PER_MM,
            "gate_lateral_mm": TARGET_GATE_LATERAL_MM,
            "ramp_time_s": TARGET_RAMP_TIME_S,
        },
        "bricksim_settings": {
            # what to actually set on BreakageThresholds / AssemblyThresholds
            "preloaded_force": round(fit["preload_n"], 3),
            "required_force": TARGET_INSERTION_PEAK_N,
            "position_tolerance": TARGET_GATE_LATERAL_MM / 1000.0,
            "note": "preloaded_force is fitted; the rest are the report's "
                    "§2.3.3 gate values, tighter than BrickSim's defaults",
        },
        "measurement": {
            "method": "pure axial pull-off, bisected on clutch utilisation "
                      "crossing 1.0, via the patched static_solve",
            "transfer_function": fit["families"],
            "verification_at_fitted_preload": fit["verify"],
        },
        "residual_rmse_N": round(fit["residual_rmse_n"], 4),
        "phase_regime": None,
        "deformable_check_run": False,
        "limitations": [
            "literature-calibrated, not derived: the [MechanicsSnapFit] "
            "closed-form relations were not applied and no mechanical phase "
            "was identified (§WP2 steps 1-2 not done, fallback taken)",
            "asymmetry is IMPOSED by setting required_force (insertion) below "
            "the clutch capacity (extraction); BrickSim's static model does "
            "not produce insertion/extraction asymmetry emergently",
            "1xN bricks do not share the 2xN per-stud capacity and are fitted "
            "separately; the plan library must not assume one number",
            "calibrated against the static solver only; the in-sim PhysX path "
            "is assumed to share BreakageThresholds but is unverified",
        ],
        "envelope_check": {
            "envelope_N_per_stud": list(PULL_OFF_ENVELOPE_N),
            "f_break_in_envelope": in_envelope,
            "source": "master_report §2.3, citing [SnapFitFEA1]/[SnapFitFEA2]",
        },
    }
    path.write_text(json.dumps(doc, indent=1))
    return doc


# --- WP2b: adopt the DYNAMIC calibration -------------------------------------
DYNAMIC_JSONL = Path(__file__).resolve().parents[2] / "scripts" / "02_dynamic_calibration.jsonl"


def load_dynamic(path=DYNAMIC_JSONL, brick="2x4", ramp=0.1):
    """Clean dynamic sweep rows for one article, sorted by preload."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("brick") == brick and r.get("release_n")
                and abs(r.get("ramp_n_per_step", 0) - ramp) < 1e-9):
            rows.append(r)
    rows.sort(key=lambda r: r["preload_n"])
    return rows


def fit_dynamic(rows):
    """Power-law fit per_stud = a * preload**b, in log space.

    The dynamic relation is visibly sublinear -- unlike the static one, which is
    linear through the origin -- so a single multiplier would misdescribe it.
    """
    import math
    xs = [math.log(r["preload_n"]) for r in rows]
    ys = [math.log(r["per_stud_n"]) for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - b * mx)
    pred = [a * math.exp(x) ** b for x in xs]
    resid = [math.exp(y) - p for y, p in zip(ys, pred)]
    rmse = (sum(r * r for r in resid) / n) ** 0.5
    return {"a": round(a, 6), "b": round(b, 6),
            "rmse_n_per_stud": round(rmse, 4), "n_points": n,
            "form": "per_stud_n = a * preload_n ** b"}


def invert_dynamic(rows, target=TARGET_F_BREAK_PER_STUD_N):
    """Preload giving `target` N/stud, by monotone interpolation between the
    two bracketing measurements. No functional form is assumed; the power-law
    fit is reported alongside only as a cross-check."""
    for lo, hi in zip(rows, rows[1:]):
        if lo["per_stud_n"] <= target <= hi["per_stud_n"]:
            span = hi["per_stud_n"] - lo["per_stud_n"]
            if span <= 0:
                continue
            frac = (target - lo["per_stud_n"]) / span
            return (lo["preload_n"] + frac * (hi["preload_n"] - lo["preload_n"]),
                    (lo["preload_n"], hi["preload_n"]))
    return None, None


def adopt_dynamic(path=OUT, target=TARGET_F_BREAK_PER_STUD_N):
    """Fit the clean dynamic sweep and make it the calibration of record."""
    rows = load_dynamic()
    if len(rows) < 3:
        raise RuntimeError("need at least 3 clean dynamic points, have %d" % len(rows))
    fit = fit_dynamic(rows)
    preload, bracket = invert_dynamic(rows, target)
    unreachable = None
    if preload is None:
        # The dynamic response saturates, so the target can sit above every
        # measurement. Extrapolating a saturating curve would invent capacity
        # the model does not have; adopt the best measured point and say so.
        best = max(rows, key=lambda r: r["per_stud_n"])
        preload, bracket = best["preload_n"], None
        unreachable = {"target_n_per_stud": target,
                       "best_measured_n_per_stud": best["per_stud_n"],
                       "at_preload_n": best["preload_n"],
                       "shortfall_pct": round(100.0 * (target - best["per_stud_n"])
                                              / target, 2)}
    powerlaw_preload = (target / fit["a"]) ** (1.0 / fit["b"])

    doc = json.loads(path.read_text())
    doc["dynamic"] = {
        "status": "ADOPTED -- calibration of record",
        "why": "WP5 trains in the live PhysX path. The static fit predicted "
               "90.4 N release on a 2x4 where the dynamic path gives ~36.8 N, "
               "so the static preload would hand the policy a joint about "
               "2.4x too weak.",
        "method": "one Kit process per measurement (scripts/02_dynamic_sweep.sh); "
                  "pure axial ramp at 0.1 N per physics step, release detected "
                  "from get_disassembled_connections filtered to the article",
        "measurements": rows,
        "power_law_fit": fit,
        "adopted_preload_n": round(preload, 3),
        "interpolated_between": bracket,
        "power_law_cross_check_preload_n": round(powerlaw_preload, 3),
        "target_per_stud_n": target,
        "target_unreachable": unreachable,
    }
    doc["bricksim_settings"]["preloaded_force"] = round(preload, 3)
    doc["bricksim_settings"]["note"] = (
        "preloaded_force is the DYNAMIC calibration (WP2b); the static fit of "
        "18.832 N is kept in measurement.transfer_function for reference but "
        "is not what the sim is run with")
    path.write_text(json.dumps(doc, indent=1))
    return doc["dynamic"]


def _self_check():
    # capacity must scale with stud count within a family
    base = pull_off_force(pair_topology(2, 2))
    wide = pull_off_force(pair_topology(2, 4))
    assert base and wide, (base, wide)
    assert abs((wide / 8) / (base / 4) - 1.0) < 0.05, (base, wide)

    # ... and must rise with preload
    low = pull_off_force(pair_topology(2, 2), thresholds={"preloaded_force": 3.5})
    high = pull_off_force(pair_topology(2, 2), thresholds={"preloaded_force": 35.0})
    assert high > low * 5, (low, high)
    print("calibration self-check OK (2x2 %.2f N, 2x4 %.2f N, preload 3.5->35 "
          "raises 2x2 pull-off %.2f -> %.2f N)" % (base, wide, low, high))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--adopt-dynamic", action="store_true",
                    help="fit scripts/02_dynamic_calibration.jsonl and make it "
                         "the calibration of record")
    args = ap.parse_args()
    if args.check:
        _self_check()
        raise SystemExit
    if args.adopt_dynamic:
        dyn = adopt_dynamic()
        print("-- WP2b dynamic calibration adopted --")
        for r in dyn["measurements"]:
            print("  preload %6.2f -> %7.3f N release (%6.3f N/stud)"
                  % (r["preload_n"], r["release_n"], r["per_stud_n"]))
        print("  power law: per_stud = %.4f * preload^%.4f (rmse %.4f, n=%d)"
              % (dyn["power_law_fit"]["a"], dyn["power_law_fit"]["b"],
                 dyn["power_law_fit"]["rmse_n_per_stud"],
                 dyn["power_law_fit"]["n_points"]))
        print("  adopted preloaded_force = %.3f N for %.1f N/stud "
              "(interpolated between %s; power law says %.3f)"
              % (dyn["adopted_preload_n"], dyn["target_per_stud_n"],
                 dyn["interpolated_between"],
                 dyn["power_law_cross_check_preload_n"]))
        raise SystemExit

    _self_check()
    doc = emit()
    print("\n-- WP2 calibration --")
    for family, f in doc["measurement"]["transfer_function"].items():
        print("  %s: %.4f N/stud per N preload, rmse %.4f (n=%d), "
              "free intercept %.4f N"
              % (family, f["slope_n_per_stud_per_preload_n"],
                 f["rmse_n_per_stud"], f["n_points"],
                 f["free_intercept_check"]["intercept_n"]))
    print("  fitted preloaded_force: %.3f N -> target %.1f N/stud"
          % (doc["bricksim_settings"]["preloaded_force"],
             doc["fitted"]["f_break_per_stud_N"]))
    print("  verification at that preload:")
    for v in doc["measurement"]["verification_at_fitted_preload"]:
        print("    %-4s overlap %d (%2d studs) -> %s N/stud"
              % (v["brick"], v["overlap_j"], v["studs"], v["per_stud_n"]))
    print("  residual RMSE %.4f N, f_break in envelope: %s"
          % (doc["residual_rmse_N"], doc["envelope_check"]["f_break_in_envelope"]))
    print("  -> %s" % OUT)
