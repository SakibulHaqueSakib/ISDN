"""BrickSim's static solver, as a force query this project can call (WP1, 3a).

`static_solve` takes a lego_topology@2 JSON and returns per-connection
`clutch_utilizations` -- the load fraction on each clutch -- in about 0.4 ms,
with no simulator. That is the §3.6 "internal force distribution" query.

    python sim/joint_model/bricksim_adapter.py      # self-check on S1 and S3

Upstream static_solve applied gravity only. We patch it to take external loads
(patches/static_solve_applied_load.patch), so it can be asked the question
bracing actually has: which joint is most loaded by the INSERTION PRESS.
Without a load it is byte-identical to upstream.
"""

import json
import os
import subprocess
from pathlib import Path

DEFAULT_BINARY = (Path(__file__).resolve().parents[3]
                  / "BrickSim/native/.build/release/static_solve")

# WP2 calibration knobs: static_solve reads these from the environment
# (static_solve.cpp:85-93). Defaults are BrickSim's own, from
# native/modules/bricksim/physx/breakage.cppm:228.
THRESHOLD_ENV = {
    "contact_regularization": "BREAKAGE_CONTACT_REGULARIZATION",
    "clutch_axial_compliance": "BREAKAGE_CLUTCH_AXIAL_COMPLIANCE",
    "clutch_radial_compliance": "BREAKAGE_CLUTCH_RADIAL_COMPLIANCE",
    "clutch_tangential_compliance": "BREAKAGE_CLUTCH_TANGENTIAL_COMPLIANCE",
    "friction_coefficient": "BREAKAGE_FRICTION_COEFFICIENT",
    "preloaded_force": "BREAKAGE_PRELOADED_FORCE",
    "slack_fraction_warn": "BREAKAGE_SLACK_FRACTION_WARN",
    "slack_fraction_b_floor": "BREAKAGE_SLACK_FRACTION_B_FLOOR",
}
BRICKSIM_DEFAULTS = {
    "contact_regularization": 0.1, "clutch_axial_compliance": 1.0,
    "clutch_radial_compliance": 1.0, "clutch_tangential_compliance": 1.0,
    "friction_coefficient": 0.2, "preloaded_force": 3.5,
    "slack_fraction_warn": 0.1, "slack_fraction_b_floor": 1e-9,
}

CALIBRATION = Path(__file__).resolve().parents[2] / "joint_calibration.json"


def calibrated_thresholds(path="static_solve"):
    """WP2's fitted thresholds for a given path, or {} if uncalibrated.

    BrickSim's own defaults give about 2.1 N per stud of pull-off against the
    8-15 N the report cites, so anything force-dependent running on the
    defaults is measuring the wrong joint.

    The path argument is not decoration. static_solve and the live PhysX
    solver need DIFFERENT preloads to represent the same joint -- 18.832 N and
    120 N both give ~11.3 N/stud on a 2x4 -- so using one value in both
    misrepresents the clutch by about 6.4x. This module only ever drives
    static_solve, so it defaults accordingly; the sim reads "live_sim".
    """
    try:
        doc = json.loads(CALIBRATION.read_text())
    except (OSError, ValueError):
        return {}
    settings = doc.get("bricksim_settings", {})
    out = {k: v for k, v in settings.items() if k in THRESHOLD_ENV}
    out.update({k: v for k, v in settings.get(path, {}).items()
                if k in THRESHOLD_ENV})
    return out


def binary():
    """Path to static_solve, overridable with BRICKSIM_STATIC_SOLVE."""
    return Path(os.environ.get("BRICKSIM_STATIC_SOLVE", DEFAULT_BINARY))


def available():
    return binary().is_file()


def solve(topology, thresholds=None, loads=None, timeout=60):
    """Run static_solve on a topology dict. Returns its parsed JSON result.

    thresholds: overrides keyed as in THRESHOLD_ENV. Omitted means WP2's
        calibration; pass {} explicitly for BrickSim's raw defaults.
    loads: [(part_id, (fx, fy, fz)[, (px, py, pz)]), ...] in newtons, root
        frame. Without a point the force acts at the part's centre of mass and
        adds no couple. Needs the applied-load patch; upstream ignores it.
    """
    path = binary()
    if not path.is_file():
        raise FileNotFoundError(
            "static_solve not built at %s -- see README, `pixi run build-native` "
            "after deleting .prebuilt-native/manifest.env" % path)

    env = dict(os.environ)
    if thresholds is None:
        thresholds = calibrated_thresholds()
    for key, value in thresholds.items():
        env[THRESHOLD_ENV[key]] = repr(float(value))

    # It takes a file path, not stdin, despite what the repo's own
    # evaluate_stability.py does with "-".
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / ("bricksim_%d.json" % os.getpid())
    tmp.write_text(json.dumps(topology))

    argv = [str(path), str(tmp)]
    for load in (loads or []):
        vals = [load[0]] + list(load[1]) + (list(load[2]) if len(load) > 2 else [])
        argv += ["--load", ",".join(repr(float(v)) for v in vals)]

    try:
        proc = subprocess.run(argv, env=env, text=True,
                              capture_output=True, timeout=timeout)
    finally:
        tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError("static_solve exited %d: %s"
                           % (proc.returncode, proc.stderr.strip()[-500:]))
    start = proc.stdout.find("{")
    if start == -1:
        raise RuntimeError("static_solve produced no JSON")
    return json.loads(proc.stdout[start:])


def press_utilizations(topology, part_id, force_n, point=None, **kw):
    """Load fractions with `force_n` newtons pressed straight down on a part.

    This is the §3.6 query: the brick being placed carries the insertion press,
    and the reaction propagates through every clutch beneath it.
    """
    load = (part_id, (0.0, 0.0, -abs(force_n)))
    if point is not None:
        load = load + (point,)
    return utilizations(topology, loads=[load], **kw)


def utilizations(topology, **kw):
    """{connection_id (int): clutch load fraction}, or {} if the QP did not solve.

    static_solve emits clutch_utilizations as null when the solution carries no
    utilisation vector, i.e. the solve did not converge. That is not the same
    as "nothing is loaded", so it must not be read as zeros.
    """
    result = solve(topology, **kw)
    return {int(k): v for k, v in (result["clutch_utilizations"] or {}).items()}


def axial_forces(topology, **kw):
    """{connection_id (int): axial force on that clutch, newtons}."""
    return {int(k): v["axial_n"]
            for k, v in (solve(topology, **kw)["clutch_forces"] or {}).items()}


def release_force(topology, part_id, lo=0.0, hi=500.0, tol=0.05, **kw):
    """Press on `part_id` at which some clutch first reaches capacity (newtons).

    Bisects on max utilisation crossing 1.0. Returns None if the structure is
    already over capacity at `lo` or still under it at `hi`.
    """
    def over(f):
        util = press_utilizations(topology, part_id, f, **kw)
        if not util:
            raise RuntimeError("QP did not solve at %.3f N press" % f)
        return max(util.values()) > 1.0

    if over(lo) or not over(hi):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if over(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def weakest_connection(topology, **kw):
    """(connection_id, utilization) for the most-loaded clutch, or None if all are zero."""
    util = utilizations(topology, **kw)
    conn, value = max(util.items(), key=lambda kv: kv[1])
    return None if value <= 0.0 else (conn, value)


def joint_of(topology, connection_id):
    """(lower_brick_id, upper_brick_id) for a connection, using our _brick_ids map."""
    names = topology.get("_brick_ids", {})
    conn = next(c for c in topology["connections"] if c["id"] == connection_id)
    return (names.get(str(conn["stud_id"]), "baseplate"),
            names.get(str(conn["hole_id"]), "baseplate"))


def _self_check():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import planner as P

    if not available():
        print("static_solve not built -- skipping (%s)" % binary())
        return

    # S1 is a straight column: every clutch is in pure compression.
    s1 = solve(P.to_bricksim_topology("S1"))
    assert s1["stable"] is True, s1
    assert all(v == 0.0 for v in s1["clutch_utilizations"].values()), s1

    # S3 is a cantilever: the joint under the overhang carries the most.
    topo = P.to_bricksim_topology("S3")
    worst = weakest_connection(topo)
    assert worst is not None, "cantilever loaded nothing"
    assert joint_of(topo, worst[0]) == ("b_002", "b_003"), joint_of(topo, worst[0])

    # ... and it agrees with the lever model, which reasons about the press.
    order = P.sequence(P.STRUCTURES["S3"])
    lever = P.weakest_joint(order[-1], order[:-1])
    assert tuple(lever[0]) == joint_of(topo, worst[0]), (lever[0], worst)

    # Applied loads: with no load the patched tool must reproduce upstream,
    # and the press must raise the load monotonically.
    pid = {v: int(k) for k, v in topo["_brick_ids"].items()}["b_004"]
    assert press_utilizations(topo, pid, 0.0) == utilizations(topo), \
        "zero press changed the gravity-only answer"
    series = [max(press_utilizations(topo, pid, f).values())
              for f in (0.0, 1.0, 2.0, 4.0)]
    assert series == sorted(series), series

    # §WP3 preflight: a structure MUST fail single-arm, or ablation A6 has no
    # subject. Asserted at WP2's calibrated clutch (11.3 N/stud pull-off), and
    # across the whole 8-15 N/stud envelope so the result does not depend on
    # where in that envelope the real value sits.
    press_n = 4 * P.F_INSERT_PER_STUD
    assert calibrated_thresholds(), "run sim/joint_model/calibration.py first"

    long_topo = P.to_bricksim_topology("S3L")
    long_pid = {v: int(k) for k, v in long_topo["_brick_ids"].items()}["b_005"]
    envelope = {}
    for per_stud in (8.0, 11.3, 15.0):
        thr = {"preloaded_force": per_stud / 0.6}    # WP2 transfer function
        envelope[per_stud] = (
            max(press_utilizations(topo, pid, press_n, thresholds=thr).values()),
            max(press_utilizations(long_topo, long_pid, press_n,
                                   thresholds=thr).values()))
    for per_stud, (s3_u, s3l_u) in envelope.items():
        assert s3_u > 1.0, ("S3 survives at %.1f N/stud" % per_stud, s3_u)
        assert s3l_u > s3_u, ("S3L has less margin than S3 at %.1f N/stud" % per_stud)
    s3_util, s3l_util = envelope[11.3]

    # G1 "equal and opposite wrench": the QP balances the summed force and
    # torque on the root by construction, so the invariant is structural and
    # what matters is the constraint residual ||Ax-b||/||b||.
    for name, load in (("unloaded", None), ("pressed", [(pid, (0.0, 0.0, -35.6))])):
        residual = solve(topo, loads=load)["slack_fraction"]
        assert residual < 1e-3, (name, residual)

    # Axial force scales linearly with the applied press -- the check that makes
    # clutch_forces trustworthy as newtons rather than as arbitrary units.
    a1 = axial_forces(topo, loads=[(pid, (0.0, 0.0, -35.6))])
    a2 = axial_forces(topo, loads=[(pid, (0.0, 0.0, -71.2))])
    for conn in a1:
        if abs(a1[conn]) > 0.1:
            assert abs(a2[conn] / a1[conn] - 2.0) < 0.01, (conn, a1[conn], a2[conn])

    print("bricksim_adapter self-check OK "
          "(S1 unloaded; S3 weakest = %s at utilisation %.5f, lever model agrees)"
          % (" -> ".join(joint_of(topo, worst[0])), worst[1]))
    print("  under a %.1f N press at WP2's calibration (11.3 N/stud): S3 %.2f, "
          "S3L %.2f -- both fail single-arm, as A6 needs" % (press_n, s3_util, s3l_util))
    print("  across the 8-15 N/stud envelope: " + ", ".join(
        "%.0f N/stud S3 %.2f / S3L %.2f" % (k, v[0], v[1])
        for k, v in sorted(envelope.items())))


if __name__ == "__main__":
    _self_check()
