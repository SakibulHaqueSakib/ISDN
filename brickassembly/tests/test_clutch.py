"""Gate G1 for the MuJoCo twin's joint model -- master_report §WP1 acceptance.

    ../mjenv/bin/python -m pytest tests/test_clutch.py -q          # ~3 min
    ... -m "not slow"                                               # ~40 s

One test per §WP1 criterion (docstrings quote it), plus the checks that keep
the harness honest: the analytic wrench read-back is verified against
MuJoCo's own J^T f, and every dynamic test first shows the sim is live.
Written against the report's acceptance text, not against the
implementation's behaviour -- a failure here is a finding, not a threshold
to move (guardrail §5.4.3).
"""

import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import planner as P  # noqa: E402
from sim.joint_model import clutch as CL  # noqa: E402
from sim.joint_model.capacity import F_BREAK_PER_STUD, F_INSERT_PER_STUD  # noqa: E402
from sim.mj import scene as S  # noqa: E402

RESULTS = ROOT / "tests" / "test_clutch.json"
_results = {}

# Structures used only here: a 2x2 on a 2x4 base, so the pull-off joint (4
# studs) is weaker than the joint under it (8 studs) and fails first.
P.STRUCTURES.setdefault("T_PULL", [("b_000", "2x4", 0, 0, 0, 0), ("b_001", "2x2", 0, 1, 1, 0)])


def record(name, **kv):
    _results[name] = kv
    try:
        old = json.loads(RESULTS.read_text())
    except (OSError, ValueError):
        old = {}
    old.update(_results)
    RESULTS.write_text(json.dumps(old, indent=1, default=float))


def make(structure, preplaced=(), spawn=None, with_arms=False):
    S.centre_plan_origin(P.STRUCTURES[structure])
    plan = P.build_plan(structure, "none")
    cell = S.build_cell(plan, preplaced=set(preplaced), spawn=spawn, with_arms=with_arms)
    cm = CL.ClutchModel(cell)
    for b in preplaced:
        cm.mate_now(b)
    return plan, cell, cm


def step(cell, cm, n):
    for _ in range(n):
        mujoco.mj_step(cell.model, cell.data)
        cm.update(cell.data)


def free_dofs(m, body):
    return m.jnt_dofadr[m.body_jntadr[body]]


def jtf_on(m, d, rows, body):
    """Generalised force of efc rows on a free body: (F world, tau local)."""
    f = np.zeros(d.nefc)
    f[rows] = d.efc_force[rows]
    q = np.zeros(m.nv)
    mujoco.mj_mulJacTVec(m, d, q, f)
    a = free_dofs(m, body)
    return q[a:a + 3].copy(), q[a + 3:a + 6].copy()


def weld_rows(d, eq_id):
    n = d.nefc
    return np.nonzero((d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                      & (d.efc_id[:n] == eq_id))[0]


def tilt_deg(q):
    return 2 * math.degrees(math.acos(min(1.0, abs(q[0]))))


# --------------------------------------------------------------------------------
def test_liveness_control():
    """Control: an unconnected brick falls -- proves the harness sees a live sim
    (the prior harness passed on stale USD poses; ledger g1_harness_false_pass)."""
    plan, cell, cm = make("S1", spawn={"b_000": (0.0, 0.0, 0.10)})
    b = cell.model.body("b_000").id
    z0 = cell.data.xpos[b][2]
    step(cell, cm, 200)
    fell = (z0 - cell.data.xpos[b][2]) * 1000
    record("liveness_control", fell_mm=fell)
    assert fell > 50


def test_wrench_readback_matches_JTf():
    """Harness check: clutch.py's analytic weld read-back equals MuJoCo's J^T f."""
    plan, cell, cm = make("S3", preplaced=[f"b_00{i}" for i in range(5)])
    m, d = cell.model, cell.data
    tip = m.body("b_004").id
    d.xfrc_applied[tip] = [1.5, -0.8, -3.0, 0.004, 0.0, 0.0]
    step(cell, cm, 300)
    worst = 0.0
    for st in cm.conn_states:
        U = m.body(st.conn.upper).id
        F, tau = jtf_on(m, d, weld_rows(d, st.conn.eq_id), U)
        R = d.xmat[U].reshape(3, 3)
        Fl = R.T @ F
        Ml = tau - np.cross(st.conn.patch_center_u, Fl)
        ref = np.concatenate([Fl, Ml])
        scale = max(1e-3, np.abs(ref).max())
        worst = max(worst, float(np.abs(ref - st.wrench).max() / scale))
    record("wrench_readback", worst_rel_err=worst)
    assert worst < 0.02


def test_equal_and_opposite():
    """§2.3.3 invariant 1: any joint wrench is equal and opposite on both bodies
    (sum of forces ~ 0 across each mated pair, and for the insertion tendon)."""
    plan, cell, cm = make("S3", preplaced=[f"b_00{i}" for i in range(5)])
    m, d = cell.model, cell.data
    # 5 N at the cantilever tip: loaded, below capacity (20 N pries it off)
    d.xfrc_applied[m.body("b_004").id] = [0.8, 0.0, -5.0, 0.0, 0.0, 0.0]
    step(cell, cm, 300)
    worst = 0.0
    assert not cm.breaks()
    for st in cm.conn_states:
        if st.conn.lower is None:
            continue                               # the world absorbs its half
        U, L = m.body(st.conn.upper).id, m.body(st.conn.lower).id
        rows = weld_rows(d, st.conn.eq_id)
        tot = np.zeros(6)
        for body in (U, L):
            F, tau = jtf_on(m, d, rows, body)
            tot[:3] += F
            tot[3:] += d.xmat[body].reshape(3, 3) @ tau + np.cross(d.xpos[body], F)
        mag = max(1e-6, float(np.abs(d.efc_force[rows]).max()))
        worst = max(worst, float(np.abs(tot).max() / mag))
    record("equal_and_opposite", weld_worst_rel=worst)
    assert worst < 1e-6


def test_mated_pair_holds_30s():
    """A mated pair holds 30 s under gravity with drift < 0.05 mm."""
    plan, cell, cm = make("S1", preplaced=["b_000", "b_001"])
    b = cell.model.body("b_001").id
    p0 = cell.data.xpos[b].copy()
    step(cell, cm, 30000)
    drift = float(np.linalg.norm(cell.data.xpos[b] - p0) * 1000)
    record("mated_pair_holds_30s", drift_mm=drift, connected=cm.connected("b_001"))
    assert cm.connected("b_001") and drift < 0.05


def _pull_until_break(rate=10.0):
    plan, cell, cm = make("T_PULL", preplaced=["b_000", "b_001"])
    m, d = cell.model, cell.data
    b = m.body("b_001").id
    step(cell, cm, 150)          # past the settle window
    t0 = d.time
    while d.time - t0 < 20.0:
        F = rate * (d.time - t0)
        d.xfrc_applied[b] = [0, 0, F, 0, 0, 0]
        mujoco.mj_step(m, d)
        cm.update(d)
        if cm.breaks():
            return F, cm
    return None, cm


def test_release_at_f_break():
    """Release occurs at f_break +- 15% (4 studs x 11.3 N = 45.2 N), and it is
    the weaker joint that lets go."""
    F, cm = _pull_until_break()
    expected = 4 * F_BREAK_PER_STUD
    who = cm.breaks()[0][2] if cm.breaks() else None
    record("release_at_f_break", release_N=F, expected_N=expected, broke=who)
    assert F is not None and abs(F - expected) / expected < 0.15 and who == "b_001"


def test_extraction_exceeds_insertion():
    """Extraction force > insertion force (asymmetry, [MechanicsSnapFit])."""
    # insertion: a free, aligned brick 0.5 mm into its studs, pushed at its centre
    plan, cell, cm = make("T_PULL", preplaced=["b_000"])
    m, d = cell.model, cell.data
    b = m.body("b_001").id
    tgt = cell.targets["b_001"][0]
    a = m.jnt_qposadr[m.body_jntadr[b]]
    d.qpos[a:a + 3] = tgt + [0, 0, 0.0013]
    mujoco.mj_forward(m, d)
    step(cell, cm, 50)
    z0 = d.xpos[b][2]
    ins = None
    t0 = d.time
    while d.time - t0 < 10.0:
        F = 10.0 * (d.time - t0)
        d.xfrc_applied[b] = [0, 0, -F, 0, 0, 0]
        mujoco.mj_step(m, d)
        cm.update(d)
        if z0 - d.xpos[b][2] > 0.00005:
            ins = F
            break
    ext, _ = _pull_until_break()
    record("asymmetry", insertion_N=ins, extraction_N=ext,
           ratio=(ext / ins) if ins and ext else None,
           expected_insertion_N=4 * F_INSERT_PER_STUD)
    assert ins is not None and ext is not None
    assert abs(ins - 4 * F_INSERT_PER_STUD) / (4 * F_INSERT_PER_STUD) < 0.15
    assert ext > ins


def test_gate_boundaries():
    """Gate fires inside tolerance and not outside -- 6 boundary cases per gate
    condition in §2.3.3 (lateral, vertical gap, misalignment, yaw, force, dwell)."""
    p = CL.ClutchParams()
    base = dict(lat=0.0, dz=0.0003, tilt=0.0, yaw=0.0, force=10.0, dwell=0.1)
    cases = []
    for f in (0.5, 0.9, 0.99, 1.01, 1.1, 1.5):
        inside = f < 1
        cases += [("lateral", dict(base, lat=f * p.gate_lateral), inside),
                  ("tilt", dict(base, tilt=f * p.gate_tilt_deg), inside),
                  ("yaw", dict(base, yaw=f * p.gate_yaw_deg), inside),
                  ("force", dict(base, force=p.gate_force / f), inside),
                  # dwell is a minimum: it must be held AT LEAST 40 ms
                  ("dwell", dict(base, dwell=f * p.dwell), not inside)]
    # vertical gap in [-0.3, +1.0] mm: three cases at each end
    lo, hi = p.gate_dz
    for dz, ok in ((hi * 0.5, True), (hi * 0.99, True), (hi * 1.05, False),
                   (lo * 0.5, True), (lo * 0.99, True), (lo * 1.3, False)):
        cases.append(("vertical_gap", dict(base, dz=dz), ok))
    wrong = []
    for cond, c, expect in cases:
        fired = _gate_fires(**c)
        if fired != expect:
            wrong.append((cond, c, expect, fired))
    record("gate_boundaries", n_cases=len(cases), wrong=wrong)
    assert len(cases) == 36 and not wrong, wrong


def _gate_fires(lat, dz, tilt, yaw, force, dwell):
    """Hold b_001 kinematically at an offset from its seat and run the state
    machine for `dwell` seconds with a controlled axial push."""
    plan, cell, cm = make("T_PULL", preplaced=["b_000"])
    m, d = cell.model, cell.data
    b = m.body("b_001").id
    tgt = cell.targets["b_001"][0]
    a = m.jnt_qposadr[m.body_jntadr[b]]
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.radians([tilt, 0.0, yaw]), "xyz")
    cm.axial_push = lambda d_, name: force
    dt = m.opt.timestep
    for _ in range(int(math.floor(dwell / dt + 1e-9))):     # each update adds dt of dwell
        d.qpos[a:a + 3] = tgt + [lat, 0.0, dz]
        d.qpos[a + 3:a + 7] = q
        mujoco.mj_kinematics(m, d)
        cm.update(d)
        d.time += dt
    return cm.bricks["b_001"].state != CL.DISENGAGED


def test_two_bricks_cannot_share_a_stud():
    """Two bricks cannot mate to the same stud: every planned stud belongs to at
    most one upper brick, and a rival brick at an occupied seat never mates."""
    for sid in P.STRUCTURES:
        if sid.startswith("T_"):
            continue
        plan = P.build_plan(sid, "none")
        seen = set()
        bricks = {b["id"]: b for b in plan["bricks"]}
        for s in plan["sequence"]:
            b = bricks[s["brick_id"]]
            nx, ny = P.footprint(b["type"], b["yaw_index"])
            i, j, k = b["grid_pos"]
            for c in ((i + x, j + y, k) for x in range(nx) for y in range(ny)):
                assert c not in seen, (sid, c)
                seen.add(c)
    # dynamic: b_002 of S1 spawned onto b_000's seat (b_001's place) can never mate there
    plan, cell, cm = make("S1", preplaced=["b_000", "b_001"])
    m, d = cell.model, cell.data
    rival = m.body("b_002").id
    a = m.jnt_qposadr[m.body_jntadr[rival]]
    d.qpos[a:a + 3] = cell.targets["b_001"][0] + [0, 0, 0.0096 + 0.0005]
    mujoco.mj_forward(m, d)
    for _ in range(500):
        d.xfrc_applied[rival] = [0, 0, -60.0, 0, 0, 0]
        mujoco.mj_step(m, d)
        cm.update(d)
    record("no_double_mate", rival_state=cm.bricks["b_002"].state)
    assert cm.bricks["b_002"].state == CL.DISENGAGED


def _energy(m, d):
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    return float(d.energy[0] + d.energy[1])


def test_energy_non_increasing():
    """Energy test: total system energy is non-increasing over 10,000 steps with
    no actuation. A nudge is applied first so there is energy to pump or not
    (the prior version's nudge was absorbed: ledger g1 'PASS but weak').

    Resolution: MuJoCo's energy omits the elastic energy held in soft
    constraints, so single steps can trade a few nJ between that store and
    kinetic energy. The criterion is read over the run: the maximum energy
    of each successive 1000-step window must not increase, and the run must
    end no higher than it started."""
    out = {}
    for label, pre, free in (("mated_S3", [f"b_00{i}" for i in range(5)], None),
                             ("interference_lock", ["b_000"], "b_001")):
        struct = "S3" if label == "mated_S3" else "T_PULL"
        plan, cell, cm = make(struct, preplaced=pre)
        m, d = cell.model, cell.data
        m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        if free:
            b = m.body(free).id
            a = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[a:a + 3] = cell.targets[free][0] + [0, 0, 0.0012]
            mujoco.mj_forward(m, d)
            step(cell, cm, 300)                    # settles onto the lock
        top = m.body(pre[-1] if not free else free).id
        dof = free_dofs(m, top)
        d.qvel[dof:dof + 3] += [0.02, -0.01, 0.01]  # the nudge
        mujoco.mj_forward(m, d)
        e0 = _energy(m, d)
        es = []
        for _ in range(10000):
            mujoco.mj_step(m, d)
            cm.update(d)
            es.append(_energy(m, d))
        es = np.array(es)
        wmax = es.reshape(10, 1000).max(axis=1)
        step_gain = float(np.max(es[1:] - es[:-1]))
        out[label] = {"E0_J": e0, "E_end_J": float(es[-1]), "window_max_J": wmax.tolist(),
                      "largest_single_step_gain_J": step_gain,
                      "nudge_J": 0.5 * m.body_mass[top] * (0.02 ** 2 + 0.01 ** 2 + 0.01 ** 2)}
    record("energy", **out)
    for label, r in out.items():
        assert r["E_end_J"] <= r["E0_J"], label
        w = np.array(r["window_max_J"])
        assert np.all(np.diff(w) <= 1e-12), (label, w)


def _det_worker(_):
    F, cm = _pull_until_break()
    return F, cm.cell.data.qpos.copy()


def test_identical_alone_and_batched():
    """Kernel result identical at 1 env and at max envs -- for the twin, a run
    alone and the same run in each of N parallel worker processes."""
    F0, q0 = _det_worker(0)
    with mp.get_context("spawn").Pool(4) as pool:
        outs = pool.map(_det_worker, range(4))
    same = all(F == F0 and np.array_equal(q, q0) for F, q in outs)
    record("identical_alone_and_batched", release_N=F0, identical=same, workers=4)
    assert same


@pytest.mark.slow
def test_scripted_insertion_from_2mm():
    """A scripted top-down insertion snaps reliably from <= 2 mm initial error and
    holds 30 s (the Panda, the §WP4.4 inserter, 10 trials, one held 30 s)."""
    from motion import skills as K
    from sim.mj import runtime as RT
    rng = np.random.default_rng(7)
    ok, rows = 0, []
    for trial in range(10):
        plan, cell, cm = make("S1", preplaced=["b_000"], with_arms=True)
        sim = RT.Sim(cell, clutch=cm)
        s = plan["sequence"][1]
        bid, tgt = s["brick_id"], cell.targets[s["brick_id"]][0]
        yaw = math.radians(s["grasp"]["yaw_offset_deg"])
        press = 1.3 * 4 * F_INSERT_PER_STUD
        K.pick(sim, "B", bid, yaw, grip_force=K.grip_for(press))
        r, th = 0.002 * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        err = np.array([r * math.cos(th), r * math.sin(th), 0.0])
        K.transport(sim, "B", bid, tgt + err, yaw, clearance_z=0.12)
        res = K.insert(sim, "B", bid, tgt, cm, press_force=press)
        K.release(sim, "B")
        hold = 30.0 if trial == 0 else 2.0
        b = cell.model.body(bid).id
        p0 = cell.data.xpos[b].copy()
        sim.run(hold)
        drift = float(np.linalg.norm(cell.data.xpos[b] - p0) * 1000)
        good = res["success"] and cm.connected(bid) and drift < 0.05
        ok += good
        rows.append({"err_mm": round(r * 1000, 3), "success": res["success"], "held_s": hold,
                     "drift_mm": drift, "peak_N": res["peak_force_N"], "searches": res["searches"]})
    record("scripted_insertion_from_2mm", successes=ok, trials=10, rows=rows)
    assert ok >= 9
