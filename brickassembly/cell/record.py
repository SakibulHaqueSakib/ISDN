"""Per-frame run recorder (plan_v4 r5 S4a, `dual_arm_sim.py --record-all PATH.npz`) and its reader.

Read-only on the sim: only .numpy() reads of state, control and solver.mjw_data. One row per frame, taken after
the frame's simulate() and the break rule, so row i is frame i + 1 (`ex.frame` after the increment).

Fields (`load_record`): frame, sim_time; q_cmd, q_exec (N, 18: the control joint targets sent to both arms
[7 IK joints + finger command each] and the executed joint_q); a_phase, b_phase (int codes into
phase_names); b_step (-1 = none yet); brick_ids, brick_q (N, nb, 7 body_q of every brick), welded (N, nb: all
of that brick's welds on); flag_ab (any A-B arm contact pair with d < 0), flag_arm_welded (any arm body, or
the brick B is holding, in contact at d < 0 with a welded brick). d is solver.mjw_data.contact.dist of the
last substep, as cell/contacts.py reads it (same geom -> shape -> body mapping); arm bodies are
body < 2 * n_arm_bodies, the held brick is contacts.HELD_PHASES of the step in progress. Weld events per
pair: fire_pairs / fire_frame (first frame simulated with the weld on), break_pairs / break_frame (the
frame whose end-of-frame sample broke it); fire_frame_by_brick (-1 = never). failure: the run's failure.

    bash scripts/run.sh cell/record.py PATH.npz [--cube]     # self-check of a recorded run
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cell.contacts import HELD_PHASES  # noqa: E402


class Recorder:
    def __init__(self, ex):
        self.n_arm = ex.n_arm_bodies
        self.sb = ex.model.shape_body.numpy()
        self.ids = list(ex.body)
        self.bodies = np.array([ex.body[b] for b in self.ids])
        self.pairs = [(bid, w[1]) for bid, ws in ex.welds.items() for w in ws]
        self.on = {p: False for p in self.pairs}
        self.rows = {k: [] for k in ("frame", "sim_time", "q_cmd", "q_exec", "a_phase", "b_phase", "b_step",
                                     "brick_q", "welded", "flag_ab", "flag_arm_welded")}
        self.phase_names = []
        self.fire, self.brk = [], []

    def _code(self, name):
        if name not in self.phase_names:
            self.phase_names.append(name)
        return self.phase_names.index(name)

    def _flags(self, ex, welded):
        mjw = ex.solver.mjw_data
        n = int(mjw.nacon.numpy()[0])
        if not n:
            return False, False
        neg = np.where(mjw.contact.dist.numpy()[:n] < 0)[0]
        if not len(neg):
            return False, False
        geom, world = mjw.contact.geom.numpy()[neg], mjw.contact.worldid.numpy()[neg]
        g2s = ex.solver.mjc_geom_to_newton_shape.numpy()
        b0, b1 = self.sb[g2s[world, geom[:, 0]]], self.sb[g2s[world, geom[:, 1]]]
        na = self.n_arm
        isA = lambda b: (b >= 0) & (b < na)
        isB = lambda b: (b >= na) & (b < 2 * na)
        ab = bool((isA(b0) & isB(b1) | isB(b0) & isA(b1)).any())
        held = -2
        if ex.b_step is not None and ex.B.phase in HELD_PHASES:
            held = ex.body[ex.plan["sequence"][ex.b_step]["brick_id"]]
        wb = self.bodies[welded]
        agent = lambda b: (b >= 0) & (b < 2 * na) | (b == held)
        hit = agent(b0) & np.isin(b1, wb) | agent(b1) & np.isin(b0, wb)
        return ab, bool(hit.any())

    def step(self, ex):
        r = self.rows
        welded = np.array([all(w[3] for w in ex.welds[b]) for b in self.ids])
        for p in self.pairs:                      # weld events, from the weld pool's own state
            on = next(w[3] for w in ex.welds[p[0]] if w[1] == p[1])
            if on != self.on[p]:
                (self.fire if on else self.brk).append(("%s<-%s" % p, ex.frame))
                self.on[p] = on
        ab, aw = self._flags(ex, welded)
        r["frame"].append(ex.frame)
        r["sim_time"].append(ex.sim_time)
        r["q_cmd"].append(ex.control.joint_target_pos.numpy()[:18].copy())
        r["q_exec"].append(ex.state_0.joint_q.numpy()[:18].copy())
        r["a_phase"].append(self._code(ex.A.phase))
        r["b_phase"].append(self._code(ex.B.phase))
        r["b_step"].append(-1 if ex.b_step is None else ex.b_step)
        r["brick_q"].append(ex.state_0.body_q.numpy()[self.bodies].copy())
        r["welded"].append(welded)
        r["flag_ab"].append(ab)
        r["flag_arm_welded"].append(aw)

    def save(self, path, ex):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        r, first = self.rows, {}
        for name, f in self.fire:
            first.setdefault(name.split("<-")[0], f)
        np.savez_compressed(
            path, frame=np.array(r["frame"], np.int32), sim_time=np.array(r["sim_time"], np.float64),
            q_cmd=np.array(r["q_cmd"], np.float32).reshape(-1, 18), q_exec=np.array(r["q_exec"], np.float32).reshape(-1, 18),
            a_phase=np.array(r["a_phase"], np.int8), b_phase=np.array(r["b_phase"], np.int8),
            phase_names=np.array(self.phase_names), b_step=np.array(r["b_step"], np.int16),
            brick_ids=np.array(self.ids), brick_q=np.array(r["brick_q"], np.float32).reshape(-1, len(self.ids), 7),
            welded=np.array(r["welded"], bool).reshape(-1, len(self.ids)),
            flag_ab=np.array(r["flag_ab"], bool), flag_arm_welded=np.array(r["flag_arm_welded"], bool),
            fire_pairs=np.array([n for n, _ in self.fire]), fire_frame=np.array([f for _, f in self.fire], np.int32),
            break_pairs=np.array([n for n, _ in self.brk]), break_frame=np.array([f for _, f in self.brk], np.int32),
            fire_frame_by_brick=np.array([first.get(b, -1) for b in self.ids], np.int32),
            failure=np.array(ex.failure or ""))


def load_record(path):
    """{field: array} of a --record-all file."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def check(path, cube=False):
    """Self-check of a recorded run: one row per frame in every per-frame field, boolean flags, fire frames;
    cube=True (arm A only parks): fire frames == 8 and no A-B contact flags."""
    d = load_record(path)
    n = len(d["frame"])
    assert n and (d["frame"] == np.arange(1, n + 1)).all(), "frames are not 1..N"
    for k in ("sim_time", "q_cmd", "q_exec", "a_phase", "b_phase", "b_step", "brick_q", "welded", "flag_ab", "flag_arm_welded"):
        assert len(d[k]) == n, (k, len(d[k]), n)
    assert d["flag_ab"].dtype == bool and d["flag_arm_welded"].dtype == bool and d["welded"].dtype == bool
    assert d["q_cmd"].shape == (n, 18) and d["brick_q"].shape == (n, len(d["brick_ids"]), 7)
    assert d["a_phase"].max() < len(d["phase_names"]) and d["b_phase"].max() < len(d["phase_names"])
    fired = int((d["fire_frame_by_brick"] >= 0).sum())
    assert len(d["fire_pairs"]) == len(d["fire_frame"]) and len(d["break_pairs"]) == len(d["break_frame"])
    if cube:
        assert fired == 8, fired
        assert not d["flag_ab"].any(), int(d["flag_ab"].sum())
    print("record OK: %d frames, %d bricks fired (%d weld pairs), %d breaks, flag_ab %d, flag_arm_welded %d frames, failure %r" % (
        n, fired, len(d["fire_pairs"]), len(d["break_pairs"]), d["flag_ab"].sum(), d["flag_arm_welded"].sum(), str(d["failure"])))
    return d


if __name__ == "__main__":
    check(sys.argv[1], cube="--cube" in sys.argv)
