"""WP4 in the MuJoCo twin: the §WP4 behaviour tree over the scripted skills.

    python -m orchestration.twin_executor --structure S2 --strategy weakest_joint --seed 3
    assemble(sid, strategy, seed)         -> summary, one §2.6 episode per attempt
    step_trial(sid, step, strategy, seed) -> one placement onto the pre-placed
                                             structure (ablation A6's unit)

The tree is §WP4 item 5's, in py_trees (the report forbids a hand-rolled
state machine):

    Sequence AssembleStructure
     └── per step: Retry(<=3) PlaceBrick
          └── Selector
               ├── Sequence NominalPlacement
               │    Brace(A, if requires_brace) -> Grasp(B) -> Transport(B)
               │    -> Insert(B, the policy under test; 10 s) -> VerifySnap
               │    -> ReleaseAndRetract(B) -> ReleaseBrace(A)
               └── Sequence Recovery
                    LogFailure -> LiftAndRegrasp(B) -> ScriptedInsertionFallback(B)
                    -> VerifySnap -> ReleaseAndRetract(B) -> ReleaseBrace(A)

The skills block (they step the physics themselves), so each leaf finishes in
the tick that starts it. The brace goes on before the placer's transport
(§WP4). Poses are ground truth; the handoff to the inserter carries an
injected error (§2.5 phase handoff) drawn from DR.handoff_*.

A joint that breaks during a step is a structural failure even if the
popped brick re-seats under the press afterwards: it ends the assembly (no
retry can un-break the structure, and a real one would have shifted).
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import py_trees
from py_trees.common import Status

import planner as P
from motion import skills as K
from sim.joint_model import clutch as CL
from sim.joint_model.capacity import F_INSERT_PER_STUD
from sim.mj import control as C
from sim.mj import runtime as RT
from sim.mj import scene as S

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "mujoco_twin"
JOINT_MODEL = "clutch_capacity_v1"      # sim/joint_model/clutch.py + capacity.py
PRESS_MARGIN = 1.3                      # press force = 1.3 x n_studs x f_insert
                                        # (bracing.FF_PRESS_MARGIN sizes the brace for it)
BUDGET_MARGIN = 1.6                     # force budget = 1.6 x n_studs x f_insert
RETRIES = 3
VIEW = None                             # --view: playback speed of a live viewer (runtime.watch)


@dataclass
class DR:
    """Per-trial randomisation (ground-truth poses; §WP4 G3 and A6)."""
    feeder_xy_mm: float = 1.0           # brick placement in the feeder, uniform square
    feeder_yaw_deg: float = 0.0         # a fixtured feeder presents bricks square (at 3 deg
                                        # the grasp flips to the other wrist configuration,
                                        # where touchdown shoves the brick 0.3 mm along the grip
                                        # axis and the base joint of S1 breaks 2 runs in 3)
    handoff_xy_mm: float = 0.3          # pre-insertion error, uniform disc (cuRobo-like
    handoff_yaw_deg: float = 0.3        # placement error: A3 decides if vision is needed;
                                        # a calibrated arm's yaw -- at 1 deg a 2x4 wedges on
                                        # diagonal studs, since 0.4 deg uses its 0.1 mm clearance)
    ft_noise: tuple = (0.1, 0.005)      # N, N*m white noise on the wrist sensors


NO_DR = DR(0.0, 0.0, 0.0, 0.0, None)


PLANS = ROOT / "plans" / "twin"


def load_plan(sid, strategy):
    """The §2.6 assembly_plan for the twin (planner.py --frame twin writes
    them; planning S5 with the hand-clearance check takes a minute). The
    caller must have set planner.VOXEL_ORIGIN (scene.centre_plan_origin)."""
    path = PLANS / ("%s_%s.json" % (sid, strategy))
    if path.exists():
        plan = json.loads(path.read_text())
        assert np.allclose(plan["voxel_origin"], P.VOXEL_ORIGIN), "stale plan frame: %s" % path
        return plan
    from sim.mj.checks import BraceClearance
    return P.build_plan(sid, strategy, clearance=BraceClearance(sid) if strategy != "none" else None)


class Ctx:
    """What the leaves share (a blackboard, kept as a plain object)."""

    def __init__(self, sid, strategy, seed, dr=DR(), preplace_upto=0, run_id=None,
                 inserter=None, policy_name="scripted_spiral_v1", stop_on_break=True,
                 coordinated=True):
        S.centre_plan_origin(P.STRUCTURES[sid])
        self.plan = load_plan(sid, strategy)
        self.sid, self.strategy, self.seed, self.dr = sid, strategy, seed, dr
        self.rng = np.random.default_rng(seed)
        seq = self.plan["sequence"]
        pre = [s["brick_id"] for s in seq[:preplace_upto]]
        self.cell = S.build_cell(self.plan, preplaced=set(pre))
        self.cm = CL.ClutchModel(self.cell)
        for b in pre:
            self.cm.mate_now(b)
        self._jitter_feeder(set(pre))
        self.sim = RT.Sim(self.cell, clutch=self.cm, ft_noise=dr.ft_noise, rng=self.rng)
        if VIEW:
            RT.watch(self.sim, VIEW)
        self.sim.run(0.2)
        self.placed = list(pre)
        self.top_z = max(x["target_pose"][2] for x in seq) + P.BRICK_H
        self.run_id = run_id or "%s_%s_seed%02d" % (sid, strategy, seed)
        self.inserter = inserter          # callable(ctx, step, target, press) -> result dict
        self.policy_name = policy_name if inserter else "scripted_spiral_v1"
        self.stop_on_break = stop_on_break
        self.episodes = []
        self.in_hand = None
        self.braced = None                # step index whose brace arm A holds
        self.coordinated = coordinated    # brace feeds forward the LP wrench (skills)
        self.brace_hook = None
        self.fatal = None
        self.cur = {}                     # per-attempt scratch
        self.t_wall = time.time()

    def _jitter_feeder(self, pre):
        m, d, dr = self.cell.model, self.cell.data, self.dr
        for b in self.plan["bricks"]:
            if b["id"] in pre or not (dr.feeder_xy_mm or dr.feeder_yaw_deg):
                continue
            j = m.body(b["id"]).jntadr[0]
            a = m.jnt_qposadr[j]
            d.qpos[a:a + 2] += self.rng.uniform(-1, 1, 2) * dr.feeder_xy_mm / 1000
            yaw = math.radians(self.rng.uniform(-1, 1) * dr.feeder_yaw_deg)
            q = d.qpos[a + 3:a + 7].copy()
            dq = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, dq, q)
            d.qpos[a + 3:a + 7] = out
        mujoco.mj_forward(m, d)

    def step_dict(self, n):
        return self.plan["sequence"][n]

    def target(self, s):
        return self.cell.targets[s["brick_id"]][0]

    def press(self, s):
        return PRESS_MARGIN * self.n_studs(s) * F_INSERT_PER_STUD

    def n_studs(self, s):
        n = sum(m[1] for m in s["mating_studs"])
        if n == 0:                        # on the baseplate: its whole footprint
            b = next(x for x in self.plan["bricks"] if x["id"] == s["brick_id"])
            nx, ny = P.footprint(b["type"], b["yaw_index"])
            n = nx * ny
        return n

    def structure_intact(self):
        return all(self.cm.connected(b) for b in self.placed)


# --- leaves ---------------------------------------------------------------------------

class Leaf(py_trees.behaviour.Behaviour):
    def __init__(self, name, ctx, n):
        super().__init__(name)
        self.ctx, self.n = ctx, n

    def update(self):
        if self.ctx.fatal:
            return Status.FAILURE
        if VIEW:
            print("%6.2f s  %s" % (self.ctx.sim.t, self.name), flush=True)
        try:
            ok = self.run(self.ctx, self.ctx.step_dict(self.n))
        except FloatingPointError as e:
            self.ctx.fatal = "diverged: %s" % e
            return Status.FAILURE
        return Status.SUCCESS if ok else Status.FAILURE

    def run(self, ctx, s):
        raise NotImplementedError


class Brace(Leaf):
    def run(self, ctx, s):
        if not s["requires_brace"] or ctx.braced == self.n:
            return True
        ok = K.brace(ctx.sim, "A", s["brace"])
        ctx.braced = self.n
        ctx.cur["brace_ft0"] = ctx.sim.ft("A").copy()
        ctx.cur["brace_rel0"] = _brace_rel(ctx, s)
        if not ok:
            ctx.cur["failure_mode"] = "brace_missed"
        return ok


class Grasp(Leaf):
    def run(self, ctx, s):
        bid = s["brick_id"]
        if ctx.in_hand == bid:
            return True
        if ctx.cm.bricks[bid].state != "DISENGAGED":
            ctx.cur["failure_mode"] = "brick_stuck_in_structure"
            return False
        yaw = _feeder_yaw(ctx, bid) + math.radians(s["grasp"]["yaw_offset_deg"])
        # tcp_offset_m is in the brick's frame at its target (yaw 0); pick()
        # applies the brick's current orientation
        ok = K.pick(ctx.sim, "B", bid, yaw, grip_force=K.grip_for(ctx.press(s)),
                    offset=s["grasp"]["tcp_offset_m"])
        if ok:
            ctx.in_hand = bid
        else:
            ctx.cur["failure_mode"] = "grasp_failed"
            K.release(ctx.sim, "B")
        return ok


class Transport(Leaf):
    def run(self, ctx, s):
        bid, dr = s["brick_id"], ctx.dr
        yaw = math.radians(s["grasp"]["yaw_offset_deg"])
        tgt = ctx.target(s).copy()
        K.transport(ctx.sim, "B", bid, tgt, yaw, clearance_z=_clearance(ctx))
        # the handoff error (§2.5): the inserter starts from here, not from truth
        r = dr.handoff_xy_mm / 1000 * math.sqrt(ctx.rng.uniform())
        th = ctx.rng.uniform(0, 2 * math.pi)
        dyaw = math.radians(ctx.rng.uniform(-1, 1) * dr.handoff_yaw_deg)
        arm = ctx.sim.arms["B"]
        if r or dyaw:
            ctx.sim.goto("B", arm.x_d + [r * math.cos(th), r * math.sin(th), 0],
                         C.rotvec_to_mat([0, 0, dyaw]) @ arm.R_d, duration=0.3, settle=0.3)
        return _held(ctx, bid)


class Insert(Leaf):
    def __init__(self, name, ctx, n, scripted=False):
        super().__init__(name, ctx, n)
        self.scripted = scripted

    def run(self, ctx, s):
        bid = s["brick_id"]
        if ctx.in_hand != bid or not _held(ctx, bid):
            ctx.cur["failure_mode"] = "dropped"
            return False
        tgt = ctx.target(s)
        if K.hand_tilt_deg(ctx.sim, "B", bid) > K.MAX_HAND_TILT_DEG:
            ctx.cur["failure_mode"] = "tilted_in_hand"      # recovery regrasps it
            return False
        ep = _episode_start(ctx, s, "scripted_spiral_v1" if self.scripted or not ctx.inserter
                            else ctx.policy_name)
        brace_log = []
        hook = None
        if ctx.braced == self.n:
            # re-latch the hold where the structure is now (no spring preload
            # from whatever moved since the grasp), then feed forward
            ctx.sim.arms["A"].hold(ctx.sim.d)
            ctx.sim.run(0.05)
            ctx.cur["brace_ft0"] = ctx.sim.ft("A").copy()
            if ctx.coordinated and ctx.brace_hook is None:
                ctx.brace_hook = K.brace_feedforward(ctx.sim, "A", s["brace"])
                ctx.sim.hooks.append(ctx.brace_hook)
            ft0 = ctx.cur.get("brace_ft0", np.zeros(6))
            hook = lambda sim: brace_log.append((sim.t, sim.ft("A") - ft0)) \
                if sim.k % 10 == 0 else None
            ctx.sim.hooks.append(hook)
        n_breaks = len(ctx.cm.breaks())
        flog = []
        if self.scripted or ctx.inserter is None:
            res = K.insert(ctx.sim, "B", bid, tgt, ctx.cm, press_force=ctx.press(s), log=flog)
        else:
            res = ctx.inserter(ctx, s, tgt, ctx.press(s), flog)
        if hook is not None:
            ctx.sim.hooks.remove(hook)
        _episode_end(ctx, s, ep, res, flog, brace_log, n_breaks)
        if ctx.stop_on_break and len(ctx.cm.breaks()) > n_breaks:
            ctx.fatal = "joint_break: %s" % [(e[2], e[3]["lower"])
                                             for e in ctx.cm.breaks()[n_breaks:][:3]]
            return False
        return res["success"]


class VerifySnap(Leaf):
    def run(self, ctx, s):
        ctx.sim.run(0.1)
        bid = s["brick_id"]
        ok = ctx.cm.connected(bid)
        broke = [e for e in ctx.cm.breaks() if e[0] >= ctx.cur.get("t_step0", 0.0)]
        if broke and ctx.stop_on_break:
            ctx.fatal = "joint_break: %s" % [(e[2], e[3]["lower"]) for e in broke[:3]]
        if not ctx.structure_intact():
            ctx.fatal = ctx.fatal or "structure_disconnected"
        if not ok:
            ctx.cur["failure_mode"] = ctx.cur.get("failure_mode") or "no_snap"
        return ok and not ctx.fatal


class ReleaseAndRetract(Leaf):
    def run(self, ctx, s):
        K.release(ctx.sim, "B")
        ctx.in_hand = None
        ctx.sim.run(0.1)
        ok = ctx.cm.connected(s["brick_id"])
        if not ok:
            ctx.cur["failure_mode"] = "pulled_off_on_release"
        return ok


class ReleaseBrace(Leaf):
    def run(self, ctx, s):
        if ctx.braced is not None:
            if ctx.brace_hook in ctx.sim.hooks:
                ctx.sim.hooks.remove(ctx.brace_hook)
            ctx.brace_hook = None
            K.unbrace(ctx.sim, "A")
            ctx.braced = None
        broke = [e for e in ctx.cm.breaks() if e[0] >= ctx.cur.get("t_step0", 0.0)]
        if broke and ctx.stop_on_break:
            ctx.fatal = "joint_break: %s" % [(e[2], e[3]["lower"]) for e in broke[:3]]
        if ctx.fatal:
            return False
        ctx.placed.append(s["brick_id"])
        return True


class LogFailure(Leaf):
    """Every attempt is logged (§2.6); one that failed before the inserter ran
    gets a record here."""

    def run(self, ctx, s):
        _log_unlogged(ctx, s)
        return True


def _log_unlogged(ctx, s):
    if not ctx.cur.get("attempt_logged"):
        ep = _episode_start(ctx, s, ctx.policy_name)
        ep.pop("t0")
        ep.update({"success": False, "failure_mode": ctx.cur.get("failure_mode") or "unknown",
                   "brace_active": ctx.braced == s["step"]})
        ctx.episodes.append(ep)
    ctx.cur["attempt_logged"] = False
    ctx.cur["failure_mode"] = None


class LiftAndRegrasp(Leaf):
    """Back off 15 mm and re-centre on ground truth; if the brick is no longer
    in the hand, pick it up again from wherever it lies."""

    def run(self, ctx, s):
        bid = s["brick_id"]
        arm = ctx.sim.arms["B"]
        if ctx.in_hand == bid and _held(ctx, bid):
            if ctx.cm.bricks[bid].state != "DISENGAGED":
                # half-seated; lifting would tear it. After a failed RL attempt
                # (a force-budget stop mid-press) the fallback presses it home
                # (§WP7.1); the scripted-only tree keeps its G3/A6 behaviour
                return ctx.inserter is not None

            ctx.sim.goto("B", arm.x_d + [0, 0, 0.015], arm.R_d, duration=0.4)
            if K.hand_tilt_deg(ctx.sim, "B", bid) <= K.MAX_HAND_TILT_DEG:
                yaw = math.radians(s["grasp"]["yaw_offset_deg"])
                K.transport(ctx.sim, "B", bid, ctx.target(s), yaw, clearance_z=_clearance(ctx))
                return True
            # tilted in the fingers: set it down on its (now empty) feeder slot
            # and pick it up again, level
            K.put_down(ctx.sim, "B", bid, ctx.cell.feeder[bid][:2])
            ctx.in_hand = None
        K.release(ctx.sim, "B")
        ctx.in_hand = None
        if ctx.cm.bricks[bid].state != "DISENGAGED":
            return False
        p, Rb = K.brick_pose(ctx.sim, bid)
        if p[2] < -0.002 or Rb[2, 2] < 0.95:       # off the table, or on its side
            return False
        yaw = math.atan2(Rb[1, 0], Rb[0, 0]) + math.radians(s["grasp"]["yaw_offset_deg"])
        ok = K.pick(ctx.sim, "B", bid, yaw, grip_force=K.grip_for(ctx.press(s)),
                    offset=s["grasp"]["tcp_offset_m"])
        if not ok:
            return False
        ctx.in_hand = bid
        return Transport("t", ctx, self.n).run(ctx, s)


def _clearance(ctx):
    """Travel height for the placer: above the finished structure, and above
    the stabilizer's hand while it braces (it leans 45 deg up and back from
    the brace point; carrying the brick at structure height + 60 mm drove it
    into A's hand and broke 25 joints before the press began)."""
    z = ctx.top_z + 0.06
    if ctx.braced is not None:
        m, d = ctx.sim.m, ctx.sim.d
        top = max(d.geom_xpos[g][2] + m.geom_rbound[g] for g in range(m.ngeom)
                  if m.body(m.geom_bodyid[g]).name.startswith("A/") and m.geom_contype[g])
        z = max(z, top + 0.04)
    return z


def _feeder_yaw(ctx, bid):
    _, Rb = K.brick_pose(ctx.sim, bid)
    return math.atan2(Rb[1, 0], Rb[0, 0])


def _held(ctx, bid):
    """The brick is between the fingers of B."""
    x, _ = ctx.sim.arms["B"].tcp(ctx.sim.d)
    p, _ = K.brick_pose(ctx.sim, bid)
    return np.linalg.norm(p + [0, 0, K.GRASP_TCP_ABOVE_BOTTOM] - x) < 0.015


def _brace_rel(ctx, s):
    """Arm A's TCP in the frame of the brick it braces (for brace_slipped)."""
    x, _ = ctx.sim.arms["A"].tcp(ctx.sim.d)
    p, Rb = K.brick_pose(ctx.sim, s["brace"]["target_brick_id"])
    return Rb.T @ (x - p)


def _pose_err(ctx, bid, tgt):
    p, Rb = K.brick_pose(ctx.sim, bid)
    yaw = math.atan2(Rb[1, 0], Rb[0, 0])
    sym = math.pi / 2 if ctx.cm.square[bid] else math.pi
    dyaw = (yaw + sym / 2) % sym - sym / 2
    tilt = math.degrees(math.acos(np.clip(Rb[2, 2], -1, 1)))
    return (float(np.linalg.norm((p - tgt)[:2]) * 1000), float(abs(p[2] - tgt[2]) * 1000),
            float(math.hypot(math.degrees(dyaw), tilt)))


def _episode_start(ctx, s, policy):
    bid = s["brick_id"]
    exy, ez, edeg = _pose_err(ctx, bid, ctx.target(s))
    att = sum(1 for e in ctx.episodes if e["brick_id"] == bid)
    ctx.cur.setdefault("t_step0", ctx.sim.t)
    return {"run_id": ctx.run_id, "structure_id": ctx.sid, "step": s["step"], "brick_id": bid,
            "attempt": att, "policy": policy, "backend": BACKEND, "joint_model": JOINT_MODEL,
            "bracing_strategy": ctx.strategy, "seed": ctx.seed,
            "initial_error_mm": round(exy, 3), "initial_error_deg": round(edeg, 3),
            "t0": ctx.sim.t}


def _episode_end(ctx, s, ep, res, flog, brace_log, n_breaks0):
    bid = s["brick_id"]
    exy, ez, edeg = _pose_err(ctx, bid, ctx.target(s))
    f = np.array([r[1] for r in flog]) if flog else np.zeros(1)
    budget = BUDGET_MARGIN * ctx.n_studs(s) * F_INSERT_PER_STUD
    breaks = ctx.cm.breaks()[n_breaks0:]
    ep.update({
        "success": bool(res["success"]),
        "steps": int(round(res["duration_s"] / (RT.CTRL_EVERY * ctx.sim.m.opt.timestep))),
        "duration_s": res["duration_s"],
        "final_error_mm": round(math.hypot(exy, ez), 3), "final_error_deg": round(edeg, 3),
        "peak_force_N": res["peak_force_N"],
        "mean_force_N": round(float(np.mean(np.clip(f, 0, None))), 2),
        "impulse_Ns": res["impulse_Ns"],
        "force_budget_N": round(budget, 1),
        "budget_violated": bool(res["peak_force_N"] > budget),
        "brace_active": ctx.braced == s["step"],
        "brace_peak_reaction_N": None, "brace_slipped": None,
        "predicted_success_prob": res.get("predicted_success_prob"),
        "snap_time_s": res["duration_s"] if res["success"] else None,
        "n_studs_engaged": ctx.n_studs(s) if res["success"] else 0,
        "failure_mode": None if res["success"] else
        ("joint_break" if breaks else "insert_timeout_in_%s" % res["phase"]),
        "searches": res.get("searches"),
        "joint_breaks": [{"upper": e[2], "lower": e[3]["lower"], "util": e[3]["util"]}
                         for e in breaks],
        "max_joint_util": round(float(max((c.peak_util for c in ctx.cm.conn_states
                                           if c.conn.upper in ctx.placed), default=0.0)), 3),
        "predicted_util_unbraced": s.get("predicted_util_unbraced"),
    })
    if ep["brace_active"]:
        br = s["brace"]
        w = np.array([r[1] for r in brace_log]) if brace_log else np.zeros((1, 6))
        # the stabilizer's push on the structure is minus the environment's on the hand
        on_structure = -w
        k = int(np.argmax(np.linalg.norm(on_structure[:, :3], axis=1)))
        rel = _brace_rel(ctx, s)
        ep.update({
            "brace_peak_reaction_N": round(float(np.linalg.norm(on_structure[k, :3])), 2),
            "brace_measured_wrench": [round(float(v), 3) for v in on_structure[k]],
            "brace_expected_wrench": br["expected_reaction_wrench"],
            "brace_slipped": bool(np.linalg.norm(rel - ctx.cur["brace_rel0"]) > 0.001),
            "brace_gripped_bricks": br["gripped_bricks"],
            "predicted_util_braced": br.get("predicted_util_braced"),
        })
    ep.pop("t0")
    ctx.episodes.append(ep)
    ctx.cur["attempt_logged"] = True
    ctx.cur["failure_mode"] = ep["failure_mode"]


# --- trees ------------------------------------------------------------------------------

def place_brick(ctx, n, nominal_inserter=True):
    bid = ctx.step_dict(n)["brick_id"]

    def seq(name, leaves):
        return py_trees.composites.Sequence(name, memory=True, children=leaves)

    nominal = seq("NominalPlacement", [
        Brace("[A] Brace", ctx, n), Grasp("[B] Grasp", ctx, n),
        Transport("[B] TransportToPreInsertion", ctx, n),
        Insert("[B] RunInsertionSkill", ctx, n, scripted=not nominal_inserter),
        VerifySnap("VerifySnap", ctx, n), ReleaseAndRetract("[B] ReleaseAndRetract", ctx, n),
        ReleaseBrace("[A] ReleaseBrace", ctx, n)])
    recovery = seq("Recovery", [
        LogFailure("LogFailure", ctx, n), Brace("[A] Brace", ctx, n),
        LiftAndRegrasp("[B] LiftAndRegrasp", ctx, n),
        Insert("[B] ScriptedInsertionFallback", ctx, n, scripted=True),
        VerifySnap("VerifySnap", ctx, n), ReleaseAndRetract("[B] ReleaseAndRetract", ctx, n),
        ReleaseBrace("[A] ReleaseBrace", ctx, n)])
    sel = py_trees.composites.Selector("PlaceBrick %s" % bid, memory=True,
                                       children=[nominal, recovery])
    return py_trees.decorators.Retry("Retry %s" % bid, sel, num_failures=RETRIES)


def tick_to_end(root, max_ticks=50):
    for _ in range(max_ticks):
        root.tick_once()
        if root.status != Status.RUNNING:
            break
    return root.status


def _new_step(ctx):
    ctx.cur = {}


class StepMarker(py_trees.behaviour.Behaviour):
    """Resets the per-step scratch before a PlaceBrick."""

    def __init__(self, ctx):
        super().__init__("NextStep")
        self.ctx = ctx

    def update(self):
        _new_step(self.ctx)
        return Status.SUCCESS


def assemble(sid, strategy="weakest_joint", seed=0, dr=DR(), inserter=None,
             policy_name=None, out=None, steps=None, preplace_upto=0, coordinated=True):
    """Build `sid` (or just `steps` of it) with the tree. Returns a summary."""
    ctx = Ctx(sid, strategy, seed, dr=dr, preplace_upto=preplace_upto, inserter=inserter,
              coordinated=coordinated,
              policy_name=policy_name or "scripted_spiral_v1")
    todo = list(steps if steps is not None else range(preplace_upto, len(ctx.plan["sequence"])))
    children = []
    for n in todo:
        children += [StepMarker(ctx), place_brick(ctx, n)]
    root = py_trees.composites.Sequence("AssembleStructure", memory=True, children=children)
    status = tick_to_end(root, max_ticks=10 * len(todo) + 10)
    if status != Status.SUCCESS and todo:
        failed = next((n for n in todo if ctx.step_dict(n)["brick_id"] not in ctx.placed), None)
        if failed is not None and ctx.cur.get("failure_mode") and not ctx.cur.get("attempt_logged"):
            _log_unlogged(ctx, ctx.step_dict(failed))
    placed = [b for b in ctx.placed if b not in {s["brick_id"] for s in ctx.plan["sequence"][:preplace_upto]}]
    summary = {
        "run_id": ctx.run_id, "structure_id": sid, "strategy": strategy, "seed": seed,
        "policy": ctx.policy_name, "success": status == Status.SUCCESS and not ctx.fatal,
        "placed": len(placed), "of": len(todo), "fatal": ctx.fatal,
        "breaks": len(ctx.cm.breaks()), "structure_intact": ctx.structure_intact(),
        "attempts": len(ctx.episodes),
        "sim_s": round(ctx.sim.t, 2), "wall_s": round(time.time() - ctx.t_wall, 1),
        "dr": asdict(dr),
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as fh:
            for e in ctx.episodes:
                fh.write(json.dumps(e) + "\n")
    summary["episodes"] = ctx.episodes
    return summary


def step_trial(sid, step, strategy, seed=0, dr=DR(), inserter=None, policy_name=None, out=None,
               coordinated=True):
    """Ablation A6's unit: place step `step` onto the structure pre-placed and
    mated up to it. A break of ANY joint during the step is a failure."""
    r = assemble(sid, strategy, seed, dr=dr, inserter=inserter, policy_name=policy_name,
                 out=out, steps=[step], preplace_upto=step, coordinated=coordinated)
    ep = [e for e in r["episodes"]]
    last = ep[-1] if ep else {}
    r.update({"step": step, "success": r["success"] and r["breaks"] == 0,
              "brace_active": bool(last.get("brace_active")),
              "brace_peak_reaction_N": last.get("brace_peak_reaction_N"),
              "brace_measured_wrench": last.get("brace_measured_wrench"),
              "brace_expected_wrench": last.get("brace_expected_wrench"),
              "brace_slipped": last.get("brace_slipped"),
              "max_joint_util": max((e.get("max_joint_util", 0.0) for e in ep), default=None),
              "peak_force_N": max((e.get("peak_force_N", 0.0) for e in ep), default=None)})
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="S1", choices=list(P.STRUCTURES))
    ap.add_argument("--strategy", default="weakest_joint", choices=["none", "nearest", "weakest_joint"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=None, help="one placement onto the pre-placed structure")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--passive-brace", action="store_true",
                    help="hold the brace pose stiffly without the LP feed-forward")
    ap.add_argument("--out", default=str(ROOT / "results" / "twin_episodes.jsonl"))
    ap.add_argument("--view", type=float, nargs="?", const=1.0, default=None, metavar="SPEED",
                    help="watch it live in MuJoCo's viewer at SPEED x real time (default 1; 0.25 = slow motion)")
    a = ap.parse_args()
    VIEW = a.view
    dr = NO_DR if a.no_dr else DR()
    if a.step is not None:
        r = step_trial(a.structure, a.step, a.strategy, a.seed, dr=dr, out=a.out,
                       coordinated=not a.passive_brace)
    else:
        r = assemble(a.structure, a.strategy, a.seed, dr=dr, out=a.out,
                     coordinated=not a.passive_brace)
    eps = r.pop("episodes")
    print(json.dumps(r, indent=1, default=str))
    for e in eps:
        print("  step %2d %s att %d: %s peak %.1f N, %s%s" % (
            e["step"], e["brick_id"], e["attempt"], "OK " if e["success"] else "FAIL",
            e.get("peak_force_N", 0.0), e["failure_mode"],
            "  brace %s N (pred %s)" % (e["brace_peak_reaction_N"], e["brace_expected_wrench"][:3])
            if e.get("brace_peak_reaction_N") is not None else ""))
    if VIEW:
        print("done - close the viewer window to exit", flush=True)
        RT.wait_viewers()
