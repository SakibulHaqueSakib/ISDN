"""Gymnasium wrappers of the dual-arm LEGO cell (dual_arm_sim.Example, headless Newton): one simulator core, two environments.

    Sim                 the core: builds an Example (ViewerNull), steps physics, builds the observation, scores
    BrickStepEnv        one decision per brick, one env step per episode (plan_v4_rl_brace 2.3-2.5, plan_v4_rl_place)
    BrickReachEnv       low-level control of both hands, many steps per episode (10 Hz), shaped reward

    bash scripts/run.sh rl_env.py --demo            # the scripted reach episode (privileged info inside the script only) and a timing line

Observation (the same Dict for both envs; BrickStepEnv adds "features"), vision and proprioception only:
    proprio   (54,)  joint q (18) and qd (18) of both arms, both hands' FK pose (pos + quat, sign fixed to the parked hand's), finger openings (4)
    bricks    (N_MAX, 20)  one row per top_detect brick, held ones included: valid, x, y, z_top, sin/cos 2*yaw, L, W, colour one-hot (8),
                           hidden, held_by_A, held_by_B, on_plate. Padded with valid = 0
    plate     (8,)   found, x, y, sin/cos 2*yaw, L, W, top_z
    task      (8,)   the current plan step's target pose x, y, z, sin/cos 2*yaw, brick L, W, step / N
    images    (optional) per camera: <cam>_rgb uint8 (H/stride, W/stride, 3) and <cam>_depth float32 (H/stride, W/stride), from the NOISED
              Cameras.capture at the nominal pose (RL sees the sensor noise); depth 0 = no return
    features  (BrickStepEnv) brace_bandit.features(.., action=None): the 15 context numbers of the RL plan, a function of the plan alone

Firewall: the observation is built by ObsSpec.encode(ObsInputs), a pure function of ObsInputs (fields below), and Sim.sense() is the only
place that fills ObsInputs. The inputs are the camera captures, top_detect on the top capture (detect -> remember -> track_held, as
Example.refresh_feeds does), the joint state and its FK, and the plan. Simulator ground truth (body_q of bricks, welds, contacts, the true
camera pose, shape-index labels) is read only by Sim.privileged() / the envs' reward code, and reaches the caller through info["privileged"].
tests/test_rl_env.py checks it: moving the bricks does not change a non-image entry when the captures are frozen, and sense() mentions no
ground-truth name.
"""

import argparse
import contextlib
import gc
import io
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blueprint as B  # noqa: E402
import dual_arm_sim as D  # noqa: E402
import newton  # noqa: E402
import planner as P  # noqa: E402
import top_detect as TD  # noqa: E402
from cell import liveview as LV  # noqa: E402
from cell.cameras import CAMS, Cameras  # noqa: E402
from cell.contacts import ContactMonitor  # noqa: E402
from tasks import brace_bandit as BB  # noqa: E402

CAM_NAMES = ("top", "wrist_A", "wrist_B")
N_COL = len(TD.COLOURS)
BRICK_F = 1 + 3 + 2 + 2 + N_COL + 4                  # valid, x y z_top, sin cos, L W, colour, hidden held_A held_B on_plate
PLATE_F, TASK_F, PROPRIO_F = 8, 8, 18 + 18 + 14 + 4
N_CTX = BB.FEATURE_NAMES.index("is_none")            # the context part of brace_bandit.features
FPS = 60


def bricks_of(structure):
    """Planner tuples of a sample shape (blueprint.SAMPLES: cube, arch, ...) or a hand-authored structure (planner.STRUCTURES: S3, ...)."""
    return list(B.load(structure)) if structure in B.SAMPLES else list(P.STRUCTURES[structure])


# --- the observation: a pure function of ObsInputs -----------------------------------------------------------------
class ObsInputs(NamedTuple):
    """Everything an observation may read; Sim.sense() fills it, ObsSpec.encode() reads nothing else."""
    q: np.ndarray                    # (18,) joint positions of both arms (proprioception)
    qd: np.ndarray                   # (18,) joint velocities
    hands: list                      # per arm: FK pose [x y z qx qy qz qw] of the TCP, from q alone
    det: dict                        # top_detect result on the noised top capture, with remember and track_held
    images: Optional[dict]           # {camera: Cameras.capture result} (noised, nominal pose), or None
    task: dict                       # the plan: target x, y, z, yaw, L, W of the current step; step, n
    features: Optional[np.ndarray]   # brace_bandit.features context part (BrickStepEnv), or None


ALLOWED_INPUTS = ObsInputs._fields


class ObsSpec:
    def __init__(self, n_max=16, images=True, cameras=CAM_NAMES, stride=4, features=False):
        self.n_max, self.images, self.cameras, self.stride, self.features = n_max, images, tuple(cameras), stride, features
        assert all(c in CAMS for c in self.cameras) and stride >= 1
        self.park_quat = [np.array([math.cos(b[1] / 2), math.sin(b[1] / 2), 0.0, 0.0]) for b in (D.ARM_A, D.ARM_B)]

    def space(self):
        inf, f32 = np.inf, np.float32
        sp = {"proprio": spaces.Box(-inf, inf, (PROPRIO_F,), f32),
              "bricks": spaces.Box(-10.0, 10.0, (self.n_max, BRICK_F), f32),
              "plate": spaces.Box(-10.0, 10.0, (PLATE_F,), f32),
              "task": spaces.Box(-10.0, 10.0, (TASK_F,), f32)}
        if self.features:
            sp["features"] = spaces.Box(-1e3, 1e3, (N_CTX,), f32)
        if self.images:
            im = {}
            for c in self.cameras:
                h, w = -(-CAMS[c]["h"] // self.stride), -(-CAMS[c]["w"] // self.stride)
                im[c + "_rgb"] = spaces.Box(0, 255, (h, w, 3), np.uint8)
                im[c + "_depth"] = spaces.Box(0.0, 100.0, (h, w), f32)
            sp["images"] = spaces.Dict(im)
        return spaces.Dict(sp)

    def encode(self, inp):
        hands = []
        for k, h in enumerate(inp.hands):
            h = np.array(h, float)
            if h[3:] @ self.park_quat[k] < 0:        # q and -q are one rotation: take the sign that is continuous over the arm's yaw range
                h[3:] *= -1
            hands.append(h)
        proprio = np.concatenate([inp.q, inp.qd, *hands, inp.q[7:9], inp.q[16:18]])
        det = inp.det
        rows = np.zeros((self.n_max, BRICK_F), np.float32)
        for i, b in enumerate((det["bricks"] + list(det.get("held", {}).values()))[:self.n_max]):   # ponytail: bricks past n_max are dropped (largest build is 24 vs 16)
            held = b.get("held_by")
            rows[i] = [1, b["x"], b["y"], b["z_top"], math.sin(2 * b["yaw"]), math.cos(2 * b["yaw"]), b["L"], b["W"],
                       *np.eye(N_COL)[TD.COLOURS.index(b["colour"])], b["hidden"], held == "A", held == "B", b["on_plate"]]
        pl, plate = det["plate"], np.zeros(PLATE_F, np.float32)
        if pl:
            plate[:] = [1, pl["x"], pl["y"], math.sin(2 * pl["yaw"]), math.cos(2 * pl["yaw"]), *pl["size"], pl["top_z"]]
        t = inp.task
        task = np.array([t["x"], t["y"], t["z"], math.sin(2 * t["yaw"]), math.cos(2 * t["yaw"]), t["L"], t["W"], t["step"] / t["n"]], np.float32)
        obs = {"proprio": proprio.astype(np.float32), "bricks": rows, "plate": plate, "task": task}
        if self.features:
            obs["features"] = np.asarray(inp.features, np.float32)
        if self.images:
            obs["images"] = {}
            for c in self.cameras:
                obs["images"][c + "_rgb"] = np.ascontiguousarray(inp.images[c]["rgb"][::self.stride, ::self.stride])
                obs["images"][c + "_depth"] = np.ascontiguousarray(inp.images[c]["depth"][::self.stride, ::self.stride], np.float32)
        return obs


# --- the simulator core --------------------------------------------------------------------------------------------
class Sim:
    """One Example at a time. build() makes it (about 1.7 s: plan, scene, CUDA graphs), sense() reads what a policy may see,
    privileged() reads the rest (reward, termination, info)."""

    def __init__(self, spec, exe="E0", verbose=False):
        self.spec, self.exe, self.verbose = spec, exe, verbose
        self.ex = self.cams = self.det = self.mon = None

    def build(self, structure, step, seed, rl_control, only_step, press=None):
        """A fresh Example at the nominal start of `step` (earlier bricks welded on the plate, the rest on the feeder).
        ponytail: a reset re-creates the Example (measured ~1.7 s, 4 s for the first one of a process); restoring a saved state would also need the
        welds, collision groups, breaker, arm queues and the recaptured CUDA graphs, and a new context needs a new model anyway.
        `press`: (scale, fx, fy) of the design-press fixture. It lives in the captured CUDA graph, and re-capturing a live Example crashes (CUDA
        illegal access, tried), so a fixture means building the Example with it."""
        self.ex = self.cams = self.mon = self.det = None
        gc.collect()
        args = argparse.Namespace(front=None, side=None, top=None, width=None, depth=None, name=structure, shape=None, structure=None,
                                  bricks=bricks_of(structure), strategy="none", brace_model="grasp_lp", plan=None, spec="rl_env",
                                  start_step=step, only_step=only_step, brace_json=None, press_scale=press and press[0], press_lateral=press and tuple(press[1:]),
                                  fixture_hold=False, exe=self.exe, viewer="null", legacy_brace=False, show_cameras=False,
                                  seed=seed, rl_control=rl_control)           # spec: no plans/ and results/ writes
        with contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO()):
            self.ex = D.Example(newton.viewer.ViewerNull(num_frames=10 ** 9), args)
            self.mon = self.ex.monitor = ContactMonitor(self.ex)
            self.cams = Cameras(self.ex.model, seed=seed)
        self.step_n, self.seed = step, seed
        self.seq = tuple(P.sequence(bricks_of(structure)))
        return self.ex

    def advance(self, frames, until=None):
        """`frames` physics frames (1/60 s each); stops early when `until()` is true. Returns the frames run."""
        ex = self.ex
        with contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO()):
            for i in range(frames):
                ex.step()
                if until is not None and until():
                    return i + 1
        return frames

    # -- what a policy may see -------------------------------------------------------------------------------------
    def sense(self, features=None):
        """The ObsInputs. Reads: joint state and FK (proprioception), the cameras' captures, the plan. Nothing else."""
        ex = self.ex
        q, qd = ex.state_0.joint_q.numpy()[:18], ex.state_0.joint_qd.numpy()[:18]     # the two arms' 9 joints each (the bricks' free joints follow)
        fk = [ex.hand_fk(0), ex.hand_fk(1)]
        imgs = {"top": self.cams.capture("top", state=ex.state_0)}
        top = imgs["top"]
        hands = [dict(name="AB"[k], pos=fk[k][:3], R=Rot.from_quat(fk[k][3:7]).as_matrix(), opening=q[9 * k + 7:9 * k + 9]) for k in range(2)]
        res = TD.detect(top["rgb"], top["depth"], top["intrinsics"], top["pos"], Rot.from_quat(top["quat_xyzw"]))
        self.det = TD.track_held(TD.remember(self.det, res), self.det, hands)
        det = dict(self.det, bricks=[dict(b) for b in self.det["bricks"]], held={k: dict(b) for k, b in self.det["held"].items()})
        if self.spec.images:
            for c in self.spec.cameras:
                if c not in imgs:
                    imgs[c] = self.cams.capture(c, fk[int(c[-1] == "B")], state=ex.state_0)
        s = ex.plan["sequence"][self.step_n]
        tgt, yaw, _ = ex.target[s["brick_id"]]
        L, W = ex.dims[s["brick_id"]]
        task = dict(x=tgt[0], y=tgt[1], z=tgt[2], yaw=yaw, L=L, W=W, step=self.step_n, n=len(ex.plan["sequence"]))
        return ObsInputs(q, qd, fk, det, imgs if self.spec.images else None, task, features)

    def observe(self, features=None):
        return self.spec.encode(self.sense(features))

    # -- ground truth: reward, termination, info only --------------------------------------------------------------
    def privileged(self):
        """Ground truth for scoring: the current brick's true pose, B's TCP, the gate numbers."""
        ex = self.ex
        bid = ex.plan["sequence"][self.step_n]["brick_id"]
        q = ex.state_0.body_q.numpy()
        b = q[ex.body[bid]]
        tcp = q[D.EE + ex.n_arm_bodies][:3]
        tgt, yaw, square = ex.target[bid]
        dyaw = abs(D.wrap(D.yaw_of(b[3:]) - yaw, math.pi / 2 if square else math.pi))
        tilt = math.acos(float(np.clip(D.rotate(b[3:], np.array([0, 0, 1.0]))[2], -1, 1)))
        return dict(brick_id=bid, brick_pos=b[:3].copy(), brick_yaw_err=dyaw, tilt=tilt, tcp_B=tcp.copy(), target=np.array(tgt),
                    slot=ex.feeder[bid].copy(), gate=ex.gate(bid)[1:6], placed=self.step_n in ex.placed, failure=ex.failure,
                    arm_arm=sum(e["class"] == "arm_arm" for e in self.mon.raw) + sum(k[0] == "arm_arm" for k in self.mon._active),
                    sim_time=ex.sim_time, frame=ex.frame)


def _spec(name, kw):
    """The EnvSpec of an instance (how to rebuild it). nondeterministic: the GPU solver is not bitwise reproducible, identical runs differ by ~1e-6."""
    return EnvSpec(name + "-v0", "rl_env:%sEnv" % name, nondeterministic=True, kwargs={k: v for k, v in kw.items() if k not in ("self", "__class__")})


def _sim_seed(env, seed):
    return int(seed) if seed is not None else int(env.np_random.integers(2 ** 31 - 1))


# --- env 1: one decision per brick ---------------------------------------------------------------------------------
class BrickStepEnv(gym.Env):
    """One step-episode of the RL-bracing / RL-placing plans (plan_v4_rl_brace 2.1-2.5): reset builds the scene at the nominal start of a context
    (structure, step n: bricks < n welded on the plate, step n runs alone from park); step(action) sets that step's brace and press
    fixture, runs the step to completion with the Example's own queue (as --start-step N --only-step does) and returns terminated = True.

    action  Dict(brace  Discrete(2)       0 = no brace, 1 = brace
                 brace_pose Box(3)        (y, z, lean_deg) in the cell frame, the units of brace_bandit.actions; used when brace = 1. With
                                          snap_to_candidates (default) it is moved to the nearest member of A(c) (the candidate (y, z) by distance, the
                                          lean to the nearest of LEANS), so the executed action stays in the plan's frame; info records both
                 press  Box(3))           (scale, fx, fy): the design-press fixture of --press-scale / --press-lateral (the load of the step's
                                          supports at `scale` x design press, N lateral force). scale <= 0 = no fixture
    reward  s - 0.05 * 1[brace applied], s = 1 iff no failure, the step ran to its end and the contact monitor logged no arm_arm event
    obs     the common Dict + "features" (brace_bandit.features with action None, the 15 context numbers)
    info    success, failure, and under "privileged": u2_peak (max over welds and frames of min(u_t, u_t-1)), arm_arm_events, gate numbers, ...

    The drop-and-press placement primitive of plan_v4_rl_place is NOT implemented in the simulator (queue_place has only the committed insert).
    A `primitive` action is added to this Dict when that primitive exists; nothing here fakes it.
    """
    metadata = {"render_modes": []}
    BRACE_BOUNDS = ((-0.06, 0.0152, -45.0), (0.06, 0.09, 45.0))      # y, z, lean over the candidates of cube, arch, S3 and the corbel family (|y| <= 0.044, z <= 0.083)
    PRESS_BOUNDS = ((0.0, -3.0, -3.0), (1.3, 3.0, 3.0))               # the band of plan_v4_rl_brace 2.2 (lambda 0.5-1.3) and lateral magnitude <= 3 N
    MAX_FRAMES = 6000                                                  # 100 s of sim (experiments/v4_brace.NUM_FRAMES); a step-episode takes 12-20 s

    def __init__(self, contexts=(("cube", 7), ("arch", 10), ("S3", 13)), images=True, cameras=CAM_NAMES, image_stride=4, n_max=16,
                 exe="E0", snap_to_candidates=True, verbose=False):
        self.spec = _spec("BrickStep", locals())
        self.contexts, self.snap, self.exe_name = list(contexts), snap_to_candidates, exe
        self.obs_spec = ObsSpec(n_max, images, cameras, image_stride, features=True)
        self.sim = Sim(self.obs_spec, exe, verbose)
        self.observation_space = self.obs_spec.space()
        self.action_space = spaces.Dict({"brace": spaces.Discrete(2),
                                         "brace_pose": spaces.Box(np.array(self.BRACE_BOUNDS[0], np.float32), np.array(self.BRACE_BOUNDS[1], np.float32)),
                                         "press": spaces.Box(np.array(self.PRESS_BOUNDS[0], np.float32), np.array(self.PRESS_BOUNDS[1], np.float32))})
        self.ctx = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o = options or {}
        if "structure" in o:
            structure, step = o["structure"], int(o["step"])
        else:
            structure, step = self.contexts[int(self.np_random.integers(len(self.contexts)))]
        self.ctx = (structure, step)
        self.sim.build(structure, step, _sim_seed(self, seed), rl_control=False, only_step=True)
        self.bricks = self.sim.seq
        # step 0 has no placed brick: brace_bandit's contexts start at step 1, so its features are zeros
        self._feat = BB.features(self.bricks, step, None, BB.EXE[self.exe_name])[:N_CTX] if step > 0 else np.zeros(N_CTX)
        self.done_step = False
        return self.sim.observe(self._feat), {"structure": structure, "step": step}

    def candidates(self):
        """A(c) of the current context (brace_bandit.actions): None and the (y, z, lean_deg) triples."""
        return BB.actions({"bricks": self.bricks, "step": self.ctx[1]}) if self.ctx[1] else [None]     # step 0: nothing to brace

    def _applied(self, brace_pose):
        """The brace executed for a brace_pose request: the nearest candidate (or the request as it is), None if A(c) has no brace."""
        req = [float(x) for x in brace_pose]
        cands = [a for a in self.candidates() if a is not None]
        if not cands:
            return None
        if not self.snap:
            return req
        c = min(cands, key=lambda a: (a[0] - req[0]) ** 2 + (a[1] - req[1]) ** 2)
        same = [a for a in cands if a[:2] == c[:2]]
        return list(min(same, key=lambda a: abs(a[2] - req[2])))

    def step(self, action):
        assert not self.done_step, "BrickStepEnv: one step per episode; call reset()"
        sim = self.sim
        n = self.ctx[1]
        lam, fx, fy = (float(x) for x in action["press"])
        if lam > 0:                                            # the fixture is part of the captured graph: rebuild the Example with it (~2 s)
            sim.build(self.ctx[0], n, sim.seed, rl_control=False, only_step=True, press=(lam, fx, fy))
        ex = sim.ex
        s = ex.plan["sequence"][n]
        applied = self._applied(action["brace_pose"]) if int(action["brace"]) else None
        ex.set_brace(s, applied, ex.exe)                       # None = no brace (--brace-json null)
        diverged, t0 = None, time.perf_counter()
        try:
            sim.advance(self.MAX_FRAMES, until=lambda: (ex.done or ex.stopped) and ex.B.idle() and ex.A.idle())
        except RuntimeError as e:                              # the solver diverged (Example.step raises)
            diverged = str(e)
        sim.mon.finalize()
        row = ex.episode_row(sim.mon.events)
        success = bool(row["success"]) and diverged is None
        reward = float(success) - BB.BRACE_COST * (applied is not None)
        failure = ex.failure or ("divergence" if diverged else None) or (None if ex.done else "timeout")
        self.done_step = True
        info = {"structure": self.ctx[0], "step": n, "success": success, "failure": failure, "wall_s": time.perf_counter() - t0,
                "brace": {"requested": bool(int(action["brace"])), "applied": applied, "pose_in": [float(x) for x in action["brace_pose"]]},
                "press": [lam, fx, fy],
                "privileged": {"u2_peak": row["u2_peak"], "arm_arm_events": row["arm_arm_events"], "arm_arm_peak_N": row["arm_arm_peak_N"],
                               "last_gate": dict(ex.last_gate), "frames": ex.frame, "sim_time": ex.sim_time, "breaks": list(ex.breaker.breaks),
                               "brace_row": row["brace_row"], "episode_row": row}}
        return sim.observe(self._feat), reward, True, False, info


# --- env 2: low-level control of both hands -----------------------------------------------------------------------
class BrickReachEnv(gym.Env):
    """Pick the current step's brick with arm B and seat it on its target, driving both hands directly.

    action  Box(10) in [-1, 1]: for arm A then arm B, (dx, dy, dz, dyaw, grip). The first four scale to a per-tick delta of the hand target
            (dpos m, dyaw rad; world x, y, z and the hand's yaw about its base), grip is an ABSOLUTE finger target: -1 = shut (0 m per
            finger), +1 = open (cell/liveview.GRIP_OPEN, 0.03 m). Executed by cell/liveview.ManualCtl (workspace clamp, slew limit V_MAX, leash on
            the measured hand) through Example(rl_control=True) for `control_every` physics frames (default 6 = 10 Hz).
            The target is not pulled back toward the measured hand: a hand blocked by a contact keeps its (clamped) target.
    start   nominal start of `step` (bricks < step welded on the plate, the rest on the feeder), both hands parked, fingers 10 mm.
    snap    every frame the Example's own gate (Example.gate / snap) reads the current brick; when it passes the brick is welded and the episode ends.
    reward  per tick, privileged state allowed (all weights are constructor args):
              w_reach  * -|B's TCP - the brick|                  until the brick is lifted for the first time
              w_grasp  * +1                                      once, the first tick the brick is lifted (>= lift_dz over its slot, in the fingers)
              w_place  * -(|brick - target| + w_yaw * yaw error) while it is lifted
              w_time   per tick (negative)
              w_success on the tick the gate passes (terminates)
              w_break / w_arm_arm / w_drop  (negative) on a joint break / the first arm_arm contact / the brick dropped or knocked off (each terminates)
            dropped: lifted before, now out of the fingers for 3 ticks; knocked off: tilted > 45 deg or moved > 50 mm from its slot unlifted.
    truncation  at max_steps ticks (default 600 = 60 s).
    obs     the common Dict. info: privileged numbers under "privileged"; "events" lists what ended the episode.
    """
    metadata = {"render_modes": []}

    def __init__(self, structure="cube", step=0, control_every=6, max_steps=600, dpos=0.01, dyaw=0.1, images=True, cameras=CAM_NAMES,
                 image_stride=4, n_max=16, exe="E0", w_reach=0.5, w_grasp=5.0, w_place=1.0, w_yaw=0.05, w_time=-0.01, w_success=100.0,
                 w_break=-20.0, w_arm_arm=-20.0, w_drop=-20.0, lift_dz=0.006, verbose=False):
        self.spec = _spec("BrickReach", locals())
        self.structure, self.step_n, self.every, self.max_steps, self.dpos, self.dyaw = structure, step, control_every, max_steps, dpos, dyaw
        self.w = dict(reach=w_reach, grasp=w_grasp, place=w_place, yaw=w_yaw, time=w_time, success=w_success, brk=w_break, aa=w_arm_arm, drop=w_drop)
        self.lift_dz, self.exe_name = lift_dz, exe
        self.obs_spec = ObsSpec(n_max, images, cameras, image_stride)
        self.sim = Sim(self.obs_spec, exe, verbose)
        self.observation_space = self.obs_spec.space()
        self.action_space = spaces.Box(-1.0, 1.0, (10,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o = options or {}
        self.structure, self.step_n = o.get("structure", self.structure), int(o.get("step", self.step_n))
        self.sim.build(self.structure, self.step_n, _sim_seed(self, seed), rl_control=True, only_step=False)
        self.t, self.lifted, self.out_ticks, self.arm_arm0 = 0, False, 0, 0
        self.slot = self.sim.ex.feeder[self.sim.ex.plan["sequence"][self.step_n]["brick_id"]].copy()
        return self.sim.observe(), {"structure": self.structure, "step": self.step_n}

    def _apply(self, a):
        man = self.sim.ex.man
        a = np.clip(np.asarray(a, np.float64), -1, 1).reshape(2, 5)
        for k in (0, 1):
            man.nudge(k, a[k, :3] * self.dpos, a[k, 3] * self.dyaw, grip=LV.GRIP_SHUT + (a[k, 4] + 1) / 2 * (LV.GRIP_OPEN - LV.GRIP_SHUT))

    def step(self, action):
        sim, ex, w = self.sim, self.sim.ex, self.w
        self._apply(action)
        events, diverged = [], False
        try:
            sim.advance(self.every, until=lambda: self.step_n in ex.placed or ex.failure is not None)
        except RuntimeError:
            diverged = True
        self.t += 1
        pv = sim.privileged()
        dist_tcp = float(np.linalg.norm(pv["tcp_B"] - (pv["brick_pos"] + [0, 0, D.GRASP_DZ])))
        in_hand = dist_tcp < 0.025
        lifted = pv["brick_pos"][2] > self.slot[2] + 0.0005 + self.lift_dz and in_hand
        r = w["time"]
        if not (self.lifted or lifted):
            r -= w["reach"] * float(np.linalg.norm(pv["tcp_B"] - pv["brick_pos"]))
        if lifted and not self.lifted:
            r += w["grasp"]
        self.lifted |= lifted
        if lifted:
            r -= w["place"] * (float(np.linalg.norm(pv["brick_pos"] - pv["target"])) + w["yaw"] * pv["brick_yaw_err"])
        self.out_ticks = 0 if (in_hand or not self.lifted) else self.out_ticks + 1
        if pv["placed"]:
            r += w["success"]
            events.append("success")
        if diverged or pv["failure"] == "joint_break":
            r += w["brk"]
            events.append("joint_break" if not diverged else "divergence")
        if pv["arm_arm"] > self.arm_arm0:
            r += w["aa"]
            events.append("arm_arm")
        self.arm_arm0 = pv["arm_arm"]
        slot_move = float(np.linalg.norm(pv["brick_pos"][:2] - self.slot[:2]))
        if self.out_ticks >= 3 or pv["tilt"] > math.pi / 4 or (not self.lifted and slot_move > 0.05):
            r += w["drop"]
            events.append("dropped" if self.out_ticks >= 3 else "knocked_off")
        terminated = bool(events)
        truncated = not terminated and self.t >= self.max_steps
        info = {"success": "success" in events, "events": events, "failure": ex.failure,
                "privileged": {k: pv[k] for k in ("brick_pos", "tcp_B", "target", "gate", "arm_arm", "sim_time")} | {"lifted": self.lifted, "dist_tcp": dist_tcp}}
        return sim.observe(), float(r), terminated, truncated, info


# --- a scripted controller (privileged info lives here only) ------------------------------------------------------------
class ScriptedReach:
    """Drives arm B to the current brick, grasps, carries it to the target, descends and seats it: the sanity check of the reward and the snap
    path. Reads the Example's ground truth (brick pose, plan grasp) because it is a test script, not a policy."""

    def __init__(self, env):
        self.env, self.phase, self.hold = env, "over", 0

    def act(self):
        env = self.env
        ex = env.sim.ex
        n = env.step_n
        s = ex.plan["sequence"][n]
        pv = env.sim.privileged()
        man, tgt_p, slot = ex.man, ex.man.tgt[1], ex.feeder[s["brick_id"]]
        width = s["grasp"]["grip_width_m"] + 0.0012
        g_open, g_shut = width / 2 + 0.002, width / 2 - 0.0015
        yaw = D.wrap(math.radians(s["grasp"]["yaw_offset_deg"]) - D.ARM_B[1], math.pi)   # hand yaw about B's base (a gripper is 180-degree symmetric)
        tcp, brick, tgt = pv["tcp_B"], pv["brick_pos"], pv["target"]
        zt = ex.z_travel

        def go(goal, grip, tol=0.0025):
            d = np.clip((goal - tgt_p["p"]) / env.dpos, -1, 1)
            dy = np.clip((yaw - tgt_p["yaw"]) / env.dyaw, -1, 1)
            g = 2 * (grip - LV.GRIP_SHUT) / (LV.GRIP_OPEN - LV.GRIP_SHUT) - 1
            a = np.zeros(10)
            a[5:] = [*d, dy, g]
            return a, np.linalg.norm(tcp - goal) < tol and abs(yaw - tgt_p["yaw"]) < 0.02

        ph = self.phase
        if ph == "over":                                                    # rise, then over the brick
            goal = np.array([tcp[0], tcp[1], zt]) if tcp[2] < zt - 0.01 and np.linalg.norm(tcp[:2] - slot[:2]) > 0.02 else np.array([slot[0], slot[1], zt])
            a, ok = go(goal, g_open, 0.004)
            nxt = "hover"
        elif ph == "hover":
            a, ok = go(slot + [0, 0, D.GRASP_DZ + D.SETTLE_DZ], g_open, 0.0008)
            nxt = "descend"
        elif ph == "descend":
            a, ok = go(slot + [0, 0, D.GRASP_DZ], g_open, 0.0008)
            nxt = "close"
        elif ph == "close":
            a, _ = go(slot + [0, 0, D.GRASP_DZ], g_shut)
            self.hold += 1
            ok, nxt = self.hold >= 8, "lift"
        elif ph == "lift":
            a, ok = go(np.array([tcp[0], tcp[1], zt]), g_shut, 0.004)
            nxt = "transport"
        else:                                                               # the brick, not the hand, goes on the target: hand = target + (hand - brick)
            off = tcp - brick - [0, 0, D.GRASP_DZ]
            xy = tgt[:2] + off[:2]
            if ph == "transport":
                a, ok = go(np.array([*xy, zt]), g_shut, 0.004)
                ok, nxt = ok and np.linalg.norm(brick[:2] - tgt[:2]) < 0.004, "pre"
            elif ph == "pre":
                a, ok = go(np.array([*xy, tgt[2] + 0.025 + D.GRASP_DZ]), g_shut, 0.002)
                ok, nxt = ok and np.linalg.norm(brick[:2] - tgt[:2]) < 0.0007, "insert"
            else:
                a, ok = go(np.array([*xy, tgt[2] + D.GRASP_DZ - D.PRESS]), g_shut, 0.0005)
                nxt = "insert"
        if ok:
            self.phase = nxt
        return a.astype(np.float32)


def run_scripted(env, max_ticks=None):
    """One scripted episode: (return, ticks, info of the last step)."""
    env.reset(seed=0)
    pol, ret, info = ScriptedReach(env), 0.0, {}
    for t in range(max_ticks or env.max_steps):
        _, r, term, trunc, info = env.step(pol.act())
        ret += r
        if term or trunc:
            return ret, t + 1, info, pol.phase
    return ret, max_ticks or env.max_steps, info, pol.phase


def demo():
    t0 = time.perf_counter()
    env = BrickReachEnv(images=False)
    ret, ticks, info, phase = run_scripted(env)
    print("BrickReachEnv scripted: return %.2f in %d ticks (%.1f s), events %s, last phase %s" % (ret, ticks, time.perf_counter() - t0, info["events"], phase))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        print(__doc__)
