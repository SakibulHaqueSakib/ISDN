"""Stepping the twin: controllers at 500 Hz, joint model every physics step.

    sim = Sim(cell)                     # arms at READY_Q, grippers open
    sim.goto("B", pos, R, duration)     # min-jerk Cartesian move (blocking)
    sim.run(seconds)

Motions are open-loop min-jerk trajectories fed to the impedance law with
velocity and acceleration feed-forward; contact moves stop on a predicate.
"""

import math

import mujoco
import numpy as np

from sim.mj import control as C
from sim.mj import scene as S

CTRL_EVERY = 2          # physics steps per control update -> 500 Hz
FT_TAU = 0.005          # s, F/T low-pass: a real wrist sensor is band-limited, and the
                        # raw constraint force spikes for a single step when a joint
                        # engages (measured 3.2 kN for 2 ms on a 46 N press)
PARK = (0.30, 0.0, 0.30)  # TCP in each arm's base frame while idle


class Sim:
    def __init__(self, cell, clutch=None, ft_noise=None, rng=None):
        self.cell, self.m, self.d = cell, cell.model, cell.data
        self.clutch = clutch
        self.arms = {k: C.Arm(self.m, k) for k in ("A", "B")}
        self.park = {}
        for k, arm in self.arms.items():
            (bx, by, bz), yaw = S.ARM_BASE[k]
            c, s_ = math.cos(yaw), math.sin(yaw)
            p = np.array([bx + PARK[0] * c, by + PARK[0] * s_, PARK[2]])
            R = C.down_rot(yaw + math.pi / 2)        # fingers close along the arm's x
            q, _ = C.ik(self.m, arm, p, R)
            self.park[k] = (p, R, q)
            self.d.qpos[arm.qadr] = q
            arm.q_null = q.copy()
            self.d.qpos[arm.finger_q] = 0.04
        mujoco.mj_forward(self.m, self.d)
        for arm in self.arms.values():
            arm.hold(self.d)
        self.Mfull = np.zeros((self.m.nv, self.m.nv))
        self._scratch = mujoco.MjData(self.m)
        self.k = 0
        self.hooks = []          # callables(sim) run every physics step
        self.ft_filt = {k: np.zeros(6) for k in self.arms}
        self.rng = rng or np.random.default_rng(0)
        self.ft_noise = ft_noise  # (sigma_N, sigma_Nm) or None
        self.diverged = False

    @property
    def t(self):
        return self.d.time

    def step(self):
        m, d = self.m, self.d
        if self.k % CTRL_EVERY == 0:
            mujoco.mj_fullM(m, self.Mfull, d.qM)
            for arm in self.arms.values():
                arm.apply(d, self.Mfull)
        mujoco.mj_step(m, d)
        self.k += 1
        if self.k % CTRL_EVERY == 0:
            dt = m.opt.timestep * CTRL_EVERY
            a = dt / (FT_TAU + dt)
            for k, arm in self.arms.items():
                self.ft_filt[k] += a * (arm.ft(d) - self.ft_filt[k])
        if self.clutch is not None:
            self.clutch.update(d)
        for h in self.hooks:
            h(self)

    def run(self, seconds, until=None):
        """Step for `seconds` of sim time, or until until(sim) is true.
        Returns True if `until` fired."""
        n = int(round(seconds / self.m.opt.timestep))
        for _ in range(n):
            self.step()
            if until is not None and self.k % CTRL_EVERY == 0 and until(self):
                return True
        if not np.isfinite(self.d.qpos).all():
            self.diverged = True
            raise FloatingPointError("simulation diverged at t=%.3f" % self.t)
        return False

    def noslip(self, on):
        """MuJoCo's no-slip post-solver, on while forces are high (a press, a
        brace hold): contact friction otherwise creeps -- a gripped brick slid
        3.7 mm under a 46 N press -- but it triples the cost of a step."""
        self.m.opt.noslip_iterations = 5 if on else 0

    def ft(self, name, raw=False):
        """Wrist wrench on the hand from the environment (world frame), low-
        passed at FT_TAU; with ft_noise set, plus sensor noise (§5.5)."""
        w = self.arms[name].ft(self.d) if raw else self.ft_filt[name].copy()
        if self.ft_noise is not None:
            w = w + np.concatenate([self.rng.normal(0, self.ft_noise[0], 3),
                                    self.rng.normal(0, self.ft_noise[1], 3)])
        return w

    def goto(self, name, pos, R=None, duration=1.0, until=None, settle=0.0, posture=True):
        """Min-jerk move of the TCP to pos (and orientation R). Blocking.

        posture: steer the null-space posture toward the IK solution of the
        goal (seeded from the current joints), so a sequence of moves keeps
        the arm in one consistent configuration instead of drifting toward
        its joint limits.
        until(sim) -> True aborts the move (and latches the current pose).
        Returns True if the move was aborted by `until`.
        """
        arm = self.arms[name]
        p0 = arm.x_d.copy()
        R0 = arm.R_d.copy()
        R1 = R0 if R is None else np.asarray(R)
        if posture:
            q, err = C.ik(self.m, arm, np.asarray(pos, float), R1, q0=arm.q(self.d), iters=60)
            if err < 0.005:
                arm.q_null = q
        dr = C.rot_error(R1, R0)
        p1 = np.asarray(pos, float)
        dt = self.m.opt.timestep * C_EVERY
        n = max(1, int(round(duration / dt)))
        for i in range(1, n + 1):
            s = i / n
            sd = C.min_jerk(s)
            # derivative of min-jerk for velocity feed-forward
            vel = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
            acc = (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2
            arm.x_d = p0 + sd * (p1 - p0)
            arm.R_d = C.rotvec_to_mat(sd * dr) @ R0
            arm.v_d = np.concatenate([vel * (p1 - p0), vel * dr])
            arm.a_d = np.concatenate([acc * (p1 - p0), acc * dr])
            for _ in range(C_EVERY):
                self.step()
            if until is not None and until(self):
                arm.v_d[:] = 0
                arm.a_d[:] = 0
                return True
        arm.v_d[:] = 0
        arm.a_d[:] = 0
        if settle:
            self.run(settle)
        return False


    def goto_joint(self, name, pos, R, duration=None, seed=None, settle=0.0):
        """Large move: IK the goal (seeded from `seed`, default the park
        configuration, so solutions stay on one branch), interpolate in JOINT
        space with min-jerk, and feed FK of that path to the impedance law with
        the posture target riding along. This is how a motion planner's
        output (cuRobo's, §2.5) would be executed; straight Cartesian lines
        from an arbitrary configuration drive the wrist into its limits.
        Returns the IK residual (m)."""
        arm = self.arms[name]
        q0 = arm.q(self.d)
        seed = self.park[name][2] if seed is None else seed
        q1, err = C.ik(self.m, arm, np.asarray(pos, float), np.asarray(R), q0=seed)
        if err > 0.002:
            q1b, err_b = C.ik(self.m, arm, np.asarray(pos, float), np.asarray(R), q0=q0)
            if err_b < err:
                q1, err = q1b, err_b
        if duration is None:
            duration = float(np.clip(np.abs(q1 - q0).max() / 0.8, 0.6, 3.0))
        scratch = self._scratch
        dt = self.m.opt.timestep * C_EVERY
        n = max(1, int(round(duration / dt)))
        prev = None
        for i in range(1, n + 1):
            q = q0 + C.min_jerk(i / n) * (q1 - q0)
            scratch.qpos[arm.qadr] = q
            mujoco.mj_kinematics(self.m, scratch)
            x = scratch.site_xpos[arm.site].copy()
            Rm = scratch.site_xmat[arm.site].reshape(3, 3).copy()
            arm.x_d, arm.R_d, arm.q_null = x, Rm, q
            if prev is not None:
                arm.v_d = np.concatenate([(x - prev[0]) / dt, C.rot_error(Rm, prev[1]) / dt])
            prev = (x, Rm)
            for _ in range(C_EVERY):
                self.step()
        arm.v_d[:] = 0
        arm.a_d[:] = 0
        arm.x_d = np.asarray(pos, float).copy()
        arm.R_d = np.asarray(R).copy()
        if settle:
            self.run(settle)
        return err


C_EVERY = CTRL_EVERY
