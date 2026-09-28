"""Task-space impedance control for the Pandas -- master_report §2.5.

    RL policy / scripted skill  20 Hz  -> delta pose in EE frame (PLAI below)
    task-space impedance       500 Hz  -> joint torques
    MuJoCo                    1000 Hz

Impedance, not full inertia shaping: F = K e - D (v - v_d) + Lambda a_d + f_ff,
tau = J^T F + N^T tau_posture + bias. K is the §2.5 stiffness in physical units
(1200/1200/600 N/m, 40 N*m/rad), so "600 N/m in z" means what it says: a 6 N
contact deflects the hand 10 mm, which is what lets a brick find its socket
instead of jamming. D = 2 zeta sqrt(K Lambda_ii) per axis, zeta = 1.

f_ff is the force feedforward a press uses (hybrid impedance/force): the
scripted inserter and the brace hold both command a force along the stud
axis while keeping the lateral axes compliant.
"""

import math

import mujoco
import numpy as np

KP_POS = np.array([1200.0, 1200.0, 600.0])     # N/m, §2.5 -- deliberately soft in z
KP_ROT = np.array([40.0, 40.0, 40.0])          # N*m/rad
ZETA = 1.0
NULL_KP = 25.0                                  # 1/s^2 posture task toward mid-range
READY_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
NULL_RATE = 0.5              # rad/s, posture reference slew
CTRL_DT = 0.002              # s, the controller period (runtime.CTRL_EVERY x timestep)
GRIP_SPEED = 0.08            # m/s per finger while closing (Franka hand: <= 0.1 m/s)
JOINT_DAMPING = 1.0                             # menagerie panda passive damping, compensated
_EYE6, _EYE7 = np.eye(6), np.eye(7)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def rot_error(R_d, R):
    """Rotation vector (world frame) taking R to R_d."""
    q = mat_to_quat(R_d @ R.T)
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rotvec_to_mat(r):
    q = np.zeros(4)
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.eye(3)
    mujoco.mju_axisAngle2Quat(q, np.asarray(r, float) / a, a)
    return quat_to_mat(q)


def down_rot(yaw, tilt=(0.0, 0.0, 0.0)):
    """Hand orientation: fingers pointing down, rotated `yaw` about world z,
    then leaned by the rotation vector `tilt` (world frame).

    Hand frame: z along the fingers, y along the finger travel. With yaw = 0
    the fingers close along world y.
    """
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # z down
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return rotvec_to_mat(tilt) @ Rz @ Rx


class Arm:
    """Index bookkeeping and the impedance law for one Panda in a Cell."""

    def __init__(self, model, prefix, kp_pos=KP_POS, kp_rot=KP_ROT):
        self.m, self.prefix = model, prefix
        self.joint_ids = [model.joint("%s/joint%d" % (prefix, k)).id for k in range(1, 8)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dofs = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.finger_q = np.array([model.jnt_qposadr[model.joint("%s/finger_joint%d" % (prefix, k)).id]
                                  for k in (1, 2)])
        self.finger_v = np.array([model.jnt_dofadr[model.joint("%s/finger_joint%d" % (prefix, k)).id]
                                  for k in (1, 2)])
        self.motors = np.array([model.actuator("%s/motor%d" % (prefix, k)).id for k in range(1, 8)])
        self.grip_act = model.actuator(prefix + "/grip").id
        self.limits = model.actuator_ctrlrange[self.motors, 1].copy()
        self.site = model.site(prefix + "/tcp").id
        self.ft_site = model.site(prefix + "/ft").id
        self.hand = model.body(prefix + "/hand").id
        self.ft_force_adr = model.sensor_adr[model.sensor(prefix + "/ft_force").id]
        self.ft_torque_adr = model.sensor_adr[model.sensor(prefix + "/ft_torque").id]
        # the hand subtree's weight, for taring the F/T (quasi-static)
        self.hand_mass = float(model.body_subtreemass[self.hand])
        self.kp_pos, self.kp_rot = np.array(kp_pos, float), np.array(kp_rot, float)
        self.q_null = READY_Q.copy()
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))
        self._jdp = np.zeros((3, model.nv))
        self._jdr = np.zeros((3, model.nv))
        # commands
        self.x_d = None
        self.R_d = None
        self.v_d = np.zeros(6)
        self.a_d = np.zeros(6)
        self.f_ff = np.zeros(6)
        self.grip_open = 0.04          # per-finger opening target, m
        self.grip_force = 15.0         # N per finger when closing on something
        self._close_ref = None         # ramped closing target (grip_ctrl)
        self._q_ref = None             # slewed posture reference (torque)

    # --- state -------------------------------------------------------------
    def q(self, d):
        return d.qpos[self.qadr].copy()

    def tcp(self, d):
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def opening(self, d):
        """Per-finger opening (m); the pad gap is about 2x this."""
        return float(d.qpos[self.finger_q].mean())

    def jac(self, d):
        mujoco.mj_jacSite(self.m, d, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dofs], self._jr[:, self.dofs]])

    def tcp_vel(self, d):
        return self.jac(d) @ d.qvel[self.dofs]

    def ft(self, d):
        """Contact wrench ON the hand from the environment, world frame, about
        the F/T site: sensor reading minus the hand's own weight (tared)."""
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        f_s = d.sensordata[self.ft_force_adr:self.ft_force_adr + 3]
        t_s = d.sensordata[self.ft_torque_adr:self.ft_torque_adr + 3]
        # MuJoCo's force sensor is the force the child subtree receives from
        # its parent, in the site frame. The environment's force on the hand is
        # (subtree weight + inertial) minus that; quasi-static tare:
        mg = self.hand_mass * self.m.opt.gravity
        f_env = -(R @ f_s) - mg
        r = d.subtree_com[self.hand] - d.site_xpos[self.ft_site]
        rxg = np.array([r[1] * mg[2] - r[2] * mg[1], r[2] * mg[0] - r[0] * mg[2],
                        r[0] * mg[1] - r[1] * mg[0]])
        t_env = -(R @ t_s) - rxg
        return np.concatenate([f_env, t_env])

    # --- control ---------------------------------------------------------------
    def hold(self, d):
        """Latch the current TCP pose as the target (no motion)."""
        self.x_d, self.R_d = self.tcp(d)
        self.v_d[:] = 0
        self.a_d[:] = 0
        self.f_ff[:] = 0

    def torque(self, d, Mfull):
        J = self.jac(d)
        a, b = self.dofs[0], self.dofs[-1] + 1       # an arm's dofs are contiguous
        M = Mfull[a:b, a:b]
        Minv = np.linalg.inv(M)
        MinvJT = Minv @ J.T
        Lam = np.linalg.inv(J @ MinvJT + 1e-6 * _EYE6)
        x, R = self.tcp(d)
        qd = d.qvel[self.dofs]
        v = J @ qd
        e = np.concatenate([self.x_d - x, rot_error(self.R_d, R)])
        K = np.concatenate([self.kp_pos, self.kp_rot])
        D = 2 * ZETA * np.sqrt(K * np.clip(np.diag(Lam), 1e-6, None))
        # -Lambda Jdot qdot: without it, null-space motion leaks into the task
        # (0.23 mm at 0.46 rad/s of posture swing)
        mujoco.mj_jacDot(self.m, d, self._jdp, self._jdr, d.site_xpos[self.site], self.m.site_bodyid[self.site])
        Jdqd = np.concatenate([self._jdp[:, self.dofs] @ qd, self._jdr[:, self.dofs] @ qd])
        F = K * e - D * (v - self.v_d) + Lam @ (self.a_d - Jdqd) + self.f_ff
        tau = J.T @ F
        # null-space posture (dynamically consistent)
        Jbar = MinvJT @ Lam
        N = _EYE7 - J.T @ Jbar.T
        q = d.qpos[self.qadr]
        # the posture reference slews at NULL_RATE toward q_null (goto()
        # re-targets the posture at every move; a 0.4 rad step would whip the
        # elbow)
        if self._q_ref is None:
            self._q_ref = q.copy()
        step = NULL_RATE * CTRL_DT
        self._q_ref += np.clip(self.q_null - self._q_ref, -step, step)
        tau0 = M @ (NULL_KP * (self._q_ref - q) - 2 * math.sqrt(NULL_KP) * qd)
        tau += N @ tau0
        tau += d.qfrc_bias[self.dofs] + JOINT_DAMPING * qd
        return np.clip(tau, -self.limits, self.limits)

    def grip_ctrl(self, d):
        """Force on the split tendon (half of it reaches each finger).

        Closing (grip_open below the current opening): the position target
        ramps shut at GRIP_SPEED, and once the pads are held 2 mm behind it
        (they are on the brick) the law is pure FORCE control at the full
        grip_force per finger -- a position loop never saturates on a 16 mm
        brick and squeezed 15.6 N where 27.6 N was asked for. The ramp is the
        Franka hand's speed limit: unlimited force control shut the fingers
        at 1.4 m/s and flicked a 2x4 out of the feeder (S3 step 12); a
        velocity servo chatters on fingers this light at 500 Hz.
        Opening is a position loop.
        """
        o = self.opening(d)
        od = float(d.qvel[self.finger_v].mean())
        if self.grip_open < o - 0.0005 or (self._close_ref is not None and self.grip_open < o):
            if self._close_ref is None:
                self._close_ref = o
            self._close_ref = max(self.grip_open, self._close_ref - GRIP_SPEED * 0.002)
            if o - self._close_ref > 0.002:
                return -2.0 * self.grip_force - 20.0 * od
            f = 4000.0 * (self._close_ref - o) - 60.0 * od
            return float(np.clip(f, -2 * self.grip_force, 2 * 40.0))
        self._close_ref = None
        f = 4000.0 * (self.grip_open - o) - 60.0 * od
        return float(np.clip(f, -2 * self.grip_force, 2 * 40.0))

    def apply(self, d, Mfull):
        d.ctrl[self.motors] = self.torque(d, Mfull)
        d.ctrl[self.grip_act] = self.grip_ctrl(d)


class PLAI:
    """Policy-level action integrator ([IndustReal]): accumulate clipped
    pose increments into the impedance target instead of commanding absolute
    poses, so a noisy policy cannot command a jump."""

    def __init__(self, arm, max_dpos=0.002, max_drot=math.radians(2.0), leash=0.01):
        self.arm, self.max_dpos, self.max_drot, self.leash = arm, max_dpos, max_drot, leash

    def step(self, d, action, frame_R=None):
        """action: 6-vector in [-1, 1] (dx dy dz droll dpitch dyaw), EE frame."""
        a = np.clip(np.asarray(action, float), -1, 1)
        R = frame_R if frame_R is not None else self.arm.tcp(d)[1]
        dp = R @ (a[:3] * self.max_dpos)
        dr = R @ (a[3:] * self.max_drot)
        x, _ = self.arm.tcp(d)
        target = self.arm.x_d + dp
        # keep the target on a leash around the hand so an integrated target
        # cannot wind up far past a contact
        off = target - x
        n = np.linalg.norm(off)
        if n > self.leash:
            target = x + off / n * self.leash
        self.arm.x_d = target
        self.arm.R_d = rotvec_to_mat(dr) @ self.arm.R_d


def min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def ik(model, arm, pos, R, q0=None, iters=200, tol=1e-5):
    """Damped least-squares IK for one arm's TCP on a scratch MjData.

    Other bodies are ignored (their qpos stays at qpos0). Returns the 7 joint
    angles and the residual position error (m).
    """
    d = mujoco.MjData(model)
    q = READY_Q.copy() if q0 is None else np.array(q0, float)
    lo, hi = model.jnt_range[arm.joint_ids, 0], model.jnt_range[arm.joint_ids, 1]
    err = np.inf
    for _ in range(iters):
        d.qpos[arm.qadr] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        x = d.site_xpos[arm.site]
        Rc = d.site_xmat[arm.site].reshape(3, 3)
        e = np.concatenate([pos - x, rot_error(R, Rc)])
        err = float(np.linalg.norm(e[:3]))
        if err < tol and np.linalg.norm(e[3:]) < 1e-4:
            break
        J = arm.jac(d)
        lam = 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        # gentle pull toward mid-range in the null space
        N = np.eye(7) - np.linalg.pinv(J) @ J
        dq += N @ (0.05 * (READY_Q - q))
        q = np.clip(q + dq, lo + 1e-3, hi - 1e-3)
    return q, err
