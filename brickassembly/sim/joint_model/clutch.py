"""The stud clutch as a joint model for the MuJoCo twin -- master_report §2.3.3.

    DISENGAGED --(gate)--> ENGAGING --(ramp 50 ms)--> MATED --(u >= 1)--> DISENGAGED

State is kept per BRICK for the gate (a brick seats onto all its supports at
once, as it does physically) and per CONNECTION for failure (one support can
let go while another holds).

DISENGAGED  studs inside the cavity (the brick's anti-stud face within
            STUD_H of seated and laterally aligned) engage the stud/cavity
            interference a placer must push through, n_studs * f_insert, as
            Coulomb friction on the connection tendon (static = kinetic):
              STICK  a lower LIMIT on the tendon's length, locked where the
                     brick is -- a position constraint, so a 1 g brick cannot
                     creep through it the way it creeps through MuJoCo's
                     velocity-based frictionloss below saturation;
              SLIP   once the lock's reaction exceeds n_studs * f_insert it
                     is released and frictionloss = n_studs * f_insert (which
                     saturates, i.e. is exactly Coulomb, while sliding) resists
                     until the brick stops, when it locks again.
            Push harder than the interference and the brick goes in; push
            less and it does not move. Both are internal constraints between
            the two bricks, so their force is equal and opposite on both.
gate        every §2.3.3 condition, relative to each support's seated frame:
            |dxy| < 1.2 mm, dz in [-0.3, +1.0] mm, tilt < 4 deg, yaw error to
            the brick's symmetry < 5 deg, axial push > 4 N, all held for 40 ms.
ENGAGING    the pre-allocated welds switch on soft and stiffen over 50 ms,
            pulling the brick into its seated pose.
MATED       welds at full stiffness; the brick's walls stop colliding with
            anything but hands (§2.3.2 contact filtering, bricks.py). Each
            step the weld's constraint force is read back as the connection
            wrench and capacity.Patch.utilization() decides whether it holds.
            u >= 1 for BREAK_HOLD_S -> that connection's weld switches off.

Invariants (§2.3.3): both the weld and the tendon act through MuJoCo
constraints between the two bodies, so every joint force is equal and
opposite by construction (tests check it numerically). Extraction >
insertion because f_break (11.3 N/stud) > f_insert (8.9 N/stud).
"""

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from sim.joint_model.capacity import F_BREAK_PER_STUD, F_INSERT_PER_STUD, Patch
from sim.mj import bricks as Bk

DISENGAGED, ENGAGING, MATED = "DISENGAGED", "ENGAGING", "MATED"


@dataclass
class ClutchParams:
    f_break_per_stud: float = F_BREAK_PER_STUD
    f_insert_per_stud: float = F_INSERT_PER_STUD
    gate_lateral: float = 0.0012          # m
    gate_dz: tuple = (-0.0003, 0.0010)    # m, anti-stud face above seated
    gate_tilt_deg: float = 4.0
    gate_yaw_deg: float = 5.0
    gate_force: float = 4.0               # N along the stud axis
    dwell: float = 0.040                  # s
    ramp: float = 0.050                   # s
    ramp_start_timeconst: float = 0.05    # s, soft weld at the start of the ramp
    weld_timeconst: float = 0.003         # s, MATED stiffness
    break_hold: float = 0.005             # s over capacity before letting go (filters the
                                          # one-step impulses of contacts engaging)
    settle: float = 0.100                 # s after MATED before capacity is enforced: the
                                          # stiffening weld pulls the last <= 1 mm home
                                          # against the gripper and loads its own joint
    zone_lateral: float = 0.0015          # studs-in-cavity test for the insertion friction
    zone_entry: float = 0.0003            # interference starts this far below the stud tops
                                          # (the tops are filleted; above it the walls
                                          # already guide the brick laterally)
    stick_speed: float = 0.001            # m/s: a sliding brick below this sticks again
    rearm: float = 0.00002                # m a brick must back off before the lock follows it
                                          # (re-arming on nm jitter ratchets it upward and
                                          # pumped ~1e-9 J per event: tests/test_clutch.py)

    @classmethod
    def from_calibration(cls, calib):
        f = calib.get("fitted", {})
        return cls(f_break_per_stud=f.get("f_break_per_stud_N", F_BREAK_PER_STUD),
                   f_insert_per_stud=f.get("insertion_peak_N", F_INSERT_PER_STUD),
                   gate_lateral=f.get("gate_lateral_mm", 1.2) / 1000,
                   ramp=f.get("ramp_time_s", 0.050))


@dataclass
class ConnState:
    conn: object                  # scene.Connection
    patch: Patch
    state: str = DISENGAGED
    over_since: float = -1.0
    mated_at: float = -1.0
    lock: float = -1.0            # tendon length the static-friction limit holds, or -1
    util: float = 0.0
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))   # on U, patch frame
    peak_util: float = 0.0


@dataclass
class BrickState:
    name: str
    body: int
    conns: list
    walls: list                   # geom ids
    state: str = DISENGAGED
    dwell: float = 0.0
    near: bool = False
    ramp_t: float = 0.0
    last_gate: dict = field(default_factory=dict)


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rel_pose(d, body_l, body_u):
    """Pose of U in L's frame (L = -1 means world)."""
    pu, qu = d.xpos[body_u], d.xquat[body_u]
    if body_l < 0:
        return pu.copy(), qu.copy()
    pl, ql = d.xpos[body_l], d.xquat[body_l]
    Rl = d.xmat[body_l].reshape(3, 3)
    return Rl.T @ (pu - pl), _quat_mul(_quat_conj(ql), qu)


class ClutchModel:
    def __init__(self, cell, params=None):
        self.cell, self.m = cell, cell.model
        self.p = params or ClutchParams()
        m = self.m
        self.bricks = {}
        by_upper = {}
        for c in cell.connections:
            by_upper.setdefault(c.upper, []).append(c)
        self.conn_states = []
        for name, conns in by_upper.items():
            body = m.body(name).id
            walls = [m.geom("%s/wall%d" % (name, k)).id for k in range(4)]
            cs = []
            for c in conns:
                xs = c.studs_u - c.patch_center_u[:2]
                xmin, xmax, ymin, ymax = c.rect_u
                rect = (xmin - c.patch_center_u[0], xmax - c.patch_center_u[0],
                        ymin - c.patch_center_u[1], ymax - c.patch_center_u[1])
                st = ConnState(conn=c, patch=Patch(xs, rect, self.p.f_break_per_stud))
                cs.append(st)
                self.conn_states.append(st)
            self.bricks[name] = BrickState(name=name, body=body, conns=cs, walls=walls)
        self.lower_body = {id(st): (-1 if st.conn.lower is None else m.body(st.conn.lower).id)
                           for st in self.conn_states}
        self.square = {n: cell.dims[n][0] == cell.dims[n][1] for n in self.bricks}
        self.events = []          # (t, kind, brick, detail)
        self.enabled = True

    # --- external control ---------------------------------------------------------
    def mate_now(self, name):
        """Put a brick straight into MATED at full stiffness (pre-placed bricks)."""
        b = self.bricks[name]
        for st in b.conns:
            self._weld_on(st, self.p.weld_timeconst)
            st.state = MATED
            st.mated_at = -1e9
        b.state = MATED
        self._walls(b, mated=True)
        self._friction(b, 0.0)

    def connected(self, name):
        return self.bricks[name].state == MATED and all(st.state == MATED for st in self.bricks[name].conns)

    def all_mated(self, names=None):
        names = names or list(self.bricks)
        return all(self.connected(n) for n in names)

    # --- internals ----------------------------------------------------------------
    def _weld_on(self, st, timeconst):
        """Switch a connection's weld on at its seated pose, with the brick's
        own symmetry: a 2x2 turned 90 deg (or a 2x4 turned 180) seats the same,
        so the weld holds the symmetric orientation nearest the brick's actual
        one. Welding to the nominal one instead spun a 2x2 that had arrived
        turned 180 deg and launched it when the gripper let go."""
        m, d = self.m, self.cell.data
        c = st.conn
        _, q_now = _rel_pose(d, self.lower_body[id(st)], self.bricks[c.upper].body)
        period = math.pi / 2 if self.square[c.upper] else math.pi
        best, best_ang = c.rel_quat, 9.0
        for k in range(int(round(2 * math.pi / period))):
            qz = np.array([math.cos(k * period / 2), 0.0, 0.0, math.sin(k * period / 2)])
            cand = _quat_mul(c.rel_quat, qz)
            ang = 2 * math.acos(min(1.0, abs(float(np.dot(cand, q_now)))))
            if ang < best_ang:
                best, best_ang = cand, ang
        m.eq_data[c.eq_id, 3:6] = c.rel_pos
        m.eq_data[c.eq_id, 6:10] = best
        m.eq_solref[c.eq_id] = [timeconst, 1.0]
        d.eq_active[c.eq_id] = 1

    def _weld_off(self, st):
        self.cell.data.eq_active[st.conn.eq_id] = 0

    def _walls(self, b, mated):
        self.m.geom_conaffinity[b.walls] = Bk.MATED_WALL_AFFINITY if mated else Bk.WALL_AFFINITY

    def _friction(self, b, per_stud):
        """Interference off (the brick left the zone, or mated)."""
        assert per_stud == 0.0
        for st in b.conns:
            self.m.tendon_frictionloss[st.conn.tendon_id] = 0.0
            if st.lock >= 0:
                self.m.tendon_limited[st.conn.tendon_id] = 0
                st.lock = -1.0

    def _interference(self, d, b):
        """Coulomb stud/cavity interference on each connection tendon."""
        p = self.p
        for st in b.conns:
            tid = st.conn.tendon_id
            limit = p.f_insert_per_stud * st.conn.n_studs
            self.m.tendon_frictionloss[tid] = limit          # kinetic, while sliding
            L, v = float(d.ten_length[tid]), float(d.ten_velocity[tid])
            if st.lock >= 0:
                if L > st.lock + p.rearm:
                    st.lock = L                              # backed off: re-arm here
                elif self._limit_force(d, tid) > limit:
                    st.lock = -1.0                           # breaks loose: SLIP
                    self.m.tendon_limited[tid] = 0
                    continue
            elif abs(v) < p.stick_speed:
                st.lock = L                                  # stopped: STICK
            if st.lock >= 0:
                self.m.tendon_range[tid] = [st.lock, 10.0]
                self.m.tendon_limited[tid] = 1

    @staticmethod
    def _limit_force(d, tid):
        n = d.nefc
        rows = np.nonzero((d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
                          & (d.efc_id[:n] == tid))[0]
        return float(np.abs(d.efc_force[rows]).sum()) if len(rows) else 0.0

    def _gate_one(self, d, st):
        """Relative-pose errors of one connection vs its seated pose."""
        c = st.conn
        pos, quat = _rel_pose(d, self.lower_body[id(st)], self.bricks[c.upper].body)
        dp = pos - c.rel_pos
        dq = _quat_mul(_quat_conj(c.rel_quat), quat)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, dq)
        R = R.reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
        yaw = math.atan2(R[1, 0], R[0, 0])
        period = math.pi / 2 if self.square[c.upper] else math.pi
        yaw_err = abs((yaw + period / 2) % period - period / 2)
        return float(np.hypot(dp[0], dp[1])), float(dp[2]), tilt, math.degrees(yaw_err)

    def axial_push(self, d, name):
        """Force pushing brick `name` onto its supports along the stud axis:
        insertion-friction reaction + contact normal forces from the supports."""
        b = self.bricks[name]
        lowers = {self.lower_body[id(st)] for st in b.conns}
        total = 0.0
        # interference: static lock or kinetic friction rows on the tendons
        tids = {st.conn.tendon_id for st in b.conns}
        n = d.nefc
        if n:
            kinds = (mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON, mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
            rows = np.nonzero(np.isin(d.efc_type[:n], kinds))[0]
            for r in rows:
                if int(d.efc_id[r]) in tids:
                    total += abs(float(d.efc_force[r]))
        # contacts between the brick and its supports (world = baseplate):
        # the normal runs geom1 -> geom2 and pushes geom2 along +n
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if b.body == b2:
                other, sign = b1, 1.0
            elif b.body == b1:
                other, sign = b2, -1.0
            else:
                continue
            if other not in lowers and not (other == 0 and -1 in lowers):
                continue
            mujoco.mj_contactForce(self.m, d, i, f6)
            total += max(0.0, sign * f6[0] * con.frame[2])
        return total

    def _connection_wrenches(self, d, mated):
        """Wrench on U at each mated connection's patch centre, patch frame.

        A weld's first three efc rows are the anchor-point constraint (a
        force at the anchor -- U's origin -- world frame); the last three the
        orientation constraint, whose efc force is -2x the torque on body2 for
        small errors (quaternion-derivative scaling). The wrench is then moved
        from U's origin to the patch centre. Verified against J^T f in
        tests/test_clutch.py.
        """
        if not mated:
            return
        n = d.nefc
        eq_rows = np.nonzero(d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
        ids, idx = np.unique(d.efc_id[eq_rows], return_index=True)
        first = dict(zip(ids.tolist(), eq_rows[idx].tolist()))
        for st in mated:
            r = first.get(st.conn.eq_id)
            if r is None:
                continue
            f = d.efc_force[r:r + 6]
            R = d.xmat[self.bricks[st.conn.upper].body].reshape(3, 3)
            F = R.T @ (-f[:3])
            M = R.T @ (-0.5 * f[3:6]) - np.cross(st.conn.patch_center_u, F)
            st.wrench = np.concatenate([F, M])

    # --- the per-step update ---------------------------------------------------------
    def update(self, d):
        if not self.enabled:
            return
        dt = self.m.opt.timestep
        t = d.time
        p = self.p
        mated = [st for st in self.conn_states if st.state == MATED]
        self._connection_wrenches(d, mated)
        for st in mated:
            F, M = st.wrench[:3], st.wrench[3:]
            st.util = float(st.patch.utilization(F[2], M[0], M[1]))
            if t - st.mated_at < p.settle:
                continue
            st.peak_util = max(st.peak_util, st.util)
            if st.util >= 1.0:
                if st.over_since < 0:
                    st.over_since = t
                if t - st.over_since >= p.break_hold:
                    self._break(d, st)
            else:
                st.over_since = -1.0

        for b in self.bricks.values():
            if b.state == MATED:
                continue
            if b.state == ENGAGING:
                b.ramp_t += dt
                x = min(1.0, b.ramp_t / p.ramp)
                tc = math.exp((1 - x) * math.log(p.ramp_start_timeconst) + x * math.log(p.weld_timeconst))
                for st in b.conns:
                    self.m.eq_solref[st.conn.eq_id, 0] = tc
                if x >= 1.0:
                    b.state = MATED
                    for st in b.conns:
                        st.state = MATED
                        st.over_since = -1.0
                        st.mated_at = t
                    self._walls(b, mated=True)
                    self._friction(b, 0.0)
                    self.events.append((t, "mated", b.name, dict(b.last_gate)))
                continue
            # DISENGAGED: cheap proximity screen first
            tgt = self.cell.targets[b.name][0]
            if np.linalg.norm(d.xpos[b.body] - tgt) > 0.006:
                if b.dwell or b.near:
                    b.dwell = 0.0
                    b.near = False
                    self._friction(b, 0.0)
                continue
            b.near = True
            errs = [self._gate_one(d, st) for st in b.conns]
            in_zone = all(lat < p.zone_lateral and -0.0005 < dz < Bk.STUD_H - p.zone_entry
                          for lat, dz, _, _ in errs)
            if in_zone:
                self._interference(d, b)
            else:
                self._friction(b, 0.0)
            ok = all(lat < p.gate_lateral and p.gate_dz[0] <= dz <= p.gate_dz[1]
                     and tilt < p.gate_tilt_deg and yaw < p.gate_yaw_deg
                     for lat, dz, tilt, yaw in errs)
            push = self.axial_push(d, b.name) if ok else 0.0
            worst = max(errs, key=lambda e: e[0])
            b.last_gate = {"lateral_mm": round(worst[0] * 1e3, 3), "dz_mm": round(worst[1] * 1e3, 3),
                           "tilt_deg": round(worst[2], 2), "yaw_deg": round(worst[3], 2),
                           "push_N": round(push, 2)}
            if ok and push > p.gate_force:
                b.dwell += dt
                if b.dwell >= p.dwell:
                    b.state = ENGAGING
                    b.ramp_t = 0.0
                    # the weld pulls it home; the lock goes, kinetic friction
                    # stays so the brick slides rather than slams into its seat
                    for st in b.conns:
                        st.lock = -1.0
                        self.m.tendon_limited[st.conn.tendon_id] = 0
                    for st in b.conns:
                        self._weld_on(st, p.ramp_start_timeconst)
                        st.state = ENGAGING
                    self.events.append((t, "engaging", b.name, dict(b.last_gate)))
            else:
                b.dwell = 0.0

    def _break(self, d, st):
        self._weld_off(st)
        st.state = DISENGAGED
        st.over_since = -1.0
        b = self.bricks[st.conn.upper]
        self.events.append((d.time, "break", b.name,
                            {"lower": st.conn.lower, "util": round(st.util, 3),
                             "wrench": [round(float(v), 4) for v in st.wrench]}))
        if all(s.state != MATED for s in b.conns):
            b.state = DISENGAGED
            b.dwell = 0.0
            self._walls(b, mated=False)

    # --- reporting -------------------------------------------------------------------
    def max_util(self):
        return max((st.util for st in self.conn_states if st.state == MATED), default=0.0)

    def breaks(self):
        return [e for e in self.events if e[1] == "break"]
