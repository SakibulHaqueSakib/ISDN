"""Physical plan checks in the twin -- master_report §WP3.3.

    drop_test(plan) -> {"max_disp_mm": ..., "breaks": ..., ...}

§3.3: "instantiate the full structure with all joints mated, hold 2 s under
gravity, apply a 0.5 N lateral impulse at the top, accept if max displacement
< 1.0 mm." The "0.5 N lateral impulse" is applied as a 0.5 N force for
PUSH_S on the top brick, along +x and then +y (the structures are 2 studs
deep in x, so x is their weak direction).
"""

import mujoco
import numpy as np

from sim.joint_model import clutch as CL
from sim.mj import scene as S

HOLD_S = 2.0
PUSH_N = 0.5
PUSH_S = 0.1
SETTLE_S = 0.5
ACCEPT_MM = 1.0


def drop_test(plan, push_n=PUSH_N):
    ids = [b["id"] for b in plan["bricks"]]
    cell = S.build_cell(plan, preplaced=ids, with_arms=False)
    m, d = cell.model, cell.data
    cm = CL.ClutchModel(cell)
    for b in ids:
        cm.mate_now(b)
    body = {b: m.body(b).id for b in ids}

    def run(sec, top=None, f=None):
        worst = 0.0
        for _ in range(int(round(sec / m.opt.timestep))):
            if f is not None:
                d.xfrc_applied[body[top], :3] = f
            mujoco.mj_step(m, d)
            cm.update(d)
            worst = max(worst, max(np.linalg.norm(d.xpos[body[b]] - p0[b]) for b in ids))
        return worst

    mujoco.mj_forward(m, d)
    p0 = {b: d.xpos[body[b]].copy() for b in ids}
    disp = run(HOLD_S)
    top = max(ids, key=lambda b: d.xpos[body[b]][2])
    pushes = {}
    for axis in (0, 1):
        f = np.zeros(3)
        f[axis] = push_n
        pushes["xy"[axis]] = run(PUSH_S, top, f)
        d.xfrc_applied[body[top], :3] = 0
        pushes["xy"[axis]] = max(pushes["xy"[axis]], run(SETTLE_S))
    worst = max([disp] + list(pushes.values()))
    return {"max_disp_mm": round(1000 * float(worst), 4),
            "hold_disp_mm": round(1000 * float(disp), 4),
            "push_disp_mm": {k: round(1000 * float(v), 4) for k, v in pushes.items()},
            "breaks": len(cm.breaks()),
            "all_mated": cm.all_mated(),
            "passed": bool(worst * 1000 < ACCEPT_MM and not cm.breaks())}


class BraceClearance:
    """Do the two hands fit? -- §3.4 accessibility for the brace, in the twin.

        ok = BraceClearance(sid)(brick, y, z, tilt_rotvec)

    Arm B holds `brick` (planner tuple) at its target and at its
    pre-insertion pose (+25 mm); arm A grasps the structure at (y, z) with
    the brace's lean. Both arms are posed by IK and the smallest distance
    from any of A's colliding geoms to B's hand, fingers and the carried brick
    must exceed MARGIN. This replaces bracing.HAND_CLEARANCE_Y (28 mm between
    grasp centres at any height), which kept the stabilizer's pads off the
    brick carrying the press (S3 step 12: 4.5 mm of pad on it).
    """

    MARGIN = 0.0015
    OPEN = 0.0079          # per-finger opening on a 2-stud (16 mm) face

    def __init__(self, structure_id):
        import planner as P
        from sim.mj import control as C
        self.P, self.C = P, C
        plan = P.build_plan(structure_id, "none")
        self.plan = plan
        self.cell = S.build_cell(plan, with_arms=True)
        self.m = self.cell.model
        self.d = mujoco.MjData(self.m)
        self.arms = {k: C.Arm(self.m, k) for k in ("A", "B")}
        self.grasp = {s["brick_id"]: s["grasp"] for s in plan["sequence"]}
        m = self.m
        self.geoms = {k: [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]
                          if m.body(m.geom_bodyid[g]).name.startswith(k + "/")] for k in ("A", "B")}
        # This cell is private: rewire its collision filter so the narrowphase
        # reports only A-vs-(B, carried brick) pairs, within MARGIN.
        # (mj_geomDistance returned 0.0 for box-box and mesh-mesh pairs 3-10 cm
        # apart, so distances are not used.)
        self._brick_geoms = {}
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        for g in self.geoms["A"]:
            m.geom_contype[g] = 16
            m.geom_margin[g] = self.MARGIN
        self._ik = {}
        self._cache = {}

    def _pose_arm(self, name, pos, R, key):
        """Pose the arm by IK; returns the pose error (position m, or 1.0 if
        the orientation is off by more than 1 deg -- ik() reports position
        only, and a tilted brace can converge in position with a twisted
        wrist)."""
        arm = self.arms[name]
        if key not in self._ik:
            best = None
            for seed in (None, self.C.READY_Q + [0, 0, 0, 0, 0, 0, np.pi / 2],
                         self.C.READY_Q - [0, 0, 0, 0, 0, 0, np.pi / 2]):
                q, err = self.C.ik(self.m, arm, np.asarray(pos, float), R, q0=seed, iters=300)
                self.d.qpos[arm.qadr] = q
                mujoco.mj_kinematics(self.m, self.d)
                rerr = np.linalg.norm(self.C.rot_error(R, self.d.site_xmat[arm.site].reshape(3, 3)))
                e = err if rerr < np.radians(1.0) else 1.0
                if best is None or e < best[1]:
                    best = (q, e)
            self._ik[key] = best
        q, err = self._ik[key]
        self.d.qpos[arm.qadr] = q
        self.d.qpos[arm.finger_q] = self.OPEN
        return err

    def hits(self, bid):
        """Any of A's geoms within MARGIN of B's hand or of brick `bid`?"""
        m, d = self.m, self.d
        others = self.geoms["B"] + [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(bid).id]
        m.geom_conaffinity[:] = 0
        m.geom_conaffinity[others] = 16
        mujoco.mj_collision(m, d)
        return d.ncon > 0

    def __call__(self, brick, y, z, tilt):
        from motion import skills as K
        key = (brick[0], round(y, 5), round(z, 5), tuple(np.round(tilt, 4)))
        if key in self._cache:
            return self._cache[key]
        P, C, m, d = self.P, self.C, self.m, self.d
        bid = brick[0]
        g = self.grasp[bid]
        tx, ty, tz = P.brick_pose(brick)
        off = g["tcp_offset_m"]
        yaw = np.radians(g["yaw_offset_deg"])
        x_brace = P.VOXEL_ORIGIN[0] + P.PITCH
        RA = C.down_rot(np.pi / 2, tilt)
        ok = True
        for dz in (0.0, 0.025):
            d.qpos[:] = m.qpos0
            errA = self._pose_arm("A", [x_brace, y, z], RA, ("A", round(y, 5), round(z, 5), key[3]))
            pB = [tx + off[0], ty + off[1], tz + dz + K.GRASP_TCP_ABOVE_BOTTOM]
            errB = self._pose_arm("B", pB, C.down_rot(yaw), ("B", bid, dz))
            j = m.body(bid).jntadr[0]
            a = m.jnt_qposadr[j]
            d.qpos[a:a + 3] = [tx, ty, tz + dz]
            d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            mujoco.mj_kinematics(m, d)
            if errA > 1e-3 or errB > 1e-3:
                ok = False
                break
            if self.hits(bid):
                ok = False
                break
        self._cache[key] = ok
        return ok
