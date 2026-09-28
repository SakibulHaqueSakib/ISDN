"""Scripted skills for the twin -- master_report §WP4 steps 1-4.

    pick(sim, "B", brick)               grasp from the feeder, lift
    transport(sim, "B", brick, target)  carry the BRICK (not the hand) above its socket
    insert(sim, "B", brick, ...)        §WP4.4 scripted insertion
    release(sim, "B")                   open and retract
    brace(sim, "A", brace)              stabilizer grasp-and-hold (see bracing.py)

Poses are ground truth (§WP4: "a complete pipeline with ground-truth poses").

The scripted insertion is §WP4.4, and it is written to be the residual base
controller of WP5: descend at 5 mm/s; on contact, a 2 mm Archimedean spiral at
3 mm/s holding 8 N down with +-1.5 deg wiggle about x/y at 2 Hz; once the
studs drop into the cavity, press with force feed-forward up to the press
force until the joint model reports ENGAGING/MATED, or time out (10 s).
"""

import math

import numpy as np

from sim.mj import control as C
from sim.mj import bricks as Bk

GRASP_TCP_ABOVE_BOTTOM = 0.012    # pad bottoms (TCP - 9 mm) stay above neighbouring studs
APPROACH = 0.06                   # m above a grasp or place before descending
GRIP_FORCE = 15.0                 # N per finger, §WP4.1 -- the floor, see grip_for()
DESCEND_SPEED = 0.005             # m/s, §WP4.4, over the last APPROACH_SLOW
FAST_DESCEND = 0.020              # m/s until then (25 mm at 5 mm/s would spend half the
APPROACH_SLOW = 0.003             # 10 s skill budget before the brick touches anything)
SEARCH_FORCE = 8.0                # N held down during the spiral, §WP4.4
SPIRAL_R = 0.002                  # m
SPIRAL_SPEED = 0.003              # m/s
SPIRAL_PITCH = 0.0010             # m per turn (the capture window is ~1.4 mm wide)
SPIRAL_TIME = 5.0                 # s before pressing again (reaches the 2 mm radius)
SEARCH_KP_XY = 4000.0             # N/m laterally while searching: at 1200 N/m the
                                  # 4.8 N of friction under 8 N down absorbs 4 mm of spiral
INSERT_KP_ROT = 150.0             # N*m/rad while inserting: at §2.5's 40 a brick pressed
                                  # on one wall tips ~2.7 deg and reads as "in" by depth; the
                                  # scripted skill commands its own wiggle, so it holds level
LEVEL_TORQUE = 0.12               # N*m: wrist moment above this = one-sided support (tipped)
IN_CAVITY = 0.00148               # m: anti-stud face this far above seated = the studs are
                                  # inside the cavity, into the interference (which starts
                                  # 0.3 mm below the 1.8 mm stud tops). The robot knows this
                                  # from its own TCP height; it cannot know lateral error.
WIGGLE_DEG, WIGGLE_HZ = 1.5, 2.0
CONTACT_FORCE = 3.0               # N on the wrist that counts as contact
SEARCH_KP_ROT = 20.0              # N*m/rad during the spiral: the +-1.5 deg wiggle at the
                                  # 150 of INSERT_KP_ROT put up to 3.9 N*m into the structure,
                                  # and a 2x2 pier's joints hold ~0.4 (S3 step 8); compliant,
                                  # it is ~0.5 N*m
PRESS_SPEED = 0.008               # m/s: pressing faster than this sheds force...
PRESS_DAMPING = 6000.0            # ...at this rate (N per m/s)
MAX_HAND_TILT_DEG = 1.5           # a brick tilted more than this in the fingers is regrasped
INSERT_TIMEOUT = 10.0             # s, §WP4 behaviour tree


def grip_yaw(sim, name, yaw, pos=None, tilt=(0.0, 0.0, 0.0)):
    """A parallel gripper is 180-deg symmetric: of yaw and yaw + pi, pick the
    one whose IK solution keeps every joint farthest from its limits (joint 7
    has only +-166 deg and winds up otherwise)."""
    arm = sim.arms[name]
    pos = arm.x_d if pos is None else np.asarray(pos, float)
    lo = sim.m.jnt_range[arm.joint_ids, 0]
    hi = sim.m.jnt_range[arm.joint_ids, 1]

    def margin(y):
        q, err = C.ik(sim.m, arm, pos, C.down_rot(y, tilt), q0=sim.park[name][2])
        return min(np.min(q - lo), np.min(hi - q)) - 10 * err
    return max((yaw, yaw + math.pi), key=margin)


def grip_for(press_force, mu=1.0):
    """Grip force per finger that can transmit the seating press.

    §WP4.1's 15 N cannot: two pads at mu = 1 transmit 30 N axially, and a
    calibrated 2x2 needs 4 x 8.9 = 35.6 N to seat (measured: the brick slips
    and tilts 7.5 deg in the grasp). 1.2 x press gives a 2.4x margin at mu = 1;
    a real Franka hand sustains 70 N.
    """
    return float(np.clip(1.2 * press_force / mu, 20.0, 70.0))


def brick_pose(sim, brick):
    b = sim.m.body(brick).id
    return sim.d.xpos[b].copy(), sim.d.xmat[b].reshape(3, 3).copy()


def pick(sim, name, brick, yaw, grip_force=GRIP_FORCE, offset=(0.0, 0.0)):
    """Grasp `brick` where it lies, `offset` (m, world xy) from its centre.
    Returns True if it left the table with the hand."""
    arm = sim.arms[name]
    p, Rb = brick_pose(sim, brick)
    grasp = p + Rb @ np.array([offset[0], offset[1], 0.0]) + [0, 0, GRASP_TCP_ABOVE_BOTTOM]
    yaw = grip_yaw(sim, name, yaw, grasp)
    R = C.down_rot(yaw)
    arm.grip_open = 0.04
    arm.grip_force = grip_force
    x, _ = arm.tcp(sim.d)
    if x[2] < grasp[2] + APPROACH - 0.01:              # lift clear before swinging over
        sim.goto(name, [x[0], x[1], grasp[2] + APPROACH], arm.R_d, duration=0.5)
    sim.goto_joint(name, grasp + [0, 0, APPROACH], R)
    sim.goto(name, grasp, R, duration=0.8, settle=0.6)
    arm.grip_open = 0.0                      # close: force-limited at grip_force
    close(sim, name)
    z0 = brick_pose(sim, brick)[0][2]
    sim.goto(name, grasp + [0, 0, APPROACH], R, duration=0.8, settle=0.1)
    lifted = brick_pose(sim, brick)[0][2] - z0 > 0.5 * APPROACH
    return lifted


def close(sim, name, tmax=1.2):
    """Wait for the fingers (closing at control.GRIP_SPEED) to stop on
    something, then let the grip force settle."""
    arm = sim.arms[name]
    t0 = sim.t
    sim.run(0.1)
    sim.run(tmax, until=lambda s_: abs(float(s_.d.qvel[arm.finger_v].mean())) < 0.001)
    sim.run(0.1)
    return sim.t - t0


def in_hand_offset(sim, name, brick):
    """Brick pose in the TCP frame (measured once it is held)."""
    x, R = sim.arms[name].tcp(sim.d)
    p, Rb = brick_pose(sim, brick)
    return R.T @ (p - x), R.T @ Rb


def tcp_for_brick(offset, brick_target, R_tcp):
    """TCP position that puts the brick's frame origin at brick_target."""
    off_p, _ = offset
    return brick_target - R_tcp @ off_p


def transport(sim, name, brick, target, yaw, clearance_z, dz=0.025):
    """Carry the held brick to `dz` above target via a travel height."""
    arm = sim.arms[name]
    R = C.down_rot(grip_yaw(sim, name, yaw, target + [0, 0, dz + GRASP_TCP_ABOVE_BOTTOM]))
    off = in_hand_offset(sim, name, brick)
    x, _ = arm.tcp(sim.d)
    travel = max(clearance_z, x[2])
    up = arm.x_d.copy()
    up[2] = travel
    sim.goto(name, up, arm.R_d, duration=0.6)
    above = tcp_for_brick(off, target + [0, 0, dz], R)
    hi = above.copy()
    hi[2] = travel
    sim.goto_joint(name, hi, R)
    sim.goto(name, above, R, duration=0.8, settle=0.8)
    # correct residual brick error (ground truth), once
    off = in_hand_offset(sim, name, brick)
    above = tcp_for_brick(off, target + [0, 0, dz], R)
    sim.goto(name, above, R, duration=0.4, settle=0.6)
    return off


def insert(sim, name, brick, target, clutch, press_force, timeout=INSERT_TIMEOUT,
           search=True, log=None):
    """§WP4.4 scripted insertion from the pre-insertion pose.

    descend at 5 mm/s -> contact -> PRESS (ramp to press_force over 0.5 s).
    While the brick keeps going down, keep pressing until the joint model
    reports MATED. If it has not moved 0.03 mm in the last 0.4 s at full
    force (and the joint is not engaging) it is stuck -- on the stud tops, or
    tipped with one wall on them -- so back off to 8 N and SEARCH
    (Archimedean spiral to 2 mm at 3 mm/s, +-1.5 deg wiggle at 2 Hz, stiffer
    laterally) until the TCP height says the studs are in, then press again.
    Depth alone is not enough to call it in: a brick 1.7 mm off tips 2.7 deg
    with one wall past the edge, and its centre reads 1.36 mm.

    The press comes first because the stud/cavity interference holds an
    ALIGNED brick at the stud tops too (sim/joint_model/clutch.py): only
    pushing past n_studs x f_insert tells aligned from misaligned.

    Returns: success, peak_force_N, impulse_Ns, duration_s, phase, searches.
    """
    arm = sim.arms[name]
    b = clutch.bricks[brick]
    bid = sim.m.body(brick).id
    t0 = sim.t
    peak, impulse = 0.0, 0.0
    R0 = arm.R_d.copy()
    dt = sim.m.opt.timestep * 2
    phase = "descend"
    contact_z = None
    phase_t = 0.0
    spiral_t = 0.0
    centre = None
    searches = 0
    press_ramp = 0.5
    hist = []                         # (t, dz) during the press, for stuck detection
    arm.kp_rot[:] = INSERT_KP_ROT

    def fz():
        # the environment pushes the hand UP when the hand presses down
        return sim.ft(name)[2]

    while sim.t - t0 < timeout:
        if b.state == "MATED":
            phase = "mated"
            break
        f = fz()
        peak = max(peak, f)
        impulse += max(f, 0.0) * dt
        zb = sim.d.xpos[bid][2]
        if log is not None:
            log.append((sim.t - t0, f, zb - target[2], phase))
        x, _ = arm.tcp(sim.d)
        phase_t += dt
        if phase == "descend":
            v = DESCEND_SPEED if zb - target[2] < Bk.STUD_H + APPROACH_SLOW else FAST_DESCEND
            arm.x_d = arm.x_d + [0, 0, -v * dt]
            arm.f_ff[:] = 0
            if f > CONTACT_FORCE:
                contact_z = zb
                centre = arm.x_d.copy()
                sim.noslip(True)          # the grip must not creep under the press
                # the plan says where the stud tops are: touching down on them
                # (rather than at the interference, 0.3 mm lower) means misaligned
                if search and zb - target[2] > IN_CAVITY + 0.00015:
                    phase, phase_t, spiral_t = "search", 0.0, 0.0
                    arm.kp_rot[:] = SEARCH_KP_ROT
                    searches += 1
                    arm.kp_pos[:2] = SEARCH_KP_XY
                else:
                    phase, phase_t = "press", 0.0
                    arm.kp_rot[:] = INSERT_KP_ROT
        elif phase == "press":
            ramp = min(1.0, phase_t / press_ramp)
            # laterally free (the target follows the hand): once the studs are
            # on the chamfer the wedge centres the brick; a lateral spring
            # holding it 0.45 mm off made it jam with the press half-absorbed
            arm.x_d = np.array([x[0], x[1], x[2]])
            arm.R_d = R0.copy()
            arm.f_ff[:] = 0
            # a force-controlled press with a speed limit: when the stud
            # interference lets go, a constant 93 N drove the hand into the
            # seat at 40 mm/s, and the 676 N impact tipped the brick 5 deg in
            # the grasp before the gate's 40 ms dwell (S2 step 2)
            vz = float(arm.tcp_vel(sim.d)[2])
            relief = PRESS_DAMPING * max(0.0, -vz - PRESS_SPEED)
            arm.f_ff[2] = -max(SEARCH_FORCE, ramp * press_force - relief)
            if phase_t >= press_ramp:
                hist.append((phase_t, zb))
                while hist and hist[0][0] < phase_t - 0.4:
                    hist.pop(0)
            moving = len(hist) < 2 or hist[0][1] - hist[-1][1] > 0.00003 or phase_t < press_ramp + 0.4
            if b.state == "ENGAGING" or moving:
                pass                                         # keep pushing
            elif search:
                phase, phase_t, spiral_t = "search", 0.0, 0.0
                arm.kp_rot[:] = SEARCH_KP_ROT
                hist.clear()
                searches += 1
                centre = arm.x_d.copy()
                arm.kp_pos[:2] = SEARCH_KP_XY
        elif phase == "search":
            spiral_t += dt
            s = SPIRAL_SPEED * spiral_t
            a = SPIRAL_PITCH / (2 * math.pi)                 # Archimedean r = a*theta
            th = math.sqrt(2 * s / a) if s > 0 else 0.0
            r = min(a * th, SPIRAL_R)
            arm.x_d = np.array([centre[0] + r * math.cos(th), centre[1] + r * math.sin(th), x[2]])
            arm.f_ff[:] = 0
            arm.f_ff[2] = -SEARCH_FORCE
            w = math.radians(WIGGLE_DEG) * math.sin(2 * math.pi * WIGGLE_HZ * spiral_t)
            arm.R_d = C.rotvec_to_mat([w, 0.7 * w, 0]) @ R0
            tau = sim.ft(name)[3:5]
            dropped = zb - target[2] < IN_CAVITY and np.hypot(*tau) < LEVEL_TORQUE
            if dropped or spiral_t > SPIRAL_TIME:
                contact_z = min(contact_z, zb)
                phase, phase_t = "press", 0.0
                arm.kp_rot[:] = INSERT_KP_ROT
                hist.clear()
                arm.kp_pos[:] = C.KP_POS
        for _ in range(2):
            sim.step()
    arm.f_ff[:] = 0
    arm.R_d = R0.copy()
    arm.kp_pos[:] = C.KP_POS
    arm.kp_rot[:] = C.KP_ROT
    x, _ = arm.tcp(sim.d)
    arm.x_d = x.copy()
    sim.noslip(False)
    return {"success": b.state == "MATED", "peak_force_N": round(float(peak), 2),
            "impulse_Ns": round(float(impulse), 3), "duration_s": round(sim.t - t0, 3),
            "phase": phase, "searches": searches}


def release(sim, name, retract=0.06):
    arm = sim.arms[name]
    arm.f_ff[:] = 0
    arm.grip_open = 0.04
    sim.run(0.25)
    x, R = arm.tcp(sim.d)
    sim.goto(name, x + [0, 0, retract], arm.R_d, duration=0.6)


def hand_tilt_deg(sim, name, brick):
    """How far the held brick is tilted relative to the fingers' frame."""
    _, R = sim.arms[name].tcp(sim.d)
    _, Rb = brick_pose(sim, brick)
    return float(np.degrees(np.arccos(np.clip(abs((R.T @ Rb)[2, 2]), -1, 1))))


def put_down(sim, name, brick, spot):
    """Set the held brick down at `spot` (world xy on the table) and let go --
    the first half of a regrasp (§WP4 LiftAndRegrasp)."""
    arm = sim.arms[name]
    x, R = arm.tcp(sim.d)
    off = in_hand_offset(sim, name, brick)
    spot = np.array([spot[0], spot[1], 0.0])
    up = arm.x_d.copy()
    up[2] = max(up[2], x[2] + 0.04)
    sim.goto(name, up, arm.R_d, duration=0.5)
    above = tcp_for_brick(off, spot + [0, 0, APPROACH], R)
    sim.goto_joint(name, above, R)
    low = tcp_for_brick(off, spot + [0, 0, 0.002], R)
    sim.goto(name, low, R, duration=1.0, settle=0.2,
             until=lambda s_: s_.ft(name)[2] > CONTACT_FORCE)
    release(sim, name)
    sim.run(0.3)


def park(sim, name):
    p, R, _ = sim.park[name]
    arm = sim.arms[name]
    x, _ = arm.tcp(sim.d)
    if x[2] < p[2]:
        sim.goto(name, [x[0], x[1], p[2]], arm.R_d, duration=0.5)
    sim.goto_joint(name, p, R)


# --- the stabilizer ------------------------------------------------------------
BRACE_KP_POS = np.array([15000.0, 15000.0, 15000.0])  # N/m while holding: an anchor, not a spring
BRACE_KP_ROT = np.array([300.0, 300.0, 300.0])        # N*m/rad
BRACE_APPROACH = 0.05                                  # m back along the tool axis


def brace_pose(brace):
    """TCP position and orientation for a §2.6 brace (bracing.py fields)."""
    p = np.array(brace["brace_pose"][:3], float)
    R = C.down_rot(math.pi / 2, brace.get("brace_tilt_rotvec", (0.0, 0.0, 0.0)))  # fingers along x
    return p, R


def brace(sim, name, brace, width=0.0158):
    """Arm `name` grasps the structure at the brace pose and holds it stiffly.

    Approach along the (tilted) tool axis, close with brace_force_N per
    finger, then latch the pose with BRACE_KP gains: the LP modelled the brace
    as an anchor bounded by pad friction, so the arm must not yield like the
    §2.5 insertion impedance (600 N/m would give 70 mm under 40 N).
    Returns True if the pads closed on something.
    """
    arm = sim.arms[name]
    p, R = brace_pose(brace)
    tool = R[:, 2]                                   # hand z = toward the fingertips
    back = p - tool * BRACE_APPROACH
    arm.grip_open = width / 2 + 0.006
    x, _ = arm.tcp(sim.d)
    sim.goto(name, [x[0], x[1], max(x[2], back[2] + 0.05)], arm.R_d, duration=0.4)
    sim.goto_joint(name, back + [0, 0, 0.04], R)
    sim.goto(name, back, R, duration=0.5)
    sim.goto(name, p, R, duration=0.8, settle=0.3)
    arm.grip_force = float(brace.get("brace_force_N", 40.0))
    arm.grip_open = 0.0
    close(sim, name)
    closed_on_something = arm.opening(sim.d) > width / 2 - 0.002
    arm.hold(sim.d)
    arm.kp_pos[:] = BRACE_KP_POS
    arm.kp_rot[:] = BRACE_KP_ROT
    sim.noslip(True)
    sim.run(0.2)
    return closed_on_something


def brace_feedforward(sim, name, brace, placer="B"):
    """Coordinated bracing: a hook (append to sim.hooks) that makes arm `name`
    push the structure with the plan's feed-forward wrench, scaled by the
    press the placer is COMMANDING (the dual-arm controller shares it; the
    measured press spikes on contact). The stiff hold takes only what the LP
    did not foresee."""
    arm = sim.arms[name]
    w = np.array(brace.get("feedforward_wrench_per_N", np.zeros(6)), float)
    top = float(brace.get("press_nominal_N", 0.0))

    def hook(s_):
        if s_.k % 2 == 0:
            press = float(np.clip(-s_.arms[placer].f_ff[2], 0.0, top))
            arm.f_ff[:] = w * press
    return hook


def unbrace(sim, name):
    arm = sim.arms[name]
    arm.kp_pos[:] = C.KP_POS
    arm.kp_rot[:] = C.KP_ROT
    arm.hold(sim.d)
    arm.grip_open = 0.03
    sim.run(0.3)
    sim.noslip(False)
    x, R = arm.tcp(sim.d)
    sim.goto(name, x - R[:, 2] * BRACE_APPROACH, arm.R_d, duration=0.5)
    park(sim, name)
