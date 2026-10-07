"""Live-viewer aids of dual_arm_sim (--show-cameras, --manual). Display and hand control only: nothing here feeds the executor, and the
camera frames drawn here use their own noise stream, so a run with the panels on is the same run.

    depth_rgb, tile, overlay_look      camera frames -> uint8 tiles for ViewerBase.log_image; the V5 result drawn on the wrist look
    clamp_target, ManualCtl            the manual-control targets: workspace clamp, rate limit and a leash on the measured hand
"""

import math

import numpy as np

# --- camera tiles -----------------------------------------------------------------------------------------------
SHOW_STRIDE = 2           # tiles are the 1280x960 frames at 1/2 (the upload is the cost, not the render)


def enlarge_image_window(px=400):
    """ViewerGL opens an image window with 192 px tiles; ask for `px` (first appearance only; the window is resizable). Best effort: it is a private constant."""
    try:
        import newton._src.viewer.gl.image_logger as m
        m._INITIAL_TILE_PX = px
    except Exception:
        pass


def tile(rgb):
    return np.ascontiguousarray(rgb[::SHOW_STRIDE, ::SHOW_STRIDE])


def depth_rgb(depth, stride=SHOW_STRIDE):
    """Depth [m] -> a turbo colour tile (1/`stride` size), near = blue, scaled to the 2-98 % range of the valid pixels; 0 (no return) = black."""
    from matplotlib import colormaps
    d = depth[::stride, ::stride]
    ok = d > 0
    out = np.zeros(d.shape + (3,), np.uint8)
    if ok.any():
        lo, hi = np.percentile(d[ok], [2, 98])
        out[ok] = (colormaps["turbo"](np.clip((d[ok] - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3] * 255).astype(np.uint8)
    return out


def box_corners(x, y, z, yaw, L, W):
    """The L x W footprint at height z, centred on (x, y) and turned by yaw: (4, 3) world points."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[x + c * a - s * b, y + s * a + c * b, z] for a, b in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])


def overlay_look(rgb, cam_pose, rec, T_box, B_box):
    """The wrist frame of one look with V5's result on it: the estimated target footprint (cyan) and held-brick footprint (magenta) projected
    through the nominal camera, a frame green if accepted and red if not, and a caption (look, estimator, studs, reason). `T_box`, `B_box`:
    (4, 3) world corners or None (the fit failed). The image is the tile-sized one; the pose and corners are in the full camera's pixels."""
    from PIL import Image, ImageDraw, ImageFont
    from cell.cameras import project
    pos, R = cam_pose
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for box, col in ((T_box, (0, 255, 255)), (B_box, (255, 0, 255))):
        if box is not None and np.isfinite(box).all():
            px, py, _ = project("wrist_B", pos, R, box)
            dr.line([(x / SHOW_STRIDE, y / SHOW_STRIDE) for x, y in zip(px, py)] + [(px[0] / SHOW_STRIDE, py[0] / SHOW_STRIDE)], fill=col, width=2)
    ok = rec.get("accepted")
    w, h = im.size
    dr.rectangle([0, 0, w - 1, h - 1], outline=(0, 220, 0) if ok else (230, 0, 0), width=4)
    dr.rectangle([4, 4, w - 5, 30], fill=(0, 0, 0))
    dr.text((10, 7), "look %d  %s  target studs %s  held studs %s  %s" % (
        rec["k"], "ACCEPTED " + str(rec.get("estimator")) if ok else "REJECTED", rec.get("n_target_studs"), rec.get("n_held_studs"),
        "" if ok else rec.get("reject_reason", "")), fill=(255, 255, 255), font=ImageFont.load_default(size=16))
    return np.asarray(im)


# --- manual control ---------------------------------------------------------------------------------------------
Z_MIN, Z_MAX = 0.012, 0.55    # hand TCP height (m): fingertips are 8.9 mm below it, the plate top is 3.2 mm
REACH = 0.76                  # TCP to the shoulder (base + 0.333 up): the farthest feeder slot, with the hand down, is 0.73 m
R_MIN = 0.18                  # horizontal distance from the arm's base column
CROSS = 0.12                  # an arm may cross the mid-plane x = 0 by this much
YAW_MAX = 1.7                 # hand yaw about the arm's base frame (rad), 0 = parked: the wrist (q7 0.785 at park, +-2.897) reaches both brick yaws, 0 and +-pi/2
SHOULDER_Z = 0.333
V_MAX, W_MAX, G_MAX = 0.20, 1.5, 0.08     # slew rates of the command: m/s, rad/s, finger m/s
LEASH = 0.015                 # the command stays within this of the measured hand: no wind-up against a limit or a contact
GRIP_OPEN, GRIP_SHUT = 0.03, 0.0         # finger target (m per finger); 0.04 is the joint limit


def clamp_target(base, p, yaw):
    """The nearest allowed hand target (p, yaw) for the arm at `base` (world); p in the world, yaw relative to the arm's base frame (the gripper is
    180-degree symmetric, so for the hand this is also the world yaw mod pi)."""
    b, p = np.asarray(base, float), np.array(p, float)
    p[2] = np.clip(p[2], Z_MIN, Z_MAX)
    p[0] = max(p[0], -CROSS) if b[0] > 0 else min(p[0], CROSS)
    s = b + [0, 0, SHOULDER_Z]
    n = np.linalg.norm(p - s)
    if n > REACH:
        p = s + (p - s) * REACH / n
    h = p[:2] - b[:2]
    if np.linalg.norm(h) < R_MIN:
        p[:2] = b[:2] + (h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-9 else [-np.sign(b[0]), 0.0]) * R_MIN
    return p, float(np.clip(yaw, -YAW_MAX, YAW_MAX))


def toward(cur, tgt, step):
    """`cur` moved toward `tgt` by at most `step` (vectors by their norm, scalars by their size)."""
    d = np.asarray(tgt, float) - cur
    n = np.linalg.norm(d)
    return np.asarray(tgt, float) if n <= step else cur + d * step / n


class ManualCtl:
    """Targets of the two arms' hands: world position, yaw relative to the arm's base (0 = parked) and finger target. `nudge` moves a target
    (clamped), `advance` is the per-frame command: slewed toward the target and held within LEASH of the measured hand."""

    def __init__(self, bases, parks):
        self.bases = [np.array(b[0], float) for b in bases]                # world base of A, B
        self.yaws = [b[1] for b in bases]
        self.parks = [np.array(p, float) for p in parks]
        self.sel = 1                                                       # the selected arm: 0 = A, 1 = B
        self.tgt = [dict(p=p.copy(), yaw=0.0, grip=0.01) for p in self.parks]

    def nudge(self, k, d=(0, 0, 0), dyaw=0.0, grip=None):
        t = self.tgt[k]
        t["p"], t["yaw"] = clamp_target(self.bases[k], t["p"] + np.asarray(d, float), t["yaw"] + dyaw)
        if grip is not None:
            t["grip"] = float(np.clip(grip, 0.0, 0.04))

    def home(self, k):
        self.tgt[k].update(p=clamp_target(self.bases[k], self.parks[k], 0.0)[0], yaw=0.0)

    def advance(self, k, cmd, ee, dt):
        """The new Arm.cmd (pos, yaw, grip, tilt) of arm k: `cmd` slewed toward the target, then kept within LEASH of `ee` (its measured hand)."""
        p, yaw, g, tilt = cmd
        t = self.tgt[k]
        p = toward(p, t["p"], V_MAX * dt)
        lag = p - ee
        if np.linalg.norm(lag) > LEASH:
            p = ee + lag * LEASH / np.linalg.norm(lag)
        yaw = float(toward(np.array(yaw), self.yaws[k] + t["yaw"], W_MAX * dt))        # the command is the hand's world yaw
        g = float(toward(np.array(g), t["grip"], G_MAX * dt))
        return p, yaw, g, tilt
