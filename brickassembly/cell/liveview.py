"""Live-viewer aids of dual_arm_sim (--show-cameras, --manual). Display and hand control only: nothing here feeds the executor, and the
camera frames drawn here use their own noise stream, so a run with the panels on is the same run.

    depth_rgb, tile, overlay_look      camera frames -> uint8 tiles for ViewerBase.log_image; the V5 result drawn on the wrist look
    overlay_top                        top_detect's plate and bricks drawn on the top tile
    overlay_wrist                      the same read-out projected into a wrist tile, with each brick's label
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


HELD = (0, 255, 0)        # outline of a brick in a gripper (top_detect.track_held)


def overlay_top(rgb, det, stride=SHOW_STRIDE):
    """The top tile with top_detect's read-out on it: the plate outline (cyan), each brick's top-face outline (yellow; magenta if it is on the plate; gray if hidden, i.e. remembered; green if a gripper holds it)
    and its id. `det` is detect()'s result in the pixels of the image it ran on; the tile is that image at 1/`stride`."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=14)
    if det and det["plate"]:
        pts = [(x / stride, y / stride) for x, y in det["plate"]["outline"]]
        dr.line(pts + pts[:1], fill=(0, 255, 255), width=1)
    for b in (det["bricks"] + list(det.get("held", {}).values()) if det else []):
        pts = [(x / stride, y / stride) for x, y in b["outline"]]
        dr.line(pts + pts[:1], fill=HELD if b.get("held_by") else (200, 200, 200) if b["hidden"] else (255, 0, 255) if b["on_plate"] else (255, 255, 0), width=1)
        dr.text((b["px"] / stride + 4, b["py"] / stride - 16), str(b["id"]), fill=(0, 255, 0), font=f)
    return np.asarray(im)


def overlay_wrist(rgb, cam, cam_pose, det, stride=SHOW_STRIDE):
    """A wrist tile with the TOP camera's read-out `det` (top_detect.detect) projected through this wrist camera's pose `cam_pose` = (pos, R)
    (the nominal one: display only, nothing from ground truth): the plate outline (cyan) and each brick's top-face outline (yellow; magenta on
    the plate; gray and "(hidden)" when remembered, TD.remember; green and "(held by B)" in a gripper, TD.track_held) with a two-line label on black, "#id colour size" and "x, y mm  yaw deg" (+ " L<layer> (i,j)" on the plate). Labels are moved off each other (a leader line to a far one). A shape with a corner
    behind the camera or wholly outside the image is skipped. The tile is the camera's frame at 1/`stride`; None `det`: unchanged."""
    if not det:
        return rgb
    from PIL import Image, ImageDraw, ImageFont
    from cell.cameras import CAMS, project
    pos, R = cam_pose
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=15)
    w, h = CAMS[cam]["w"] / stride, CAMS[cam]["h"] / stride
    boxes = []                                                       # the label boxes drawn so far

    def pts(corners):
        if (R.inv().apply(corners - pos)[:, 2] >= 0).any():          # a corner behind the camera: project() clamps it and would not say
            return None
        px, py, _ = project(cam, pos, R, corners)
        px, py = px / stride, py / stride
        if px.max() < 0 or px.min() > w or py.max() < 0 or py.min() > h:
            return None
        return list(zip(px, py))

    if det["plate"]:
        q = pts(np.c_[det["plate"]["corners"][:, :2], np.full(4, det["plate"]["top_z"])])
        if q:
            dr.line(q + q[:1], fill=(0, 255, 255), width=2)
    for b in det["bricks"] + list(det.get("held", {}).values()):
        q = pts(box_corners(b["x"], b["y"], b["z_top"], b["yaw"], b["L"], b["W"]))
        if not q:
            continue
        dr.line(q + q[:1], fill=HELD if b.get("held_by") else (200, 200, 200) if b["hidden"] else (255, 0, 255) if b["on_plate"] else (255, 255, 0), width=2)
        txt = ("#%d %s %s" + (" (held by %s)" % b["held_by"] if b.get("held_by") else " (hidden)" if b["hidden"] else "") + "\n%.0f, %.0f mm  yaw %d deg") % (b["id"], b["colour"], b["studs"], 1e3 * b["x"], 1e3 * b["y"], round(math.degrees(b["yaw"])))
        if b["on_plate"]:
            txt += " L%d (%d,%d)" % (b["layer"], *b["cell"])
        ox, oy = min(q, key=lambda p: p[1])                           # above the outline's top edge, kept in the tile
        x0, y0, x1, y1 = dr.multiline_textbbox((0, 0), txt, font=f)
        tx, ty0 = min(max(ox, 0), w - x1), min(max(oy - y1 - 2, 0), h - y1)
        for d in (y1 + 2, -(y1 + 2)):                                 # clear of the labels already drawn: down, else up, a label height at a time
            ty = ty0
            while any(tx < bx1 and tx + x1 > bx0 and ty < by1 and ty + y1 > by0 for bx0, by0, bx1, by1 in boxes) and 0 <= ty + d <= h - y1:
                ty += d
            if not any(tx < bx1 and tx + x1 > bx0 and ty < by1 and ty + y1 > by0 for bx0, by0, bx1, by1 in boxes):
                break
        boxes.append((tx, ty, tx + x1, ty + y1))
        if abs(ty - ty0) > y1:                                        # far from its brick: a leader to the outline's top point
            dr.line([(tx + x1 / 2, ty + y1 / 2), (ox, oy)], fill=(255, 255, 255), width=1)
        dr.rectangle(dr.multiline_textbbox((tx, ty), txt, font=f), fill=(0, 0, 0))
        dr.multiline_text((tx, ty), txt, fill=(255, 255, 255), font=f)
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
