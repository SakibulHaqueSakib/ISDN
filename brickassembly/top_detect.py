"""Top-camera scene read-out: every brick's pose, colour and size, and the baseplate (the grid base), from one top image alone.

Same firewall as vision.py: pure numpy / scipy / cv2 (vision for the rays and nothing else); no newton, warp, torch,
dual_arm_sim or cell.*, and never ground truth. Display and planning aid: nothing in the executor reads it.

    res = detect(rgb, depth, intrinsics_or_rays, cam_pos, cam_R)
    res["bricks"]   [dict(id, colour, x, y, z_top, yaw, studs "LxW", L, W, layer, px, py, outline, on_plate, cell, hidden), ...]
    res["plate"]    dict(x, y, yaw, size, studs (NI, NJ), pitch, top_z, corners, outline, origin, occupied, hidden) or None
    res["height"]   (H, W) metres over the table of each pixel, nan where the depth is invalid
    res = remember(prev_res, res)     # object permanence: bricks (and the plate) an arm hides stay, with hidden=True, until the top camera sees their spot empty
    res = track_held(res, prev_res, hands)    # bricks in a gripper (hands' FK pose + finger openings) leave res["bricks"] for res["held"] = {"A"/"B": brick dict with held_by}, followed with the hand

Inputs are what cell.cameras Cameras.capture / preview return for the top camera: rgb uint8 (H, W, 3), depth = ray distance [m]
(0 = invalid), the nominal pose (OpenGL camera: looks -z, up +y). `intrinsics_or_rays` is the intrinsics dict of the FULL image
(any stride of it works: the rays are subsampled to the depth's shape) or the (H, W, 3) unit rays of exactly this image. Metres and
radians, world z up, table z = 0. x, y, yaw are the top face's centre and its long axis (mod pi; squares mod pi/2); outline is the
brick's 4 top-face corners and px, py its centroid, in pixels of the given image (for drawing); layer 0 = the lowest. Plate cells (i, j):
i along the plate's yaw from its `origin` corner, j along the left normal; a brick's `cell` is its min-corner cell.
"""

import functools
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import vision as V

MM = 1e-3
PITCH, BRICK_H, STUD_H, PLATE_TOP = 8 * MM, 9.6 * MM, 1.7 * MM, 3.2 * MM      # LEGO constants (dual_arm_sim: Z0, ex.BRICK_HEIGHT)
H_BRICK = (6 * MM, 0.15)            # brick pixels: above the plate studs (4.9 mm), below the arms
H_PLATE = (2.0 * MM, 7 * MM)        # plate pixels: the plate face and its studs
TOL_STUDS = 2.5 * MM                # a brick side must be a whole number of studs within this
FILL_MIN, AREA_MIN = 0.8, 40e-6     # rectangularity and size (m^2) of an accepted brick
MAX_STUDS = (6, 2)                  # the largest SKU (planner.BRICKS: 2x6): long side, short side
COLOURS = ("red", "orange", "yellow", "green", "blue", "white", "gray", "other")


def colour_class(rgb, white_v=0.5):
    """Colour name index (into COLOURS) of an (n, 3) float rgb in 0..1 array, by hue / saturation / value: unsaturated and
    brighter than `white_v` is white, else gray."""
    if len(rgb) == 0:
        return np.zeros(0, int)
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    k = np.select([h < 12, h < 42, h < 80, h < 170, h < 265, h < 340], [0, 1, 2, 3, 4, 7], 0)    # red / orange / yellow / green / blue / purple / red
    return np.where(s < 0.22, np.where(v >= white_v, 5, 6), k)


def _rect(P, s):
    """minAreaRect of (n, 2) world points of pixel centres: (centre, L, W, yaw of the long side). Pixel centres span one pixel
    pitch `s` less than the area, so both sides gain `s`. The long axis comes from the second moments when the aspect is clear
    (sub-pixel), else from the rect. yaw: mod pi for a long rectangle, mod pi / 2 for a square one."""
    (cx, cy), (a, b), ang = cv2.minAreaRect((P / MM).astype(np.float32))
    yaw = math.radians(ang) + (0.0 if a >= b else math.pi / 2)
    L, W = max(a, b) * MM + s, min(a, b) * MM + s
    if L > 1.4 * W:
        yaw = math.atan2(*np.linalg.eigh(np.cov(P.T))[1][::-1, 1])      # principal axis (x, y) of the points
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    else:
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return np.array([cx, cy]) * MM, L, W, yaw


def _corners(c, L, W, yaw):
    ca, sa = math.cos(yaw), math.sin(yaw)
    return np.array([[c[0] + ca * a - sa * b, c[1] + sa * a + ca * b] for a, b in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])


_rays = functools.lru_cache(maxsize=4)(V.rays_from_intrinsics)          # 25 ms for the full image, and the same every call


def _table_level(d, rays, pos, R, step=8):
    """The pose (pos, R) corrected so that the table, the commonest surface, is the plane z = 0: a plane fit to the depth of the table
    (the nominal pose is off by up to ~0.2 deg and 1 mm: 4 mm of tilt over the field, more than the height of a plate stud). Only the
    tilt about the camera and the height are corrected; the camera's yaw and xy error stay."""
    d, r = d[::step, ::step], rays[::step, ::step]
    P = (pos + d[..., None] * (r @ R.T))[d > 0]
    if len(P) < 100:
        return pos, R
    cnt, edges = np.histogram(P[:, 2], bins=np.arange(-0.05, 0.05, MM))
    inl = np.abs(P[:, 2] - edges[np.argmax(cnt)]) < 8 * MM
    A = np.c_[P[:, :2], np.ones(len(P))]
    for _ in range(3):
        if inl.sum() < 50:
            return pos, R
        coef = np.linalg.lstsq(A[inl], P[inl, 2], rcond=None)[0]
        inl = np.abs(P[:, 2] - A @ coef) < 1.5 * MM
    n = np.array([-coef[0], -coef[1], 1.0]) / math.hypot(1, math.hypot(coef[0], coef[1]))
    ax = np.cross(n, [0, 0, 1])
    Q = Rot.from_rotvec(ax / max(np.linalg.norm(ax), 1e-12) * math.acos(min(n[2], 1.0))).as_matrix()      # the smallest turn that makes the table level
    z1 = np.mean(pos[2] + (P[inl] - pos) @ Q.T[:, 2])                    # the table's height after it
    return pos - [0, 0, z1], Q @ R


def detect(rgb, depth, intrinsics_or_rays, cam_pos, cam_R):
    rgb, depth = np.asarray(rgb), np.asarray(depth, np.float32)
    Hh, Ww = depth.shape
    rays = intrinsics_or_rays
    if isinstance(rays, dict):                                        # the full-image intrinsics: the matching rays of this stride
        I = rays
        rays = _rays(I["width"], I["height"], I["vfov_deg"])[::I["height"] // Hh, ::I["width"] // Ww]
    rays = np.asarray(rays)
    pos, R = np.asarray(cam_pos, float), (cam_R.as_matrix() if isinstance(cam_R, Rot) else np.asarray(cam_R, float))
    d = cv2.medianBlur(depth, 5 if Hh >= 720 else 3)                   # the depth noise (2 mm at 1 m) is much more than a stud
    pos, R = _table_level(d, rays, pos, R)                             # the camera's pose error tilts the table: level it
    dirs_z = rays @ R[2]                                                # world z of each pixel's ray
    h = np.where(d > 0, pos[2] + d * dirs_z, np.nan)                   # height over the table
    rgbf = rgb.astype(np.float32) / 255

    def world_xy(py, px, z):                                           # the pixels' rays meet the plane z
        t = (z - pos[2]) / dirs_z[py, px]
        return pos[:2] + (rays[py, px] * t[:, None]) @ R[:2].T

    def pitch_px(py, px):                                              # one pixel at the plane of the component, in metres
        r = np.nanmedian(d[py, px])
        return float(r * np.linalg.norm(rays[Hh // 2, Ww // 2 + 1] - rays[Hh // 2, Ww // 2]))

    # --- the plate: the largest slab at the plate's height ----------------------------------------------------------
    plate = None
    pm = ((h > H_PLATE[0]) & (h < H_PLATE[1])).astype(np.uint8)
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(pm)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        py, px = np.nonzero(lab == k)
        c, L, W, yaw = _rect(world_xy(py, px, PLATE_TOP), pitch_px(py, px))
        if L * W > 4 * PITCH * PITCH:
            ni, nj = max(1, round(L / PITCH)), max(1, round(W / PITCH))
            plate = dict(x=float(c[0]), y=float(c[1]), yaw=yaw, size=(L, W), studs=(ni, nj), pitch=PITCH,
                         top_z=float(np.percentile(h[py, px], 30)), corners=_corners(c, L, W, yaw), occupied=set())
            plate["outline"] = _outline(plate["corners"], PLATE_TOP, pos, R, rays)
            plate["origin"] = plate["corners"][0]                      # the corner of cell (0, 0); i along yaw, j along its left normal

    # a white brick is about twice as bright as the plate (reflectance 0.9 vs 0.45) under the same light, whatever the exposure
    white_v = 1.25 * float(np.median(rgbf[py, px].max(-1))) if plate else 0.5

    # --- the bricks: a component of one colour class, above the plate studs, that is a whole-stud rectangle -------------
    bm = cv2.morphologyEx(((h > H_BRICK[0]) & (h < H_BRICK[1])).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cls = np.full((Hh, Ww), -1, np.int8)
    cls[bm > 0] = colour_class(rgbf[bm > 0], 0.0)                     # to split bricks: all the unsaturated pixels are one class (white or gray is the mean's)
    bricks, k3 = [], 5 if Hh >= 720 else 3
    n, reg, st, _ = cv2.connectedComponentsWithStats(bm)
    for r in range(1, n):
        x0, y0, w, hh = st[r, :4]
        x0, y0 = max(x0 - k3, 0), max(y0 - k3, 0)
        sl = (slice(y0, y0 + hh + 2 * k3), slice(x0, x0 + w + 2 * k3))
        for k in np.unique(cls[sl][reg[sl] == r]):
            m = cv2.morphologyEx(((reg[sl] == r) & (cls[sl] == k)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            # ponytail: same-colour bricks touching side by side merge into one component and fail the whole-stud test; split them with the depth edge / the studs' lattice when a build needs it
            nc, lab = cv2.connectedComponents(m)
            for j in range(1, nc):
                py, px = np.nonzero(lab == j)
                py, px = py + y0, px + x0
                s = pitch_px(py, px)
                if len(py) * s * s < AREA_MIN:
                    continue
                zt = float(np.percentile(h[py, px], 40))               # the top face (the studs are 28 % of it, above)
                mine = lab == j
                ring = cv2.dilate(mine.astype(np.uint8), np.ones((k3, k3), np.uint8)).astype(bool) & ~mine & np.isfinite(h[sl])
                if (h[sl][ring] > zt - 3 * MM).mean() > 0.5:           # not a prism standing on lower ground: an arm part
                    continue
                top = h[py, px] > zt - 4 * MM                          # a tall stack shows the side of its top brick, at a slant, below the top face
                py, px = py[top], px[top]
                c, L, W, yaw = _rect(world_xy(py, px, zt), s)
                nl, nw = round(L / PITCH), round(W / PITCH)
                if not (1 <= min(nl, nw) <= MAX_STUDS[1] and max(nl, nw) <= MAX_STUDS[0]) or abs(L - nl * PITCH) > TOL_STUDS or abs(W - nw * PITCH) > TOL_STUDS or len(py) * s * s < FILL_MIN * L * W:
                    continue
                col = COLOURS[int(colour_class(rgbf[py, px].mean(0)[None], white_v)[0])]
                bricks.append(dict(colour=col, x=float(c[0]), y=float(c[1]), z_top=zt, yaw=yaw, L=L, W=W, studs="%dx%d" % (max(nl, nw), min(nl, nw)),
                                   px=float(px.mean()), py=float(py.mean()), outline=_outline(_corners(c, L, W, yaw), zt, pos, R, rays), hidden=False))
    _number(bricks, plate)
    return dict(bricks=bricks, plate=plate, height=h, cam=(pos, R, rays))


def _number(bricks, plate, held=()):
    """Sort the bricks (so ids stay stable while the set does), number them, and place each on the plate: on_plate, cell, layer, and the plate's occupied cells.
    `held` (the bricks in a gripper) are numbered in the same order but never placed on the plate."""
    key = lambda b: (round(b["x"], 3), b["y"])
    bricks.sort(key=key)
    ids = {id(b): i for i, b in enumerate(sorted(bricks + list(held), key=key))}
    for b in held:
        b["id"], b["on_plate"], b["cell"], b["layer"] = ids[id(b)], False, None, int(round((b["z_top"] - BRICK_H) / BRICK_H))
    if plate:
        plate["occupied"] = set()
    for b in bricks:
        b["id"], b["on_plate"], b["cell"], b["layer"] = ids[id(b)], False, None, int(round((b["z_top"] - BRICK_H) / BRICK_H))     # 0 = the lowest layer
        if plate:
            ca, sa = math.cos(plate["yaw"]), math.sin(plate["yaw"])
            u = np.array([[ca, sa], [-sa, ca]]) @ (np.array([b["x"], b["y"]]) - plate["origin"])    # in the plate's cell frame
            if 0 <= u[0] <= plate["size"][0] and 0 <= u[1] <= plate["size"][1]:
                dy = b["yaw"] - plate["yaw"]
                hx, hy = (abs(math.cos(dy)) * b["L"] + abs(math.sin(dy)) * b["W"]) / 2, (abs(math.sin(dy)) * b["L"] + abs(math.cos(dy)) * b["W"]) / 2
                i0, j0 = int(round((u[0] - hx) / PITCH)), int(round((u[1] - hy) / PITCH))
                nx, ny = int(round(2 * hx / PITCH)), int(round(2 * hy / PITCH))
                b["on_plate"], b["cell"] = True, (i0, j0)
                plate["occupied"] |= {(i, j) for i in range(i0, i0 + nx) for j in range(j0, j0 + ny)}
                b["layer"] = int(round((b["z_top"] - PLATE_TOP - BRICK_H) / BRICK_H))


def _inside(a, b):
    """Brick a's centre lies within brick b's top face."""
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    u, v = c * (a["x"] - b["x"]) + s * (a["y"] - b["y"]), -s * (a["x"] - b["x"]) + c * (a["y"] - b["y"])
    return abs(u) <= b["L"] / 2 and abs(v) <= b["W"] / 2


def remember(prev, res, win=4):
    """Object permanence over `detect` results (`prev` = the last merged result, or None): a previous brick that no new brick matches (same colour,
    centre within half its long side, z_top within 4 mm) is kept, flagged hidden=True, if the top camera now sees at or above its top there (the median
    of res["height"] in a +-`win` px window at its centroid: an arm or a brick over it); if it sees lower, the brick is gone and is dropped. A plate that
    is missing or clearly smaller (an arm over it) is kept from `prev` with hidden=True. Ids, on_plate, cell, layer, occupied are recomputed for the merged set.
    A new brick that is smaller than a previous one and lies inside it (same height) is that brick seen in part (an arm over the rest): the previous one is kept, hidden=True.
    A brick a gripper holds is not in `prev["bricks"]` (see track_held), so this never judges it."""
    for b in res["bricks"]:
        b["hidden"] = False
    if res["plate"]:
        res["plate"]["hidden"] = False
    if prev is None:
        return res
    h = res["height"]
    for i, n in enumerate(res["bricks"]):                                    # a hand over part of a brick leaves a smaller whole-stud rectangle in view: the brick stays whole
        for o in prev["bricks"]:
            if n["L"] * n["W"] < 0.9 * o["L"] * o["W"] and _inside(n, o) and abs(n["z_top"] - o["z_top"]) < 4 * MM:
                res["bricks"][i] = dict(o, hidden=True)
                break
    for o in prev["bricks"]:
        if any(abs(n["z_top"] - o["z_top"]) < 4 * MM and (n["colour"] == o["colour"] and math.hypot(n["x"] - o["x"], n["y"] - o["y"]) < o["L"] / 2
                                                          or _inside(o, n) or _inside(n, o)) for n in res["bricks"]):
            continue                                                         # re-detected (a partial view, or the colour flipping under a shadow, still overlaps it)
        py, px = int(round(o["py"])), int(round(o["px"]))
        w = h[max(py - win, 0):py + win + 1, max(px - win, 0):px + win + 1]
        w = w[np.isfinite(w)]
        if len(w) == 0 or np.median(w) > o["z_top"] - 4 * MM:
            res["bricks"].append(dict(o, hidden=True))
    p, q = res["plate"], prev["plate"]
    if q and (p is None or p["size"][0] * p["size"][1] < 0.8 * q["size"][0] * q["size"][1]):
        p = res["plate"] = dict(q, hidden=True)
    _number(res["bricks"], p)
    return res


GRASP_XY = 15 * MM                  # a gripped brick's top-face centre is this close (xy) to the TCP: the planner slides the grip up to ~14 mm along a long brick
GRASP_Z = (-2 * MM, 9 * MM)         # TCP height over a gripped brick's top face: GRASP_DZ 13 mm over the bottom face = 3.4 mm over the top, with a band for FK / depth error
OPEN_TOL, FINGER_MIN = 1.5 * MM, 2 * MM    # fingers "closed on it": each within OPEN_TOL of half a brick side (8 mm for the 16 mm side); the grasp command is 1.5 mm inside it and
                                    # the open one 2.6 mm outside, so OPEN_TOL splits them; below FINGER_MIN the hand shut on nothing
OVERLAP_DZ = 6 * MM


def _symm(a, b):
    """Yaw a wrapped to brick b's symmetry (mod pi; squares mod pi / 2), as detect reports it."""
    p = math.pi / 2 if abs(b["L"] - b["W"]) < MM else math.pi
    return (a + p / 2) % p - p / 2


def _same(a, b):
    """Two detections of one brick: footprints overlap (either centre inside the other) at about the same height."""
    return abs(a["z_top"] - b["z_top"]) < OVERLAP_DZ and (_inside(a, b) or _inside(b, a))


def track_held(det, prev, hands):
    """Bricks in a gripper, from the hand's pose and finger joints (proprioception, FK only; no ground truth). `hands`: per arm dict(name "A"/"B",
    pos (3,) and R (3x3 or Rotation) = the TCP from FK, opening = per-finger opening [m] from the measured finger joints). Held bricks live in
    det["held"] = {arm name: brick dict + d (hand-frame offset of the top-face centre), yaw_rel, half, held_by}, carried over from `prev`.
    Grasp: an arm with nothing held, fingers closed on a brick side (see OPEN_TOL), takes the brick of det["bricks"] (hidden ones too) whose top face
    is at the TCP (GRASP_XY, GRASP_Z): removed from det["bricks"], its hand-frame offset stored. Hold: its pose follows the hand each call and a new
    detection of the same brick is dropped. Release: fingers opened, or shut on nothing: it goes back into det["bricks"] at the hand's pose with
    hidden=True, for remember to match when the camera sees it again. Renumbers bricks and held together.
    ponytail: identity comes from the grasp (the nearest brick at the TCP); the held pose is the grasp-time offset, so slip in the fingers isn't seen
    (V5's B_hat at the looks could refine it)."""
    held = {k: dict(b) for k, b in ((prev or {}).get("held") or {}).items()}
    for hd in hands or ():
        k = hd["name"]
        pos, R = np.asarray(hd["pos"], float), hd["R"].as_matrix() if isinstance(hd["R"], Rot) else np.asarray(hd["R"], float)
        o, hyaw = np.asarray(hd["opening"], float), math.atan2(R[1, 0], R[0, 0])
        b = held.get(k)
        if b is None and o.min() >= FINGER_MIN:
            near = [(math.hypot(c["x"] - pos[0], c["y"] - pos[1]), n, c) for n, c in enumerate(det["bricks"])
                    if math.hypot(c["x"] - pos[0], c["y"] - pos[1]) <= GRASP_XY and GRASP_Z[0] <= pos[2] - c["z_top"] <= GRASP_Z[1]]
            for _, _, c in sorted(near, key=lambda t: t[:2]):
                nl, nw = map(int, c["studs"].split("x"))
                half = min((nw * PITCH / 2, nl * PITCH / 2), key=lambda w: np.abs(o - w).max())
                if np.abs(o - half).max() <= OPEN_TOL:
                    b = held[k] = dict(c, d=R.T @ (np.array([c["x"], c["y"], c["z_top"]]) - pos), yaw_rel=c["yaw"] - hyaw, half=half)
                    det["bricks"] = [n for n in det["bricks"] if n is not c]
                    break
        if b is None:
            continue
        c3 = pos + R @ b["d"]
        b.update(x=float(c3[0]), y=float(c3[1]), z_top=float(c3[2]), yaw=_symm(hyaw + b["yaw_rel"], b), hidden=False, held_by=k)
        if det.get("cam") is not None:
            b["outline"] = _outline(_corners(c3[:2], b["L"], b["W"], b["yaw"]), c3[2], *det["cam"])
            b["px"], b["py"] = map(float, b["outline"].mean(0))
        if o.mean() > b["half"] + OPEN_TOL or o.min() < FINGER_MIN:          # opened, or shut on nothing (slipped)
            del held[k]
            if not any(_same(n, b) for n in det["bricks"]):
                det["bricks"].append({a: v for a, v in b.items() if a not in ("d", "yaw_rel", "half", "held_by")} | dict(hidden=True))
    det["bricks"] = [n for n in det["bricks"] if not any(_same(n, b) for b in held.values())]       # the held one wins
    det["held"] = held
    _number(det["bricks"], det["plate"], list(held.values()))
    return det


def _outline(c2, z, pos, R, rays):
    """World corners (n, 2) at height z -> pixels (n, 2) of the image the rays belong to (the ray slopes are linear in the pixel)."""
    q = (np.c_[c2, np.full(len(c2), z)] - pos) @ R                      # camera frame, OpenGL: looks -z
    Hh, Ww = rays.shape[:2]
    sx, sy = rays[Hh // 2, :, 0] / -rays[Hh // 2, :, 2], rays[:, Ww // 2, 1] / -rays[:, Ww // 2, 2]
    return np.c_[(q[:, 0] / -q[:, 2] - sx[0]) / (sx[-1] - sx[0]) * (Ww - 1), (q[:, 1] / -q[:, 2] - sy[0]) / (sy[-1] - sy[0]) * (Hh - 1)]
