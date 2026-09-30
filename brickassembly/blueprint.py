"""Images -> brick blueprint: silhouettes -> voxel grid -> tiled, bonded bricks.

Multi-view silhouette carving (a visual hull on the 8 x 8 x 9.6 mm LEGO grid),
then a per-layer greedy tiler that bridges overhangs first and staggers seams.
Output is the same brick list planner.STRUCTURES holds, so the planner, the
bracing assignment and the simulator take it unchanged.

    python blueprint.py --samples                   # (re)draw the sample images
    python blueprint.py --shape arch                # carve + tile + print a sample
    python blueprint.py --front my.png --top ring.png --width 8

Image conventions (dark shape on light background, or the reverse):
    front: columns = j (world +y), rows = layers k, top row is the highest layer
    side:  columns = i (world +x), rows = layers k
    top:   columns = j,            rows = i, top row is i = 0

ponytail: silhouettes only. Works for extrusion-like shapes (cube, hollow box,
arch, wall, L, stairs); a concavity no view can see (a closed cavity) is filled
in. A single photo of an arbitrary object needs image-to-3D first (plan §4.4).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import planner as P

HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "blueprints"
# Sample drawings keep the physical aspect: one stud is 8.0 mm, one layer 9.6 mm.
PX_STUD, PX_LAYER = 20, 24

SAMPLES = {
    "cube": {"front": "cube_front.png", "side": "cube_side.png"},
    "hollow_box": {"front": "hollow_box_front.png", "top": "hollow_box_top.png"},
    "arch": {"front": "arch_front.png", "depth": 2},
}

# Largest first. 2-wide bricks grip best; 1-wide ones fill what is left.
LIBRARY = ["2x6", "2x4", "2x3", "2x2", "1x6", "1x4", "1x2", "1x1"]


def mask(path):
    """Binary silhouette, cropped to the shape. Polarity is taken from the border."""
    g = np.asarray(Image.open(path).convert("L"), dtype=float)
    t = (g.min() + g.max()) / 2
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]).mean()
    m = g < t if border > t else g > t
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if not len(rows):
        raise ValueError("%s: no shape found" % path)
    return m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def resample(m, n_rows, n_cols):
    """Majority vote per grid cell."""
    r = np.linspace(0, m.shape[0], n_rows + 1).round().astype(int)
    c = np.linspace(0, m.shape[1], n_cols + 1).round().astype(int)
    return np.array([[m[r[a]:r[a + 1], c[b]:c[b + 1]].mean() > 0.5
                      for b in range(n_cols)] for a in range(n_rows)])


def carve(front, side=None, top=None, width=None, depth=None):
    """Visual hull V[i, j, k] from up to three silhouettes (paths).

    width (studs along j) defaults to the sample drawings' scale, PX_STUD
    pixels per stud -- pass it for anything else. The layer count follows from
    the front image's aspect at 8.0 mm studs and 9.6 mm layers.
    """
    f = mask(front)
    width = width or max(1, round(f.shape[1] / PX_STUD))
    layers = max(1, round(width * f.shape[0] / f.shape[1] * P.PITCH / P.BRICK_H))
    s = mask(side) if side else None
    t = mask(top) if top else None
    if depth is None:
        if s is not None:
            depth = max(1, round(layers * s.shape[1] / s.shape[0] * P.BRICK_H / P.PITCH))
        elif t is not None:
            depth = max(1, round(width * t.shape[0] / t.shape[1]))
        else:
            depth = 2
    F = resample(f, layers, width)[::-1]                    # [k, j], k up
    S = resample(s, layers, depth)[::-1] if s is not None else np.ones((layers, depth), bool)
    T = resample(t, depth, width) if t is not None else np.ones((depth, width), bool)
    return F.T[None, :, :] & S.T[:, None, :] & T[:, :, None]


def tile(V):
    """Per-layer greedy tiling -> [(id, type, i, j, k, yaw)].

    Cells with nothing below (overhangs) are covered first, by the brick that
    rests on the most distinct bricks below: that is what ties the two halves
    of an arch together. The rest go in raster order. Ties prefer fewer seams
    stacked over seams, then bigger bricks, then the layer's long axis, which
    alternates per layer (cross-bond).

    ponytail: greedy, one pass. Beam search over J(T) (plan §4.2) if a shape
    comes out badly bonded.
    """
    NI, NJ, NK = V.shape
    bricks, below = [], {}
    for k in range(NK):
        todo = {(i, j) for i in range(NI) for j in range(NJ) if V[i, j, k]}
        if not todo:
            break
        centre = np.mean(sorted(todo), axis=0)
        along_j = k % 2 == 0
        here = {}
        anchors = sorted(todo, key=lambda c: (k > 0 and c in below, c))
        for anchor in anchors:
            if anchor in here:
                continue
            best = None
            for btype in LIBRARY:
                for yaw in (0, 1):
                    nx, ny = P.footprint(btype, yaw)
                    if yaw and nx == ny:
                        continue
                    for di in range(nx):
                        for dj in range(ny):
                            i0, j0 = anchor[0] - di, anchor[1] - dj
                            cells = {(i0 + a, j0 + b) for a in range(nx) for b in range(ny)}
                            if not cells <= todo or any(c in here for c in cells):
                                continue
                            sup = {below[c] for c in cells if c in below}
                            if k and not sup:
                                continue
                            seams = sum(
                                1 for a, b in cells
                                for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1))
                                if n not in cells and (a, b) in below and n in below
                                and below[(a, b)] != below[n])
                            mid = (i0 + nx / 2 - 0.5 - centre[0], j0 + ny / 2 - 0.5 - centre[1])
                            score = (len(sup), -seams, len(cells), (ny >= nx) == along_j,
                                     -abs(mid[0]) - abs(mid[1]))
                            if best is None or score > best[0]:
                                best = (score, btype, i0, j0, yaw, cells)
            if best is None:
                raise ValueError("layer %d: cell %s cannot be reached by a supported "
                                 "brick -- the shape overhangs too far" % (k, anchor))
            _, btype, i0, j0, yaw, cells = best
            for c in cells:
                here[c] = len(bricks)
            bricks.append(("b_%03d" % len(bricks), btype, i0, j0, k, yaw))
        below = here
    # planner ids follow build order
    order = P.sequence(bricks)
    return [("b_%03d" % n,) + b[1:] for n, b in enumerate(order)]


def components(bricks):
    """Number of stud-connected groups, ignoring the baseplate."""
    parent = {b[0]: b[0] for b in bricks}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for n, b in enumerate(bricks):
        for s in P.supports(b, bricks[:n]):
            parent[root(s)] = root(b[0])
    return len({root(b[0]) for b in bricks})


def pieces(bricks):
    """Number of separate objects in the drawing: bricks joined by studs or touching side by
    side on a layer count as one (a stepped pyramid's outer base ring is tied to the rest only
    through the baseplate, so components() calls it several groups)."""
    cell = {(i, j, k): b[0] for b in bricks for (i, j), k in [(c, P.cells(b)[1]) for c in P.cells(b)[0]]}
    parent = {b[0]: b[0] for b in bricks}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for (i, j, k), a in cell.items():
        for n in ((i + 1, j, k), (i, j + 1, k), (i, j, k + 1)):
            if n in cell:
                parent[root(cell[n])] = root(a)
    return len({root(b[0]) for b in bricks})


def ascii(bricks):
    """Top-down text view of each layer, one letter per brick."""
    NI = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    NJ = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    out = []
    for k in range(max(b[4] for b in bricks), -1, -1):
        grid = [["." for _ in range(NJ)] for _ in range(NI)]
        for n, b in enumerate(bricks):
            if b[4] != k:
                continue
            cells, _ = P.cells(b)
            for i, j in cells:
                grid[i][j] = chr(ord("A") + n % 26)
        out.append("k=%d  %s" % (k, "  ".join("".join(r) for r in grid)))
    return "\n".join(out)


def load(shape):
    """Bricks for one of the sample shapes."""
    spec = SAMPLES[shape]
    paths = {v: SAMPLE_DIR / spec[v] for v in ("front", "side", "top") if v in spec}
    if not all(p.exists() for p in paths.values()):
        write_samples()
    return tile(carve(**paths, depth=spec.get("depth")))


def write_samples():
    """Draw the sample silhouettes: black shape on white, 20 px margin."""
    SAMPLE_DIR.mkdir(exist_ok=True)

    def canvas(studs, layers):
        img = Image.new("L", (studs * PX_STUD + 40, layers * PX_LAYER + 40), 255)
        return img, ImageDraw.Draw(img)

    def rect(d, studs, layers):
        d.rectangle([20, 20, 20 + studs * PX_STUD - 1, 20 + layers * PX_LAYER - 1], fill=0)

    for name in ("cube_front", "cube_side"):          # 4 studs x 4 layers
        img, d = canvas(4, 4)
        rect(d, 4, 4)
        img.save(SAMPLE_DIR / (name + ".png"))

    img, d = canvas(6, 3)                               # 6 studs x 3 layers
    rect(d, 6, 3)
    img.save(SAMPLE_DIR / "hollow_box_front.png")
    # 6 x 6 ring, 1-stud walls. The 4 x 4 hole matters: a fingertip is
    # 17.5 mm wide, so a gripper can reach into it; a 2 x 2 hole (16 mm) it cannot.
    img = Image.new("L", (6 * PX_STUD + 40,) * 2, 255)
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 20 + 6 * PX_STUD - 1, 20 + 6 * PX_STUD - 1], fill=0)
    d.rectangle([20 + PX_STUD, 20 + PX_STUD,
                 20 + 5 * PX_STUD - 1, 20 + 5 * PX_STUD - 1], fill=255)
    img.save(SAMPLE_DIR / "hollow_box_top.png")

    # Round arch, 10 studs x 5 layers: a 6-stud opening under an elliptical
    # crown. The grid turns the curve into a corbelled arch whose upper
    # courses overhang -- the case the stabilizer arm exists for.
    img, d = canvas(10, 5)
    rect(d, 10, 5)
    x0, x1 = 20 + 2 * PX_STUD, 20 + 8 * PX_STUD - 1
    y_bottom, y_spring = 20 + 5 * PX_LAYER - 1, 20 + 3 * PX_LAYER
    d.rectangle([x0, y_spring, x1, y_bottom], fill=255)
    d.ellipse([x0, y_spring - int(1.6 * PX_LAYER), x1, y_spring + int(1.6 * PX_LAYER)],
              fill=255)
    img.save(SAMPLE_DIR / "arch_front.png")


def _self_check():
    cube = load("cube")
    assert len(cube) == 8 and all(b[1] == "2x4" for b in cube), cube
    # alternate layers cross: long axis along j on k=0, along i on k=1
    assert {b[5] for b in cube if b[4] == 0} != {b[5] for b in cube if b[4] == 1}
    box = load("hollow_box")
    V = carve(SAMPLE_DIR / "hollow_box_front.png", top=SAMPLE_DIR / "hollow_box_top.png")
    assert V.shape == (6, 6, 3) and not V[1:5, 1:5].any() and V.sum() == 3 * 20
    assert sum(len(P.cells(b)[0]) for b in box) == V.sum()     # 100% coverage
    arch = load("arch")
    assert components(arch) == 1, ascii(arch)                  # the crown ties it
    for shape in SAMPLES:
        bricks = load(shape)
        occ = set()
        for b in bricks:                                       # no overlaps
            c, k = P.cells(b)
            keyed = {(i, j, k) for i, j in c}
            assert not occ & keyed, shape
            occ |= keyed
    print("blueprint self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true", help="redraw the sample images")
    ap.add_argument("--shape", choices=list(SAMPLES))
    ap.add_argument("--front", type=Path)
    ap.add_argument("--side", type=Path)
    ap.add_argument("--top", type=Path)
    ap.add_argument("--width", type=int, help="studs along the front view")
    ap.add_argument("--depth", type=int, help="studs deep, if no side/top view")
    args = ap.parse_args()

    if args.samples:
        write_samples()
        print("samples -> %s" % SAMPLE_DIR)
    if args.front:
        bricks = tile(carve(args.front, args.side, args.top, args.width, args.depth))
    elif args.shape:
        bricks = load(args.shape)
    else:
        _self_check()
        raise SystemExit
    print(ascii(bricks))
    print("%d bricks, %d connected group(s)" % (len(bricks), components(bricks)))
