"""Joint-stiffness metrics of a run, offline (plan_v4 r5 S4): rows JSONL + the --record-all .npz of each run.

    bash scripts/run.sh experiments/j_tables.py results/v4/j/acc/rows.jsonl [-o out.json]

Each row's record is <rows dir>/<structure>_r<repeat>.npz. Per welded brick, at the 8 corners of its body box
(a rigid transform of the box, so rotation counts; studs ignored): fire pose = the state entering the first
frame the weld is on; unloaded window = >= 30 consecutive frames with flag_arm_welded false; reference = mean
corner positions over the last 10 frames of the first unloaded window after the fire.
M1 snap correction: fire pose -> reference.  M2: peak excursion over frames >= fire + 0.1 s vs the reference.
M3: unloaded residual vs the reference at every later unloaded window and at build end (each: mean over its
last 10 frames).  A brick's series ends at its first weld break (the break itself is G2).
Gates: G2 0 breaks, G3 M2 <= 1.0 mm, G4 M3 <= 0.1 mm, G6 no divergence/timeout, G7 built == n.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PITCH, BRICK_H = 0.008, 0.0096
WINDOW, MEAN_N, SETTLE = 30, 10, 6          # frames; 6 frames = 0.1 s at 60 fps


def corners(q, L, W):
    """(F, 8, 3) world corners of the body box (mesh x = long side, centred in xy, bottom at z = 0)."""
    box = np.array([[sx * L * PITCH / 2, sy * W * PITCH / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                    for z in (0, BRICK_H)])
    x, y, z, w = q[:, 3], q[:, 4], q[:, 5], q[:, 6]          # newton transform: p, quat xyzw
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    return np.einsum("fij,cj->fci", R, box) + q[:, None, :3]


def windows(free, start):
    """[(a, b)) runs of >= WINDOW consecutive free frames, clipped to begin at `start`."""
    out, a = [], None
    for i in range(start, len(free) + 1):
        if i < len(free) and free[i]:
            a = i if a is None else a
        elif a is not None:
            if i - a >= WINDOW:
                out.append((a, i))
            a = None
    return out


def dev(c, ref):
    """max over the 8 corners of the displacement, in mm."""
    return np.linalg.norm(c - ref, axis=-1).max(-1) * 1000


def run_metrics(d, types):
    free = ~d["flag_arm_welded"]
    breaks = {}
    for name, f in zip(d["break_pairs"], d["break_frame"]):
        breaks.setdefault(str(name).split("<-")[0], int(f))
    bricks, m2 = {}, (0.0, None, None, None)
    m3 = 0.0
    for n, bid in enumerate(d["brick_ids"]):
        bid, f = str(bid), int(d["fire_frame_by_brick"][n])
        if f < 1:
            continue
        L, W = sorted((int(x) for x in types[bid].split("x")), reverse=True)
        end = min(breaks.get(bid, len(free) + 1) - 1, len(free))
        c = corners(d["brick_q"][:end, n], L, W)
        i0 = max(f - 2, 0)                                     # row of the last frame before the fire
        wins = windows(free[:end], f - 1)
        if not wins:
            bricks[bid] = {"fire_frame": f, "note": "no unloaded window after the fire"}
            continue
        ref = c[wins[0][1] - MEAN_N:wins[0][1]].mean(0)
        e2 = dev(c[f - 1 + SETTLE:], ref)
        k = int(e2.argmax()) if len(e2) else None
        res = [dev(c[b - MEAN_N:b].mean(0), ref) for _, b in wins[1:]] + [dev(c[-MEAN_N:].mean(0), ref)]
        bricks[bid] = {"fire_frame": f, "m1_mm": round(float(dev(c[i0], ref)), 4),
                       "m2_mm": round(float(e2[k]), 4) if k is not None else None,
                       "m2_frame": f + SETTLE + k if k is not None else None,
                       "m3_mm": round(float(max(res)), 4)}
        if k is not None and e2[k] > m2[0]:
            m2 = (float(e2[k]), bid, f + SETTLE + k, str(d["phase_names"][d["b_phase"][f + SETTLE + k - 1]]))
        m3 = max(m3, max(res))
    return bricks, m2, m3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    out = []
    print("%-14s %-6s %-14s %-6s %-9s %-26s %-8s %-5s  %s" % (
        "run", "built", "failure", "breaks", "u_peak", "M2 max mm (brick, frame, B phase)", "M3 max mm", "rtf", "G2 G3 G4 G6 G7"))
    for r in rows:
        tag = "%s_r%d" % (r["structure"], r["repeat"])
        npz = a.rows.parent / (tag + ".npz")
        if not npz.exists():
            print("%-14s no record %s" % (tag, npz.name))
            continue
        with np.load(npz) as z:
            d = {k: z[k] for k in z.files}
        types = (r.get("preflight") or {}).get("types")
        if not types:
            print("%-14s row has no preflight.types (run before j_tables)" % tag)
            continue
        bricks, m2, m3 = run_metrics(d, types)
        u = max(r["u_peak_by_step"].values(), default=0.0)
        fail = r["failure"]
        g = {"G2": r["breaks"] == 0 if isinstance(r["breaks"], int) else not r["breaks"],
             "G3": bool(m2[0] <= 1.0), "G4": bool(m3 <= 0.1),
             "G6": not r["error"] and not (fail or "").startswith(("divergence", "wait_timeout")),
             "G7": r["built"] == r["total"]}
        res = {"run": tag, "built": r["built"], "n": r["total"], "failure": fail, "breaks": r["breaks"],
               "u_peak_max": u, "m2_max_mm": m2[0], "m2_brick": m2[1], "m2_frame": m2[2], "m2_b_phase": m2[3],
               "m3_max_mm": float(m3), "rtf": r.get("rtf"), "gates": g, "bricks": bricks}
        out.append(res)
        print("%-14s %d/%-4d %-14s %-6s %-9.3f %-26s %-8.4f %-5s  %s" % (
            tag, r["built"], r["total"], fail, r["breaks"] if isinstance(r["breaks"], int) else len(r["breaks"]), u,
            "%.3f (%s, %s, %s)" % (m2[0], m2[1], m2[2], m2[3]), m3, r.get("rtf"),
            " ".join("pass" if v else "FAIL" for v in g.values())))
    path = a.out or a.rows.with_name(a.rows.stem + "_j_tables.json")
    path.write_text(json.dumps(out, indent=1))
    print("-> %s" % path)


def _self_check():
    q = np.tile([0, 0, 0.1, 0, 0, 0, 1.0], (2, 1))
    q[1, 2] += 0.001                                          # 1 mm up
    c = corners(q, 4, 2)
    assert abs(dev(c[1], c[0]) - 1.0) < 1e-6
    q[1, 2] = 0.1
    q[1, 3:7] = [0, 0, np.sin(0.005), np.cos(0.005)]            # 0.01 rad yaw: corner at (16, 8) mm moves ~ 0.01*17.9 mm
    assert abs(dev(c[0], corners(q, 4, 2)[1]) - 0.01 * np.hypot(16, 8)) < 0.01
    assert windows(np.array([1] * 40 + [0] + [1] * 30, bool), 5) == [(5, 40), (41, 71)]


if __name__ == "__main__":
    _self_check() if len(sys.argv) == 1 else main()
