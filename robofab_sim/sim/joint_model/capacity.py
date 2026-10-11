"""Clutch capacity of one stud connection under a general axial/prying load.

The model is the per-connection force balance StableLego [StableLego] and
BrickSim's static solver [BrickSim] both build on: across the overlap patch
between an upper brick U and its support L,

  * compression (U bearing down on L) can act ANYWHERE in the overlap
    rectangle -- the cavity rim rests on the support's top face;
  * tension can act only AT THE STUDS, at most f_break per stud -- the clutch.

Given the connection wrench on U, (F_z, M_x, M_y) at the patch centre in the
patch frame, the utilisation u is the smallest s such that some admissible
distribution carries the wrench with every stud at <= s * f_break. It is an LP
(min s; A_c c + A_t t = w; 0 <= t <= s f_b; c >= 0) whose dual has only four
non-trivial vertices -- one per edge of the rectangle -- so it has a closed
form:

    u = max(0, max_e [ (n_e . q_e) F_z - n_ey M_x + n_ex M_y ] / (f_b D_e))

with n_e the inward normal of edge e, q_e a point on it, and
D_e = sum_i n_e . (p_i - q_e) the studs' summed lever arms about that edge.
Read physically: the prying moment about edge e divided by the most moment
the studs can resist pivoting about e. Pure pull-off gives F/(n f_b); a load
beyond an edge levers the patch open about that edge.

This is a plastic (lower-bound) limit analysis -- friction clutches keep
their force while slipping, so the whole stud row reaches f_break together.
Checked against BrickSim's static_solve in tests/test_capacity.py: S3's
cantilever at 35.6 N gives u = 2.36 here and 2.36 in BrickSim; a half-lapped
2x4 pull-off fails at 4.24 N/stud here and 4.06 in BrickSim's measured sweep.

Not modelled: shear and torsion about the stud axis (stud-wall bearing makes
LEGO very strong in both; a top-down insertion loads neither), and the
per-geometry 1xN vs 2xN difference BrickSim's calibration found.
"""

import numpy as np

F_BREAK_PER_STUD = 11.3      # N, joint_calibration.json (literature envelope 8-15, §2.3)
F_INSERT_PER_STUD = 8.9      # N, extraction > insertion (asymmetry 1.27)


class Patch:
    """Overlap patch of one connection, in its own frame (origin at the patch
    centre, z along the stud axis).

    studs: (n, 2) stud centres; rect: (xmin, xmax, ymin, ymax) of the region
    compression may act on (the overlap rectangle).
    """

    def __init__(self, studs, rect, f_break=F_BREAK_PER_STUD):
        self.studs = np.asarray(studs, float).reshape(-1, 2)
        self.rect = tuple(float(v) for v in rect)
        self.f_break = float(f_break)
        xmin, xmax, ymin, ymax = self.rect
        rows = []
        for n, q in (((1, 0), (xmin, 0)), ((-1, 0), (xmax, 0)),
                     ((0, 1), (0, ymin)), ((0, -1), (0, ymax))):
            n, q = np.array(n, float), np.array(q, float)
            D = float(((self.studs - q) @ n).sum())
            rows.append(np.array([n @ q, -n[1], n[0]]) / (self.f_break * D))
        self.coef = np.array(rows)            # (4, 3): (Fz, Mx, My) -> util per edge

    @property
    def n(self):
        return len(self.studs)

    def utilization(self, Fz, Mx, My):
        """u >= 1 means the clutch lets go. Scalar or broadcast arrays."""
        w = np.stack(np.broadcast_arrays(Fz, Mx, My), axis=-1)
        return np.maximum(0.0, (w @ self.coef.T).max(axis=-1))

    def critical_edge(self, Fz, Mx, My):
        """Index of the edge the patch pries open about (0:-x 1:+x 2:-y 3:+y)."""
        return int(np.argmax(self.coef @ np.array([Fz, Mx, My])))

    def pull_capacity(self):
        """Concentric pull-off force (N)."""
        return 1.0 / self.utilization(-1.0, 0.0, 0.0)


def grid_patch(nx, ny, pitch=0.008, f_break=F_BREAK_PER_STUD):
    """Patch of a full nx x ny overlap centred on the origin."""
    studs = [((i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch)
             for i in range(nx) for j in range(ny)]
    return Patch(studs, (-nx * pitch / 2, nx * pitch / 2, -ny * pitch / 2, ny * pitch / 2), f_break)
