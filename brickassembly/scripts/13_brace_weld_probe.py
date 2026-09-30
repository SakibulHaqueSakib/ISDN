"""P4 probe (evidence for a plan revision): brace force, weld stiffness, weld load readout.

Measurement only. dual_arm_sim.py is imported and subclassed, never edited. Results go to
results/v4/p4_brace_weld/ (one file per part). Run from anywhere:

  Part A -- what the stabilizer arm A really does to the structure (default weld, grasp_lp,
  weakest_joint, 1 run). Per frame while A.phase in (brace, hold): A<->brick contact force,
  and each welded brick's displacement from the pose it had when its weld fired.
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part a --shape arch --viewer null --test --num-frames 30000 --quiet
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part a --structure S3 --viewer null --test --num-frames 30000 --quiet
  the same run under another weld setting, or a repeat (writes part_a_<name>_<tag>.json):
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part a --shape arch --tag solimp9999 \
        --weld-solimp 0.9999,0.9999,0.001,0.5,2.0 --viewer null --test --num-frames 30000 --quiet
    (also run: --tag solimp99 --weld-solimp 0.99,0.9999,0.001,0.5,2.0 ; --structure S3 --tag r2 / solimp9999)
  Part A leaves plans/*.json and results/proto_episodes.jsonl alone (episodes go to OUT); the plan file
  make_plan rewrites is put back. A_normal is A (links+fingers) vs ALL bricks, incl. the one B carries;
  A_on_B_arm is the arm-arm normal force. Forces are the last substep of each frame.

  Part B -- weld stiffness in a static scene (4 bricks stacked / 3-brick staircase cantilever on
  the baseplate, welded), force ladder 1..50 N, weld settings x {16, 32} substeps:
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part b        (-> part_b.jsonl, part_b_summary.json)

  Part C -- read each weld's constraint wrench from solver.mjw_data.efc and validate it on the
  column (welds only, brick collisions off; and the default scene for the load share):
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part c        (-> part_c.json)

Plan v4 r5, step S1 (structure group STRUCT_GROUP = -2 on the baseplate and every welded brick), results in
results/v4/j/. Run from brickassembly/; parts B/C/S leave plans/*.json and results/proto_episodes.jsonl alone
(after the dual_arm_sim.py smoke below, restore them):
  J-a  weld static, grouped, solimp9999, 16 substeps (-> j_a.jsonl, j_a.json with the gate verdicts):
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part b --group --viewer null
    (pre-registered branch, only if lateral fails: add --torquescale 10 / 100 --suffix _ts10 / _ts100)
  J-b  grouped readout: column, staircase (cantilever), yawed-90 column, 5-degree column, bridge, the Jacobian
       check against qfrc_constraint and the bridge sum (-> j_b.json; every case is also sampled 30 more frames):
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part c --group --viewer null
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part c --group --substeps 32 --suffix _sub32 --viewer null   (diagnostic)
  Cube smoke build with the new weld + group (must place 8/8), and the contact count with / without the group
  (-> smoke_cube_group.json, smoke_cube_nogroup.json):
    bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 20000
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part s --shape cube --viewer null --test --num-frames 20000 --quiet
    bash scripts/run.sh scripts/13_brace_weld_probe.py --part s --shape cube --no-group --viewer null --test --num-frames 20000 --quiet
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import dual_arm_sim as D  # noqa: E402
import stability as ST  # noqa: E402
from dual_arm_sim import ex, P  # noqa: E402

OUT = HERE / "results" / "v4" / "p4_brace_weld"
BRACE = ("brace", "hold")
SETTINGS = {                       # weld attrs: name -> (eq_solref, eq_solimp)
    "current": ((0.002, 1.0), (0.95, 0.99, 0.001, 0.5, 2.0)),
    "solimp99": ((0.002, 1.0), (0.99, 0.9999, 0.001, 0.5, 2.0)),
    "solimp9999": ((0.002, 1.0), (0.9999, 0.9999, 0.001, 0.5, 2.0)),
    "solref004": ((0.004, 1.0), (0.95, 0.99, 0.001, 0.5, 2.0)),
}


def attrs(name):
    r, i = SETTINGS[name]
    return {"mujoco:eq_solref": r, "mujoco:eq_solimp": i}


def rnd(x, n=3):
    return None if x is None else round(float(x), n)


# =============================================================================
# Part A -- the real runs
# =============================================================================
class ProbeExample(D.Example):
    """dual_arm_sim.Example plus read-only recording; behaviour is unchanged."""

    def __init__(self, viewer, args, out_stem):
        super().__init__(viewer, args)
        self.log_path = OUT / (out_stem + "_episodes.jsonl")     # not results/proto_episodes.jsonl
        self.model.request_contact_attributes("force")
        self.pcontacts = self.pipeline.contacts()                 # own buffer, as cell/contacts.py
        self.sb = self.model.shape_body.numpy()
        self.ref = {}                                             # brick id -> (pos, quat) at weld fire
        self.first_miss = None                                    # frame of the first failed gate
        self.rows = {k: [] for k in ("frame", "a_phase", "b_phase", "fA_normal", "fA_on_placing", "fA_net",
                                     "fB_normal", "fA_on_B_arm", "max_disp_mm", "max_disp_brick")}
        self.cur, self.pending, self.events, self.ins_cur, self.ins_events = None, [], [], None, []
        self.out_brace = {"max_mm": 0.0, "brick": None, "frame": None, "b_phase": None, "by_b_phase": {}}

    def snap(self, s):
        super().snap(s)
        bid = s["brick_id"]
        if self.welds[bid][0][3]:
            self.ref[bid] = self.state_0.body_q.numpy()[self.body[bid]].copy()
        elif self.first_miss is None:
            self.first_miss = self.frame

    def contact_force(self):
        """Contact forces in N from the last substep, as a dict: A<->bricks normal sum (A_all), of which
        A<->the brick B is placing (A_placing), A's net force vector norm, B<->bricks normal sum,
        A<->B arm normal sum."""
        self.solver.update_contacts(self.pcontacts, self.state_0)
        c = self.pcontacts
        n = int(c.rigid_contact_count.numpy()[0])
        z = dict(A_all=0.0, A_placing=0.0, A_net=0.0, B_all=0.0, AB=0.0)
        if not n:
            return z
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        nrm, fv = c.rigid_contact_normal.numpy()[:n], c.force.numpy()[:n, :3]
        ok = (s0 >= 0) & (s1 >= 0)
        b0, b1 = np.where(ok, self.sb[np.maximum(s0, 0)], -9), np.where(ok, self.sb[np.maximum(s1, 0)], -9)
        na = self.n_arm_bodies
        isA = lambda b: (b >= 0) & (b < na)
        isB = lambda b: (b >= na) & (b < 2 * na)
        isBr = lambda b: b >= 2 * na
        fn = np.abs((fv * nrm).sum(1))
        mA = (isA(b0) & isBr(b1)) | (isBr(b0) & isA(b1))
        mB = (isB(b0) & isBr(b1)) | (isBr(b0) & isB(b1))
        mAB = (isA(b0) & isB(b1)) | (isB(b0) & isA(b1))
        pl = self.body[self.plan["sequence"][self.b_step]["brick_id"]] if self.b_step is not None else -7
        mP = mA & ((b0 == pl) | (b1 == pl))
        netA = (np.where(isA(b0)[:, None], fv, -fv) * mA[:, None]).sum(0)     # force on A (either shape order)
        return dict(A_all=float(fn[mA].sum()), A_placing=float(fn[mP].sum()), A_net=float(np.linalg.norm(netA)),
                    B_all=float(fn[mB].sum()), AB=float(fn[mAB].sum()))

    def disp(self, body_q):
        """{brick: displacement mm from its weld-fire pose} for every welded brick."""
        return {b: float(np.linalg.norm(body_q[self.body[b]][:3] - r[:3])) * 1000 for b, r in self.ref.items()}

    def step(self):
        super().step()
        body_q = self.state_0.body_q.numpy()
        d = self.disp(body_q)
        mb = max(d, key=d.get) if d else None
        md = d[mb] if d else 0.0
        a_ph, b_ph = self.A.phase, self.B.phase
        in_brace, in_ins = a_ph in BRACE, b_ph == "insert"
        fc = self.contact_force() if (in_brace or in_ins) else None
        fA, fAp, fAn, fB, fAB = (fc["A_all"], fc["A_placing"], fc["A_net"], fc["B_all"], fc["AB"]) if fc else (None,) * 5
        r = self.rows
        for k, v in (("frame", self.frame), ("a_phase", a_ph), ("b_phase", b_ph), ("fA_normal", rnd(fA)),
                     ("fA_on_placing", rnd(fAp)), ("fA_net", rnd(fAn)), ("fB_normal", rnd(fB)),
                     ("fA_on_B_arm", rnd(fAB)), ("max_disp_mm", rnd(md, 4)), ("max_disp_brick", mb)):
            r[k].append(v)
        # placer-induced: welded-brick motion outside A's brace phases
        if not in_brace and d:
            o = self.out_brace
            if md > o["max_mm"]:
                o.update(max_mm=rnd(md, 4), brick=mb, frame=self.frame, b_phase=b_ph)
            o["by_b_phase"][b_ph] = max(o["by_b_phase"].get(b_ph, 0.0), rnd(md, 4))
        # brace events
        if in_brace and self.cur is None:
            self.cur = dict(step=self.b_step, brick=self.plan["sequence"][self.b_step]["brick_id"],
                            start_frame=self.frame, fA=[], fAp=[], fAn=[], fAB=[], peak=dict(d), start=dict(d))
        if in_brace:
            c = self.cur
            c["fA"].append(fA)
            c["fAp"].append(fAp)
            c["fAn"].append(fAn)
            c["fAB"].append(fAB)
            for b, v in d.items():
                c["peak"][b] = max(c["peak"].get(b, 0.0), v)
        elif self.cur is not None:
            self.cur.update(end_frame=self.frame, end=dict(d))
            self.pending.append(self.cur)
            self.cur = None
        for e in list(self.pending):
            if self.frame >= e["end_frame"] + 60:                 # 1 s after A left the brace
                e["after"] = dict(d)
                self.events.append(self.close(e))
                self.pending.remove(e)
        # insert events (B pressing)
        if in_ins:
            if self.ins_cur is None:
                self.ins_cur = dict(step=self.b_step, start_frame=self.frame, fB=[])
            self.ins_cur["fB"].append(fB)
        elif self.ins_cur is not None:
            c, self.ins_cur = self.ins_cur, None
            self.ins_events.append(dict(step=c["step"], brick=self.plan["sequence"][c["step"]]["brick_id"],
                                        frames=len(c["fB"]), peak_N=rnd(max(c["fB"]), 2),
                                        mean_N=rnd(np.mean(c["fB"]), 2)))

    def close(self, e):
        top = lambda dd: max(dd.items(), key=lambda kv: kv[1]) if dd else (None, 0.0)
        pk, after, start = top(e["peak"]), top(e.get("after", {})), top(e["start"])
        s = self.plan["sequence"][e["step"]]
        return {"step": e["step"], "placing": e["brick"], "brace_target": (s["brace"] or {}).get("target_brick_id"),
                "gripped": (s["brace"] or {}).get("gripped_bricks"),
                "frames": len(e["fA"]), "start_frame": e["start_frame"],
                "A_normal_peak_N": rnd(max(e["fA"]), 2), "A_normal_mean_N": rnd(np.mean(e["fA"]), 2),
                "A_net_peak_N": rnd(max(e["fAn"]), 2), "A_net_mean_N": rnd(np.mean(e["fAn"]), 2),
                "A_on_placing_brick_peak_N": rnd(max(e["fAp"]), 2), "A_on_placing_brick_mean_N": rnd(np.mean(e["fAp"]), 2),
                "A_on_B_arm_peak_N": rnd(max(e["fAB"]), 2), "A_on_B_arm_mean_N": rnd(np.mean(e["fAB"]), 2),
                "disp_at_start_mm": rnd(start[1], 4), "disp_at_start_brick": start[0],
                "peak_disp_mm": rnd(pk[1], 4), "peak_disp_brick": pk[0],
                "disp_1s_after_retract_mm": rnd(after[1], 4), "disp_1s_after_brick": after[0],
                "springback_mm": rnd(pk[1] - after[1], 4),
                "peak_disp_per_brick_mm": {b: rnd(v, 3) for b, v in sorted(e["peak"].items())},
                "after_disp_per_brick_mm": {b: rnd(v, 3) for b, v in sorted(e.get("after", {}).items())}}


def part_a(args, viewer):
    name_tag = (args.shape or args.structure) + ("_" + args.tag if args.tag else "")
    stem = "part_a_" + name_tag
    if args.weld_solref or args.weld_solimp:                    # override before the scene is built
        r, i = SETTINGS["current"]
        D.WELD_ATTRS = {"mujoco:eq_solref": tuple(args.weld_solref or r),
                        "mujoco:eq_solimp": tuple(args.weld_solimp or i)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (stem + "_episodes.jsonl")).unlink(missing_ok=True)
    # make_plan rewrites plans/sim_*.json; restore what was there
    plan_f = HERE / "plans" / ("sim_%s_%s.json" % (args.shape or args.structure, args.strategy))
    bak = plan_f.read_bytes() if plan_f.exists() else None
    t0 = time.time()
    e = ProbeExample(viewer, args, stem)
    if bak is None:
        plan_f.unlink()
    else:
        plan_f.write_bytes(bak)
    err = None
    try:
        newton.examples.run(e, args)
    except BaseException as x:                                   # a failed run still leaves its file
        err = "%s: %s" % (type(x).__name__, str(x)[:400])
        print("RUN ERROR", err)
    n = len(e.plan["sequence"])
    ok = sum(x["success"] for x in e.episodes)
    # placer-induced motion that is not a brace's springback and not after the structure failed:
    # outside brace phases, >1 s after any brace ended, before the first failed gate
    r = e.rows
    skip = np.zeros(len(r["frame"]), bool)
    for v in e.events + [dict(start_frame=p["start_frame"], frames=len(p["fA"]) + 300) for p in e.pending]:
        skip |= (np.array(r["frame"]) >= v["start_frame"]) & (np.array(r["frame"]) <= v["start_frame"] + v["frames"] + 60)
    if e.first_miss is not None:
        skip |= np.array(r["frame"]) >= e.first_miss
    md = np.array([x if x is not None else 0.0 for x in r["max_disp_mm"]])
    idx = np.where(~skip & np.array([p not in BRACE for p in r["a_phase"]]))[0]
    clean = {"max_mm": None}
    if len(idx):
        j = idx[np.argmax(md[idx])]
        clean = {"max_mm": rnd(md[j], 4), "brick": r["max_disp_brick"][j], "frame": r["frame"][j],
                 "b_phase": r["b_phase"][j], "frames_counted": int(len(idx)),
                 "by_b_phase": {ph: rnd(md[idx][np.array(r["b_phase"])[idx] == ph].max(), 4)
                                for ph in sorted(set(np.array(r["b_phase"])[idx]))}}
    res = {"config": {"structure": e.name, "strategy": args.strategy, "brace_model": args.brace_model,
                      "weld_attrs": {k: list(v) for k, v in D.WELD_ATTRS.items()},
                      "substeps": e.substeps, "fps": e.fps},
           "completed": err is None and e.done and ok == n, "error": err, "done": e.done,
           "placed": "%d/%d" % (ok, n), "frames": e.frame, "sim_time_s": rnd(e.sim_time, 2),
           "wall_s": rnd(time.time() - t0, 1),
           "welded_disp_outside_brace": e.out_brace,
           "welded_disp_outside_brace_clean": clean, "first_failed_gate_frame": e.first_miss,
           "brace_events": e.events,
           "brace_events_unresolved_at_end": len(e.pending) + (e.cur is not None),
           "insert_events": e.ins_events,
           "note": "A_normal = A(links+fingers) vs all bricks incl. the brick B carries; forces are the last substep of each frame (update_contacts); disp = |dpos| of a welded "
                   "brick from the pose it had when its weld fired",
           "frames_data": e.rows}
    (OUT / (stem + ".json")).write_text(json.dumps(res))
    print("\n== PART A %s: completed=%s placed=%s err=%s" % (name_tag, res["completed"], res["placed"], err))
    print("outside-brace welded max disp (whole run): %s" % e.out_brace)
    print("outside-brace welded max disp (clean: before first failed gate, >1 s after each brace): %s" % clean)
    print("%4s %-6s %-6s %6s %6s %6s %6s %6s | %7s %-6s | %7s %-6s | %7s" % (
        "step", "place", "target", "Apk", "Amean", "Anetpk", "Aplcpk", "ABpk", "peakmm", "brick", "after1s", "brick",
        "spring"))
    for v in e.events:
        print("%4d %-6s %-6s %6.1f %6.1f %6.1f %6.1f %6.1f | %7.3f %-6s | %7.3f %-6s | %7.3f" % (
            v["step"], v["placing"], v["brace_target"], v["A_normal_peak_N"], v["A_normal_mean_N"],
            v["A_net_peak_N"], v["A_on_placing_brick_peak_N"], v["A_on_B_arm_peak_N"], v["peak_disp_mm"],
            v["peak_disp_brick"], v["disp_1s_after_retract_mm"], v["disp_1s_after_brick"], v["springback_mm"]))
    if e.ins_events:
        print("B insert force: peak max %.2f N, mean of means %.2f N over %d inserts" % (
            max(v["peak_N"] for v in e.ins_events), np.mean([v["mean_N"] for v in e.ins_events]),
            len(e.ins_events)))
    print("-> %s" % (OUT / (stem + ".json")))


# =============================================================================
# Parts B, C -- static scene: welded bricks on the baseplate, no arms
# =============================================================================
_MESH = {}
PITCH, BH = P.PITCH, P.BRICK_H
LAYOUT = {                           # kind -> [(x_stud_offset, layer)] of 2x4 bricks (long side along x)
    "column": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "cantilever": [(0, 0), (2, 1), (4, 2)],      # staircase: each brick overhangs the one below by 2 studs
}
OUTJ = HERE / "results" / "v4" / "j"
TILT_LIFT = 0.005                    # the 5-degree column is lifted so its low edge stays off the ground
CONJ = lambda q: np.array([-q[0], -q[1], -q[2], q[3]])


def layout(kind):
    """kind -> (NI, NJ, bricks): plate studs and bricks (L, W, origin p at the bottom-face centre, quat q, parents;
    a parent is a brick index or -1 for the baseplate). Old kinds are the P4 part-B/C scenes, unchanged."""
    ID = np.array([0.0, 0.0, 0.0, 1.0])
    Z0, BHt = D.Z0, BH
    if kind in LAYOUT:
        lay = LAYOUT[kind]
        NI = 4 + max(x for x, _ in lay)
        return NI, 2, [dict(L=4, W=2, p=np.array([(xo + 2) * PITCH, PITCH, Z0 + layer * BHt]), q=ID,
                            parents=[n - 1]) for n, (xo, layer) in enumerate(lay)]
    if kind == "bridge":                                        # one 2x4 on two 2x2 piers
        piers = [dict(L=2, W=2, p=np.array([(xo + 1) * PITCH, PITCH, Z0]), q=ID, parents=[-1]) for xo in (0, 2)]
        return 4, 2, piers + [dict(L=4, W=2, p=np.array([2 * PITCH, PITCH, Z0 + BHt]), q=ID, parents=[0, 1])]
    if kind == "column_yaw":                                    # lower brick yawed 90 deg: a 2x2 patch under brick 1
        qy = D.qz(math.pi / 2)
        return 4, 4, [dict(L=4, W=2, p=np.array([2 * PITCH, 2 * PITCH, Z0 + n * BHt]), q=qy if n == 0 else ID,
                           parents=[n - 1]) for n in range(4)]
    if kind == "column_tilt":                                   # the whole column leaned 5 deg about y
        R = D.qrot([0.0, math.radians(5), 0.0])
        piv = np.array([2 * PITCH, PITCH, Z0 + TILT_LIFT])
        return 4, 2, [dict(L=4, W=2, p=piv + D.rotate(R, np.array([0, 0, n * BHt])), q=R, parents=[n - 1])
                      for n in range(4)]
    raise KeyError(kind)


class Static:
    def __init__(self, kind, setting, substeps, collide=True, fps=60, group=False, torquescale=None):
        self.kind, self.substeps, self.fps = kind, substeps, fps
        self.dt = 1.0 / fps / substeps
        NI, NJ, bricks = layout(kind)
        self.NI, self.NJ, self.bricks = NI, NJ, bricks
        s = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(s)
        cx, cy = NI * PITCH / 2, NJ * PITCH / 2
        plate = []
        if collide:
            plate.append(s.add_shape_box(-1, hx=(NI + 4) * PITCH / 2, hy=(NJ + 4) * PITCH / 2, hz=D.Z0 / 2,
                                         xform=wp.transform((cx, cy, D.Z0 / 2), wp.quat_identity()), cfg=D.PROXY_CFG))
            hh = 0.5 * ex.STUD_HEIGHT - ex.COLLIDER_INSET
            for i in range(NI):
                for j in range(NJ):
                    plate.append(s.add_shape_cylinder(
                        -1, radius=ex.STUD_COLLIDER_RADIUS, half_height=hh, cfg=D.PROXY_CFG,
                        xform=wp.transform(((i + 0.5) * PITCH, (j + 0.5) * PITCH, D.Z0 + hh), wp.quat_identity())))
        cfg = D.BRICK_CFG if collide else newton.ModelBuilder.ShapeConfig(
            density=ex.BRICK_DENSITY, has_shape_collision=False, has_particle_collision=False)
        solimp = s.custom_attributes["mujoco:geom_solimp"]
        solimp.values = solimp.values or {}
        self.body, self.pos = [], []
        for n, br in enumerate(bricks):
            if (br["L"], br["W"]) not in _MESH:
                mv, mf = ex._make_brick_mesh(br["L"], br["W"])
                _MESH[(br["L"], br["W"])] = ex._build_mesh_with_sdf(mv, mf, color=D.COLORS[0])
            bd = s.add_body(xform=wp.transform(br["p"], br["q"]), label="b%d" % n)
            sh = s.add_shape_mesh(bd, mesh=_MESH[(br["L"], br["W"])], cfg=cfg, color=D.COLORS[n % len(D.COLORS)])
            solimp.values[sh] = (0.6, 0.95, 0.00075, 0.5, 2.5)
            if collide:
                D.add_proxies(s, bd, br["L"], br["W"], wp.transform_identity())
            self.body.append(bd)
            self.pos.append(br["p"])
        self.welds, self.weld_info = [], []                        # welds[n] = brick n's first weld
        for n, br in enumerate(bricks):
            for par in br["parents"]:
                if par < 0:
                    rel = wp.transform(br["p"], br["q"])
                else:
                    pj = bricks[par]
                    rel = wp.transform(D.rotate(CONJ(pj["q"]), br["p"] - pj["p"]), D.qmul(CONJ(pj["q"]), br["q"]))
                eq = s.add_equality_constraint_weld(
                    body1=-1 if par < 0 else self.body[par], body2=self.body[n], relpose=rel, enabled=True,
                    label="w%d_%d" % (par, n), custom_attributes=attrs(setting), torquescale=torquescale)
                self.weld_info.append(dict(eq=eq, up=n, lo=par))
                if par == br["parents"][0]:
                    self.welds.append(eq)
        s.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))
        self.model = m = s.finalize()
        if group:                                                  # plate + every brick (all welded) in one negative group
            g = m.shape_collision_group.numpy()
            g[plate] = D.STRUCT_GROUP
            g[np.isin(m.shape_body.numpy(), self.body)] = D.STRUCT_GROUP
            m.shape_collision_group.assign(g)
        cm = 8192
        m.rigid_contact_max = cm
        self.pipeline = newton.CollisionPipeline(m, reduce_contacts=True, rigid_contact_max=cm, broad_phase="sap",
                                                 max_triangle_pairs=4_000_000)
        self.solver = newton.solvers.SolverMuJoCo(
            m, solver="newton", integrator="implicitfast", iterations=15, ls_iterations=100, nconmax=cm,
            njmax=cm * 2, cone="elliptic", impratio=50.0, use_mujoco_contacts=False)
        self.state_0, self.state_1 = m.state(), m.state()
        self.control, self.contacts = m.control(), self.pipeline.contacts()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.state_0)
        self.mass = m.body_mass.numpy()[self.body]
        self.com = m.body_com.numpy()[self.body]
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as c:
                self.simulate()
            self.graph = c.graph

    def simulate(self):
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):                # body_f is set on both states and never cleared
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def run(self, frames):
        for _ in range(frames):
            wp.capture_launch(self.graph) if self.graph else self.simulate()

    def set_force(self, n, f, tau=(0, 0, 0)):
        a = np.zeros((self.model.body_count, 6), np.float32)
        a[self.body[n], :3] = f
        a[self.body[n], 3:] = tau
        for st in (self.state_0, self.state_1):
            st.body_f.assign(a)

    def q(self, n):
        return self.state_0.body_q.numpy()[self.body[n]]

    def finite(self):
        return bool(np.isfinite(self.state_0.body_q.numpy()).all())

    def tilt_mrad(self, n):
        z = D.rotate(self.q(n)[3:], np.array([0, 0, 1.0]))
        return 1000 * math.atan2(z[0], z[2])


def hold(st, n, f, frames=60):
    """Apply f on brick n for `frames`; return (mean pose over the last 6 frames, pose at half time, finite)."""
    st.set_force(n, f)
    tail, mid = [], None
    for i in range(frames):
        st.run(1)
        if i == frames // 2 - 1:
            mid = st.q(n)[:3].copy()
        if i >= frames - 6:
            tail.append(np.concatenate([st.q(n)[:3], [st.tilt_mrad(n)]]))
    st.set_force(n, (0, 0, 0))
    return np.mean(tail, 0), mid, st.finite()


def part_b(args):
    grp = args.group                                  # J-a: grouped solimp9999 at 16 substeps only
    out = OUTJ if grp else OUT
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    dirs = {"column": {"lat_x": (1, 0, 0), "up": (0, 0, 1), "down": (0, 0, -1)},
            "cantilever": {"down": (0, 0, -1), "up": (0, 0, 1), "lat_x": (1, 0, 0)}}
    if grp:
        dirs["cantilever"] = {"down": (0, 0, -1), "lat_x": (1, 0, 0)}
    fout = (out / ("j_a%s.jsonl" % args.suffix if grp else "part_b.jsonl")).open("w")
    for kind in ("column", "cantilever"):
        top = len(LAYOUT[kind]) - 1
        for setting in (("solimp9999",) if grp else SETTINGS):
            for sub in ((16,) if grp else (16, 32)):
                st = Static(kind, setting, sub, group=grp, torquescale=args.torquescale)
                st.run(60)                                            # settle under gravity
                ok = st.finite()
                base = dict(scene=kind, setting=setting, substeps=sub, mass_g=rnd(st.mass[0] * 1000, 3),
                            grouped=grp, torquescale=args.torquescale)
                for dname, u in dirs[kind].items():
                    for F in (1, 5, 10, 20, 50):
                        if not ok:
                            break
                        st.run(30)
                        ref = np.concatenate([st.q(top)[:3], [st.tilt_mrad(top)]])
                        pose, mid, ok = hold(st, top, np.array(u) * F)
                        st.run(30)
                        rest = np.concatenate([st.q(top)[:3], [st.tilt_mrad(top)]])
                        d = (pose[:3] - ref[:3]) * 1000
                        along = float(d @ np.array(u))
                        row = dict(base, direction=dname, F_N=F, finite=ok,
                                   defl_mm=[rnd(x, 4) for x in d], defl_along_mm=rnd(along, 4),
                                   half_time_along_mm=rnd(((mid - ref[:3]) * 1000) @ np.array(u), 4) if ok else None,
                                   tilt_mrad=rnd(pose[3] - ref[3], 3),
                                   stiffness_N_per_mm=rnd(F / along, 3) if abs(along) > 1e-4 else None,
                                   residual_after_unload_mm=rnd(np.linalg.norm((rest[:3] - ref[:3]) * 1000), 4)
                                   if st.finite() else None)
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()
                        rows.append(row)
                        if abs(along) > 200 or not ok:                # 20 cm: gone
                            ok = False
                    if not ok:
                        rows.append(dict(base, direction=dname, DIVERGED=True))
                        fout.write(json.dumps(rows[-1]) + "\n")
                        break
                print("%s %-10s sub=%d done (%.0fs) finite=%s" % (kind, setting, sub, time.time() - t0, ok), flush=True)
                del st
    fout.close()
    summ = {}
    for r in rows:
        k = "%s|%s|sub%d|%s" % (r["scene"], r["setting"], r["substeps"], r["direction"])
        if r.get("DIVERGED"):
            summ[k] = "DIVERGED"
        elif k in summ and summ[k] != "DIVERGED":
            summ[k]["by_F"][r["F_N"]] = r["stiffness_N_per_mm"]
        else:
            summ[k] = {"by_F": {r["F_N"]: r["stiffness_N_per_mm"]}}
    if grp:
        j_a_gates(rows, summ, out / ("j_a%s.json" % args.suffix), args.torquescale)
    else:
        (out / "part_b_summary.json").write_text(json.dumps(summ, indent=1))
    print("\nstiffness N/mm at F = 1 / 5 / 10 / 20 / 50 N (None = below 0.1 um)")
    for k, v in summ.items():
        print("%-42s %s" % (k, v if v == "DIVERGED" else "  ".join(
            "%s" % ("%8.2f" % x if x is not None else "    None") for x in v["by_F"].values())))
    print("-> %s" % fout.name)


def j_a_gates(rows, summ, path, torquescale):
    """Plan r5 gate J-a on the grouped part-B rows."""
    ok = [r for r in rows if not r.get("DIVERGED")]
    sec = lambda scene, d: [r["stiffness_N_per_mm"] for r in ok if r["scene"] == scene and r["direction"] == d]
    mn = lambda v: None if not v or None in v else min(v)
    val = {"column_lateral_min": mn(sec("column", "lat_x")), "column_up_min": mn(sec("column", "up")),
           "column_down_min": mn(sec("column", "down")),
           "cantilever_down_min": mn(sec("cantilever", "down")), "cantilever_lateral_min": mn(sec("cantilever", "lat_x")),
           "set_after_50N_max_mm": max(r["residual_after_unload_mm"] for r in ok if r["F_N"] == 50),
           "set_any_max_mm": max(r["residual_after_unload_mm"] for r in ok),
           "diverged": [r for r in rows if r.get("DIVERGED")] != [] or not all(r["finite"] for r in ok)}
    gates = {"column_lateral >= 20 N/mm": val["column_lateral_min"] is not None and val["column_lateral_min"] >= 20,
             "column_vertical >= 300 N/mm (up and down)": all(
                 v is not None and v >= 300 for v in (val["column_up_min"], val["column_down_min"])),
             "cantilever >= 30 N/mm (down and lateral)": all(
                 v is not None and v >= 30 for v in (val["cantilever_down_min"], val["cantilever_lateral_min"])),
             "set <= 0.05 mm after 50 N": val["set_after_50N_max_mm"] <= 0.05,
             "no divergence": not val["diverged"]}
    res = {"gate": "J-a", "torquescale": torquescale, "values": val, "gates": gates, "pass": all(gates.values()),
           "stiffness_by_F": summ}
    path.write_text(json.dumps(res, indent=1))
    print("\nJ-a:", json.dumps(val), "\n" + "\n".join("  %-46s %s" % (k, "PASS" if v else "FAIL") for k, v in gates.items()),
          "\n  ->", "PASS" if res["pass"] else "FAIL", path)


# --- Part C: weld constraint wrench from the solver ---------------------------
EQUALITY = 0                                                    # mujoco.mjtConstraint.mjCNSTR_EQUALITY


def weld_rows(st):
    """{newton weld idx: (f6 constraint-space force, pos residual 6)} from mjw_data.efc."""
    d = st.solver.mjw_data
    nefc = int(d.nefc.numpy()[0])
    typ, ids = d.efc.type.numpy()[0, :nefc], d.efc.id.numpy()[0, :nefc]
    frc, pos = d.efc.force.numpy()[0, :nefc], d.efc.pos.numpy()[0, :nefc]
    m = st.solver.mjc_eq_to_newton_eq.numpy()[0]
    out = {}
    for mid in sorted(set(ids[typ == EQUALITY])):
        rows = np.where((typ == EQUALITY) & (ids == mid))[0]
        assert len(rows) == 6 and rows.max() - rows.min() == 5, (mid, rows)   # one 6-row block per weld
        out[int(m[mid])] = (frc[rows].astype(float), pos[rows].astype(float), int(rows.min()), int(mid))
    return out


def brick_wrench(st, k, w, eq=None):
    """Wrench the weld exerts ON brick k (body2), about the brick origin, world frame.
    MuJoCo's efc force acts on body1 as +J^T f, so on body2 it is -f. The 3 rotational rows are
    0.5 * torquescale * R2^T-rotated (quaternion-error Jacobian), hence torque = 0.5 * ts * R2 f_rot."""
    f, _, _, _ = w
    ts = float(st.model.equality_constraint_torquescale.numpy()[st.welds[k] if eq is None else eq])
    q = st.q(k)
    return -f[:3], -0.5 * ts * D.rotate(q[3:], f[3:6])


def expected_wrench(st, k, ext_n, ext_f):
    """Wrench on brick k from its weld, for a static chain: minus the (gravity + applied) load of
    bricks k..top, about brick k's origin."""
    o = st.q(k)[:3]
    F, T = np.zeros(3), np.zeros(3)
    for j in range(k, len(st.body)):
        qj = st.q(j)
        c = qj[:3] + D.rotate(qj[3:], st.com[j])
        fj = np.array([0, 0, -9.81 * st.mass[j]])
        if j == ext_n:
            fj = fj + ext_f
        F += fj
        T += np.cross(c - o, fj)
    return -F, -T


def part_c(args):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"fields": {
        "type": "solver.mjw_data.efc.type (nworld, njmax), ==0 for equality", "id": "solver.mjw_data.efc.id = MuJoCo eq id",
        "force": "solver.mjw_data.efc.force (nworld, njmax), constraint space", "pos": "solver.mjw_data.efc.pos",
        "nefc": "solver.mjw_data.nefc[0]", "eq_map": "solver.mjc_eq_to_newton_eq[world, mujoco eq] = Newton weld index"},
        "cases": []}
    for collide in (False, True):
        st = Static("column", "current", 16, collide=collide)
        st.run(120)
        top = len(st.body) - 1
        print("\n=== column, brick collisions %s; masses g=%s, torquescale=%s, anchors=%s" % (
            "ON" if collide else "OFF (welds carry everything)", np.round(st.mass * 1000, 3),
            st.model.equality_constraint_torquescale.numpy(), st.model.equality_constraint_anchor.numpy()[0]))
        w = weld_rows(st)
        print("efc rows per weld: %s" % {k: "rows %d..%d (mj eq %d)" % (v[2], v[2] + 5, v[3]) for k, v in w.items()})
        for name, u, F in (("gravity only", (0, 0, 0), 0), ("pull up 0.30 N", (0, 0, 1), 0.30),
                           ("pull up 1.0 N", (0, 0, 1), 1.0), ("push down 1.0 N", (0, 0, -1), 1.0),
                           ("lateral x 0.10 N", (1, 0, 0), 0.10), ("lateral x 0.50 N", (1, 0, 0), 0.50),
                           ("lateral y 0.20 N", (0, 1, 0), 0.20)):
            st.set_force(top, np.array(u, float) * F)
            st.run(120)                                                     # 2 s to settle
            f_ext = np.array(u, float) * F
            w = weld_rows(st)
            case = {"collide": collide, "load": name, "welds": []}
            print("-- %s" % name)
            print("   weld  measured F_on_brick [N] (x,y,z)     expected F_on_brick        |  measured M [mN.m]        expected M")
            for k in range(len(st.body)):
                fm, tm = brick_wrench(st, k, w[st.welds[k]])
                fe, te = expected_wrench(st, k, top, f_ext)
                print("   w%d   %8.4f %8.4f %8.4f          %8.4f %8.4f %8.4f    | %8.3f %8.3f %8.3f    %8.3f %8.3f %8.3f" % (
                    k, *fm, *fe, *(tm * 1000), *(te * 1000)))
                case["welds"].append({"weld": k, "F_measured": [rnd(x, 5) for x in fm],
                                      "F_expected": [rnd(x, 5) for x in fe],
                                      "M_measured_mNm": [rnd(x * 1000, 4) for x in tm],
                                      "M_expected_mNm": [rnd(x * 1000, 4) for x in te],
                                      "pos_residual_mm": [rnd(x * 1000, 4) for x in w[st.welds[k]][1][:3]]})
            res["cases"].append(case)
            st.set_force(top, (0, 0, 0))
            st.run(60)
        del st
    (OUT / "part_c.json").write_text(json.dumps(res, indent=1))
    print("-> %s" % (OUT / "part_c.json"))


# --- Part C --group: J-b, the grouped weld readout against analytic statics ----------------
G = 9.81
J_LOADS = (("gravity", (0, 0, 0)), ("+1 N z", (0, 0, 1)), ("-1 N z", (0, 0, -1)),
           ("0.5 N x", (0.5, 0, 0)), ("0.5 N y", (0, 0.5, 0)))
TOL = dict(rel=0.01, F_abs=1e-3, M_abs=1e-4)            # 1 % or 1 mN / 0.1 mN.m


def patch_of(st, n, par):
    """The overlap of brick n's studs on `par` (a brick index; -1 = the baseplate, taken as the whole footprint)
    as a capacity Patch in brick n's local axes (as stability.connections builds it), and its centre there."""
    studs = lambda b: np.array([((i - (b["L"] - 1) / 2) * PITCH, (j - (b["W"] - 1) / 2) * PITCH)
                                for i in range(b["L"]) for j in range(b["W"])])
    bn = st.bricks[n]
    up = studs(bn)
    if par >= 0:
        bl = st.bricks[par]
        lo = np.array([D.rotate(CONJ(bn["q"]), bl["p"] + D.rotate(bl["q"], np.array([x, y, 0.0])) - bn["p"])[:2]
                       for x, y in studs(bl)])
        up = up[[bool((np.abs(lo - u).max(axis=1) < 1e-6).any()) for u in up]]
    cen = up.mean(axis=0)
    rect = (up[:, 0].min() - PITCH / 2 - cen[0], up[:, 0].max() + PITCH / 2 - cen[0],
            up[:, 1].min() - PITCH / 2 - cen[1], up[:, 1].max() + PITCH / 2 - cen[1])
    return ST.Patch(up - cen, rect, ST.F_BREAK_PER_STUD), np.array([cen[0], cen[1], 0.0])


def nominal_loads(st, ext_n, f, tau):
    """[(COM world, force)] of every brick at its nominal pose, gravity plus the applied (f, couple tau) on ext_n."""
    out = []
    for j, b in enumerate(st.bricks):
        fj = np.array([0, 0, -G * st.mass[j]]) + (np.asarray(f, float) if j == ext_n else 0)
        out.append((b["p"] + D.rotate(b["q"], st.com[j]), fj))
    return out


def subtree(st, k):
    """k and every brick that rests on k (through parents[0], the chain used by all single-parent scenes)."""
    out = {k}
    for j, b in enumerate(st.bricks):
        if j != k and b["parents"][0] in out:
            out.add(j)
    return out


def analytic_on(st, k, loads, ext_n, tau):
    """Wrench on brick k from its weld(s), world, about brick k's nominal origin: minus the load of k's subtree."""
    sub = sorted(subtree(st, k))
    o = st.bricks[k]["p"]
    F = -sum(loads[j][1] for j in sub)
    T = -sum(np.cross(loads[j][0] - o, loads[j][1]) for j in sub) - (np.asarray(tau, float) if ext_n in sub else 0)
    return F, T


def to_patch(st, n, c_world, R_n, F, M, p_n):
    """(F_patch(3), M_patch(3)): wrench on brick n about its origin -> about the patch centre, in nominal axes."""
    Mc = M + np.cross(p_n - c_world, F)
    return R_n.T @ F, R_n.T @ Mc


def rotmat(q):
    return np.stack([D.rotate(q, e) for e in np.eye(3)], axis=1)


def readout(st, w, info, bricks_R):
    """Measured (F, M about upper's origin, world) of one weld on its upper brick; M is rotated by the *current*
    upper brick orientation (as the probe's brick_wrench does)."""
    return brick_wrench(st, info["up"], w[info["eq"]], eq=info["eq"])


def errs(fm, fa, mm, ma):
    eF, eM = np.linalg.norm(fm - fa), np.linalg.norm(mm - ma)
    okF = eF <= max(TOL["rel"] * np.linalg.norm(fa), TOL["F_abs"])
    okM = eM <= max(TOL["rel"] * np.linalg.norm(ma), TOL["M_abs"])
    return dict(F_err_mN=rnd(eF * 1e3, 4), M_err_mNm=rnd(eM * 1e3, 5), F_ok=bool(okF), M_ok=bool(okM))


def jacobian_qfrc(st, k):
    """qfrc_constraint of brick k's free joint: (force world, torque about the origin) both as read; the dof
    order is looked up through solver.mjc_dof_to_newton_dof."""
    m = st.model
    j = int(np.where(m.joint_child.numpy() == st.body[k])[0][0])
    q0 = int(m.joint_qd_start.numpy()[j])
    dmap = st.solver.mjc_dof_to_newton_dof.numpy()[0]
    qf = st.solver.mjw_data.qfrc_constraint.numpy()[0]
    v = np.array([qf[int(np.where(dmap == q0 + i)[0][0])] for i in range(6)], float)
    return v[:3], v[3:]


def part_c_grouped(args):
    OUTJ.mkdir(parents=True, exist_ok=True)
    res = {"gate": "J-b", "tolerance": "1 % or 1 mN (F) / 0.1 mN.m (M_patch) absolute; u 1 %; Jacobian and bridge sum 1 % or the same absolutes",
           "weld": "solimp9999, %d substeps, plate + bricks in STRUCT_GROUP" % args.substeps, "torquescale": args.torquescale,
           "statics": [], "jacobian": [], "bridge": []}
    for kind in ("column", "cantilever", "column_yaw", "column_tilt", "bridge"):
        st = Static(kind, "solimp9999", args.substeps, group=True, torquescale=args.torquescale)
        st.run(120)
        top = len(st.body) - 1
        print("\n=== %s (grouped): %d bricks, %d welds, masses g %s, torquescale %s" % (
            kind, len(st.body), len(st.weld_info), np.round(st.mass * 1000, 3),
            np.unique(st.model.equality_constraint_torquescale.numpy())))
        cases = list(J_LOADS) + ([("couple y 2 mN.m + 1 N z", (0, 0, 1))] if kind == "bridge" else [])
        for name, u in cases:
            f = np.array(u, float)
            tau = np.array([0, 2e-3, 0]) if name.startswith("couple") else np.zeros(3)
            st.set_force(top, f, tau)
            st.run(120)
            w = weld_rows(st)
            loads = nominal_loads(st, top, f, tau)
            geo = {}
            for info in st.weld_info:
                bn = st.bricks[info["up"]]
                patch, cl = patch_of(st, info["up"], info["lo"])
                R_n = rotmat(bn["q"])
                geo[info["eq"]] = (patch, bn["p"] + R_n @ cl, R_n)

            def sample(w):
                """{eq: (Fp, Mp, u, Fpa, Mpa, ua)}: measured and analytic patch-frame wrench + utilisation."""
                out = {}
                for info in st.weld_info:
                    n = info["up"]
                    patch, c_w, R_n = geo[info["eq"]]
                    Fm, Mm = readout(st, w, info, None)
                    Fp, Mp = to_patch(st, n, c_w, R_n, Fm, Mm, st.bricks[n]["p"])
                    r = [Fp, Mp, float(patch.utilization(Fp[2], Mp[0], Mp[1])), None, None, None]
                    if kind != "bridge":
                        Fa, Ma = analytic_on(st, n, loads, top, tau)
                        r[3], r[4] = to_patch(st, n, c_w, R_n, Fa, Ma, st.bricks[n]["p"])
                        r[5] = float(patch.utilization(r[3][2], r[4][0], r[4][1]))
                    out[info["eq"]] = r
                return out

            def u_ok(um, ua):
                return bool(abs(um - ua) <= max(TOL["rel"] * ua, 1e-4))  # ponytail: abs floor 1e-4 in u (u ~ 1e-3 under gravity)

            first = sample(w)
            frames = []
            for _ in range(30):                               # the readout chatters frame to frame: 30 more samples
                st.run(1)
                frames.append(sample(weld_rows(st)))
            for info in st.weld_info:
                n = info["up"]
                Fp, Mp, um, Fpa, Mpa, ua = first[info["eq"]]
                row = dict(scene=kind, load=name, weld="%s->%d" % (info["lo"], n), u_measured=rnd(um, 6),
                           F_measured=[rnd(x, 5) for x in Fp], M_patch_measured_mNm=[rnd(x * 1e3, 4) for x in Mp])
                if kind != "bridge":
                    e = errs(Fp, Fpa, Mp, Mpa)
                    e["u_analytic"] = rnd(ua, 6)
                    e["u_ok"] = u_ok(um, ua)
                    e["u_rel_err_pct"] = rnd(100 * abs(um - ua) / ua, 4) if ua > 1e-3 else None
                    ss = [errs(x[eq][0], x[eq][3], x[eq][1], x[eq][4]) for x in frames for eq in [info["eq"]]]
                    mean_e = errs(np.mean([x[info["eq"]][0] for x in frames], 0), Fpa,
                                  np.mean([x[info["eq"]][1] for x in frames], 0), Mpa)
                    e["samples30"] = dict(
                        F_err_mN_max=max(x["F_err_mN"] for x in ss), M_err_mNm_max=max(x["M_err_mNm"] for x in ss),
                        samples_failing=sum(not (x["F_ok"] and x["M_ok"]) for x in ss),
                        u_samples_failing=sum(not u_ok(x[info["eq"]][2], x[info["eq"]][5]) for x in frames),
                        mean_F_ok=mean_e["F_ok"], mean_M_ok=mean_e["M_ok"], mean_F_err_mN=mean_e["F_err_mN"],
                        mean_M_err_mNm=mean_e["M_err_mNm"])
                    row.update(F_analytic=[rnd(x, 5) for x in Fpa], M_patch_analytic_mNm=[rnd(x * 1e3, 4) for x in Mpa], **e)
                res["statics"].append(row)
                print("  %-22s %-6s F %-30s Fan %-30s M %-24s Man %-24s %s" % (
                    name, row["weld"], np.round(Fp, 4), np.round(row.get("F_analytic", [0] * 3), 4),
                    np.round(Mp * 1e3, 3), np.round(row.get("M_patch_analytic_mNm", [0] * 3), 3),
                    "u %.5f/%s ok=%s (30 frames: %d F/M and %d u samples fail)" % (
                        row["u_measured"], row.get("u_analytic"), row.get("F_ok") and row.get("M_ok") and row.get("u_ok"),
                        row["samples30"]["samples_failing"], row["samples30"]["u_samples_failing"])
                    if kind != "bridge" else "u %.5f" % row["u_measured"]))
            w = weld_rows(st)                                  # the Jacobian and bridge checks read one instant
            if kind in ("column", "column_yaw", "column_tilt"):
                # (ii) Jacobian: the constraint wrench each brick receives vs qfrc_constraint of its free joint
                for k in range(len(st.body)):
                    own = readout(st, w, st.weld_info[k], None)
                    Fn, Mn = np.array(own[0]), np.array(own[1])
                    if k + 1 < len(st.body):
                        Fu, Mu = readout(st, w, st.weld_info[k + 1], None)     # wrench on brick k+1; k receives the reaction
                        Fn = Fn - Fu
                        Mn = Mn - Mu - np.cross(st.bricks[k + 1]["p"] - st.bricks[k]["p"], Fu)
                    fq, tq = jacobian_qfrc(st, k)
                    Rk = rotmat(st.q(k)[3:])
                    cand = {"torque_local": Rk @ tq, "torque_world": tq}
                    r = dict(scene=kind, gated=kind == "column", load=name, brick=k, F_weld=[rnd(x, 5) for x in Fn], F_qfrc=[rnd(x, 5) for x in fq],
                             M_weld_mNm=[rnd(x * 1e3, 4) for x in Mn])
                    r["F_err_mN"] = rnd(np.linalg.norm(Fn - fq) * 1e3, 4)
                    r["F_ok"] = bool(np.linalg.norm(Fn - fq) <= max(TOL["rel"] * np.linalg.norm(Fn), TOL["F_abs"]))
                    for cn, tv in cand.items():
                        r["M_err_mNm_" + cn] = rnd(np.linalg.norm(Mn - tv) * 1e3, 5)
                        r["M_ok_" + cn] = bool(np.linalg.norm(Mn - tv) <= max(TOL["rel"] * np.linalg.norm(Mn), TOL["M_abs"]))
                    r["M_qfrc_local_to_world_mNm"] = [rnd(x * 1e3, 4) for x in cand["torque_local"]]
                    res["jacobian"].append(r)
                    print("    jac brick %d: F_err %.4f mN ok=%s | M_err mN.m local-frame %.5f (ok=%s) world-frame %.5f (ok=%s)" % (
                        k, r["F_err_mN"], r["F_ok"], r["M_err_mNm_torque_local"], r["M_ok_torque_local"],
                        r["M_err_mNm_torque_world"], r["M_ok_torque_world"]))
            if kind == "bridge":
                # (iii) the two welds on the top brick sum to minus its load; the piers' plate welds sum to everything
                tops = [i for i in st.weld_info if i["up"] == top]
                Fs = sum(readout(st, w, i, None)[0] for i in tops)
                Ms = sum(readout(st, w, i, None)[1] for i in tops)
                Fa, Ma = analytic_on(st, top, loads, top, tau)
                b = dict(load=name, F_sum=[rnd(x, 5) for x in Fs], F_analytic=[rnd(x, 5) for x in Fa],
                         M_sum_mNm=[rnd(x * 1e3, 4) for x in Ms], M_analytic_mNm=[rnd(x * 1e3, 4) for x in Ma],
                         **errs(Fs, Fa, Ms, Ma))
                pier = [i for i in st.weld_info if i["up"] != top]
                o = st.bricks[top]["p"]
                Fp_ = sum(readout(st, w, i, None)[0] for i in pier)
                Mp_ = sum(readout(st, w, i, None)[1] + np.cross(st.bricks[i["up"]]["p"] - o, readout(st, w, i, None)[0]) for i in pier)
                Fall = -sum(fl for _, fl in loads)
                Mall = -sum(np.cross(c - o, fl) for c, fl in loads) - tau
                b["piers_sum_vs_total"] = errs(Fp_, Fall, Mp_, Mall)
                b["split"] = [dict(weld="%s->%d" % (i["lo"], i["up"]), F=[rnd(x, 5) for x in readout(st, w, i, None)[0]],
                                   M_mNm=[rnd(x * 1e3, 4) for x in readout(st, w, i, None)[1]]) for i in tops]
                b["split_u"] = [r["u_measured"] for r in res["statics"][-len(st.weld_info):] if r["weld"].endswith("->%d" % top)]
                ss = []
                for _ in range(30):                            # end-of-frame chatter: 30 more samples of the sum
                    st.run(1)
                    w2 = weld_rows(st)
                    ss.append(errs(sum(readout(st, w2, i, None)[0] for i in tops), Fa,
                                   sum(readout(st, w2, i, None)[1] for i in tops), Ma))
                b["samples30"] = dict(F_err_mN_max=max(x["F_err_mN"] for x in ss), M_err_mNm_max=max(x["M_err_mNm"] for x in ss),
                                      samples_failing=sum(not (x["F_ok"] and x["M_ok"]) for x in ss))
                res["bridge"].append(b)
                print("    bridge sum F %s vs %s M %s vs %s -> ok=%s; split %s" % (
                    np.round(Fs, 4), np.round(Fa, 4), np.round(Ms * 1e3, 3), np.round(Ma * 1e3, 3),
                    b["F_ok"] and b["M_ok"], [(x["weld"], x["F"]) for x in b["split"]]))
            st.set_force(top, (0, 0, 0))
            st.run(60)
        del st
    res["statics_pass"] = all(r.get("F_ok", True) and r.get("M_ok", True) and r.get("u_ok", True) for r in res["statics"])
    st30 = [r["samples30"] for r in res["statics"] if "samples30" in r]
    res["statics_pass_all_30_samples"] = all(x["samples_failing"] == 0 and x["u_samples_failing"] == 0 for x in st30)
    res["statics_pass_30_frame_mean"] = all(x["mean_F_ok"] and x["mean_M_ok"] for x in st30)
    res["statics_failing"] = [(r["scene"], r["load"], r["weld"]) for r in res["statics"]
                              if not (r.get("F_ok", True) and r.get("M_ok", True) and r.get("u_ok", True))]
    res["statics_failing_samples"] = sum(x["samples_failing"] for x in st30)
    res["statics_worst_sample"] = dict(F_err_mN=max(x["F_err_mN_max"] for x in st30), M_err_mNm=max(x["M_err_mNm_max"] for x in st30))
    res["statics_samples_total"] = 30 * len(st30)
    jg = [r for r in res["jacobian"] if r["gated"]]
    res["jacobian_pass_local"] = all(r["F_ok"] and r["M_ok_torque_local"] for r in jg)
    res["jacobian_pass_world"] = all(r["F_ok"] and r["M_ok_torque_world"] for r in jg)
    res["jacobian_rotated_scenes_pass_local"] = all(r["F_ok"] and r["M_ok_torque_local"] for r in res["jacobian"] if not r["gated"])
    res["jacobian_rotated_scenes_pass_world"] = all(r["F_ok"] and r["M_ok_torque_world"] for r in res["jacobian"] if not r["gated"])
    res["bridge_pass"] = all(b["F_ok"] and b["M_ok"] for b in res["bridge"])
    res["bridge_pass_all_30_samples"] = all(b["samples30"]["samples_failing"] == 0 for b in res["bridge"])
    res["bridge_failing_samples"] = sum(b["samples30"]["samples_failing"] for b in res["bridge"])
    (OUTJ / ("j_b%s.json" % args.suffix)).write_text(json.dumps(res, indent=1))
    print("\nJ-b (i) statics single sample %s (failing %s; all-30-samples %s; 30-frame mean %s; failing samples %d/%d) | "
          "(ii) Jacobian column local-frame %s world-frame %s (yaw/tilt scenes: local %s world %s) | (iii) bridge single sample %s (all-30-samples %s, failing samples %d/%d) -> %s" % (
        res["statics_pass"], res["statics_failing"], res["statics_pass_all_30_samples"], res["statics_pass_30_frame_mean"],
        res["statics_failing_samples"], res["statics_samples_total"], res["jacobian_pass_local"], res["jacobian_pass_world"],
        res["jacobian_rotated_scenes_pass_local"], res["jacobian_rotated_scenes_pass_world"], res["bridge_pass"],
        res["bridge_pass_all_30_samples"], res["bridge_failing_samples"], 30 * len(res["bridge"]),
        OUTJ / ("j_b%s.json" % args.suffix)))


# --- Part S: contact count with and without the structure group in a real cube build ------
class SmokeExample(D.Example):
    """dual_arm_sim.Example that records solver.mjw_data.nacon and the structure-internal contacts per frame."""

    def __init__(self, viewer, args, group):
        self.group = group
        super().__init__(viewer, args)
        self.log_path = OUTJ / "smoke_episodes.jsonl"
        self.sb = self.model.shape_body.numpy()
        self.rows = []

    def set_group(self, shapes):
        if self.group:
            super().set_group(shapes)

    def step(self):
        super().step()
        c = self.contacts
        n = int(c.rigid_contact_count.numpy()[0])
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        inS = np.zeros(self.model.shape_count, bool)
        inS[self.plate_shapes] = True
        for bid, w in self.welds.items():
            if w[0][3]:
                inS[self.shapes_of[bid]] = True
        self.rows.append(dict(frame=self.frame, nacon=int(self.solver.mjw_data.nacon.numpy()[0]), pipeline=n,
                              structure_internal=int((inS[s0] & inS[s1]).sum()), welded=sum(w[0][3] for w in self.welds.values())))


def part_s(args, viewer):
    OUTJ.mkdir(parents=True, exist_ok=True)
    plan_f = HERE / "plans" / ("sim_%s_%s.json" % (args.shape, args.strategy))
    bak = plan_f.read_bytes() if plan_f.exists() else None
    (OUTJ / "smoke_episodes.jsonl").unlink(missing_ok=True)
    e = SmokeExample(viewer, args, not args.no_group)
    if bak is None:
        plan_f.unlink()
    else:
        plan_f.write_bytes(bak)
    newton.examples.run(e, args)
    by = {}
    for r in e.rows:
        by.setdefault(r["welded"], []).append(r)
    summ = {k: dict(frames=len(v), nacon_mean=rnd(np.mean([x["nacon"] for x in v]), 1),
                    structure_internal_mean=rnd(np.mean([x["structure_internal"] for x in v]), 1),
                    structure_internal_max=max(x["structure_internal"] for x in v)) for k, v in sorted(by.items())}
    n = len(e.plan["sequence"])
    res = {"shape": args.shape, "grouped": not args.no_group, "placed": "%d/%d" % (sum(x["success"] for x in e.episodes), n),
           "done": e.done, "frames": e.frame, "by_welded_bricks": summ, "rows": e.rows}
    out = OUTJ / ("smoke_%s_%s.json" % (args.shape, "nogroup" if args.no_group else "group"))
    out.write_text(json.dumps(res))
    print("\nSMOKE %s grouped=%s placed=%s done=%s" % (args.shape, not args.no_group, res["placed"], e.done))
    for k, v in summ.items():
        print("  welded=%s %s" % (k, v))
    print("->", out)


# --- Part C --group --samples: the per-frame readout series behind J-b (measurement only) -------------
def _st(x):
    x = np.abs(np.asarray(x, float))
    return dict(max=float(x.max()), p99=float(np.percentile(x, 99)), mean=float(x.mean()))


def _ac1(e):
    e = np.asarray(e, float) - np.mean(e)
    d = float((e * e).sum())
    return float((e[:-1] * e[1:]).sum() / d) if d > 1e-30 else None


def _period(e):
    """Dominant period in frames (rfft peak of the de-meaned series) and its share of the variance."""
    e = np.asarray(e, float) - np.mean(e)
    if len(e) < 8 or float((e * e).sum()) < 1e-30:
        return None
    a = np.abs(np.fft.rfft(e)) ** 2
    k = int(np.argmax(a[1:]) + 1)
    return dict(period_frames=rnd(len(e) / k, 3), variance_share=rnd(float(a[k] / a[1:].sum()), 3))


def _trail(x, k):
    """Trailing k-frame mean, aligned so entry t uses frames t-k+1..t (first k-1 entries are partial and unused)."""
    c = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    out = x.copy()
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out


def part_c_samples(args):
    OUTJ.mkdir(parents=True, exist_ok=True)
    res = {"meta": {"weld": "solimp9999, %d substeps, plate + bricks in STRUCT_GROUP" % args.substeps,
                    "sample_taken": "after each frame (%d substeps, one pipeline.collide per frame), so efc.force is the LAST "
                                    "substep (%d of %d) of that frame; consecutive samples are 1 frame = %.4f s apart" % (
                                        args.substeps, args.substeps, args.substeps, 1 / 60),
                    "axes": "F_patch, M_patch(about the patch centre) in the upper brick's nominal axes, the ones Patch.utilization uses; "
                            "wrench = weld on the upper brick, compression +Fz",
                    "frames": "30 consecutive frames starting at frame 120 after the load is applied (bridge: same), plus a 300-frame series "
                              "on 'cantilever +1 N z' and 'bridge +1 N z'",
                    "filters": "f2/f3 = trailing 2/3-frame mean of the patch-frame F and M, then u from Patch.utilization; "
                               "all three (raw, f2, f3) compared over frames 2.. only",
                    "du": "|u - u_analytic| absolute, and relative to u_analytic where u_analytic > 1e-3 (else null). "
                          "Bridge has no per-weld analytic: du is against that weld's own series mean, and the sum-wrench error "
                          "(sum of the two welds on the top brick, about its origin, vs the analytic total) is given instead"},
           "cases": [], "table": [], "series300": []}
    for kind in ("column", "cantilever", "column_yaw", "column_tilt", "bridge"):
        st = Static(kind, "solimp9999", args.substeps, group=True, torquescale=args.torquescale)
        st.run(120)
        top = len(st.body) - 1
        cases = list(J_LOADS) + ([("couple y 2 mN.m + 1 N z", (0, 0, 1))] if kind == "bridge" else [])
        for name, u in cases:
            f = np.array(u, float)
            tau = np.array([0, 2e-3, 0]) if name.startswith("couple") else np.zeros(3)
            long = (kind, name) in (("cantilever", "+1 N z"), ("bridge", "+1 N z"))
            n_fr = 300 if long else 30
            st.set_force(top, f, tau)
            st.run(120)
            loads = nominal_loads(st, top, f, tau)
            geo = {}
            for info in st.weld_info:
                bn = st.bricks[info["up"]]
                patch, cl = patch_of(st, info["up"], info["lo"])
                R_n = rotmat(bn["q"])
                geo[info["eq"]] = (patch, bn["p"] + R_n @ cl, R_n)
            ser = {i["eq"]: dict(F=[], M=[]) for i in st.weld_info}
            sumF, sumM = [], []
            for t in range(n_fr):
                if t:
                    st.run(1)
                w = weld_rows(st)
                Fs = Ms = 0
                for info in st.weld_info:
                    n = info["up"]
                    _, c_w, R_n = geo[info["eq"]]
                    Fm, Mm = readout(st, w, info, None)
                    Fp, Mp = to_patch(st, n, c_w, R_n, Fm, Mm, st.bricks[n]["p"])
                    ser[info["eq"]]["F"].append(Fp)
                    ser[info["eq"]]["M"].append(Mp)
                    if info["up"] == top:
                        Fs, Ms = Fs + Fm, Ms + Mm
                sumF.append(Fs)
                sumM.append(Ms)
            case = dict(scene=kind, load=name, frames=n_fr, welds=[])
            rows_max = []
            for info in st.weld_info:
                n = info["up"]
                patch = geo[info["eq"]][0]
                Fp, Mp = np.array(ser[info["eq"]]["F"]), np.array(ser[info["eq"]]["M"])
                if kind != "bridge":
                    Fa, Ma = analytic_on(st, n, loads, top, tau)
                    Fpa, Mpa = to_patch(st, n, geo[info["eq"]][1], geo[info["eq"]][2], Fa, Ma, st.bricks[n]["p"])
                    ua = float(patch.utilization(Fpa[2], Mpa[0], Mpa[1]))
                    ref_u = ua
                else:
                    Fpa = Mpa = ua = None
                uf = lambda F, M: np.asarray(patch.utilization(F[:, 2], M[:, 0], M[:, 1]))
                u_by = {"raw": uf(Fp, Mp), "f2": uf(_trail(Fp, 2), _trail(Mp, 2)), "f3": uf(_trail(Fp, 3), _trail(Mp, 3))}
                if kind == "bridge":
                    ref_u = float(u_by["raw"].mean())
                ent = dict(weld="%s->%d" % (info["lo"], n), n_studs=patch.n, pull_capacity_N=rnd(patch.pull_capacity(), 3),
                           u_analytic=ua, u_reference=ref_u, u_reference_is="analytic" if ua is not None else "series mean of raw u",
                           F_patch=Fp.tolist(), M_patch=Mp.tolist(),
                           F_patch_analytic=None if Fpa is None else Fpa.tolist(),
                           M_patch_analytic=None if Mpa is None else Mpa.tolist(),
                           u_measured=u_by["raw"].tolist(),
                           du_abs=(np.abs(u_by["raw"] - ref_u)).tolist(),
                           du_rel=None if ref_u <= 1e-3 else (np.abs(u_by["raw"] - ref_u) / ref_u).tolist())
                for k in ("raw", "f2", "f3"):
                    d = np.abs(u_by[k] - ref_u)[2:]
                    ent["du_abs_" + k] = _st(d)
                    ent["du_rel_" + k] = None if ref_u <= 1e-3 else _st(d / ref_u)
                    if k != "raw":
                        ent["u_" + k] = u_by[k].tolist()
                if Fpa is not None:
                    eFz = Fp[:, 2] - Fpa[2]
                    ent["Fz_err_mN"] = _st(eFz * 1e3)
                    ent["F_err_norm_mN_raw"] = _st(np.linalg.norm(Fp - Fpa, axis=1) * 1e3)
                    ent["M_err_norm_mNm_raw"] = _st(np.linalg.norm(Mp - Mpa, axis=1) * 1e3)
                else:
                    eFz = Fp[:, 2] - Fp[:, 2].mean()
                ent["Fz_err_lag1_autocorr"] = _ac1(eFz)
                ent["u_err_lag1_autocorr"] = _ac1(u_by["raw"] - ref_u)
                ent["Fz_err_period"] = _period(eFz)
                case["welds"].append(ent)
                rows_max.append(ent)
            if kind == "bridge":
                sF, sM = np.array(sumF), np.array(sumM)
                Fa, Ma = analytic_on(st, top, loads, top, tau)
                eF, eM = np.linalg.norm(sF - Fa, axis=1), np.linalg.norm(sM - Ma, axis=1)
                case["sum"] = dict(F_sum=sF.tolist(), M_sum=sM.tolist(), F_analytic=Fa.tolist(), M_analytic=Ma.tolist(),
                                   F_err_mN_raw=_st(eF * 1e3), M_err_mNm_raw=_st(eM * 1e3),
                                   F_err_mN_f2=_st((np.linalg.norm(_trail(sF, 2) - Fa, axis=1) * 1e3)[2:]),
                                   F_err_mN_f3=_st((np.linalg.norm(_trail(sF, 3) - Fa, axis=1) * 1e3)[2:]),
                                   M_err_mNm_f2=_st((np.linalg.norm(_trail(sM, 2) - Ma, axis=1) * 1e3)[2:]),
                                   M_err_mNm_f3=_st((np.linalg.norm(_trail(sM, 3) - Ma, axis=1) * 1e3)[2:]),
                                   Fz_err_lag1_autocorr=_ac1(sF[:, 2] - Fa[2]), Fz_err_period=_period(sF[:, 2] - Fa[2]))
            # one row of the table: worst weld
            row = dict(scene=kind, load=name, frames=n_fr)
            for k in ("raw", "f2", "f3"):
                worst = max(rows_max, key=lambda e: e["du_abs_" + k]["max"])
                row["du_abs_max_" + k] = worst["du_abs_" + k]["max"]
                rel = [e["du_rel_" + k]["max"] for e in rows_max if e["du_rel_" + k] is not None]
                row["du_rel_max_" + k] = max(rel) if rel else None
                row["worst_weld_" + k] = worst["weld"]
            if kind == "bridge":
                row["sum_F_err_mN_max"] = {k: case["sum"]["F_err_mN_" + k]["max"] for k in ("raw", "f2", "f3")}
                row["sum_M_err_mNm_max"] = {k: case["sum"]["M_err_mNm_" + k]["max"] for k in ("raw", "f2", "f3")}
            res["table"].append(row)
            case["table_row"] = row
            if long:                                            # keep the 300-frame series apart; case keeps its first 30 frames
                res["series300"].append(dict(scene=kind, load=name, frames=300, stats=case_stats(case)))
                case["note"] = "300 frames kept in full in this entry; the 30-frame table row uses frames 0..29 only"
                case["table_row_first30"] = table30(case)
            res["cases"].append(case)
            st.set_force(top, (0, 0, 0))
            st.run(60)
        del st
    (OUTJ / ("j_b_samples%s.json" % args.suffix)).write_text(json.dumps(res))
    print("%-11s %-24s %6s | du_abs max raw/f2/f3 | du_rel max raw/f2/f3 (%%) | worst weld (raw)" % ("scene", "load", "frames"))
    for r in res["table"]:
        print("%-11s %-24s %6d | %.5f %.5f %.5f | %s | %s" % (
            r["scene"], r["load"], r["frames"], r["du_abs_max_raw"], r["du_abs_max_f2"], r["du_abs_max_f3"],
            " ".join("%7s" % ("%.3f" % (100 * r["du_rel_max_" + k]) if r["du_rel_max_" + k] is not None else "n/a") for k in ("raw", "f2", "f3")),
            r["worst_weld_raw"]))
    for c in res["series300"]:
        print("300-frame", c["scene"], c["load"], json.dumps(c["stats"]))
    print("->", OUTJ / ("j_b_samples%s.json" % args.suffix))


def case_stats(case):
    """Stats of the whole (300-frame) series per weld: du (raw/f2/f3), Fz error and its autocorrelation / period."""
    return [dict(weld=e["weld"], u_reference=e["u_reference"], u_reference_is=e["u_reference_is"], du_abs_raw=e["du_abs_raw"],
                 du_abs_f2=e["du_abs_f2"], du_abs_f3=e["du_abs_f3"], du_rel_raw=e["du_rel_raw"], du_rel_f2=e["du_rel_f2"],
                 du_rel_f3=e["du_rel_f3"], Fz_err_mN=e.get("Fz_err_mN"), Fz_err_lag1_autocorr=e["Fz_err_lag1_autocorr"],
                 u_err_lag1_autocorr=e["u_err_lag1_autocorr"], Fz_err_period=e["Fz_err_period"]) for e in case["welds"]] + (
        [dict(weld="sum of the two welds on the top brick", **case["sum"])] if "sum" in case else [])


def table30(case):
    """The table row recomputed on the first 30 frames of a longer series."""
    row = dict(scene=case["scene"], load=case["load"], frames=30)
    for k in ("raw", "f2", "f3"):
        ds = []
        for e in case["welds"]:
            u = np.array(e["u_measured"] if k == "raw" else e["u_" + k])[:30]
            ds.append((np.abs(u - e["u_reference"])[2:], e["u_reference"], e["weld"]))
        m = max(ds, key=lambda x: x[0].max())
        row["du_abs_max_" + k] = float(m[0].max())
        rel = [float((d / r).max()) for d, r, _ in ds if r > 1e-3]
        row["du_rel_max_" + k] = max(rel) if rel else None
    return row


# --- Part C --group --diag: what is the weld-force chatter? (measurement only) -------------------------
# Hypothesis A: real micro-vibration (weld force = static load + m*a). Hypothesis B: a readout defect (efc.force is
# computed before integration, brick_wrench() rotates with the post-integration pose).
# mujoco_warp/_src/forward.py: step() -> forward() -> fwd_position() (:604 smooth.kinematics: d.xpos/xmat/xquat/xipos at the
# PRE-integration qpos; :615 constraint.make_constraint) -> solver.solve() (:1256, efc.force, qacc) -> implicit() (:577) ->
# _advance() (:261, qvel and qpos integrated; xpos/xmat/xquat are NOT updated). Newton exports body_q from the integrated qpos.
# So after a step, d.xpos/xmat are the solve-time kinematics that efc.force and d.qacc belong to; body_q is one dt later.
def _grab(st, midx):
    d = st.solver.mjw_data
    nb = len(st.body)
    w = weld_rows(st)
    nefc = int(d.nefc.numpy()[0])
    typ = d.efc.type.numpy()[0, :nefc]
    frc = d.efc.force.numpy()[0, :nefc]
    g = lambda a: a.numpy()[0]
    return dict(f6=np.array([w[i["eq"]][0] for i in st.weld_info]), xp=g(d.xpos)[midx], xm=g(d.xmat)[midx],
                xip=g(d.xipos)[midx], qacc=g(d.qacc).reshape(nb, 6), qvel=g(d.qvel).reshape(nb, 6),
                qfc=g(d.qfrc_constraint).reshape(nb, 6), bq=st.state_0.body_q.numpy()[st.body].copy(),
                n_contact_rows=int((typ != EQUALITY).sum()), contact_row_force_max=float(np.abs(frc[typ != EQUALITY]).max())
                if (typ != EQUALITY).any() else 0.0, nacon=int(d.nacon.numpy()[0]), niter=int(d.solver_niter.numpy()[0]))


def _substep_run(st, frames, cb):
    """The Static.simulate loop, ungraphed, calling cb(frame, substep) after every substep (state_0 is the newest)."""
    for fr in range(frames):
        st.pipeline.collide(st.state_0, st.contacts)
        for ss in range(st.substeps):
            st.solver.step(st.state_0, st.state_1, st.control, st.contacts, st.dt)
            st.state_0, st.state_1 = st.state_1, st.state_0
            cb(fr, ss)


def _weld_wrenches(st, ts, f6, P, R):
    """Per weld (F, M about the upper origin) on the upper brick, and per brick the sum of everything the welds do to it
    (the upper brick's own welds, minus the reaction of welds that hang on it, shifted to its origin)."""
    W = [np.zeros(3) for _ in st.body], [np.zeros(3) for _ in st.body]
    per = []
    for i, info in enumerate(st.weld_info):
        u, lo = info["up"], info["lo"]
        F, M = -f6[i, :3], -0.5 * ts[info["eq"]] * (R[u] @ f6[i, 3:6])
        per.append((F, M))
        W[0][u], W[1][u] = W[0][u] + F, W[1][u] + M
        if lo >= 0:
            W[0][lo] = W[0][lo] - F
            W[1][lo] = W[1][lo] - M + np.cross(P[u] - P[lo], -F)
    return per, W


def _body_dyn(st, sm, R, P):
    """Per brick: COM accel, angular velocity/acceleration (world) and I*alpha + w x I*w (world), from d.qacc/qvel."""
    out = []
    Ib = st.model.body_inertia.numpy()[st.body]
    for k in range(len(st.body)):
        r = sm["xip"][k] - P[k]
        ao, al_l, om_l = sm["qacc"][k, :3], sm["qacc"][k, 3:], sm["qvel"][k, 3:]
        al, om = R[k] @ al_l, R[k] @ om_l
        out.append(dict(r=r, a_c=ao + np.cross(al, r) + np.cross(om, np.cross(om, r)),
                        Ldot=R[k] @ (Ib[k] @ al_l + np.cross(om_l, Ib[k] @ om_l))))
    return out


def _u_of(patch, F, M):
    return float(patch.utilization(F[2], M[0], M[1]))


def _series_stats(x):
    x = np.asarray(x, float)
    return dict(max=float(np.abs(x).max()), p99=float(np.percentile(np.abs(x), 99)), mean=float(np.abs(x).mean()))


def _diag_series(st, samples, kind, f_ext, tau, top, dt_sample):
    """Everything derived from raw samples of one series."""
    nb = len(st.body)
    ts = st.model.equality_constraint_torquescale.numpy()
    mass, com = st.mass, st.com
    geo = {}
    for info in st.weld_info:
        bn = st.bricks[info["up"]]
        patch, cl = patch_of(st, info["up"], info["lo"])
        R_n = rotmat(bn["q"])
        geo[info["eq"]] = (patch, bn["p"] + R_n @ cl, R_n)
    N = len(samples)
    o = dict(res_F_static=np.zeros((N, nb)), res_F_dyn=np.zeros((N, nb)), res_M_static=np.zeros((N, nb)),
             res_M_dyn=np.zeros((N, nb)), maF=np.zeros((N, nb)), jtf_F=np.zeros((N, nb)), jtf_M=np.zeros((N, nb)),
             jtf_F_unm=np.zeros((N, nb)), jtf_M_unm=np.zeros((N, nb)))
    nw = len(st.weld_info)
    o.update(u_meas_m=np.zeros((N, nw)), u_meas_u=np.zeros((N, nw)), dM_patch=np.zeros((N, nw)), du_matched_unmatched=np.zeros((N, nw)),
             u_an_static_nom=np.full((N, nw), np.nan), u_an_static_m=np.full((N, nw), np.nan), u_an_dyn=np.full((N, nw), np.nan),
             Fz_err_orig=np.full((N, nw), np.nan), Fz_err_dyn=np.full((N, nw), np.nan), Fz_meas_m=np.zeros((N, nw)),
             Fz_an_dyn=np.full((N, nw), np.nan), Fz_an_static=np.full((N, nw), np.nan))
    for t, sm in enumerate(samples):
        Pm, Rm = sm["xp"], sm["xm"]
        Pu = sm["bq"][:, :3]
        Ru = np.array([rotmat(q[3:]) for q in sm["bq"]])
        perm, Wm = _weld_wrenches(st, ts, sm["f6"], Pm, Rm)
        peru, Wu = _weld_wrenches(st, ts, sm["f6"], Pu, Ru)
        dyn = _body_dyn(st, sm, Rm, Pm)
        for k in range(nb):
            fx = f_ext if k == top else np.zeros(3)
            tx = tau if k == top else np.zeros(3)
            Fw, Mo = Wm[0][k], Wm[1][k]
            r = dyn[k]["r"]
            o["res_F_static"][t, k] = np.linalg.norm(Fw + mass[k] * np.array([0, 0, -G]) + fx)
            o["maF"][t, k] = np.linalg.norm(mass[k] * dyn[k]["a_c"])
            o["res_F_dyn"][t, k] = np.linalg.norm(Fw + mass[k] * np.array([0, 0, -G]) + fx - mass[k] * dyn[k]["a_c"])
            Mc = Mo - np.cross(r, Fw) + tx
            o["res_M_static"][t, k] = np.linalg.norm(Mc)
            o["res_M_dyn"][t, k] = np.linalg.norm(Mc - dyn[k]["Ldot"])
            fq, tq = sm["qfc"][k, :3], Rm[k] @ sm["qfc"][k, 3:]
            o["jtf_F"][t, k], o["jtf_M"][t, k] = np.linalg.norm(Wm[0][k] - fq), np.linalg.norm(Wm[1][k] - tq)
            fq2, tq2 = sm["qfc"][k, :3], Ru[k] @ sm["qfc"][k, 3:]
            o["jtf_F_unm"][t, k], o["jtf_M_unm"][t, k] = np.linalg.norm(Wu[0][k] - fq2), np.linalg.norm(Wu[1][k] - tq2)
        for i, info in enumerate(st.weld_info):
            n = info["up"]
            patch, c_w, R_n = geo[info["eq"]]
            (Fm, Mm), (Fu, Mu) = perm[i], peru[i]
            Fpm, Mpm = to_patch(st, n, c_w, R_n, Fm, Mm, Pm[n])
            Fpu, Mpu = to_patch(st, n, c_w, R_n, Fu, Mu, Pu[n])
            o["u_meas_m"][t, i], o["u_meas_u"][t, i] = _u_of(patch, Fpm, Mpm), _u_of(patch, Fpu, Mpu)
            o["dM_patch"][t, i] = np.linalg.norm(Mpm - Mpu)
            o["du_matched_unmatched"][t, i] = abs(o["u_meas_m"][t, i] - o["u_meas_u"][t, i])
            o["Fz_meas_m"][t, i] = Fpm[2]
            if kind == "bridge":
                continue
            # analytic on the sub-tree of n: static (nominal geometry, as j_b_samples), static and dynamic (solve-time geometry)
            loads = nominal_loads(st, top, f_ext, tau)
            Fa, Ma = analytic_on(st, n, loads, top, tau)
            Fp0, Mp0 = to_patch(st, n, c_w, R_n, Fa, Ma, st.bricks[n]["p"])
            o["u_an_static_nom"][t, i] = _u_of(patch, Fp0, Mp0)
            o["Fz_err_orig"][t, i] = _st_first(Fpu[2] - Fp0[2])
            sub = sorted(subtree(st, n))
            for use_acc in (False, True):
                Fs, Ms = np.zeros(3), np.zeros(3)
                for j in sub:
                    fj = mass[j] * np.array([0, 0, -G]) + (f_ext if j == top else 0)
                    aj = dyn[j]["a_c"] if use_acc else np.zeros(3)
                    Lj = dyn[j]["Ldot"] if use_acc else np.zeros(3)
                    cj = Pm[j] + dyn[j]["r"]
                    Fs += mass[j] * aj - fj
                    Ms += np.cross(cj - Pm[n], mass[j] * aj - fj) + Lj - (tau if j == top else 0)
                Fpd, Mpd = to_patch(st, n, c_w, R_n, Fs, Ms, Pm[n])
                if use_acc:
                    o["u_an_dyn"][t, i] = _u_of(patch, Fpd, Mpd)
                    o["Fz_err_dyn"][t, i] = Fpm[2] - Fpd[2]
                    o["Fz_an_dyn"][t, i] = Fpd[2]
                else:
                    o["u_an_static_m"][t, i] = _u_of(patch, Fpd, Mpd)
                    o["Fz_an_static"][t, i] = Fpd[2]
    return o


def _st_first(x):
    return float(x)


def _fft_peak(x, fs):
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 8 or np.abs(x).max() < 1e-15:
        return None
    a = np.abs(np.fft.rfft(x)) ** 2
    k = int(np.argmax(a[1:]) + 1)
    return dict(freq_Hz=rnd(k * fs / len(x), 2), variance_share=rnd(float(a[k] / a[1:].sum()), 3))


def _kin(st, samples, top, fs):
    """Displacement (um), rotation (urad) about the series mean, and velocities, of every brick; FFT and decay of the top."""
    nb = len(st.body)
    pos = np.array([sm["bq"][:, :3] for sm in samples], np.float64)     # (N, nb, 3); float64, or the float32 mean biases the offsets
    qs = np.array([sm["bq"][:, 3:] for sm in samples], np.float64)      # xyzw
    qm = qs.mean(0)
    qm /= np.linalg.norm(qm, axis=1)[:, None]
    out = {}
    for k in range(nb):
        dp = (pos[:, k] - pos[:, k].mean(0)) * 1e6
        rv = np.array([2 * D.qmul(CONJ(qm[k]), q)[:3] for q in qs[:, k]]) * 1e6       # small-angle rotation vector, urad
        vl = np.array([np.linalg.norm(sm["qvel"][k, :3]) for sm in samples])
        va = np.array([np.linalg.norm(sm["qvel"][k, 3:]) for sm in samples])
        e = dict(disp_pp_um=[rnd(float(np.ptp(dp[:, a])), 4) for a in range(3)], rot_pp_urad=[rnd(float(np.ptp(rv[:, a])), 4) for a in range(3)],
                 disp_rms_um=rnd(float(np.sqrt((dp ** 2).sum(1).mean())), 5), rot_rms_urad=rnd(float(np.sqrt((rv ** 2).sum(1).mean())), 5),
                 vel_lin_max_mm_s=rnd(float(vl.max() * 1e3), 5), vel_ang_max_rad_s=rnd(float(va.max()), 6),
                 vel_lin_rms_mm_s=rnd(float(np.sqrt((vl ** 2).mean()) * 1e3), 5))
        if k == top:
            h = len(dp) // 2
            e["disp_rms_um_first_half"] = rnd(float(np.sqrt((dp[:h] ** 2).sum(1).mean())), 5)
            e["disp_rms_um_second_half"] = rnd(float(np.sqrt((dp[h:] ** 2).sum(1).mean())), 5)
            a = int(np.argmax(np.ptp(dp, axis=0)))
            e["fft_disp_axis"] = "xyz"[a]
            e["fft_disp"] = _fft_peak(dp[:, a], fs)
            e["fft_vel_lin"] = _fft_peak(vl, fs)
            t3 = len(vl) // 3
            e["vel_lin_max_mm_s_first_third"] = rnd(float(vl[:t3].max() * 1e3), 5)
            e["vel_lin_max_mm_s_last_third"] = rnd(float(vl[-t3:].max() * 1e3), 5)
            e["series_disp_um"] = dp.round(4).tolist()
            e["series_vel_lin_mm_s"] = (vl * 1e3).round(5).tolist()
        out["brick%d" % k] = e
    return out


def _contacts(st, pc):
    """Newton contacts touching a brick: per body pair count and max normal force."""
    st.solver.update_contacts(pc, st.state_0)
    n = int(pc.rigid_contact_count.numpy()[0])
    if not n:
        return dict(n=0, pairs={})
    sb = st.model.shape_body.numpy()
    s0, s1 = pc.rigid_contact_shape0.numpy()[:n], pc.rigid_contact_shape1.numpy()[:n]
    nrm, fv = pc.rigid_contact_normal.numpy()[:n], pc.force.numpy()[:n, :3]
    fn = np.abs((fv * nrm).sum(1))
    pairs = {}
    for a, b, f in zip(s0, s1, fn):
        k = "%d/%d" % (sb[a], sb[b])
        c = pairs.setdefault(k, dict(count=0, max_normal_N=0.0))
        c["count"] += 1
        c["max_normal_N"] = max(c["max_normal_N"], float(f))
    return dict(n=n, pairs=pairs)


def part_c_diag(args):
    OUTJ.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / 60 / args.substeps
    tc = max(0.002, 2 * dt)
    res = {"meta": {"question": "A: micro-vibration (weld force differs from the static load by m*a) or B: readout defect (pre- vs post-integration pose)?",
                    "weld": "solimp9999, solref (0.002, 1.0), %d substeps, dt %.6f s, effective timeconst max(0.002, 2 dt) = %.6f s "
                            "(reference 1/(2 pi tc) = %.1f Hz)" % (args.substeps, dt, tc, 1 / (2 * math.pi * tc)),
                    "buffers": "mujoco_warp/_src/forward.py: fwd_position :604 smooth.kinematics (d.xpos/xmat/xipos at the pre-integration qpos), "
                               ":615 make_constraint, :1256 solver.solve (efc.force, qacc), :577 implicit -> :261 _advance (integrates qvel/qpos, "
                               "leaves xpos/xmat). Newton exports body_q from the integrated qpos. 'matched' = d.xpos/d.xmat/d.xipos, "
                               "'unmatched' = state_0.body_q (what brick_wrench/st.q use)",
                    "balance": "brick k: F_weld_sum + m g + f_applied = m a_com;  M about COM: sum(M_o - r x F) + tau = I alpha + w x I w "
                               "(a from d.qacc [origin accel world, angular local frame], I = model.body_inertia)",
                    "static_error_orig": "unmatched weld u minus the nominal-geometry static analytic u (what j_b_samples.json reported)"},
           "cases": []}
    for kind, name, u in (("column", "0.5 N x", (0.5, 0, 0)), ("cantilever", "+1 N z", (0, 0, 1)), ("bridge", "+1 N z", (0, 0, 1))):
        st = Static(kind, "solimp9999", args.substeps, group=True, torquescale=args.torquescale)
        st.model.request_contact_attributes("force")
        pc = st.pipeline.contacts()
        mo = st.solver.mjw_model
        res["meta"]["solver"] = dict(tolerance=float(mo.opt.tolerance.numpy()[0]), iterations=int(mo.opt.iterations),
                                     ls_tolerance=float(mo.opt.ls_tolerance.numpy()[0]), ls_iterations=int(mo.opt.ls_iterations),
                                     meaninertia=float(mo.stat.meaninertia.numpy()[0]), nv=int(mo.nv))
        b2n = st.solver.mjc_body_to_newton.numpy()[0]
        midx = np.array([int(np.where(b2n == n)[0][0]) for n in range(len(st.body))])
        top = len(st.body) - 1
        f = np.array(u, float)
        tau = np.zeros(3)
        st.run(120)
        case = dict(scene=kind, load=name)
        # step response, every substep, from the moment the load is applied
        st.set_force(top, f, tau)
        if kind == "cantilever":
            sr = []
            _substep_run(st, 60, lambda fr, ss: sr.append(_grab(st, midx)))
            kin = _kin(st, sr, top, 60.0 * args.substeps)
            case["step_response_substep_60frames"] = dict(kinematics=kin, note="from load application; sampled every substep (960 Hz)")
            st.run(60)
        else:
            st.run(120)
        # 300 consecutive end-of-frame samples
        samples, cont = [], {}
        for t in range(300):
            if t:
                st.run(1)
            samples.append(_grab(st, midx))
            if t in (0, 150, 299):
                cont[str(t)] = _contacts(st, pc)
        o = _diag_series(st, samples, kind, f, tau, top, 1 / 60)
        case["frame_samples"] = _pack(st, o, samples, kind, top, 60.0)
        case["contacts"] = cont
        case["contact_rows_max"] = max(sm["n_contact_rows"] for sm in samples)
        case["contact_row_force_max"] = max(sm["contact_row_force_max"] for sm in samples)
        case["nacon"] = [min(sm["nacon"] for sm in samples), max(sm["nacon"] for sm in samples)]
        if kind == "cantilever":                                       # every substep, 60 frames, settled
            sb = []
            _substep_run(st, 60, lambda fr, ss: sb.append(_grab(st, midx)))
            o2 = _diag_series(st, sb, kind, f, tau, top, dt)
            case["substep_samples"] = _pack(st, o2, sb, kind, top, 60.0 * args.substeps)
            case["substep_samples"]["a_com_from_dqvel"] = _dq_check(st, sb, dt)
            case["substep_solver"] = _solver_stats(sb, o2)
            case["tolerance_sweep"] = _tol_sweep(st, midx, top, f, tau, kind)
        case["cpu_double_vs_gpu_float32"] = _cpu_check(st, midx, f, tau, top, kind)
        res["cases"].append(case)
        print("\n== %s %s" % (kind, name))
        _print_case(case)
        del st
    (OUTJ / ("j_b_diag%s.json" % args.suffix)).write_text(json.dumps(res))
    print("\n->", OUTJ / ("j_b_diag%s.json" % args.suffix))


def _solver_stats(sb, o):
    """Solver iterations per substep, and the balance residual split by whether qacc was recomputed (differs from the
    previous substep's) or is a bit-identical repeat (the solver returned its warm start)."""
    ni = np.array([s["niter"] for s in sb])
    res = (o["res_F_dyn"] * 1e3).max(axis=1)
    resM = (o["res_M_dyn"] * 1e3).max(axis=1)
    same = np.array([False] + [bool(np.array_equal(sb[t]["qacc"], sb[t - 1]["qacc"])) for t in range(1, len(sb))])
    big = res > 1.0
    grp = lambda m: dict(n=int(m.sum()), F_res_mN_max=rnd(float(res[m].max()), 4) if m.any() else None,
                         F_res_mN_mean=rnd(float(res[m].mean()), 4) if m.any() else None,
                         M_res_mNm_max=rnd(float(resM[m].max()), 4) if m.any() else None)
    ok = lambda m: {"n": int(m.sum())}
    return dict(niter_hist={int(k): int((ni == k).sum()) for k in np.unique(ni)}, substeps=len(sb),
                qacc_identical_to_previous=grp(same), qacc_recomputed=grp(~same),
                substeps_with_res_over_1mN=int(big.sum()), of_which_qacc_identical=int((big & same).sum()),
                of_which_niter_1=int((big & (ni == 1)).sum()), niter_when_res_over_1mN={int(k): int((ni[big] == k).sum()) for k in np.unique(ni[big])} if big.any() else {},
                residual_F_mN_distinct_levels=sorted({round(float(x), 2) for x in res[big]})[:20],
                note="opt.tolerance = %s, opt.iterations = %s; exit test mujoco_warp/_src/solver.py:3221-3224 "
                     "(improvement or gradient, rescaled by meaninertia*nv, below opt.tolerance)" % ("see meta", "15"))


def _tol_sweep(st, midx, top, f, tau, kind):
    """Same static case, solver tolerance tightened in place (the captured graph reads the array): does the chatter go?"""
    m = st.solver.mjw_model
    orig = float(m.opt.tolerance.numpy()[0])
    out = []
    ts = st.model.equality_constraint_torquescale.numpy()
    for tol in (orig, 1e-8, 1e-10, 1e-12):
        m.opt.tolerance.assign(np.array([tol], np.float32))
        st.run(60)
        sm = []
        for t in range(100):
            st.run(1)
            sm.append(_grab(st, midx))
        o = _diag_series(st, sm, kind, f, tau, top, 1 / 60)
        ni = np.array([x["niter"] for x in sm])
        du = {"%s->%d" % (w["lo"], w["up"]): _series_stats(o["u_meas_m"][:, i] - o["u_an_dyn"][:, i]) for i, w in enumerate(st.weld_info)}
        fz = {"%s->%d" % (w["lo"], w["up"]): _series_stats((o["Fz_meas_m"][:, i] - o["Fz_an_dyn"][:, i]) * 1e3) for i, w in enumerate(st.weld_info)}
        out.append(dict(tolerance=tol, frames=100, niter_hist={int(k): int((ni == k).sum()) for k in np.unique(ni)},
                        res_F_mN_max=rnd(float((o["res_F_dyn"] * 1e3).max()), 4), res_M_mNm_max=rnd(float((o["res_M_dyn"] * 1e3).max()), 4),
                        Fz_err_mN=fz, du_vs_dynamic_analytic=du,
                        top_disp_pp_um=_kin(st, sm, top, 60.0)["brick%d" % top]["disp_pp_um"]))
    m.opt.tolerance.assign(np.array([orig], np.float32))
    return out


def _cpu_check(st, midx, f, tau, top, kind, n=60):
    """Same states, same model, same solver settings, two solvers: mujoco_warp (float32, GPU; forward() re-run at a state read from
    the run) vs CPU MuJoCo (double) on st.solver.mj_model. Question: is the balance residual a property of the float32 solve?"""
    import mujoco
    import mujoco_warp as mjw
    mm = st.solver.mj_model
    md = mujoco.MjData(mm)
    mm.opt.timestep = st.dt
    mm.opt.tolerance = float(st.solver.mjw_model.opt.tolerance.numpy()[0])
    mm.opt.iterations = int(st.solver.mjw_model.opt.iterations)
    mm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d, m = st.solver.mjw_data, st.solver.mjw_model
    nb, nv = len(st.body), mm.nv
    e2n = st.solver.mjc_eq_to_newton_eq.numpy()[0]
    row_of = {int(e2n[i]): i for i in range(mm.neq)}
    xf = np.zeros((1, mm.nbody, 6), np.float32)
    xf[0, midx[top], :3] = f
    xf[0, midx[top], 3:] = tau
    cpu, gpu, dq = [], [], []
    for t in range(n):
        st.run(1)
        qpos, qvel = d.qpos.numpy()[0].copy(), d.qvel.numpy()[0].copy()
        md.qpos[:], md.qvel[:], md.qacc_warmstart[:] = qpos, qvel, 0
        md.xfrc_applied[:] = 0
        md.xfrc_applied[midx[top]] = xf[0, midx[top]]
        mujoco.mj_forward(mm, md)
        d.qpos.assign(qpos[None]); d.qvel.assign(qvel[None]); d.qacc_warmstart.assign(np.zeros((1, nv), np.float32)); d.xfrc_applied.assign(xf)
        mjw.forward(m, d)
        cs = dict(f6=np.array([md.efc_force[6 * row_of[i["eq"]]:6 * row_of[i["eq"]] + 6] for i in st.weld_info]), xp=md.xpos[midx].copy(),
                  xm=md.xmat[midx].reshape(nb, 3, 3).copy(), xip=md.xipos[midx].copy(), qacc=md.qacc.reshape(nb, 6).copy(),
                  qvel=md.qvel.reshape(nb, 6).copy(), qfc=md.qfrc_constraint.reshape(nb, 6).copy(), bq=st.state_0.body_q.numpy()[st.body].copy())
        gs = _grab(st, midx)
        cpu.append(cs)
        gpu.append(gs)
        dq.append(float(np.abs(gs["qacc"] - cs["qacc"]).max()))
    out = {}
    for tag, sm in (("cpu_double", cpu), ("gpu_float32", gpu)):
        o = _diag_series(st, sm, kind, f, tau, top, 1 / 60)
        out[tag] = dict(res_F_dyn_mN={"brick%d" % b: _series_stats(o["res_F_dyn"][:, b] * 1e3) for b in range(nb)},
                        res_M_dyn_mNm={"brick%d" % b: _series_stats(o["res_M_dyn"][:, b] * 1e3) for b in range(nb)})
        if kind != "bridge":
            out[tag]["du_vs_dynamic_analytic"] = {"%s->%d" % (w["lo"], w["up"]): _series_stats(o["u_meas_m"][:, i] - o["u_an_dyn"][:, i])
                                                  for i, w in enumerate(st.weld_info)}
            out[tag]["Fz_err_mN"] = {"%s->%d" % (w["lo"], w["up"]): _series_stats((o["Fz_meas_m"][:, i] - o["Fz_an_dyn"][:, i]) * 1e3)
                                     for i, w in enumerate(st.weld_info)}
    out["efc_force_cpu_minus_gpu_max"] = float(max(np.abs(a["f6"] - b["f6"]).max() for a, b in zip(cpu, gpu)))
    out["qacc_cpu_minus_gpu_max"] = float(max(dq))
    out["qacc_max_cpu"] = float(max(np.abs(a["qacc"]).max() for a in cpu))
    out["note"] = "%d states read from the run; both solvers restart from qacc_warmstart = 0 at that state; CPU: tolerance %g, %d iterations, contacts off" % (
        n, mm.opt.tolerance, mm.opt.iterations)
    return out


def _dq_check(st, sb, dt):
    """(qvel[t+1]-qvel[t])/dt vs qacc[t+1], per substep, max over bricks and samples (N, N*s^-2 for linear)."""
    q = np.array([s["qvel"] for s in sb])
    a = np.array([s["qacc"] for s in sb])
    dq = (q[1:] - q[:-1]) / dt
    return dict(lin_max_diff=rnd(float(np.abs(dq[:, :, :3] - a[1:, :, :3]).max()), 6), lin_max_qacc=rnd(float(np.abs(a[:, :, :3]).max()), 6),
                ang_max_diff=rnd(float(np.abs(dq[:, :, 3:] - a[1:, :, 3:]).max()), 6), ang_max_qacc=rnd(float(np.abs(a[:, :, 3:]).max()), 6))


def _pack(st, o, samples, kind, top, fs):
    r = {}
    for k in ("res_F_static", "res_F_dyn", "maF"):
        r[k + "_mN"] = {"brick%d" % b: _series_stats(o[k][:, b] * 1e3) for b in range(len(st.body))}
    for k in ("res_M_static", "res_M_dyn", "jtf_M", "jtf_M_unm"):
        r[k + "_mNm"] = {"brick%d" % b: _series_stats(o[k][:, b] * 1e3) for b in range(len(st.body))}
    for k in ("jtf_F", "jtf_F_unm"):
        r[k + "_mN"] = {"brick%d" % b: _series_stats(o[k][:, b] * 1e3) for b in range(len(st.body))}
    r["dM_patch_matched_vs_unmatched_mNm"] = {w["weld"] if False else "%s->%d" % (w["lo"], w["up"]): _series_stats(o["dM_patch"][:, i] * 1e3)
                                              for i, w in enumerate(st.weld_info)}
    r["du_matched_vs_unmatched"] = {"%s->%d" % (w["lo"], w["up"]): _series_stats(o["du_matched_unmatched"][:, i])
                                    for i, w in enumerate(st.weld_info)}
    if kind != "bridge":
        r["du_static_error_orig"] = {}
        r["du_static_error_matched"] = {}
        r["du_dyn_analytic"] = {}
        r["Fz_err_mN"] = {}
        for i, w in enumerate(st.weld_info):
            key = "%s->%d" % (w["lo"], w["up"])
            r["du_static_error_orig"][key] = _series_stats(o["u_meas_u"][:, i] - o["u_an_static_nom"][:, i])
            r["du_static_error_matched"][key] = _series_stats(o["u_meas_m"][:, i] - o["u_an_static_m"][:, i])
            r["du_dyn_analytic"][key] = _series_stats(o["u_meas_m"][:, i] - o["u_an_dyn"][:, i])
            r["Fz_err_mN"][key] = dict(orig=_series_stats(o["Fz_err_orig"][:, i] * 1e3),
                                       matched_vs_static=_series_stats((o["Fz_meas_m"][:, i] - o["Fz_an_static"][:, i]) * 1e3),
                                       matched_vs_dyn=_series_stats(o["Fz_err_dyn"][:, i] * 1e3))
    else:
        r["note_bridge"] = "no per-weld analytic decomposition (statically indeterminate); u columns omitted, brick-level balance and J^T f only"
    r["kinematics"] = _kin(st, samples, top, fs)
    r["series"] = {k: o[k].round(9).tolist() for k in ("res_F_static", "res_F_dyn", "res_M_static", "res_M_dyn", "u_meas_m", "u_meas_u")}
    if kind != "bridge":
        r["series"].update({k: o[k].round(9).tolist() for k in ("u_an_static_nom", "u_an_static_m", "u_an_dyn")})
    return r


def _print_case(c):
    for tag in ("step_response_substep_60frames", "frame_samples", "substep_samples"):
        if tag not in c:
            continue
        d = c[tag]
        print(" [%s]" % tag)
        if "kinematics" in d and tag.startswith("step"):
            for b, e in d["kinematics"].items():
                print("   %s disp pp um %s rms %s fft %s vel max mm/s %s" % (b, e["disp_pp_um"], e["disp_rms_um"], e.get("fft_disp"), e["vel_lin_max_mm_s"]))
            continue
        for b in d["res_F_static_mN"]:
            print("   %s static res F max %.3f mN -> dyn res %.3f mN (m*a max %.4f mN) | M %.4f -> %.4f mNm | Jtf F %.4f (unm %.4f) M %.4f (unm %.4f)" % (
                b, d["res_F_static_mN"][b]["max"], d["res_F_dyn_mN"][b]["max"], d["maF_mN"][b]["max"], d["res_M_static_mNm"][b]["max"],
                d["res_M_dyn_mNm"][b]["max"], d["jtf_F_mN"][b]["max"], d["jtf_F_unm_mN"][b]["max"], d["jtf_M_mNm"][b]["max"], d["jtf_M_unm_mNm"][b]["max"]))
        for k, v in d["du_matched_vs_unmatched"].items():
            extra = ""
            if "du_dyn_analytic" in d:
                extra = " | du static_orig max %.2e, static_matched %.2e, dyn %.2e" % (
                    d["du_static_error_orig"][k]["max"], d["du_static_error_matched"][k]["max"], d["du_dyn_analytic"][k]["max"])
            print("   weld %s dM_patch matched-unmatched max %.5f mNm, du %.2e%s" % (
                k, d["dM_patch_matched_vs_unmatched_mNm"][k]["max"], v["max"], extra))
        top = d["kinematics"]["brick%d" % (len(d["res_F_static_mN"]) - 1)]
        print("   top brick: disp pp um %s, rms %s (halves %s -> %s), rot pp urad %s, fft %s, vel lin max mm/s %s" % (
            top["disp_pp_um"], top["disp_rms_um"], top["disp_rms_um_first_half"], top["disp_rms_um_second_half"], top["rot_pp_urad"],
            top["fft_disp"], top["vel_lin_max_mm_s"]))
    print(" contact rows (non-equality efc) max %s, force %s, nacon range %s, newton contacts %s" % (
        c.get("contact_rows_max"), c.get("contact_row_force_max"), c.get("nacon"),
        {k: {p: v for p, v in vv["pairs"].items()} for k, vv in c.get("contacts", {}).items() if k == "0"}))


# =============================================================================
if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--part", required=True, choices=["a", "b", "c", "s"])
    parser.add_argument("--shape", choices=list(D.B.SAMPLES))
    parser.add_argument("--structure", choices=list(P.STRUCTURES))
    parser.add_argument("--strategy", default="weakest_joint", choices=["none", "nearest", "weakest_joint"])
    parser.add_argument("--brace-model", default="grasp_lp", choices=["grasp_lp", "lever_press"])
    parser.add_argument("--tag", help="suffix of the Part A output file")
    parser.add_argument("--group", action="store_true",
                        help="parts B/C: plate + welded bricks in STRUCT_GROUP; writes results/v4/j/ (J-a, J-b)")
    parser.add_argument("--torquescale", type=float, help="weld torquescale (default: Newton's), for the J-a branch")
    parser.add_argument("--samples", action="store_true", help="part c --group: write the per-frame readout series to j_b_samples.json")
    parser.add_argument("--diag", action="store_true", help="part c --group: chatter diagnostic (j_b_diag.json)")
    parser.add_argument("--no-group", action="store_true", help="part s: leave the structure ungrouped (the comparison run)")
    parser.add_argument("--substeps", type=int, default=16, help="part C --group: substeps per frame")
    parser.add_argument("--suffix", default="", help="parts B/C --group: suffix of the j_a/j_b output files")
    parser.add_argument("--weld-solref", type=lambda s: [float(x) for x in s.split(",")])
    parser.add_argument("--weld-solimp", type=lambda s: [float(x) for x in s.split(",")])
    viewer, args = newton.examples.init(parser)
    args.front = args.side = args.top = args.width = args.depth = args.name = None
    args.repeat, args.p0_out = 0, None
    if args.part == "a":
        assert bool(args.shape) != bool(args.structure), "--part a needs --shape or --structure"
        # the backed-up dirty files in results/ are never touched: the log goes to OUT (see ProbeExample)
        part_a(args, viewer)
    elif args.part == "s":
        assert args.shape, "--part s needs --shape"
        part_s(args, viewer)
    elif args.part == "b":
        part_b(args)
    elif args.group and args.diag:
        part_c_diag(args)
    elif args.group and args.samples:
        part_c_samples(args)
    elif args.group:
        part_c_grouped(args)
    else:
        part_c(args)
