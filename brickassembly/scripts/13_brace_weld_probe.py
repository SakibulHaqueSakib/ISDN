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


class Static:
    def __init__(self, kind, setting, substeps, collide=True, fps=60):
        self.kind, self.substeps, self.fps = kind, substeps, fps
        self.dt = 1.0 / fps / substeps
        lay = LAYOUT[kind]
        s = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(s)
        NI = 4 + max(x for x, _ in lay)
        NJ = 2
        cx, cy = NI * PITCH / 2, NJ * PITCH / 2
        if collide:
            s.add_shape_box(-1, hx=(NI + 4) * PITCH / 2, hy=(NJ + 4) * PITCH / 2, hz=D.Z0 / 2,
                            xform=wp.transform((cx, cy, D.Z0 / 2), wp.quat_identity()), cfg=D.PROXY_CFG)
            hh = 0.5 * ex.STUD_HEIGHT - ex.COLLIDER_INSET
            for i in range(NI):
                for j in range(NJ):
                    s.add_shape_cylinder(-1, radius=ex.STUD_COLLIDER_RADIUS, half_height=hh, cfg=D.PROXY_CFG,
                                         xform=wp.transform(((i + 0.5) * PITCH, (j + 0.5) * PITCH, D.Z0 + hh),
                                                            wp.quat_identity()))
        if "mesh" not in _MESH:
            mv, mf = ex._make_brick_mesh(4, 2)
            _MESH["mesh"] = ex._build_mesh_with_sdf(mv, mf, color=D.COLORS[0])
        cfg = D.BRICK_CFG if collide else newton.ModelBuilder.ShapeConfig(
            density=ex.BRICK_DENSITY, has_shape_collision=False, has_particle_collision=False)
        solimp = s.custom_attributes["mujoco:geom_solimp"]
        solimp.values = solimp.values or {}
        self.body, self.pos = [], []
        for n, (xo, layer) in enumerate(lay):
            p = np.array([cx - NI * PITCH / 2 + (2 + xo) * PITCH, cy, D.Z0 + layer * BH])   # x-centre of the 4-stud brick
            b = s.add_body(xform=wp.transform(p, wp.quat_identity()), label="b%d" % n)
            sh = s.add_shape_mesh(b, mesh=_MESH["mesh"], cfg=cfg, color=D.COLORS[n % len(D.COLORS)])
            solimp.values[sh] = (0.6, 0.95, 0.00075, 0.5, 2.5)
            if collide:
                D.add_proxies(s, b, 4, 2, wp.transform_identity())
            self.body.append(b)
            self.pos.append(p)
        self.welds = []
        for n in range(len(lay)):
            b1 = -1 if n == 0 else self.body[n - 1]
            rel = wp.transform(self.pos[n] - (0 if n == 0 else self.pos[n - 1]), wp.quat_identity())
            self.welds.append(s.add_equality_constraint_weld(
                body1=b1, body2=self.body[n], relpose=rel, enabled=True, label="w%d" % n,
                custom_attributes=attrs(setting)))
        s.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.75))
        self.model = m = s.finalize()
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

    def set_force(self, n, f):
        a = np.zeros((self.model.body_count, 6), np.float32)
        a[self.body[n], :3] = f
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
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    dirs = {"column": {"lat_x": (1, 0, 0), "up": (0, 0, 1), "down": (0, 0, -1)},
            "cantilever": {"down": (0, 0, -1), "up": (0, 0, 1), "lat_x": (1, 0, 0)}}
    fout = (OUT / "part_b.jsonl").open("w")
    for kind in ("column", "cantilever"):
        top = len(LAYOUT[kind]) - 1
        for setting in SETTINGS:
            for sub in (16, 32):
                st = Static(kind, setting, sub)
                st.run(60)                                            # settle under gravity
                ok = st.finite()
                base = dict(scene=kind, setting=setting, substeps=sub, mass_g=rnd(st.mass[0] * 1000, 3))
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
    (OUT / "part_b_summary.json").write_text(json.dumps(summ, indent=1))
    print("\nstiffness N/mm at F = 1 / 5 / 10 / 20 / 50 N (None = below 0.1 um)")
    for k, v in summ.items():
        print("%-42s %s" % (k, v if v == "DIVERGED" else "  ".join(
            "%s" % ("%8.2f" % x if x is not None else "    None") for x in v["by_F"].values())))
    print("-> %s" % (OUT / "part_b.jsonl"))


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


def brick_wrench(st, k, w):
    """Wrench the weld exerts ON brick k (body2), about the brick origin, world frame.
    MuJoCo's efc force acts on body1 as +J^T f, so on body2 it is -f. The 3 rotational rows are
    0.5 * torquescale * R2^T-rotated (quaternion-error Jacobian), hence torque = 0.5 * ts * R2 f_rot."""
    f, _, _, _ = w
    ts = float(st.model.equality_constraint_torquescale.numpy()[st.welds[k]])
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


# =============================================================================
if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--part", required=True, choices=["a", "b", "c"])
    parser.add_argument("--shape", choices=list(D.B.SAMPLES))
    parser.add_argument("--structure", choices=list(P.STRUCTURES))
    parser.add_argument("--strategy", default="weakest_joint", choices=["none", "nearest", "weakest_joint"])
    parser.add_argument("--brace-model", default="grasp_lp", choices=["grasp_lp", "lever_press"])
    parser.add_argument("--tag", help="suffix of the Part A output file")
    parser.add_argument("--weld-solref", type=lambda s: [float(x) for x in s.split(",")])
    parser.add_argument("--weld-solimp", type=lambda s: [float(x) for x in s.split(",")])
    viewer, args = newton.examples.init(parser)
    args.front = args.side = args.top = args.width = args.depth = args.name = None
    args.repeat, args.p0_out = 0, None
    if args.part == "a":
        assert bool(args.shape) != bool(args.structure), "--part a needs --shape or --structure"
        # the backed-up dirty files in results/ are never touched: the log goes to OUT (see ProbeExample)
        part_a(args, viewer)
    elif args.part == "b":
        part_b(args)
    else:
        part_c(args)
