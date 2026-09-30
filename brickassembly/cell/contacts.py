"""Sec 2.4 contact monitor -- read-only classification of the cell's contacts.

Attach to a running dual_arm_sim.Example after its __init__ (it needs
ex.model, ex.body, ex.plan):

    ex.monitor = ContactMonitor(ex)

dual_arm_sim.Example.step() calls `self.monitor.step(self)` once per frame
when `self.monitor` is set; nothing here writes to model/state/control, or to
the example itself -- only .numpy() reads. The one write is to the monitor's own
Contacts buffer (`solver.update_contacts`, for peak force), never the sim's.

Signed distance is `solver.mjw_data.contact.dist` (plan Sec 0 fact 5, negative
= penetration); `contact.geom` holds MuJoCo geom ids, mapped to Newton shapes
via `solver.mjc_geom_to_newton_shape` and then to bodies via
`model.shape_body`. Bricks with a 10 mm gap (SDF_MARGIN) show up as contact
rows well before they touch, so only d < 0 counts.

Body classes: A_link, A_finger, B_link, B_finger (body < 2*n_arm_bodies, the
two fingers are local body index 12/13 in build_arm()), brick:<id>, plate
(baseplate mesh/box/stud proxies, all on body -1), ground (the ground plane's
shape, always the last one added, so it is simply the highest shape index --
`Example.__init__` calls `add_ground_plane()` after every brick, and it is
never called again).

Phase comes from dual_arm_sim's own state, never re-derived:
  - B's Arm.phase: park/to feeder/descend/grasp/lift/stage/transport/pre-insert/
    insert/release/retract (Example.queue_place; "stage" only on a braced step:
    B holds its brick over the feeder until A has closed).
  - A's Arm.phase: to brace/approach/brace-in/brace-guard/close/hold/open/
    retract/park (Example.queue_brace, the r5 grasp_lp brace); BRACE_PHASES =
    brace-guard, close, hold, open are its contact segment (A leaves contact when
    "open" ends). With --legacy-brace (Example.queue_brace_legacy): to brace/
    approach/brace/hold/retract/park, contact phases LEGACY_BRACE_PHASES.

Permitted windows (plan_v4 r4 Sec 2.4): the held brick resting on the ground at
its pick-up spot (descend/grasp/lift), and support/neighbour contact as
neighbour_rub through "release" as well as "insert" (the brick is still seated
on its supports). The plan's "search and a V7 re-press" do not exist in this
prototype (insert is one contact move), so that clause is n/a here.

Merging (r4): raw fragments are contiguous d<0 runs of one pair (per brick for
displaced, one raw record per moved frame); `merge_events` folds fragments of
the same key whose start is <= GAP_FRAMES (30 = 0.5 s at 60 fps) after the
previous one's last frame. finalize() applies it, and it also works offline on
the `events` rows of results/v4/p0_baseline.jsonl (P0 stored raw fragments).
  - ex.b_step is the plan step B is currently working (Example.schedule); the
    brick B is holding is ex.plan["sequence"][ex.b_step]["brick_id"] whenever
    B.phase is at or past "descend" and not yet past "release" (HELD_PHASES).
    A's brace windows never outlive ex.b_step's own step (B starts its next step
    only when A is idle; legacy: hold waits for B's placed-flag, which fires
    before B is idle enough to advance b_step), so reading ex.b_step during A's
    brace phases is safe.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import planner as P  # noqa: E402

PENETRATION_MM = 1.0   # brick-brick d below -PENETRATION_MM*scale => penetration
DISPLACED_MM = 1.0     # a resting brick moved more than this*scale => displaced
HELD_PHASES = ("descend", "grasp", "lift", "stage", "transport", "pre-insert", "insert", "release")
BRACE_PHASES = ("brace-guard", "close", "hold", "open")   # A's contact moves in the grasp_lp brace
LEGACY_BRACE_PHASES = ("brace", "hold")                   # A's two contact=True moves with --legacy-brace
GAP_FRAMES = 30        # merge same-key fragments up to this many frames apart (0.5 s at 60 fps)


def merge_events(events, gap=GAP_FRAMES):
    """Fold raw fragments into merged events, sorted by start_frame.

    Key: (class, a, b), or (class, brick) for displaced. A fragment joins the
    open event of its key when start_frame - last_frame <= gap. Works on the
    monitor's own fragments and on P0 rows (no contact_frames/fragments/force:
    duration_frames counts the frames, one raw fragment each). Merged fields:
    first/last frame (start_frame/end_frame), duration_frames (the span),
    contact_frames, fragments, max_pen_mm, peak_force_n (if present); displaced
    add path_mm (sum of moves) and max_move_mm (largest single move).
    """
    open_, out = {}, []
    for e in sorted(events, key=lambda e: e["start_frame"]):
        key = (e["class"], e["brick"]) if e["class"] == "displaced" else (e["class"], e["a"], e["b"])
        move = e.get("moved_mm")
        r = open_.get(key)
        if r is None or e["start_frame"] - r["end_frame"] > gap:
            r = open_[key] = {k: v for k, v in e.items() if k != "moved_mm"}
            r.update(contact_frames=0, fragments=0)
            if move is not None:
                r.update(path_mm=0.0, max_move_mm=0.0)
            out.append(r)
        r["end_frame"] = max(r["end_frame"], e["end_frame"])
        r["contact_frames"] += e.get("contact_frames", e["duration_frames"])
        r["fragments"] += e.get("fragments", 1)
        if "max_pen_mm" in e:
            r["max_pen_mm"] = min(r["max_pen_mm"], e["max_pen_mm"])
        if "peak_force_n" in e:
            r["peak_force_n"] = max(r.get("peak_force_n", 0.0), e["peak_force_n"])
        if move is not None:
            r["path_mm"] = round(r["path_mm"] + move, 2)
            r["max_move_mm"] = max(r["max_move_mm"], move)
        r["duration_frames"] = r["end_frame"] - r["start_frame"] + 1
    return out


def _same_course_neighbours(bricks):
    """{brick_id: {other_id, ...}}, same layer (k) and touching/overlapping footprints."""
    cells = {}
    for b in bricks:
        nx, ny = P.footprint(b["type"], b["yaw_index"])
        i, j, k = b["grid_pos"]
        cells[b["id"]] = ({(i + a, j + c) for a in range(nx) for c in range(ny)}, k)
    out = defaultdict(set)
    for bid, (c, k) in cells.items():
        grown = c | {(x + dx, y + dy) for x, y in c for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        for oid, (oc, ok) in cells.items():
            if oid != bid and ok == k and grown & oc:
                out[bid].add(oid)
    return out


class ContactMonitor:
    """One per run. `.step(ex)` each frame; `.summary()` at the end."""

    def __init__(self, ex, scale=1.0):
        self.n_arm = ex.n_arm_bodies
        self.finger = {12, 13}                        # local body index, both arms (build_arm)
        self.ground_shape = ex.model.shape_count - 1   # add_ground_plane() is always last
        self.sb = ex.model.shape_body.numpy()
        self.body_brick = {v: k for k, v in ex.body.items()}
        self.supports = {s["brick_id"]: [m[0] for m in s["mating_studs"]] or [None]
                         for s in ex.plan["sequence"]}
        self.neighbours = _same_course_neighbours(ex.plan["bricks"])
        self.scale = scale
        self.brace_phases = LEGACY_BRACE_PHASES if getattr(ex.args, "legacy_brace", False) else BRACE_PHASES
        self.rest = {}                     # body -> last at-rest position (ground truth)
        self.counts = defaultdict(int)
        self.examples = defaultdict(list)  # class -> up to 5 dicts
        self.events = []                   # merged events, set by finalize() (P0 needs the full list)
        self.raw = []                      # contiguous fragments, in close order
        self._active = {}                  # (class, bodyA, bodyB) -> open fragment
        self.frame = 0
        # own Contacts buffer for update_contacts (peak force); the sim's ex.contacts is untouched
        ex.model.request_contact_attributes("force")
        self.contacts = ex.pipeline.contacts()

    # -- classification --------------------------------------------------------
    def _cls(self, body, shape):
        if body == -1:
            return "ground" if shape == self.ground_shape else "plate"
        arm, local = ("A", body) if body < self.n_arm else ("B", body - self.n_arm)
        if body < 2 * self.n_arm:
            return "%s_finger" % arm if local in self.finger else "%s_link" % arm
        bid = self.body_brick.get(body)
        return "brick:%s" % bid if bid else "brick:?%d" % body

    def _held(self, ex):
        if ex.b_step is None:
            return None
        s = ex.plan["sequence"][ex.b_step]
        return s["brick_id"] if ex.B.phase in HELD_PHASES else None

    def _gripped(self, ex):
        """Bricks A's fingers may touch this frame (grasp_lp's gripped_bricks,
        or lever_press's single target_brick_id), else ()."""
        if ex.b_step is None or ex.A.phase not in self.brace_phases:
            return ()
        brace = ex.plan["sequence"][ex.b_step]["brace"] or {}
        return brace.get("gripped_bricks") or (
            [brace["target_brick_id"]] if brace.get("target_brick_id") else [])

    def _classify(self, ex, held, gripped, clsA, clsB):
        """Event class for a candidate (d<0) pair, or None if permitted."""
        armA, armB = clsA.endswith(("_link", "_finger")), clsB.endswith(("_link", "_finger"))
        if armA and armB and clsA[0] != clsB[0]:
            return "arm_arm"
        held_cls = "brick:%s" % held if held else None
        if held_cls and {clsA, clsB} == {"B_finger", held_cls}:
            return None                                   # B grasping its own brick
        if gripped:
            other = clsB if clsA == "A_finger" else (clsA if clsB == "A_finger" else None)
            if other and other.startswith("brick:") and other[6:] in gripped:
                return None                               # A bracing its gripped_bricks
        if held_cls and held_cls in (clsA, clsB):
            other = clsB if clsA == held_cls else clsA
            if other == "ground" and ex.B.phase in ("descend", "grasp", "lift"):
                return None                               # still resting on its pick-up spot
        if held_cls and ex.B.phase in ("insert", "release") and held_cls in (clsA, clsB):
            other = clsB if clsA == held_cls else clsA
            oid = other[6:] if other.startswith("brick:") else None
            is_support = oid in self.supports.get(held, []) or (
                other == "plate" and None in self.supports.get(held, []))
            is_neighbour = oid in self.neighbours.get(held, ())
            if is_support or is_neighbour:
                return "neighbour_rub"
        return "unintended"

    # -- per-frame ---------------------------------------------------------------
    def step(self, ex):
        self.frame += 1
        held = self._held(ex)
        gripped = self._gripped(ex)

        body_q = ex.state_0.body_q.numpy()
        for body, bid in self.body_brick.items():         # ground truth, scoring only
            pos = body_q[body][:3]
            if bid == held:                # in B's hand: it moves by design; rest = where it lands
                self.rest[body] = pos.copy()
                continue
            prev = self.rest.get(body)
            if prev is None:
                self.rest[body] = pos.copy()
                continue
            moved_mm = float(np.linalg.norm(pos - prev)) * 1000
            if moved_mm > DISPLACED_MM * self.scale:
                self._log("displaced", ex, {"brick": bid, "moved_mm": round(moved_mm, 2)})
                self.rest[body] = pos.copy()

        mjw = ex.solver.mjw_data
        n = int(mjw.nacon.numpy()[0])
        seen = set()
        if n:
            dist = mjw.contact.dist.numpy()[:n]
            neg = np.where(dist < 0)[0]
            if len(neg):
                geom = mjw.contact.geom.numpy()[neg]
                world = mjw.contact.worldid.numpy()[neg]
                g2s = ex.solver.mjc_geom_to_newton_shape.numpy()
                s0, s1 = g2s[world, geom[:, 0]], g2s[world, geom[:, 1]]
                b0, b1 = self.sb[s0], self.sb[s1]
                d = dist[neg]
                # force per contact, from the monitor's own buffer (valid for the last substep)
                ex.solver.update_contacts(self.contacts, ex.state_0)
                fmag = np.linalg.norm(self.contacts.force.numpy()[neg][:, :3], axis=1)

                pairs = {}                       # per unordered body pair: worst (min) dist, sum of |force|
                for i in range(len(neg)):
                    key = (b0[i], b1[i], s0[i], s1[i]) if b0[i] <= b1[i] else (b1[i], b0[i], s1[i], s0[i])
                    key, sh = key[:2], key[2:]
                    if key not in pairs:
                        pairs[key] = [d[i], sh, 0.0]
                    elif d[i] < pairs[key][0]:
                        pairs[key][:2] = d[i], sh
                    pairs[key][2] += fmag[i]

                for (bA, bB), (dd, (shA, shB), ff) in pairs.items():
                    clsA, clsB = self._cls(bA, shA), self._cls(bB, shB)
                    if clsA.startswith("brick:") and clsB.startswith("brick:") \
                            and dd * 1000 < -PENETRATION_MM * self.scale:
                        self._event("penetration", bA, bB, clsA, clsB, dd, ff, ex, seen)
                    is_candidate = (clsA.endswith(("_link", "_finger")) or clsB.endswith(("_link", "_finger"))
                                   or (held and "brick:%s" % held in (clsA, clsB)))
                    if not is_candidate:
                        continue
                    evt = self._classify(ex, held, gripped, clsA, clsB)
                    if evt:
                        self._event(evt, bA, bB, clsA, clsB, dd, ff, ex, seen)
        self._close(keep=seen)

    def _event(self, cls, bA, bB, clsA, clsB, d, force, ex, seen):
        key = (cls, bA, bB)
        seen.add(key)
        rec = self._active.get(key)
        if rec is None:
            rec = self._active[key] = {
                "class": cls, "a": "%s#%d" % (clsA, bA), "b": "%s#%d" % (clsB, bB),
                "phase": {"A": ex.A.phase, "B": ex.B.phase}, "start_frame": self.frame,
                "max_pen_mm": 0.0, "peak_force_n": 0.0}
        rec["end_frame"] = self.frame
        rec["max_pen_mm"] = round(float(min(rec["max_pen_mm"], d * 1000)), 3)
        rec["peak_force_n"] = round(float(max(rec["peak_force_n"], force)), 2)

    def _close(self, keep=frozenset()):
        for key in [k for k in self._active if k not in keep]:
            rec = self._active.pop(key)
            rec["duration_frames"] = rec["end_frame"] - rec["start_frame"] + 1
            self.raw.append(rec)

    def _log(self, cls, ex, extra):
        self.raw.append(dict(extra, **{"class": cls, "phase": {"A": ex.A.phase, "B": ex.B.phase},
                                       "start_frame": self.frame, "end_frame": self.frame,
                                       "duration_frames": 1}))

    def finalize(self):
        """Call once after the run ends: close open fragments, merge them into events."""
        self._close()
        self.events = merge_events(self.raw)
        self.counts, self.examples = defaultdict(int), defaultdict(list)
        for e in self.events:
            self.counts[e["class"]] += 1
            if len(self.examples[e["class"]]) < 5:
                self.examples[e["class"]].append(e)

    def summary(self):
        return {cls: {"count": self.counts[cls], "examples": self.examples[cls]}
                for cls in self.counts}


if __name__ == "__main__":
    # Self-check: the classification logic, without a live Newton scene (fast,
    # no GPU). Bypasses __init__ (which needs a running Example) and sets the
    # handful of attributes _cls/_classify/_same_course_neighbours read.
    class _Phase:
        def __init__(self, phase):
            self.phase = phase

    class _Ex:
        def __init__(self, a_phase, b_phase, b_step):
            self.A, self.B, self.b_step = _Phase(a_phase), _Phase(b_phase), b_step

    bricks = [
        {"id": "b0", "type": "2x4", "grid_pos": (0, 0, 0), "yaw_index": 0},
        {"id": "b1", "type": "2x4", "grid_pos": (0, 4, 0), "yaw_index": 0},   # same course, touches b0
        {"id": "b2", "type": "2x4", "grid_pos": (0, 0, 1), "yaw_index": 0},   # sits on b0, next course
    ]
    nb = _same_course_neighbours(bricks)
    assert nb["b0"] == {"b1"}, nb            # touching, same course
    assert nb.get("b2", set()) == set()      # different course, not a neighbour

    m = ContactMonitor.__new__(ContactMonitor)
    m.n_arm, m.finger, m.ground_shape = 14, {12, 13}, 999
    m.body_brick = {28: "b0", 29: "b1", 30: "b2"}
    m.supports = {"b2": ["b0"]}
    m.neighbours = nb
    m.brace_phases = BRACE_PHASES
    m.events, m.counts, m.examples = [], defaultdict(int), defaultdict(list)
    m.raw, m._active, m.frame = [], {}, 3
    ex_ = type("E", (), {"A": _Phase("park"), "B": _Phase("park")})()
    for f, mv in ((3, 2.0), (20, 1.5), (60, 3.0)):     # displaced: gap 17 merges, gap 40 does not
        m.frame = f
        m._log("displaced", ex_, {"brick": "b0", "moved_mm": mv})
    # unintended: frames 3, 4 (one fragment), then 30 (gap 26: merges), then 70 (gap 40: new event)
    hits = {3: (-0.001, 1.0), 4: (-0.003, 5.0), 30: (-0.002, 2.0), 70: (-0.001, 1.0)}
    for f in range(3, 71):
        m.frame, seen = f, set()
        if f in hits:
            m._event("unintended", 1, 2, "A_link", "brick:b1", *hits[f], ex_, seen)
        m._close(keep=seen)
    m.finalize()
    assert [e["class"] for e in m.events] == ["displaced", "unintended", "displaced", "unintended"], m.events
    d0, u0 = m.events[0], m.events[1]
    assert (d0["start_frame"], d0["end_frame"], d0["fragments"], d0["contact_frames"]) == (3, 20, 2, 2), d0
    assert d0["path_mm"] == 3.5 and d0["max_move_mm"] == 2.0, d0
    assert (u0["start_frame"], u0["end_frame"], u0["fragments"], u0["contact_frames"]) == (3, 30, 2, 3), u0
    assert u0["max_pen_mm"] == -3.0 and u0["peak_force_n"] == 5.0 and u0["duration_frames"] == 28, u0
    assert m.counts["unintended"] == 2 and m.counts["displaced"] == 2, dict(m.counts)

    # P0 rows (raw fragments) re-merge to the numbers in plan_v4 r4 (row 6 / B9)
    p0 = Path(__file__).resolve().parent.parent / "results" / "v4" / "p0_baseline.jsonl"
    if p0.exists():
        import json
        rows = {r["tag"]: r for r in map(json.loads, p0.open())}
        want = {"cube_grasp_lp_r0": {"neighbour_rub": 18},
                "arch_grasp_lp_r0": {"displaced": 35}, "arch_grasp_lp_r1": {"displaced": 40},
                "arch_grasp_lp_r2": {"displaced": 38},
                "S3_grasp_lp_r0": {"displaced": 146, "unintended": 76, "penetration": 56, "arm_arm": 8}}
        for tag, exp in want.items():
            got = defaultdict(int)
            for e in merge_events(rows[tag]["events"]):
                got[e["class"]] += 1
            assert all(got[c] == v for c, v in exp.items()), (tag, dict(got), exp)

    assert m._cls(-1, 999) == "ground"
    assert m._cls(-1, 5) == "plate"
    assert m._cls(12, 0) == "A_finger" and m._cls(7, 0) == "A_link"
    assert m._cls(14 + 13, 0) == "B_finger" and m._cls(14, 0) == "B_link"
    assert m._cls(29, 0) == "brick:b1"

    plan_step = {"brick_id": "b1", "brace": {"gripped_bricks": ["b0"]}}
    ex = _Ex("close", "to feeder", 0)
    ex.plan = {"sequence": [plan_step]}
    assert m._held(ex) is None                          # before "descend": not yet held
    ex.B.phase = "grasp"
    assert m._held(ex) == "b1"
    ex.B.phase = "stage"                                # braced step: B hovers over the feeder holding its brick
    assert m._held(ex) == "b1"
    assert m._classify(ex, "b1", [], "B_finger", "brick:b1") is None
    ex.B.phase = "grasp"
    for ph in BRACE_PHASES:                              # A's contact phases of the grasp_lp brace
        ex.A.phase = ph
        assert m._gripped(ex) == ["b0"], ph
    for ph in ("to brace", "approach", "brace-in", "retract", "park", "brace"):
        ex.A.phase = ph                                  # fingers may not touch anything outside them
        assert m._gripped(ex) == (), ph
    m.brace_phases = LEGACY_BRACE_PHASES                 # --legacy-brace: "brace", "hold"
    for ph, want in (("brace", ["b0"]), ("hold", ["b0"]), ("close", ()), ("brace-guard", ())):
        ex.A.phase = ph
        assert m._gripped(ex) == want, ph
    m.brace_phases, ex.A.phase = BRACE_PHASES, "close"
    assert m._gripped(ex) == ["b0"]

    # arm-arm always fires, regardless of phase
    assert m._classify(ex, "b1", ["b0"], "A_link", "B_finger") == "arm_arm"
    # B's fingers on its own held brick: permitted
    assert m._classify(ex, "b1", [], "B_finger", "brick:b1") is None
    # A's fingers on a brick outside gripped_bricks: unintended
    assert m._classify(ex, None, ["b0"], "A_finger", "brick:b1") == "unintended"
    # A's fingers on its gripped brick: permitted
    assert m._classify(ex, None, ["b0"], "A_finger", "brick:b0") is None
    # held brick against its support, only during "insert": neighbour_rub
    ex.B.phase = "insert"
    assert m._classify(ex, "b2", [], "brick:b2", "brick:b0") == "neighbour_rub"
    ex.B.phase = "release"                               # still seated on its supports
    assert m._classify(ex, "b2", [], "brick:b2", "brick:b0") == "neighbour_rub"
    ex.B.phase = "transport"
    assert m._classify(ex, "b2", [], "brick:b2", "brick:b0") == "unintended"
    ex.B.phase = "grasp"                                 # on the table at the pick-up spot
    assert m._classify(ex, "b2", [], "brick:b2", "ground") is None
    ex.B.phase = "transport"
    assert m._classify(ex, "b2", [], "brick:b2", "ground") == "unintended"
    print("contacts.py self-check OK")
