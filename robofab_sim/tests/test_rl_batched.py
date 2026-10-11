"""rl_batched.py (BatchedReachEnv: N worlds of the cell in one Newton model) against rl_env.py's BrickReachEnv. Needs the GPU; about 3 minutes.

  - the shared maths is the single env's: liveview.ManualCtlBatch = ManualCtl, the IK targets, the hand FK, the snap gate and rl_env.pose_errors = Example.gate;
  - the sensor noise on the GPU (sense_torch) has cell/cameras.sense's statistics, one stream per world;
  - spaces and the "policy" layout equal NewtonVecEnv(BrickReachEnv)'s; N = 1 gives the same first observation (to detection noise) and the scripted
    controller snaps cube step 0 in the same number of ticks with the same return; in 16 worlds every world snaps (held brick, finger contacts);
  - per-world physics: start steps, resets that leave the other worlds bit-identical, a snap welds in its world only, worlds with different actions diverge;
  - the arm-arm term fires in the world where A is driven into B (and only there), which terminates and resets alone while the others go on;
  - the firewall (the same test as test_rl_env's: moving bricks and flipping welds under frozen captures leaves the non-image observation unchanged);
  - the images group, the rsl_rl / skrl / rl_games adapters and an rsl_rl iteration run on it unchanged; the detection worker processes agree with in-process.

    bash scripts/run.sh tests/test_rl_batched.py
"""

import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    import torch
    import dual_arm_sim as D
    import rl_batched as B
    import rl_detect
    import rl_env as R
    import rl_vec as V
    from cell import cameras as CAM
    from cell import liveview as LV
    from scipy.spatial.transform import Rotation as Rot
except Exception as e:                                                   # no newton / GPU
    pytest.skip("cell unavailable: %r" % (e,), allow_module_level=True)

FORBIDDEN = ("body_q", "body_qd", "welds", "shape_index", "ground_truth", "breaker", "failure", "privileged", "contacts", "state_1", ".gate", ".mon",
             "equality_constraint", "last_gate", "snap_gates")


@pytest.fixture(scope="module")
def env4():
    env = B.BatchedReachEnv(4, start_steps=(0, 3), procs=0)
    yield env
    env.close()


def zeros(n, rows=()):
    """(n, 10) actions of zeros with `rows` {world: {column: value}} set."""
    a = torch.zeros(n, 10, device="cuda")
    for w, cols in dict(rows).items():
        for c, v in cols.items():
            a[w, c] = v
    return a


# --- the shared maths -------------------------------------------------------------------------------------------------------
def test_manual_ctl_batch_matches_scalar():
    rng = np.random.default_rng(1)
    park = lambda base: np.array(base[0]) + D.rotate(D.qz(base[1]), D.PARK)
    bases, parks, n = (D.ARM_A, D.ARM_B), [park(D.ARM_A), park(D.ARM_B)], 5
    single, batch = [LV.ManualCtl(bases, parks) for _ in range(n)], LV.ManualCtlBatch(n, bases, parks)
    cmd = [[(parks[k].copy(), bases[k][1], 0.01, np.zeros(3)) for k in (0, 1)] for _ in range(n)]
    cp, cy, cg = np.tile(parks, (n, 1, 1)), np.tile([D.ARM_A[1], D.ARM_B[1]], (n, 1)), np.full((n, 2), 0.01)
    for t in range(60):
        d, dyaw, grip = rng.normal(0, 0.05, (n, 2, 3)), rng.normal(0, 0.4, (n, 2)), rng.uniform(-0.01, 0.05, (n, 2))
        batch.nudge(d, dyaw, grip)
        ee = cp + rng.normal(0, 0.003, cp.shape)
        cp, cy, cg = batch.advance(cp, cy, cg, ee, 1 / 60)
        for i in range(n):
            for k in (0, 1):
                single[i].nudge(k, d[i, k], dyaw[i, k], grip=grip[i, k])
                cmd[i][k] = single[i].advance(k, cmd[i][k], ee[i, k], 1 / 60)
                assert np.allclose(cmd[i][k][0], cp[i, k], atol=1e-12) and np.isclose(cmd[i][k][1], cy[i, k]) and np.isclose(cmd[i][k][2], cg[i, k])
                assert np.allclose(single[i].tgt[k]["p"], batch.tgt_p[i, k], atol=1e-12) and np.isclose(single[i].tgt[k]["yaw"], batch.tgt_yaw[i, k])
    assert (batch.tgt_p[:, 0, 0] <= LV.CROSS + 1e-12).all() and (batch.tgt_p[:, 1, 0] >= -LV.CROSS - 1e-12).all()


def test_pose_helpers_match_the_example():
    rng = np.random.default_rng(2)
    for _ in range(50):
        p, yaw = rng.uniform([-0.3, -0.3, 0.05], [0.3, 0.3, 0.4], (2, 3)), rng.uniform(-3, 3, 2)
        pos, rot = B.hand_target_pose(p, yaw)
        for k, (base, byaw) in enumerate((D.ARM_A, D.ARM_B)):
            assert np.allclose(pos[k], D.rotate(D.qz(-byaw), p[k] - base), atol=1e-12)
            assert np.allclose(rot[k], D.qmul(D.qz(-byaw), D.qmul(D.qrot(np.zeros(3)), D.qmul(D.qz(yaw[k]), D.Q_DOWN))), atol=1e-12)    # Example.step, direct mode
        t = np.concatenate([rng.normal(size=(2, 3)), Rot.random(2, random_state=int(rng.integers(1e6))).as_quat()], 1)[None]
        w = B.hand_world_pose(t)[0]
        for k, (base, byaw) in enumerate((D.ARM_A, D.ARM_B)):
            ref = np.r_[D.rotate(D.qz(byaw), t[0, k, :3]) + base, D.qmul(D.qz(byaw), t[0, k, 3:])]          # Example.hand_fk's last line
            assert np.allclose(w[k, :3], ref[:3]) and abs(w[k, 3:] @ ref[3:]) > 1 - 1e-9


def test_gate_and_pose_errors_match_the_example():
    rng = np.random.default_rng(3)
    ok_seen = set()
    for i in range(300):
        tgt, yaw, square = np.array([0.01, -0.02, 0.0032]), float(rng.choice([0.0, math.pi / 2])), bool(i % 2)
        e = rng.normal(0, [0.0015, 0.0015, 0.001, 0.03, 0.03], 5)
        q = (Rot.from_euler("z", yaw + e[3]) * Rot.from_euler("x", e[4])).as_quat()
        bq = np.array([[*(tgt[:2] + e[:2]), tgt[2] + e[2], *q]], np.float32)
        fake = SimpleNamespace(state_0=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: bq)), body={"b": 0}, target={"b": (tgt, yaw, square)})
        _, lat, dz, dyaw, tilt, ok, _ = D.Example.gate(fake, "b")
        la, dza, dya, tia = (float(x) for x in R.pose_errors(bq[0], tgt, yaw, square))
        assert np.allclose([la * 1e3, dza * 1e3, math.degrees(dya), math.degrees(tia)], [lat, dz, dyaw, tilt], atol=1e-4)      # Example.gate does the quaternion maths in float32
        assert bool(B.gate_ok(la, dza, dya, tia)) == ok
        ok_seen.add(ok)
    assert ok_seen == {True, False}


def test_sense_torch_matches_the_noise_model():
    rng = np.random.default_rng(4)
    n, h, w = 4, 240, 320
    depth = np.where(rng.random((n, h // 4, w // 4)) < 0.5, 0.9, 1.0).repeat(4, 1).repeat(4, 2).astype(np.float32)       # blocks of two depths: lots of edges
    hdr = np.full((n, h, w, 3), 0.4, np.float32)
    gens = [torch.Generator(device="cuda").manual_seed(i) for i in range(n)]
    nd, rgb = B.sense_torch(torch.as_tensor(depth, device="cuda"), torch.as_tensor(hdr, device="cuda"), torch.ones(n, device="cuda"), gens)
    nd, rgb = nd.cpu().numpy(), rgb.cpu().numpy()
    ref_d, ref_rgb = CAM.sense(dict(depth=depth[0], hdr=hdr[0]), np.random.default_rng(0), 1.0)
    edge = np.zeros(depth.shape, bool)
    jv, jh = np.abs(np.diff(depth, axis=1)) > 3e-3, np.abs(np.diff(depth, axis=2)) > 3e-3
    edge[:, :-1] |= jv
    edge[:, 1:] |= jv
    edge[:, :, :-1] |= jh
    edge[:, :, 1:] |= jh
    dropped = nd == 0
    for a in (nd, ):
        z = (a[~dropped] - depth[~dropped]) / (2e-3 * depth[~dropped] ** 2)
        assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02                                    # sigma = 2e-3 r^2
        assert 0.007 < dropped[edge].mean() < 0.013 and dropped[~edge].sum() == 0                  # 1 % of the edge pixels, none elsewhere
    assert abs(rgb.mean() - ref_rgb.mean()) < 0.5 and abs(rgb.std() / ref_rgb.std() - 1) < 0.1      # the rgb noise and the soft clip of cameras.sense
    assert np.abs(ref_d[ref_d > 0].mean() - nd[0][nd[0] > 0].mean()) < 5e-4
    assert not np.array_equal(nd[0], nd[1])                                                         # one stream per world
    nd2, _ = B.sense_torch(torch.as_tensor(depth[:1], device="cuda"), torch.as_tensor(hdr[:1], device="cuda"), torch.ones(1, device="cuda"),
                           [torch.Generator(device="cuda").manual_seed(0)])
    assert np.array_equal(nd2.cpu().numpy()[0], nd[0])                                              # reproducible from the world's seed


# --- the env against the single env ------------------------------------------------------------------------------------------------
def test_spaces_and_first_observation_match_brick_reach_env():
    single = V.make_env(R.BrickReachEnv, dict(images=False))                   # what a NewtonVecEnv worker runs
    o1, _ = single.reset(seed=0)
    env = B.BatchedReachEnv(1, procs=0)
    try:
        o2, _ = env.reset(seed=0)
        assert single.observation_space["policy"] == env.single_observation_space["policy"] and single.action_space == env.single_action_space
        assert o2["policy"].shape == (1, 390) and o2["policy"].dtype == torch.float32 and o2["policy"].device.type == "cuda"
        a, b, P = o1["policy"], o2["policy"][0].cpu().numpy(), 54
        assert np.abs(a[:P] - b[:P]).max() < 1e-5                                           # proprioception: the same joints, the same FK
        assert np.array_equal(a[P + 328:], b[P + 328:])                                     # the plan
        ra, rb = a[P:P + 320].reshape(16, 20), b[P:P + 320].reshape(16, 20)                 # top_detect: the same 8 bricks, to the detection noise (half-size render)
        assert ra[:, 0].sum() == rb[:, 0].sum() == 8
        assert np.abs(ra[:, 1:4] - rb[:, 1:4]).max() < 3e-3 and np.array_equal(ra[:, 8:16], rb[:, 8:16])
        assert abs(a[P + 321] - b[P + 321]) < 5e-3 and abs(a[P + 322] - b[P + 322]) < 5e-3     # plate x, y
    finally:
        env.close()


def test_scripted_n1_matches_brick_reach_env():
    ret, ticks, info, _ = R.run_scripted(R.BrickReachEnv(images=False))
    env = B.BatchedReachEnv(1, procs=0)
    try:
        res = B.run_scripted(env)
    finally:
        env.close()
    print("scripted cube step 0: BrickReachEnv return %.3f in %d ticks; BatchedReachEnv(1) return %.3f in %d ticks" % (ret, ticks, res["ret"][0], res["ticks"][0]))
    assert info["events"] == ["success"] and res["success"][0] and res["event"][0] == "success"
    assert abs(int(res["ticks"][0]) - ticks) <= 2 and abs(res["ret"][0] - ret) < 1.0


def test_scripted_n16_every_world_snaps():
    env = B.BatchedReachEnv(16)
    try:
        res = B.run_scripted(env)
    finally:
        env.close()
    print("N=16 ticks to success %s, return %.2f-%.2f, max slip %.2f mm" % (res["ticks"].tolist(), res["ret"].min(), res["ret"].max(), res["slip_mm"].max()))
    assert res["success"].all() and set(res["event"]) == {"success"} and (res["ticks"] < env.max_steps).all()
    assert res["ret"].min() > 50 and res["slip_mm"].max() < 3.0                              # the brick stays in the fingers while carried
    assert res["ticks"].max() - res["ticks"].min() <= 3


# --- per-world physics --------------------------------------------------------------------------------------------------------------
def test_start_steps_and_reset_of_one_world(env4):
    c = env4.cells
    env4.reset(seed=0, options={"step": [0, 3, 0, 3]})
    bq = c.body_q()
    nab, sb = c.nab, c.bpw
    z = lambda w, n: bq[w, nab * 2 + n, 2]                                       # brick n of world w: its z
    assert np.allclose([z(1, 0), z(1, 1), z(1, 2)], [env4.tgt[n][2] for n in range(3)], atol=1e-6) and z(1, 3) < 0.002 and z(0, 0) < 0.002
    en = c.model.equality_constraint_enabled.numpy().reshape(4, -1)
    n_welds = sum(len(c.c.welds[s["brick_id"]]) for s in c.steps[:3])
    assert en[1].sum() == en[3].sum() == n_welds and en[0].sum() == en[2].sum() == 0
    grp = c.model.shape_collision_group.numpy()[c.n_global:].reshape(4, -1)
    sbody = c.model.shape_body.numpy()[c.n_global:c.n_global + c.spw]
    brick0, brick5 = c.c.body[c.steps[0]["brick_id"]], c.c.body[c.steps[5]["brick_id"]]
    assert (grp[1][sbody == brick0] == D.STRUCT_GROUP).all() and (grp[0][sbody == brick0] != D.STRUCT_GROUP).all() and (grp[1][sbody == brick5] != D.STRUCT_GROUP).all()
    assert (grp[:, c.c.plate_shapes] == D.STRUCT_GROUP).all()                    # the baseplate, in every world
    assert (env4.step_n == [0, 3, 0, 3]).all()
    for _ in range(2):
        env4.step(zeros(4, {1: {5: 1.0}, 2: {0: -1.0, 7: 0.5}}))
    q0, qd0 = c.state_0.joint_q.numpy().copy(), c.state_0.joint_qd.numpy().copy()
    c.reset_worlds([1], [0])
    q1 = c.state_0.joint_q.numpy().reshape(4, -1)
    assert np.array_equal(np.delete(q0.reshape(4, -1), 1, 0), np.delete(q1, 1, 0)) and np.array_equal(np.delete(qd0.reshape(4, -1), 1, 0), np.delete(c.state_0.joint_qd.numpy().reshape(4, -1), 1, 0))
    en = c.model.equality_constraint_enabled.numpy().reshape(4, -1)
    assert np.allclose(q1[1], c.T[0]["q"]) and en[1].sum() == 0 and en[3].sum() == n_welds                # world 1 is back at start 0, world 3 keeps its welds


def test_breaker_reads_the_same_weld_wrenches_as_joint_breaker(env4):
    env4.reset(seed=0, options={"step": [3, 0, 3, 0]})
    for _ in range(8):
        env4.step(zeros(4))
    single = R.BrickReachEnv(images=False, step=3)
    single.reset(seed=0)
    for _ in range(8):
        single.step(np.zeros(10, np.float32))
    ref = single.sim.ex.breaker.sample()
    b = env4.breaker
    snap = b.snapshot()
    assert len(ref) == 4 and b.sample(1, snap) == {} and b.sample(3, snap) == {}                      # no welds before the first snap
    for w in (0, 2):
        got = b.sample(w, snap)
        assert set(got) == set(ref)
        for k in ref:
            for q in ("F", "M", "Fp", "Mp"):
                assert np.allclose(got[k][q], ref[k][q], atol=5e-4), (w, k, q, got[k][q], ref[k][q])
            assert abs(got[k]["u"] - ref[k]["u"]) < 1e-3
    assert max(r["F"][2] for r in ref.values()) > 0.005                                               # the readout is not all zeros: the bricks' weight


def test_joint_break_in_the_worlds_that_have_welds(env4, monkeypatch):
    monkeypatch.setattr(D, "U_BREAK", -1.0)                                                           # every sample is "over the limit": the rule fires as soon as it may
    env4.reset(seed=0, options={"step": [0, 3, 0, 3]})
    first = None
    for t in range(4):
        o, r, te, tr, ex = env4.step(zeros(4))
        if te.any():
            first = t
            break
    assert first == 1 and te.tolist() == [False, True, False, True], (first, te.tolist())               # settle 0.1 s = 1 tick, then 2 consecutive frames
    assert ex["events"]["joint_break"].tolist() == [False, True, False, True] and (r[[1, 3]] < -19).all() and (r[[0, 2]] > -1).all()
    assert env4.episode_length_buf.tolist() == [2, 0, 2, 0]                                           # the broken worlds restarted, the others went on


def test_worlds_with_different_actions_diverge(env4):
    env4.reset(seed=0, options={"step": 0})
    for _ in range(6):
        env4.step(zeros(4, {1: {5: 1.0}, 3: {0: 1.0, 2: 0.5}}))
    bq = env4.cells.body_q()
    hand = lambda w, k: bq[w, D.EE + k * env4.cells.nab, :3]
    assert np.abs(hand(0, 1) - hand(2, 1)).max() < 1e-3 and np.abs(hand(0, 0) - hand(2, 0)).max() < 1e-3          # equal actions, equal hands
    assert np.abs(hand(1, 1) - hand(0, 1)).max() > 0.02 and np.abs(hand(3, 0) - hand(0, 0)).max() > 0.02            # different actions, different hands
    assert not torch.equal(env4._obs["policy"][0], env4._obs["policy"][1])


def test_snap_welds_only_its_world(env4):
    c = env4.cells
    env4.reset(seed=0, options={"step": 0})
    bq = c.state_0.body_q.numpy()
    bq[1 * c.bpw + env4.body_local[0]] = [*env4.tgt[0], *D.qz(env4.tyaw[0])]       # world 1's current brick at its target
    c.state_0.body_q.assign(bq)
    env4._snap(c.body_q())
    assert env4.placed.tolist() == [False, True, False, False]
    en = c.model.equality_constraint_enabled.numpy().reshape(4, -1)
    mine = [e[0] for e in c.c.welds[c.steps[0]["brick_id"]]]
    assert en[1, mine].all() and en[1].sum() == len(mine) and en.sum() == len(mine)
    grp = c.model.shape_collision_group.numpy()[c.n_global:].reshape(4, -1)
    sel = c.model.shape_body.numpy()[c.n_global:c.n_global + c.spw] == c.c.body[c.steps[0]["brick_id"]]
    assert (grp[1][sel] == D.STRUCT_GROUP).all() and (grp[0][sel] != D.STRUCT_GROUP).all()


def test_arm_arm_term_in_one_world_and_autoreset(env4):
    env4.reset(seed=0, options={"step": 0})
    a = zeros(4, {3: {0: 1.0}})                                                       # world 3: arm A's hand toward B's
    fired, lens = None, []
    for t in range(40):
        o, r, te, tr, ex = env4.step(a)
        lens.append(env4.episode_length_buf.tolist())
        aa = ex["privileged"]["arm_arm"].tolist()
        if te.any():
            fired = t
            break
        assert aa == [0, 0, 0, 0], (t, aa)
    assert fired is not None and te.tolist() == [False, False, False, True] and ex["events"]["arm_arm"].tolist() == [False, False, False, True]
    assert ex["privileged"]["arm_arm"][3] > 0 and r[3] < -19 and (r[:3] > -1).all() and not tr.any()
    assert lens[-1] == [fired + 1] * 3 + [0] and ex["log"]["Episode/length"].tolist() == [float(fired + 1)] and ex["log"]["Episode/success"].tolist() == [0.0]
    o, r, te, tr, ex = env4.step(zeros(4))                                              # the reset world went on from its start, the others did not stop
    assert env4.episode_length_buf.tolist() == [fired + 2] * 3 + [1] and not te.any()
    assert abs(env4.cells.body_q()[3, D.EE, 0] - (-0.15)) < 0.02                         # A of world 3 is back at park


def test_per_world_truncation_and_vec_interface(env4):
    env4.reset(seed=0, options={"step": 0})
    env4.rew.max_steps = np.array([3, 50, 50, 50])
    try:
        lens, done = [], []
        for _ in range(4):
            o, r, te, tr, ex = env4.step(zeros(4))
            lens.append(env4.episode_length_buf.tolist())
            done.append((te | tr).tolist())
        assert lens == [[1, 1, 1, 1], [2, 2, 2, 2], [0, 3, 3, 3], [1, 4, 4, 4]] and done[2] == [True, False, False, False] and tr.tolist() == [False] * 4
        assert not te.any() and torch.isfinite(o["policy"]).all() and o["policy"].shape == (4, 390) and r.shape == (4,) and r.dtype == torch.float32
        assert env4.max_episode_length == 600 and env4.cfg.decimation == 6 and env4.single_action_space.shape == (10,) and env4.observation_space["policy"].shape == (4, 390)
    finally:
        env4.rew.max_steps = 600


# --- the firewall -------------------------------------------------------------------------------------------------------------------------
def flat(o, pre=""):
    for k, v in sorted(o.items()):
        if isinstance(v, dict):
            yield from flat(v, pre + k + "/")
        else:
            yield pre + k, v.cpu().numpy()


def same(a, b):
    return all(k1 == k2 and np.array_equal(x, y) for (k1, x), (k2, y) in zip(flat(a), flat(b)))


def test_firewall_static():
    assert R.ALLOWED_INPUTS == ("q", "qd", "hands", "det", "images", "task", "features")
    for fn in (B.BatchedReachEnv._observe, rl_detect.detect_world, B.hand_world_pose):
        src = inspect.getsource(fn)
        assert not [w for w in FORBIDDEN if w in src], (fn.__qualname__, [w for w in FORBIDDEN if w in src])
    assert list(inspect.signature(B.BatchedFK.__call__).parameters) == ["self", "q"] and list(inspect.signature(B.BatchedFK.__init__).parameters) == ["self", "n"]   # the FK sees the joint angles only (its own model, not the cells)


def test_firewall_dynamic(env4):
    env4.reset(seed=3, options={"step": 0})
    for _ in range(2):
        env4.step(zeros(4))
    c, live = env4.cells, env4.cams.capture
    frozen = live("top")
    env4.cams.capture = lambda cam, ee=None: frozen
    try:
        env4._observe()
        ref = env4._observe()
        assert same(ref, env4._observe()), "_observe() is not idempotent on frozen captures: the test cannot tell"
        bq = c.state_0.body_q.numpy().copy()                                               # ground truth moves: every brick, the welds, the groups
        for w in range(4):
            for b in c.c.body.values():
                bq[w * c.bpw + b][:3] += [0.07, -0.05, 0.02]
                bq[w * c.bpw + b][3:] = [0, 0, 0.38, 0.92]
        c.state_0.body_q.assign(bq)
        c.model.equality_constraint_enabled.assign(~c.model.equality_constraint_enabled.numpy())
        env4.diverged[:] = True
        env4.cells.sim_time += 5
        assert same(ref, env4._observe()), "an observation entry follows simulator ground truth"
        env4.cams.capture = live                                                          # control: live cameras do see the moved bricks
        assert not same(ref, env4._observe())
    finally:
        env4.cams.capture = live
        env4.diverged[:] = False


# --- images, workers, adapters --------------------------------------------------------------------------------------------------------------
def test_images_group():
    env = B.BatchedReachEnv(2, images=True, procs=0)
    try:
        o, _ = env.reset()
        im = o["images"]
        assert sorted(im) == sorted(c + s for c in R.CAM_NAMES for s in ("_rgb", "_depth"))
        assert all(im[c + "_rgb"].shape == (2, 240, 320, 3) and im[c + "_rgb"].dtype == torch.uint8 and im[c + "_depth"].shape == (2, 240, 320) for c in R.CAM_NAMES)
        assert all(im[c + "_depth"].dtype == torch.float32 and (im[c + "_depth"] > 0).float().mean() > 0.2 for c in R.CAM_NAMES)
        sp = env.single_observation_space["images"]
        assert all(sp[k].shape == tuple(v.shape[1:]) for k, v in im.items())
        g = V.groups(o)
        assert g["top_rgb"].shape == (2, 3, 240, 320) and g["wrist_B_depth"].shape == (2, 1, 240, 320)
        o, *_ = env.step(zeros(2))
        assert o["images"]["wrist_B_rgb"].shape == (2, 240, 320, 3)
    finally:
        env.close()


def test_detector_worker_processes_agree_with_in_process(env4):
    env4.reset(seed=1, options={"step": [0, 3, 0, 3]})
    d, rgb, npos, nR = env4.cams.capture("top")
    intr = env4.intr
    dets = []
    for procs in (0, 2):
        det = rl_detect.Detector(4, env4.top_hw, intr, CAM.TOP_POS, Rot.identity(), procs)
        try:
            assert len(det.workers) == procs and all(w.is_alive() for w in det.workers)
            det.rgb_t.copy_(rgb)
            det.depth_t.copy_(d)
            hands = [[dict(name=n, pos=np.array([0.0, 0, 0.3]), R=np.eye(3), opening=np.array([0.015, 0.015])) for n in "AB"] for _ in range(4)]
            out = det.detect(hands)
            out = det.detect(hands)
            det.reset([1])
            dets.append((out, det.detect(hands)))
        finally:
            det.close()
    (a, a2), (b, b2) = dets
    for x, y in ((a, b), (a2, b2)):
        for u, v in zip(x, y):
            assert len(u["bricks"]) == len(v["bricks"]) > 0 and u["plate"]["size"] == v["plate"]["size"]
            assert np.allclose([[p["x"], p["y"], p["yaw"], p["z_top"]] for p in u["bricks"]], [[p["x"], p["y"], p["yaw"], p["z_top"]] for p in v["bricks"]])
    assert all(len(u["bricks"]) == 8 for u in a)


def test_vec_adapters_and_an_rsl_rl_iteration(env4):
    rs = V.RslRlVecEnv(env4)
    g = rs.get_observations()
    assert list(g.keys()) == ["policy"] and g["policy"].shape == (4, 390) and rs.num_actions == 10 and rs.max_episode_length == 600
    _, rew, dones, extras = rs.step(torch.zeros(4, 10, device="cuda"))
    assert dones.dtype == torch.long and extras["time_outs"].shape == (4,)
    gm = V.RlGamesVecEnv(env4).get_env_info()
    assert gm["observation_space"].shape == (390,) and gm["action_space"].shape == (10,)
    try:
        w = V.skrl_env(env4)
    except ImportError:
        w = None
    if w is not None:
        obs, _ = w.reset()
        assert obs.shape == (4, 390) and w.num_envs == 4 and w.action_space.shape == (10,)
        obs, rew, term, trunc, info = w.step(torch.zeros(4, 10, device="cuda"))
        assert obs.shape == (4, 390) and rew.shape[0] == 4
    import train_rl as T
    T.train_rsl_rl(env4, T.RslRlCfg(max_iterations=1, num_steps_per_env=4))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q", "-s"]))
