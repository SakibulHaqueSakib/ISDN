"""rl_env.py (BrickStepEnv, BrickReachEnv) and rl_vec.py (NewtonVecEnv) on the real cell. Needs the GPU; a few minutes (every reset builds a scene, and a
BrickStepEnv step is a whole 15-40 s placement).

  - the firewall: no non-image observation entry depends on simulator ground truth. With the cameras frozen (the same captures returned), moving every
    brick and toggling welds and the failure flag leaves proprio, bricks, plate, task and the images unchanged; with the cameras live the moved
    bricks do show (the test can see a leak); and the observation code (Sim.sense / observe / ObsSpec.encode) names no ground-truth attribute;
  - gymnasium's check_env on both envs, images on and off;
  - BrickStepEnv: cube step 0, no brace: reward 1.0; arch step 10 with the LP-shared brace runs to completion and reports its outcome;
  - BrickReachEnv: the scripted controller seats the cube's step-0 brick (success, positive return); a random policy runs to truncation;
  - NewtonVecEnv: two subprocess workers, auto-reset of one while the other continues, the "images" group shapes, the rsl_rl adapter's observation groups.

    bash scripts/run.sh tests/test_rl_env.py
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    import rl_env as R
    import rl_vec as V
except Exception as e:                                                   # no newton / GPU
    pytest.skip("cell unavailable: %r" % (e,), allow_module_level=True)

FORBIDDEN = ("body_q", "body_qd", "welds", "shape_index", "ground_truth", "breaker", "failure", "privileged", "contacts", "state_1", ".gate", ".mon",
             "equality_constraint", "last_gate", "snap_gates")


def flat(o, pre=""):
    for k, v in sorted(o.items()):
        if isinstance(v, dict):
            yield from flat(v, pre + k + "/")
        else:
            yield pre + k, np.asarray(v)


def same(a, b):
    return all(k1 == k2 and np.array_equal(x, y) for (k1, x), (k2, y) in zip(flat(a), flat(b)))


def test_firewall_static():
    assert R.ALLOWED_INPUTS == ("q", "qd", "hands", "det", "images", "task", "features")
    for fn in (R.Sim.sense, R.Sim.observe, R.ObsSpec.encode):
        src = inspect.getsource(fn)
        assert not [w for w in FORBIDDEN if w in src], (fn.__qualname__, [w for w in FORBIDDEN if w in src])


def test_firewall_dynamic():
    env = R.BrickReachEnv(images=True, image_stride=8)
    env.reset(seed=3)
    sim, ex = env.sim, env.sim.ex
    for _ in range(2):
        env.step(np.zeros(10, np.float32))
    live = sim.cams.capture
    frozen = {c: live(c, ex.hand_fk(int(c[-1] == "B")) if c != "top" else None, state=ex.state_0) for c in R.CAM_NAMES}
    sim.cams.capture = lambda cam, ee=None, **kw: frozen[cam]
    sim.observe()
    ref = sim.observe()
    assert same(ref, sim.observe()), "observe() is not idempotent on frozen captures: the test cannot tell"
    bq = ex.state_0.body_q.numpy().copy()                                # ground truth moves: every brick, the welds, the failure flag
    for b in ex.body.values():
        bq[b][:3] += [0.07, -0.05, 0.02]
        bq[b][3:] = [0, 0, 0.38, 0.92]
    ex.state_0.body_q.assign(bq)
    en = ex.model.equality_constraint_enabled.numpy()
    ex.model.equality_constraint_enabled.assign(~en)
    ex.failure, ex.sim_time = "joint_break", ex.sim_time + 5
    assert same(ref, sim.observe()), "an observation entry follows simulator ground truth"
    sim.cams.capture = live                                              # control: live cameras do see the moved bricks
    assert not np.array_equal(ref["bricks"], sim.observe()["bricks"])


@pytest.mark.parametrize("images", [False, True])
def test_check_env_reach(images):
    from gymnasium.utils.env_checker import check_env
    env = R.BrickReachEnv(images=images, max_steps=3)
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("images", [False, True])
def test_check_env_step(images):
    from gymnasium.utils.env_checker import check_env
    env = R.BrickStepEnv(images=images, contexts=(("cube", 0),))
    check_env(env, skip_render_check=True)


def test_step_env_cube_unbraced():
    env = R.BrickStepEnv(images=False)
    obs, _ = env.reset(seed=1, options=dict(structure="cube", step=0))
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(dict(brace=0, brace_pose=np.zeros(3, np.float32), press=np.zeros(3, np.float32)))
    assert (r, term, trunc, info["success"], info["failure"]) == (1.0, True, False, True, None), info
    assert info["privileged"]["arm_arm_events"] == 0


def test_step_env_arch_braced():
    from tasks import brace_bandit as BB
    env = R.BrickStepEnv(images=False)
    env.reset(seed=1, options=dict(structure="arch", step=10))
    a = BB.pick_lp_shared(BB.contexts(env.bricks)[9], BB.EXE["E0"])         # the LP-shared brace of the RL plan for this critical step
    assert a is not None
    _, r, term, _, info = env.step(dict(brace=1, brace_pose=np.array(a, np.float32), press=np.zeros(3, np.float32)))
    print("arch 10 braced", a, "reward", r, {k: v for k, v in info.items() if k != "privileged"}, {k: info["privileged"][k] for k in ("u2_peak", "arm_arm_events")})
    assert term and info["brace"]["applied"] is not None and r == float(info["success"]) - 0.05 and "u2_peak" in info["privileged"]


def test_reach_scripted_success():
    env = R.BrickReachEnv(images=False)
    ret, ticks, info, phase = R.run_scripted(env)
    print("scripted reach: return %.2f, %d ticks, events %s" % (ret, ticks, info["events"]))
    assert info["events"] == ["success"] and ret > 50 and ticks < env.max_steps


def test_reach_random_truncates():
    env = R.BrickReachEnv(images=False, max_steps=40)
    env.reset(seed=5)
    for t in range(40):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if term or trunc:
            break
    assert trunc and not term and t == 39, info["events"]


def test_vec_autoreset_and_adapter():
    import torch
    env = V.NewtonVecEnv(R.BrickReachEnv, 2, images=False, per_env_kwargs=[dict(max_steps=3), dict(max_steps=50)])
    try:
        o, _ = env.reset()
        assert o["policy"].shape == (2, 390) and env.single_action_space.shape == (10,) and "images" not in o
        lens, done = [], []
        for _ in range(4):
            o, r, te, tr, ex = env.step(torch.zeros(2, 10))
            lens.append(env.episode_length_buf.tolist())
            done.append((te | tr).tolist())
        assert lens == [[1, 1], [2, 2], [0, 3], [1, 4]] and done[2] == [True, False]       # env 0 restarted at its 3rd step, env 1 went on
        assert len(ex["log"]["Episode/return"]) == 0 and torch.isfinite(o["policy"]).all()
        rs = V.RslRlVecEnv(env)
        g = rs.get_observations()
        assert list(g.keys()) == ["policy"] and g["policy"].shape == (2, 390) and rs.num_actions == 10
        _, rew, dones, extras = rs.step(torch.zeros(2, 10))
        assert dones.dtype == torch.long and extras["time_outs"].shape == (2,)
    finally:
        env.close()


def test_vec_images_group():
    import torch
    env = V.NewtonVecEnv(R.BrickReachEnv, 2, images=True, image_stride=4)
    try:
        o, _ = env.reset()
        im = o["images"]
        assert sorted(im) == sorted(c + s for c in R.CAM_NAMES for s in ("_rgb", "_depth"))
        assert all(im[c + "_rgb"].shape == (2, 240, 320, 3) and im[c + "_rgb"].dtype == torch.uint8 and im[c + "_depth"].shape == (2, 240, 320) for c in R.CAM_NAMES)
        g = V.groups(o)
        assert g["top_rgb"].shape == (2, 3, 240, 320) and g["wrist_B_depth"].shape == (2, 1, 240, 320)
    finally:
        env.close()


def test_vec_step_env_flat_action():
    import torch
    env = V.NewtonVecEnv(R.BrickStepEnv, 1, images=False, contexts=(("cube", 0),))
    try:
        assert env.single_action_space.shape == (7,)
        env.reset()
        o, r, te, tr, ex = env.step(torch.tensor([[-1.0, 0, 0, 0, -1, 0, 0]]))        # no brace, no press
        assert te.item() and r.item() == 1.0 and ex["log"]["Episode/success"].tolist() == [1.0] and o["policy"].shape == (1, 405)
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q", "-s"]))
