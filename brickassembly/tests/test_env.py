"""Gate G4's environment checks -- master_report §WP5 acceptance.

    cd brickassembly && ../mjenv/bin/python -m pytest tests/test_env.py -q

  * obs dim matches spec                              test_obs_dims
  * no NaNs over 10k random-action steps; reward      test_random_rollout
    bounded
  * all terminal conditions reachable                 test_terminals_reachable
  * "Newton and PhysX agree on 100 steps < 1e-3":     test_deterministic
    one backend here (MuJoCo), so the check is that
    two runs from one seed agree exactly
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tasks.insertion_env import ACT_DIM, OBS_DIM, PRIV_DIM, InsertionEnv

OUT = Path(__file__).with_suffix(".json")
RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def dump():
    yield
    OUT.write_text(json.dumps(RESULTS, indent=1, default=float))


def test_obs_dims():
    env = InsertionEnv(stage=0, seed=0)
    o = env.reset()
    assert o["actor"].shape == (OBS_DIM,) == (121,)
    assert o["critic"].shape == (OBS_DIM + PRIV_DIM,) == (147,)
    # R4: the vision slot is present and zero
    env = InsertionEnv(stage=0, seed=0, vision=False)
    o = env.reset()
    assert np.all(o["actor"][16 + 60 + 21:16 + 60 + 21 + 9] == 0)
    RESULTS["dims"] = {"actor": OBS_DIM, "critic": OBS_DIM + PRIV_DIM, "action": ACT_DIM}


def test_random_rollout():
    """10k random-action policy steps across stages: finite obs, bounded reward."""
    rng = np.random.default_rng(0)
    n, rmin, rmax, eps = 0, 0.0, 0.0, 0
    for stage in (0, 1, 2, 3):
        for mode, reward in (("residual", "sparse"), ("e2e", "dense")):
            env = InsertionEnv(stage=stage, mode=mode, reward=reward, seed=stage)
            o = env.reset()
            for _ in range(1250):
                o, r, done, info = env.step(rng.uniform(-1, 1, ACT_DIM))
                assert np.isfinite(o["actor"]).all() and np.isfinite(o["critic"]).all()
                assert np.isfinite(r) and -2.0 <= r <= 1.0, r
                rmin, rmax = min(rmin, r), max(rmax, r)
                n += 1
                if done:
                    eps += 1
                    o = env.reset()
    RESULTS["random_rollout"] = {"steps": n, "episodes": eps, "reward_range": [rmin, rmax]}
    assert n == 10000


def test_terminals_reachable():
    """success (scripted base, no DR), force violation (a tiny budget),
    joint break (a neighbour loaded far past f_break), timeout (hold still).
    'drop' does not exist here: the brick is a rigid child of the hand."""
    seen = {}
    env = InsertionEnv(stage=0, seed=1, dr=False)
    for _ in range(5):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(np.zeros(ACT_DIM))
        seen.setdefault(info["reason"], 0)
        seen[info["reason"]] += 1
        if "success" in seen:
            break
    assert "success" in seen
    env = InsertionEnv(stage=0, seed=1, dr=False, force_budget=20.0)
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(np.zeros(ACT_DIM))
    assert info["reason"] == "force_violation", info
    # e2e, holding still at the start pose: nothing ever happens
    env = InsertionEnv(stage=0, seed=1, dr=False, mode="e2e")
    env.reset()
    done = False
    a = np.zeros(ACT_DIM)
    a[6] = -1
    while not done:
        _, _, done, info = env.step(a)
    assert info["reason"] == "timeout"
    # a joint break: stage 3 lands among mated neighbours (the scripted base
    # under DR breaks one in ~1 of 5 episodes)
    env = InsertionEnv(stage=3, seed=11)
    broke = False
    for _ in range(40):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(np.zeros(ACT_DIM))
        if info["reason"] == "joint_break":
            broke = True
            break
    RESULTS["terminals"] = {"scripted_first": seen, "joint_break_reached": broke}
    assert broke


def test_deterministic():
    runs = []
    for _ in range(2):
        env = InsertionEnv(stage=1, seed=7)
        env.reset()
        rng = np.random.default_rng(3)
        traj = []
        for _ in range(100):
            o, r, done, _ = env.step(rng.uniform(-1, 1, ACT_DIM))
            traj.append(o["critic"])
            if done:
                env.reset()
        runs.append(np.array(traj))
    diff = float(np.abs(runs[0] - runs[1]).max())
    RESULTS["determinism_max_abs_diff"] = diff
    assert diff < 1e-9
