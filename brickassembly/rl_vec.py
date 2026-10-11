"""Isaac Lab-style vectorized interface to rl_env.py's environments, for rsl_rl, skrl and rl_games. The simulator stays this repo's Newton cell;
only the interface follows isaaclab.envs.DirectRLEnv. Nothing here imports isaaclab or isaacsim.

    NewtonVecEnv(env_cls, num_envs, device="cuda:0", **env_kwargs)     torch tensors in and out, auto-reset, one Newton model per subprocess
    RslRlVecEnv(env)        rsl_rl 5.0.1 VecEnv adapter (isaaclab_rl's RslRlVecEnvWrapper isinstance-checks DirectRLEnv / ManagerBasedRLEnv, so it cannot wrap this)
    skrl_env(env)           skrl.envs.wrappers.torch.wrap_env(env, wrapper="isaaclab"): works on NewtonVecEnv as it is
    RlGamesVecEnv(env)      rl_games IVecEnv (see the class)

NewtonVecEnv mirrors DirectRLEnv's public surface: num_envs, device, cfg (NewtonVecEnvCfg: is_finite_horizon, decimation, step_dt, seed, to_dict()),
max_episode_length, episode_length_buf (long, (num_envs,)), single_observation_space / observation_space (batched), single_action_space / action_space,
unwrapped, reset(seed=None, options=None) -> (obs, extras), step(actions) -> (obs, reward, terminated, truncated, extras), close(), seed().

Observation groups (Isaac Lab convention): obs["policy"] float32 (num_envs, D), the flat concatenation of proprio, bricks, plate, task [, features
(BrickStepEnv)]; obs["images"] (only with images=True) a dict {<cam>_rgb: uint8 (N, H, W, 3), <cam>_depth: float32 (N, H, W)}. The vectors are
unnormalised (use obs_normalization in the trainer's model config).

Actions: a flat Box in [-1, 1], (num_envs, A). BrickReachEnv: its own Box(10), (dx, dy, dz, dyaw, grip) of arm A then B. BrickStepEnv (A = 7):
    a[0]      brace on iff a[0] > 0 (the continuous value thresholded at 0)
    a[1:4]    (y, z, lean_deg) of the brace: [-1, 1] mapped linearly to BrickStepEnv.BRACE_BOUNDS (moved to the nearest candidate by the env)
    a[4]      press scale = 1.3 * max(a[4], 0) (a[4] <= 0: no fixture), a[5:7] the lateral force (fx, fy) = 3 N * a[5:7]
Out-of-range actions are clipped to [-1, 1].

Auto-reset (Isaac Lab semantics): a sub-env whose episode ends (terminated or truncated) is reset inside the same step() and the returned observation is
the new episode's first one; the others continue. reward / terminated / truncated are those of the step that ended the episode. extras["log"] /
extras["episode"]: Episode/return, Episode/length, Episode/success of the episodes that ended this step (empty tensors when none); extras["time_outs"]
= truncated is added by the rsl_rl adapter. A reset costs the env 2-5 s, so a step in which one sub-env resets takes that long for all (synchronous stepping).
"""

import dataclasses
import functools
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


# --- worker side: one env, flat obs and actions ----------------------------------------------------------------------
class FlatObs(gym.Wrapper):
    """Dict obs -> {"policy": flat float32, "images": {...}}; info -> the scalar numbers a vector env can batch."""

    def __init__(self, env):
        super().__init__(env)
        sp = env.observation_space.spaces
        self.keys = [k for k in ("proprio", "bricks", "plate", "task", "features") if k in sp]
        dim = sum(int(np.prod(sp[k].shape)) for k in self.keys)
        d = {"policy": spaces.Box(-np.inf, np.inf, (dim,), np.float32)}
        if "images" in sp:
            d["images"] = sp["images"]
        self.observation_space = spaces.Dict(d)

    def _obs(self, o):
        out = {"policy": np.concatenate([np.asarray(o[k], np.float32).ravel() for k in self.keys])}
        if "images" in o:
            out["images"] = o["images"]
        return out

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        return self._obs(o), {}

    def step(self, a):
        o, r, te, tr, info = self.env.step(a)
        return self._obs(o), r, te, tr, {"success": float(info.get("success", False))}


class FlatAction(gym.ActionWrapper):
    """BrickStepEnv's Dict action as a Box(-1, 1, (7,)) (the mapping is in the module docstring); other envs unchanged."""

    def __init__(self, env):
        super().__init__(env)
        self.dict = isinstance(env.action_space, spaces.Dict)
        if self.dict:
            self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)

    def action(self, a):
        if not self.dict:
            return a
        a = np.clip(np.asarray(a, np.float64), -1, 1)
        u = self.env.unwrapped
        lo, hi = np.array(u.BRACE_BOUNDS[0]), np.array(u.BRACE_BOUNDS[1])
        return {"brace": int(a[0] > 0), "brace_pose": (lo + (a[1:4] + 1) / 2 * (hi - lo)).astype(np.float32),
                "press": np.array([1.3 * max(a[4], 0.0), 3.0 * a[5], 3.0 * a[6]], np.float32)}


def make_env(env_cls, kwargs):
    return FlatAction(FlatObs(env_cls(**kwargs)))


# --- the vectorized env ------------------------------------------------------------------------------------------------
@dataclass
class NewtonVecEnvCfg:
    """What the Isaac Lab wrappers read from env.cfg."""
    env_name: str = ""
    is_finite_horizon: bool = False       # False: truncation is a time-out, bootstrapped (rsl_rl "time_outs")
    decimation: int = 6                   # physics frames per policy step
    step_dt: float = 0.1
    episode_length: int = 1
    seed: int = 0
    env_kwargs: dict = field(default_factory=dict)

    def to_dict(self):
        return dataclasses.asdict(self)


class NewtonVecEnv:
    def __init__(self, env_cls, num_envs, device="cuda:0", seed=0, per_env_kwargs=None, context="spawn", **env_kwargs):
        """`per_env_kwargs`: optional list of dicts merged over env_kwargs for each sub-env. Sub-envs are spawned subprocesses (CUDA and
        the Newton model must not be forked)."""
        self.num_envs, self.device = num_envs, torch.device(device)
        kws = [{**env_kwargs, **(per_env_kwargs[i] if per_env_kwargs else {})} for i in range(num_envs)]
        self.venv = gym.vector.AsyncVectorEnv([functools.partial(make_env, env_cls, k) for k in kws], context=context,
                                              autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.single_observation_space = self.venv.single_observation_space
        self.observation_space = self.venv.observation_space
        self.single_action_space = self.venv.single_action_space
        self.action_space = self.venv.action_space
        probe = env_cls(**kws[0])
        self.max_episode_length = getattr(probe, "max_steps", 1)
        self.cfg = NewtonVecEnvCfg(env_name=env_cls.__name__, decimation=getattr(probe, "every", 1), step_dt=getattr(probe, "every", 1) / 60,
                                   episode_length=self.max_episode_length, seed=seed, env_kwargs=kws[0])
        probe.close()
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._ret = torch.zeros(num_envs, device=self.device)
        self._seed, self._obs = seed, None

    @property
    def unwrapped(self):
        return self

    def seed(self, seed=-1):
        self._seed = seed
        return seed

    def _t(self, x):
        return {k: self._t(v) for k, v in x.items()} if isinstance(x, dict) else torch.as_tensor(x, device=self.device)

    def reset(self, seed=None, options=None):
        o, _ = self.venv.reset(seed=self._seed if seed is None else seed, options=options)
        self._obs = self._t(o)
        self.episode_length_buf.zero_()
        self._ret.zero_()
        return self._obs, {"log": {}}

    def _get_observations(self):
        return self._obs

    def step(self, actions):
        o, r, te, tr, info = self.venv.step(actions.detach().cpu().numpy())
        self._obs = self._t(o)
        r, te, tr = (torch.as_tensor(x, device=self.device) for x in (r, te, tr))
        r = r.float()
        self.episode_length_buf += 1
        self._ret += r
        done = te | tr
        succ = torch.zeros(self.num_envs, device=self.device)
        if done.any():                                # the sub-env's info of the step that ended its episode
            succ = torch.as_tensor(np.asarray(info.get("final_info", {}).get("success", succ.cpu()), np.float32), device=self.device)
        ep = {"Episode/return": self._ret[done].clone(), "Episode/length": self.episode_length_buf[done].float(), "Episode/success": succ[done]}
        self._ret[done] = 0
        self.episode_length_buf[done] = 0
        return self._obs, r, te, tr, {"log": ep, "episode": ep}

    def close(self):
        self.venv.close()


# --- trainer adapters --------------------------------------------------------------------------------------------------
def groups(obs):
    """Obs groups for a trainer that wants top-level tensors: "policy", and the images as one group per camera stream, channel first
    (rgb float in 0-1, (N, 3, H, W); depth (N, 1, H, W))."""
    out = {"policy": obs["policy"]}
    for k, v in obs.get("images", {}).items():
        out[k] = (v.permute(0, 3, 1, 2).float() / 255) if k.endswith("rgb") else v.unsqueeze(1)
    return out


try:
    from rsl_rl.env import VecEnv as _VecEnv
except ImportError:                       # rsl_rl not installed: the adapter is unavailable
    _VecEnv = object


class RslRlVecEnv(_VecEnv):
    """rsl_rl 5.0.1 VecEnv over a NewtonVecEnv: what isaaclab_rl.rsl_rl.RslRlVecEnvWrapper does (same extras, same reset-at-construction),
    without its isinstance check on Isaac Lab classes."""

    def __init__(self, env, clip_actions=None):
        self.env, self.clip_actions = env, clip_actions
        self.num_envs, self.device, self.max_episode_length, self.cfg = env.num_envs, env.device, env.max_episode_length, env.cfg
        self.num_actions = gym.spaces.flatdim(env.single_action_space)
        env.reset()

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, v):
        self.env.episode_length_buf = v

    def get_observations(self):
        from tensordict import TensorDict
        return TensorDict(groups(self.env._get_observations()), batch_size=[self.num_envs])

    def reset(self):
        from tensordict import TensorDict
        o, extras = self.env.reset()
        return TensorDict(groups(o), batch_size=[self.num_envs]), extras

    def step(self, actions):
        from tensordict import TensorDict
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        o, rew, te, tr, extras = self.env.step(actions)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = tr
        return TensorDict(groups(o), batch_size=[self.num_envs]), rew, (te | tr).to(torch.long), extras

    def close(self):
        self.env.close()


try:
    from rl_games.common.ivecenv import IVecEnv as _IVecEnv
except ImportError:                       # rl_games not installed: the adapter is unavailable
    _IVecEnv = object


class RlGamesVecEnv(_IVecEnv):
    """rl_games 1.6.1 IVecEnv over a NewtonVecEnv: reset() and step() return {"obs": the "policy" tensor} (rl_games' network input), get_env_info() the
    flat spaces. isaaclab_rl.rl_games.RlGamesVecEnvWrapper isinstance-checks Isaac Lab classes, so this is its small equivalent (images not passed)."""

    def __init__(self, env, clip_actions=1.0):
        self.env, self.clip = env, clip_actions
        self.cfg = env.cfg

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        import gym                        # rl_games checks its spaces with OpenAI gym's classes, not gymnasium's
        o, a = self.env.single_observation_space["policy"], self.env.single_action_space
        return {"observation_space": gym.spaces.Box(-5.0, 5.0, o.shape), "action_space": gym.spaces.Box(-self.clip, self.clip, a.shape), "state_space": None}

    def reset(self):
        return {"obs": self.env.reset()[0]["policy"]}

    def step(self, actions):
        o, rew, te, tr, extras = self.env.step(torch.clamp(actions.detach(), -self.clip, self.clip))
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = tr
        return {"obs": o["policy"]}, rew, te | tr, extras

    def close(self):
        self.env.close()


def skrl_env(env):
    """skrl's own Isaac Lab wrapper: it reads single_observation_space["policy"], single_action_space, num_envs, device, reset(seed=), step()."""
    from skrl.envs.wrappers.torch import wrap_env
    return wrap_env(env, wrapper="isaaclab")
