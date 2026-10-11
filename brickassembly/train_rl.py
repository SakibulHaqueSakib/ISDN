"""PPO on the Newton cell through the Isaac Lab-style interface (rl_vec.py): rsl_rl by default, skrl with --skrl. Neither imports isaaclab or isaacsim.

    bash scripts/run.sh train_rl.py --iterations 3 --num-envs 2                  # rsl_rl OnPolicyRunner on BrickReachEnv, images off
    bash scripts/run.sh train_rl.py --skrl --timesteps 200 --num-envs 2          # skrl PPO
    bash scripts/run.sh train_rl.py --env step --num-envs 2 --iterations 2       # BrickStepEnv (one 15-30 s step per episode)

The configs are plain dataclasses in Isaac Lab's task-config style (EnvCfg for the task, RslRlCfg / SkrlCfg for the trainer); `to_dict()` gives the
dict the trainer wants. A smoke test, not a tuned training run: the defaults are tiny.
"""

import argparse
import dataclasses
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


@dataclass
class EnvCfg:
    env: str = "reach"                  # reach: BrickReachEnv, step: BrickStepEnv
    num_envs: int = 2
    images: bool = False
    structure: str = "cube"
    step: int = 0
    max_steps: int = 600                # BrickReachEnv ticks (10 Hz)
    seed: int = 0

    def make(self, device):
        import rl_env
        import rl_vec
        kw = dict(images=self.images)
        if self.env == "reach":
            cls, kw = rl_env.BrickReachEnv, dict(kw, structure=self.structure, step=self.step, max_steps=self.max_steps)
        else:
            cls = rl_env.BrickStepEnv
        return rl_vec.NewtonVecEnv(cls, self.num_envs, device=device, seed=self.seed, **kw)


@dataclass
class RslRlCfg:
    """rsl_rl 5.0.1 OnPolicyRunner config (the layout of isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg)."""
    seed: int = 0
    num_steps_per_env: int = 24
    max_iterations: int = 3
    save_interval: int = 50
    experiment_name: str = "newton_cell"
    clip_actions: float = 1.0
    check_for_nan: bool = True
    obs_groups: dict = field(default_factory=lambda: {"actor": ["policy"], "critic": ["policy"]})
    actor: dict = field(default_factory=lambda: dict(class_name="MLPModel", hidden_dims=[128, 128], activation="elu", obs_normalization=True,
                                                     distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.5, std_type="scalar")))
    critic: dict = field(default_factory=lambda: dict(class_name="MLPModel", hidden_dims=[128, 128], activation="elu", obs_normalization=True))
    algorithm: dict = field(default_factory=lambda: dict(class_name="PPO", num_learning_epochs=5, num_mini_batches=2, learning_rate=1e-3, schedule="adaptive",
                                                         gamma=0.99, lam=0.95, entropy_coef=0.005, desired_kl=0.01, max_grad_norm=1.0, value_loss_coef=1.0,
                                                         use_clipped_value_loss=True, clip_param=0.2, rnd_cfg=None, symmetry_cfg=None))

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class SkrlCfg:
    """skrl 2.1.0 Runner config (the layout of Isaac Lab's skrl_ppo_cfg.yaml)."""
    seed: int = 0
    timesteps: int = 200
    rollouts: int = 16
    layers: tuple = (128, 128)

    def to_dict(self):
        net = [dict(name="net", input="OBSERVATIONS", layers=list(self.layers), activations="elu")]
        return dict(
            seed=self.seed,
            models=dict(separate=False,
                        policy=dict({"class": "GaussianMixin"}, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0,
                                    initial_log_std=-0.7, network=net, output="ACTIONS"),
                        value=dict({"class": "DeterministicMixin"}, clip_actions=False, network=net, output="ONE")),
            memory={"class": "RandomMemory", "memory_size": -1},
            agent=dict({"class": "PPO"}, rollouts=self.rollouts, learning_epochs=4, mini_batches=2, discount_factor=0.99, gae_lambda=0.95,
                       learning_rate=1e-3, learning_rate_scheduler="KLAdaptiveLR", learning_rate_scheduler_kwargs=dict(kl_threshold=0.008),
                       observation_preprocessor="RunningStandardScaler", observation_preprocessor_kwargs=None, value_preprocessor="RunningStandardScaler",
                       value_preprocessor_kwargs=None, random_timesteps=0, learning_starts=0, grad_norm_clip=1.0, ratio_clip=0.2,
                       value_clip=0.2, entropy_loss_scale=0.0, value_loss_scale=2.0, kl_threshold=0.0,
                       rewards_shaper_scale=0.1, time_limit_bootstrap=False,
                       experiment=dict(directory=tempfile.mkdtemp(prefix="skrl_"), experiment_name="", write_interval=0, checkpoint_interval=0)),
            trainer=dict({"class": "SequentialTrainer"}, timesteps=self.timesteps, environment_info="log", close_environment_at_exit=False))


@dataclass
class RlGamesCfg:
    """rl_games 1.6.1 Runner params (the layout of Isaac Lab's rl_games_ppo_cfg.yaml)."""
    seed: int = 0
    max_epochs: int = 2
    horizon_length: int = 16
    num_envs: int = 2
    units: tuple = (128, 128)

    def to_dict(self):
        return dict(params=dict(
            seed=self.seed, env=dict(clip_observations=5.0, clip_actions=1.0), algo=dict(name="a2c_continuous"), model=dict(name="continuous_a2c_logstd"),
            network=dict(name="actor_critic", separate=False,
                         space=dict(continuous=dict(mu_activation="None", sigma_activation="None", mu_init=dict(name="default"),
                                                    sigma_init=dict(name="const_initializer", val=-0.7), fixed_sigma=True)),
                         mlp=dict(units=list(self.units), activation="elu", d2rl=False, initializer=dict(name="default"), regularizer=dict(name="None"))),
            load_checkpoint=False, load_path="",
            config=dict(name="newton_cell", train_dir=tempfile.mkdtemp(prefix="rl_games_"), env_name="newton", device="cuda:0", device_name="cuda:0", multi_gpu=False, ppo=True, mixed_precision=False,
                        normalize_input=True, normalize_value=True, num_actors=self.num_envs, reward_shaper=dict(scale_value=0.1), normalize_advantage=True,
                        gamma=0.99, tau=0.95, learning_rate=5e-4, lr_schedule="adaptive", kl_threshold=0.008, score_to_win=20000, max_epochs=self.max_epochs,
                        save_best_after=10 ** 6, save_frequency=10 ** 6, grad_norm=1.0, entropy_coef=0.0, truncate_grads=True, e_clip=0.2,
                        horizon_length=self.horizon_length, minibatch_size=self.horizon_length * self.num_envs // 2, mini_epochs=4, critic_coef=4,
                        clip_value=True, seq_length=4, bounds_loss_coef=1e-4)))


def train_rl_games(env, cfg):
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_vec import RlGamesVecEnv
    wrapped = RlGamesVecEnv(env)
    vecenv.register("NewtonVecEnv", lambda config_name, num_actors, **kw: wrapped)
    env_configurations.register("newton", {"vecenv_type": "NewtonVecEnv", "env_creator": lambda **kw: wrapped})
    runner = Runner()
    runner.load(cfg.to_dict())
    runner.reset()
    t = time.perf_counter()
    runner.run({"train": True, "play": False})
    dt = time.perf_counter() - t
    print("rl_games: %d epochs in %.1f s = %.1f steps/s" % (cfg.max_epochs, dt, cfg.max_epochs * cfg.horizon_length * env.num_envs / dt))


def train_rsl_rl(env, cfg, log_dir=None):
    from rsl_rl.runners import OnPolicyRunner
    from rl_vec import RslRlVecEnv
    venv = RslRlVecEnv(env, clip_actions=cfg.clip_actions)
    runner = OnPolicyRunner(venv, cfg.to_dict(), log_dir=log_dir or tempfile.mkdtemp(prefix="rsl_rl_"), device=str(env.device))
    t = time.perf_counter()
    runner.learn(num_learning_iterations=cfg.max_iterations)
    dt = time.perf_counter() - t
    steps = cfg.max_iterations * cfg.num_steps_per_env * env.num_envs
    print("rsl_rl: %d iterations, %d env steps in %.1f s = %.1f steps/s" % (cfg.max_iterations, steps, dt, steps / dt))


def train_skrl(env, cfg):
    from skrl.utils.runner.torch import Runner
    from rl_vec import skrl_env
    runner = Runner(skrl_env(env), cfg.to_dict())
    t = time.perf_counter()
    runner.run()
    dt = time.perf_counter() - t
    print("skrl: %d timesteps x %d envs in %.1f s = %.1f steps/s" % (cfg.timesteps, env.num_envs, dt, cfg.timesteps * env.num_envs / dt))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", choices=("reach", "step"), default="reach")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--iterations", type=int, default=3, help="rsl_rl iterations")
    ap.add_argument("--steps-per-env", type=int, default=24, help="rsl_rl rollout length")
    ap.add_argument("--skrl", action="store_true", help="train with skrl PPO instead of rsl_rl")
    ap.add_argument("--rl-games", action="store_true", help="train with rl_games PPO (a2c_continuous) instead of rsl_rl")
    ap.add_argument("--timesteps", type=int, default=200, help="skrl timesteps")
    ap.add_argument("--log-dir", help="rsl_rl log directory (tensorboard + the printed table); default a temp dir")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    env = EnvCfg(env=a.env, num_envs=a.num_envs, images=a.images).make(a.device)
    try:
        if a.rl_games:
            train_rl_games(env, RlGamesCfg(num_envs=a.num_envs))
        elif a.skrl:
            train_skrl(env, SkrlCfg(timesteps=a.timesteps))
        else:
            train_rsl_rl(env, RslRlCfg(max_iterations=a.iterations, num_steps_per_env=a.steps_per_env), a.log_dir)
    finally:
        env.close()
    forbidden = [m for m in sys.modules if m.split(".")[0] in ("isaaclab", "isaacsim", "omni")]
    print("isaaclab / isaacsim modules imported: %s" % (forbidden or "none"))


if __name__ == "__main__":
    main()
