"""Train and evaluate a MaskablePPO policy on the factory.

MaskablePPO (sb3-contrib) rather than plain PPO because the world already publishes a
legal-action mask: without it a policy burns most of its rollout on builds it cannot
afford and has to learn the budget constraint from scratch.
"""

import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import Config
from .env import FactoryEnv
from .gym_env import FactoryGymEnv

HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=2048,
    n_epochs=10,
    gamma=0.995,  # 500-tick episodes: a factory built now pays off much later
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def _paths(out: str | Path) -> tuple[Path, Path, Path]:
    out = Path(out)
    return (
        out.with_suffix(".zip"),
        out.with_name(out.stem + "_vecnormalize.pkl"),
        out.with_name(out.stem + "_config.json"),
    )


def make_vec_env(n_envs: int, config: Config, seed: int | None = None):
    """Envs run in-process: this simulator is microseconds per step, so the IPC of
    SubprocVecEnv costs more than the parallelism buys (measured ~7k vs ~12k fps)."""

    def factory():
        return Monitor(FactoryGymEnv(config))

    venv = DummyVecEnv([factory] * n_envs)
    if seed is not None:
        venv.seed(seed)
    return VecNormalize(
        venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=HYPERPARAMS["gamma"]
    )


def train(
    timesteps: int = 3_000_000,
    n_envs: int = 32,
    device: str = "cuda",
    out: str = "models/ppo_factory",
    seed: int | None = 0,
    config: Config | None = None,
    hyperparams: dict | None = None,
) -> Path:
    config = config or Config()
    venv = make_vec_env(n_envs, config, seed)
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        device=device,
        seed=seed,
        verbose=1,
        **(HYPERPARAMS | (hyperparams or {})),
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)

    model_path, norm_path, config_path = _paths(out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    venv.save(str(norm_path))
    # The observation carries tick/max_ticks, so a policy is tied to the horizon it
    # trained on; record it and let callers warn when they disagree.
    config_path.write_text(
        json.dumps({"max_ticks": config.max_ticks, "reward_mode": config.reward_mode})
    )
    venv.close()
    return model_path


class TrainedPolicy:
    """A saved policy, callable as `policy(env)` like the scripted baselines."""

    def __init__(self, out: str | Path, device: str = "cpu", deterministic: bool = True):
        model_path, norm_path, config_path = _paths(out)
        self.model = MaskablePPO.load(model_path, device=device)
        self.deterministic = deterministic
        self.trained_on = (
            json.loads(config_path.read_text()) if config_path.exists() else {}
        )
        self.normalizer = None
        if norm_path.exists():
            self.normalizer = VecNormalize.load(
                str(norm_path), DummyVecEnv([lambda: FactoryGymEnv()])
            )
            self.normalizer.training = False

    def __call__(self, env: FactoryEnv) -> int:
        obs = env.observation()
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(
            obs,
            action_masks=env.world.legal_actions(),
            deterministic=self.deterministic,
        )
        return int(action)


def rollout(env: FactoryEnv, policy, seed: int | None = None) -> dict:
    """One full episode, returning the numbers worth comparing between policies."""
    env.reset(seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(policy(env))
        total += reward
        if terminated or truncated:
            return {
                "return": total,
                "widgets": env.world.widgets_built,
                "credits": env.world.credits,
                "ticks": env.world.tick,
                "bankrupt": float(terminated),
            }


def evaluate(policy, episodes: int = 20, config: Config | None = None) -> dict:
    env = FactoryEnv(config or Config())
    runs = [rollout(env, policy, seed=1000 + i) for i in range(episodes)]
    return {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}
