"""Gymnasium adapter, so stable-baselines3 can train on the factory."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .env import FactoryEnv


class FactoryGymEnv(gym.Env):
    """`FactoryEnv` as a Gymnasium environment with action masking.

    `action_masks()` is the hook sb3-contrib's MaskablePPO looks for; it keeps the
    policy from wasting rollouts on builds it cannot afford.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: Config | None = None, render_mode: str | None = None):
        self.factory = FactoryEnv(config)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(FactoryEnv.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(FactoryEnv.n_observations,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return self.factory.reset(seed)

    def step(self, action):
        return self.factory.step(int(action))

    def action_masks(self) -> np.ndarray:
        return self.factory.world.legal_actions()

    def render(self) -> str:
        return self.factory.render()
