"""The RL plumbing: Gymnasium adapter, training round trip, evaluation."""

import numpy as np
import pytest

from rl_world import Action, Config, FactoryEnv
from rl_world.gym_env import FactoryGymEnv

TINY = dict(n_steps=64, batch_size=64, n_epochs=1, policy_kwargs=dict(net_arch=[16]))


def test_adapter_passes_the_sb3_env_checker():
    from stable_baselines3.common.env_checker import check_env

    check_env(FactoryGymEnv(), warn=True, skip_render_check=True)


def test_adapter_exposes_the_mask_maskable_ppo_looks_for():
    env = FactoryGymEnv(Config())
    env.reset(seed=0)
    mask = env.action_masks()

    assert mask.shape == (env.action_space.n,)
    assert mask.dtype == bool
    assert mask[Action.NOOP]

    env.factory.world.credits = 0.0
    assert not env.action_masks()[Action.BUILD_ASSEMBLER]


def test_adapter_reset_is_reproducible():
    first = FactoryGymEnv().reset(seed=7)[0]
    second = FactoryGymEnv().reset(seed=7)[0]
    assert np.array_equal(first, second)


def test_observation_is_public_and_matches_reset():
    env = FactoryEnv(Config(), seed=0)
    obs, _ = env.reset(0)
    assert np.array_equal(obs, env.observation())


@pytest.mark.slow
def test_training_round_trip_produces_a_usable_policy(tmp_path):
    from rl_world.train import TrainedPolicy, rollout, train

    out = tmp_path / "tiny"
    train(
        timesteps=256,
        n_envs=2,
        device="cpu",
        out=str(out),
        seed=0,
        config=Config(max_ticks=64),
        hyperparams=TINY,
    )
    assert out.with_suffix(".zip").exists()
    assert (tmp_path / "tiny_vecnormalize.pkl").exists()

    policy = TrainedPolicy(out, device="cpu")
    # The horizon is recorded because tick/max_ticks is part of the observation.
    assert policy.trained_on == {"max_ticks": 64, "reward_mode": "profit"}
    env = FactoryEnv(Config(max_ticks=64))
    env.reset(0)
    for _ in range(20):
        action = policy(env)
        assert env.world.legal_actions()[action], "masking should exclude illegal actions"
        env.step(action)

    scores = rollout(FactoryEnv(Config(max_ticks=64)), policy, seed=0)
    assert scores["ticks"] > 0 and set(scores) >= {"return", "widgets", "bankrupt"}
