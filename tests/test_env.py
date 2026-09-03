import numpy as np

from rl_world import Action, Config, FactoryEnv, Item, Kind, World, heuristic_policy, random_policy
from rl_world.world import Machine
from rl_world.env import OBSERVATION_LABELS


def test_observation_matches_its_labels():
    env = FactoryEnv(seed=0)
    obs, info = env.reset(0)

    assert obs.shape == (len(OBSERVATION_LABELS),)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert info["action_mask"].shape == (env.n_actions,)


def test_step_returns_the_gym_five_tuple_and_truncates():
    env = FactoryEnv(Config(max_ticks=5), seed=0)
    env.reset(0)
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(Action.NOOP)
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(Action.NOOP)
    assert truncated and isinstance(reward, float)


def test_bankruptcy_terminates_with_a_penalty():
    env = FactoryEnv(Config(disaster_rate=0.0), seed=0)
    env.reset(0)
    env.world.credits = 0.1  # upkeep alone will sink it

    obs, reward, terminated, truncated, info = env.step(Action.NOOP)

    assert terminated
    assert reward < -env.cfg.bankruptcy_penalty


def test_profit_reward_is_the_change_in_credits():
    env = FactoryEnv(Config(disaster_rate=0.0), seed=0)
    env.reset(0)
    env.world.inventory[Item.WIDGET] = 2.0
    before = env.world.credits

    obs, reward, *_ = env.step(Action.SELL_WIDGETS)

    assert reward == (env.world.credits - before)


def test_widgets_reward_mode_counts_production():
    env = FactoryEnv(Config(reward_mode="widgets", disaster_rate=0.0), seed=0)
    env.reset(0)
    env.world.machines.append(Machine(Kind.ASSEMBLER))
    env.world.inventory[Item.COMP_A] = 20.0
    env.world.inventory[Item.COMP_B] = 20.0

    obs, reward, *_ = env.step(Action.NOOP)

    assert reward == env.world.inventory[Item.WIDGET] > 0


def test_render_is_printable():
    env = FactoryEnv(seed=0)
    env.reset(0)
    env.step(Action.BUILD_MINE_A)
    assert "credits" in env.render()


def _run(policy, seed, ticks=400):
    env = FactoryEnv(Config(max_ticks=ticks), seed)
    env.reset(seed)
    total = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(policy(env.world))
        total += reward
        if terminated or truncated:
            return total


def test_heuristic_beats_random():
    rng = np.random.default_rng(0)
    random_return = np.mean([_run(lambda w: random_policy(w, rng), s) for s in range(3)])
    heuristic_return = np.mean([_run(heuristic_policy, s) for s in range(3)])
    assert heuristic_return > random_return


def test_random_policy_only_picks_legal_actions():
    world = World(Config(), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        action = random_policy(world, rng)
        assert world.legal_actions()[action]
        world.step(action)
        if world.credits < 0:
            break
