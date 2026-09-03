"""Gym-style single-agent interface over the factory world."""

import numpy as np

from .config import MACHINES, Config, Item, Kind
from .world import Action, World

_ORES = (Item.ORE_A, Item.ORE_B)


def _labels() -> list[str]:
    labels = ["credits"]
    labels += [f"stock_{item.value}" for item in Item]
    labels += [f"count_{kind.value}" for kind in MACHINES]
    labels += [f"condition_{kind.value}" for kind in MACHINES]
    labels += ["power_supply", "power_demand", "power_efficiency"]
    labels += [f"reserves_{ore.value}" for ore in _ORES]
    labels += [f"richness_{ore.value}" for ore in _ORES]
    labels += ["price_index", "progress"]
    return labels


OBSERVATION_LABELS = _labels()


class FactoryEnv:
    """Discrete-action environment wrapping :class:`World`.

    Follows the Gymnasium calling convention without depending on it::

        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)
    """

    observation_labels = OBSERVATION_LABELS
    n_actions = len(Action)
    n_observations = len(OBSERVATION_LABELS)

    def __init__(self, config: Config | None = None, seed: int | None = None):
        self.cfg = config or Config()
        self.world = World(self.cfg, seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.world.reset(seed)
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        world = self.world
        credits_before = world.credits
        report = world.step(action)

        if self.cfg.reward_mode == "widgets":
            reward = report.widgets_built
        else:
            reward = world.credits - credits_before

        terminated = world.credits < 0
        if terminated:
            reward -= self.cfg.bankruptcy_penalty
        truncated = world.tick >= self.cfg.max_ticks

        return self._observe(), float(reward), terminated, truncated, self._info()

    # ---------------------------------------------------------------- views

    def _observe(self) -> np.ndarray:
        world = self.world
        report = world.report
        values = [world.credits / 1000.0]
        values += [world.inventory[item] / 100.0 for item in Item]
        values += [world.count(kind) / 10.0 for kind in MACHINES]
        values += [world.mean_condition(kind) for kind in MACHINES]
        values += [
            report.power_supply / 100.0,
            report.power_demand / 100.0,
            report.power_efficiency,
        ]
        values += [world.reserves(ore) / 1000.0 for ore in _ORES]
        values += [world.best_richness(ore) for ore in _ORES]
        values += [world.price_index, world.tick / self.cfg.max_ticks]
        return np.array(values, dtype=np.float32)

    def _info(self) -> dict:
        world = self.world
        return {
            "tick": world.tick,
            "credits": world.credits,
            "widgets_built": world.widgets_built,
            "action_mask": world.legal_actions(),
            "events": list(world.report.events),
        }

    def render(self) -> str:
        world, report = self.world, self.world.report
        lines = [
            f"tick {world.tick:>4}/{self.cfg.max_ticks}   "
            f"credits {world.credits:>10,.1f}   "
            f"price index {world.price_index:.2f}   "
            f"widgets built {world.widgets_built:,.1f}",
            f"power {report.power_supply:>6.1f} supply / {report.power_demand:>6.1f} demand"
            f"   efficiency {report.power_efficiency:>5.0%}"
            f"   upkeep {report.upkeep:.1f}/tick",
            "stock  " + "  ".join(
                f"{item.value} {world.inventory[item]:>8.1f} @{world.price(item):>5.2f}"
                for item in Item
            ),
            "plant  " + "  ".join(
                f"{kind.value} {world.count(kind)}"
                f"({world.mean_condition(kind):.0%})"
                for kind in MACHINES
            ),
        ]
        for ore in _ORES:
            seams = [d for d in world.deposits if d.ore is ore and d.remaining > 0]
            detail = " ".join(f"[{d.remaining:.0f}u x{d.richness:.2f}]" for d in seams)
            lines.append(f"seams  {ore.value}: {detail or 'exhausted'}")
        for event in report.events:
            lines.append(f"  ! {event}")
        return "\n".join(lines)
