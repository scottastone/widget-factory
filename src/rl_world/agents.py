"""Baseline policies for the factory world."""

import math

import numpy as np

from .config import MACHINES, Item, Kind
from .world import Action, World

# Machines needed per assembler to keep the chain fed.
RATIO: dict[Kind, float] = {
    Kind.MINE_A: 1.0,
    Kind.MINE_B: 0.5,
    Kind.SMELTER_A: 4.0 / 3.0,
    Kind.SMELTER_B: 2.0 / 3.0,
}
BUILD_FOR: dict[Kind, Action] = {
    Kind.MINE_A: Action.BUILD_MINE_A,
    Kind.MINE_B: Action.BUILD_MINE_B,
    Kind.SMELTER_A: Action.BUILD_SMELTER_A,
    Kind.SMELTER_B: Action.BUILD_SMELTER_B,
}
REPAIR_FOR: dict[Kind, Action] = {
    Kind.MINE_A: Action.REPAIR_MINE_A,
    Kind.MINE_B: Action.REPAIR_MINE_B,
    Kind.SMELTER_A: Action.REPAIR_SMELTER_A,
    Kind.SMELTER_B: Action.REPAIR_SMELTER_B,
    Kind.ASSEMBLER: Action.REPAIR_ASSEMBLER,
    Kind.GENERATOR: Action.REPAIR_GENERATOR,
}


def random_policy(world: World, rng: np.random.Generator) -> Action:
    """Uniform over the currently legal actions."""
    legal = np.flatnonzero(world.legal_actions())
    return Action(int(rng.choice(legal)))


SELL_FOR: dict[Item, Action] = {
    Item.ORE_A: Action.SELL_ORE_A,
    Item.ORE_B: Action.SELL_ORE_B,
    Item.COMP_A: Action.SELL_COMP_A,
    Item.COMP_B: Action.SELL_COMP_B,
    Item.WIDGET: Action.SELL_WIDGETS,
}


def _best_sale(world: World) -> Action | None:
    """The stockpile worth the most credits right now, if any."""
    values = {item: world.inventory[item] * world.price(item) for item in SELL_FOR}
    item = max(values, key=lambda i: values[i])
    return SELL_FOR[item] if values[item] > 0 else None


def heuristic_policy(world: World) -> Action:
    """A hand-written baseline: keep the plant repaired, powered and balanced."""
    legal = world.legal_actions()
    reserve = 60.0  # credits kept back so upkeep never bankrupts us

    def affordable(cost: float) -> bool:
        return world.credits - cost >= reserve

    # Cash first: an idle plant with an empty till cannot recover from a disaster.
    if world.credits < reserve * 2.5:
        sale = _best_sale(world)
        if sale is not None:
            return sale

    if world.inventory[Item.WIDGET] >= 5:
        return Action.SELL_WIDGETS

    # Fix the worst machine first; a dead generator stops everything.
    for kind, action in REPAIR_FOR.items():
        if legal[action] and world.mean_condition(kind) < 0.6:
            if affordable(world.repair_cost(kind)):
                return action

    if world.report.power_demand > world.report.power_supply and affordable(
        MACHINES[Kind.GENERATOR].cost
    ):
        return Action.BUILD_GENERATOR

    assemblers = max(world.count(Kind.ASSEMBLER), 1)
    deficits = {
        kind: math.ceil(assemblers * ratio) - world.count(kind)
        for kind, ratio in RATIO.items()
    }
    kind = max(deficits, key=lambda k: deficits[k])
    if deficits[kind] > 0 and affordable(MACHINES[kind].cost):
        return BUILD_FOR[kind]

    for ore, action in ((Item.ORE_A, Action.PROSPECT_A), (Item.ORE_B, Action.PROSPECT_B)):
        if world.reserves(ore) < 250 and affordable(world.cfg.prospect_cost):
            return action

    # Without an assembler nothing sells for real money, so buy the first one
    # as soon as it is affordable and only expand out of surplus after that.
    headroom = 1 if world.count(Kind.ASSEMBLER) == 0 else 2
    if affordable(MACHINES[Kind.ASSEMBLER].cost * headroom):
        return Action.BUILD_ASSEMBLER

    # Nothing to build: liquidate whatever the chain cannot consume.
    for item, action in (
        (Item.COMP_A, Action.SELL_COMP_A),
        (Item.COMP_B, Action.SELL_COMP_B),
        (Item.ORE_A, Action.SELL_ORE_A),
        (Item.ORE_B, Action.SELL_ORE_B),
    ):
        if world.inventory[item] > 200:
            return action
    return Action.NOOP
