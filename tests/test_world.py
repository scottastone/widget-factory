import numpy as np
import pytest

from rl_world import Action, Config, Item, Kind, World
from rl_world.config import MACHINES
from rl_world.world import Machine


def stocked_world(**overrides) -> World:
    """No disasters and a frozen market, so arithmetic in the tests is exact."""
    cfg = Config(disaster_rate=0.0, price_volatility=0.0, price_reversion=0.0, **overrides)
    return World(cfg, seed=0)


def build(world: World, kind: Kind, n: int = 1) -> None:
    world.machines.extend(Machine(kind) for _ in range(n))


def test_same_seed_same_trajectory():
    def rollout(seed):
        world = World(Config(), seed)
        rng = np.random.default_rng(7)
        trace = []
        for _ in range(120):
            world.step(int(rng.integers(len(Action))))
            trace.append((round(world.credits, 6), dict(world.inventory)))
        return trace

    assert rollout(42) == rollout(42)
    assert rollout(42) != rollout(43)


def test_widget_recipe_consumes_two_a_and_one_b():
    world = stocked_world()
    build(world, Kind.ASSEMBLER)
    build(world, Kind.GENERATOR)
    world.inventory[Item.COMP_A] = 10.0
    world.inventory[Item.COMP_B] = 10.0

    world.step(Action.NOOP)

    made = world.inventory[Item.WIDGET]
    assert made == pytest.approx(MACHINES[Kind.ASSEMBLER].rate)
    assert world.inventory[Item.COMP_A] == pytest.approx(10.0 - 2 * made)
    assert world.inventory[Item.COMP_B] == pytest.approx(10.0 - made)


def test_assembler_is_limited_by_the_scarcer_component():
    world = stocked_world()
    build(world, Kind.ASSEMBLER, 4)
    build(world, Kind.GENERATOR, 2)
    world.inventory[Item.COMP_A] = 3.0
    world.inventory[Item.COMP_B] = 100.0

    world.step(Action.NOOP)

    assert world.inventory[Item.WIDGET] == pytest.approx(1.5)  # 3 comp_a / 2 per widget
    assert world.inventory[Item.COMP_A] == pytest.approx(0.0)


def test_chain_takes_one_tick_per_stage():
    world = stocked_world()
    for kind in (Kind.MINE_A, Kind.MINE_B, Kind.SMELTER_A, Kind.SMELTER_B, Kind.ASSEMBLER):
        build(world, kind, 2)
    build(world, Kind.GENERATOR, 3)

    world.step(Action.NOOP)
    assert world.inventory[Item.ORE_A] > 0
    assert world.inventory[Item.COMP_A] == 0  # ore only just arrived
    world.step(Action.NOOP)
    assert world.inventory[Item.COMP_A] > 0
    assert world.inventory[Item.WIDGET] == 0
    world.step(Action.NOOP)
    assert world.inventory[Item.WIDGET] > 0


def test_mining_depletes_deposits_without_going_negative():
    world = stocked_world()
    build(world, Kind.MINE_A, 5)
    build(world, Kind.GENERATOR, 2)
    for deposit in world.deposits:
        deposit.remaining = 3.0

    before = world.reserves(Item.ORE_A)
    for _ in range(20):
        world.step(Action.NOOP)

    assert world.reserves(Item.ORE_A) == 0
    assert all(d.remaining >= 0 for d in world.deposits)
    assert world.inventory[Item.ORE_A] == pytest.approx(before)


def test_brownout_scales_production_down():
    world = stocked_world()
    build(world, Kind.MINE_A, 10)  # 40 power demanded against one 30-power generator
    report = world.step(Action.NOOP)

    assert report.power_demand == 40
    assert report.power_efficiency == pytest.approx(0.75)
    assert report.mined[Item.ORE_A] < 10 * MACHINES[Kind.MINE_A].rate


def test_worn_out_machines_stop_running():
    world = stocked_world()
    build(world, Kind.MINE_A)
    build(world, Kind.GENERATOR)
    world.machines[-2].condition = 0.05

    report = world.step(Action.NOOP)

    assert report.power_demand == 0
    assert Item.ORE_A not in report.mined


def test_selling_empties_the_stockpile_at_market_price():
    world = stocked_world()
    world.inventory[Item.WIDGET] = 4.0
    world.price_index = 1.0
    before = world.credits

    world.step(Action.SELL_WIDGETS)

    assert world.inventory[Item.WIDGET] == 0
    assert world.credits == pytest.approx(before + 100.0 - world.report.upkeep)


def test_illegal_actions_do_nothing():
    world = stocked_world()
    world.credits = 0.0
    before = len(world.machines)

    assert not world.legal_actions()[Action.BUILD_ASSEMBLER]
    world.step(Action.BUILD_ASSEMBLER)

    assert len(world.machines) == before
    assert world.report.spent == 0


def test_repair_restores_condition_and_costs_more_when_damaged():
    world = stocked_world()
    build(world, Kind.ASSEMBLER)
    world.machines[-1].condition = 0.25
    world.credits = 500.0

    expected = MACHINES[Kind.ASSEMBLER].cost * world.cfg.repair_cost_fraction * 0.75
    assert world.repair_cost(Kind.ASSEMBLER) == pytest.approx(expected)

    world.step(Action.REPAIR_ASSEMBLER)
    assert world.machines[-1].condition == 1.0
    assert world.report.spent == pytest.approx(expected)


@pytest.mark.parametrize(
    "name", ["cave_in", "breakdown", "power_surge", "market_crash", "quake"]
)
def test_each_disaster_hurts(name):
    cfg = Config(disaster_rate=1.0, disaster_weights={name: 1.0})
    world = World(cfg, seed=3)
    build(world, Kind.ASSEMBLER, 3)
    build(world, Kind.GENERATOR, 3)

    before = (
        world.reserves(Item.ORE_A) + world.reserves(Item.ORE_B),
        sum(m.condition for m in world.machines),
        len(world.machines),
        world.price_index,
    )
    world.step(Action.NOOP)
    after = (
        world.reserves(Item.ORE_A) + world.reserves(Item.ORE_B),
        sum(m.condition for m in world.machines),
        len(world.machines),
        world.price_index,
    )

    assert world.report.events, "disaster should be reported"
    assert any(a < b for a, b in zip(after, before)), "disaster should cost something"


def test_prospecting_can_open_a_new_seam():
    world = World(Config(disaster_rate=0.0, prospect_success=1.0), seed=1)
    world.credits = 1000.0
    before = world.reserves(Item.ORE_A)

    world.step(Action.PROSPECT_A)

    assert world.reserves(Item.ORE_A) > before
    assert world.credits < 1000.0


def test_upkeep_is_charged_every_tick():
    world = stocked_world()
    build(world, Kind.GENERATOR, 4)
    world.credits = 100.0

    world.step(Action.NOOP)

    assert world.credits == pytest.approx(100.0 - 5 * MACHINES[Kind.GENERATOR].upkeep)
