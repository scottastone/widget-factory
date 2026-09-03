"""Tunable parameters for the factory world."""

from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    """Machine kinds."""

    MINE_A = "mine_a"
    MINE_B = "mine_b"
    SMELTER_A = "smelter_a"
    SMELTER_B = "smelter_b"
    ASSEMBLER = "assembler"
    GENERATOR = "generator"


class Item(str, Enum):
    """Tradeable inventory items."""

    ORE_A = "ore_a"
    ORE_B = "ore_b"
    COMP_A = "comp_a"
    COMP_B = "comp_b"
    WIDGET = "widget"


@dataclass(frozen=True)
class MachineSpec:
    cost: float  # credits to build
    power: float  # power drawn per tick while running
    upkeep: float  # credits per tick, paid whether running or not
    rate: float  # units produced per tick (power capacity for generators)
    wear: float  # condition lost per tick while running


MACHINES: dict[Kind, MachineSpec] = {
    Kind.MINE_A: MachineSpec(cost=40, power=4, upkeep=0.30, rate=2.0, wear=0.004),
    Kind.MINE_B: MachineSpec(cost=40, power=4, upkeep=0.30, rate=2.0, wear=0.004),
    Kind.SMELTER_A: MachineSpec(cost=60, power=6, upkeep=0.40, rate=1.5, wear=0.005),
    Kind.SMELTER_B: MachineSpec(cost=60, power=6, upkeep=0.40, rate=1.5, wear=0.005),
    Kind.ASSEMBLER: MachineSpec(cost=80, power=8, upkeep=0.50, rate=1.0, wear=0.006),
    Kind.GENERATOR: MachineSpec(cost=100, power=0, upkeep=1.20, rate=30.0, wear=0.003),
}

# Ore -> component smelting is 1:1; a widget needs RECIPE components.
RECIPE: dict[Item, float] = {Item.COMP_A: 2.0, Item.COMP_B: 1.0}

BASE_PRICES: dict[Item, float] = {
    Item.ORE_A: 1.0,
    Item.ORE_B: 1.0,
    Item.COMP_A: 3.0,
    Item.COMP_B: 3.0,
    Item.WIDGET: 25.0,
}


@dataclass
class Config:
    # episode
    max_ticks: int = 500
    reward_mode: str = "profit"  # "profit" (delta credits) or "widgets" (units built)
    bankruptcy_penalty: float = 100.0

    # starting position
    start_credits: float = 400.0
    start_generators: int = 1

    # machines
    min_condition: float = 0.10  # below this a machine cannot run
    repair_cost_fraction: float = 0.5  # of build cost, scaled by damage

    # deposits
    start_deposits: int = 2  # per ore type
    deposit_size: tuple[float, float] = (600.0, 1400.0)
    deposit_richness: tuple[float, float] = (0.6, 1.4)
    prospect_cost: float = 60.0
    prospect_success: float = 0.6  # chance a survey finds anything

    # market: price index is mean-reverting around 1.0
    price_reversion: float = 0.05
    price_volatility: float = 0.03
    price_bounds: tuple[float, float] = (0.4, 1.8)

    # disasters
    disaster_rate: float = 0.05  # chance per tick that something goes wrong
    disaster_weights: dict[str, float] = field(
        default_factory=lambda: {
            "cave_in": 1.0,  # a deposit partially collapses
            "breakdown": 1.5,  # one machine is wrecked
            "power_surge": 0.7,  # a generator is destroyed outright
            "market_crash": 0.8,  # price index plunges
            "quake": 0.5,  # every machine takes damage
        }
    )
    cave_in_loss: tuple[float, float] = (0.2, 0.6)  # fraction of deposit lost
    quake_damage: tuple[float, float] = (0.1, 0.35)  # condition lost by all machines
    crash_factor: tuple[float, float] = (0.4, 0.7)  # price index multiplier
