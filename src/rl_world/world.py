"""The simulated economy: mines, smelters, assemblers, power, market, disasters."""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import BASE_PRICES, MACHINES, RECIPE, Config, Item, Kind

MINE_ORE: dict[Kind, Item] = {Kind.MINE_A: Item.ORE_A, Kind.MINE_B: Item.ORE_B}
SMELTER_IO: dict[Kind, tuple[Item, Item]] = {
    Kind.SMELTER_A: (Item.ORE_A, Item.COMP_A),
    Kind.SMELTER_B: (Item.ORE_B, Item.COMP_B),
}


class Action(IntEnum):
    NOOP = 0
    BUILD_MINE_A = 1
    BUILD_MINE_B = 2
    BUILD_SMELTER_A = 3
    BUILD_SMELTER_B = 4
    BUILD_ASSEMBLER = 5
    BUILD_GENERATOR = 6
    REPAIR_MINE_A = 7
    REPAIR_MINE_B = 8
    REPAIR_SMELTER_A = 9
    REPAIR_SMELTER_B = 10
    REPAIR_ASSEMBLER = 11
    REPAIR_GENERATOR = 12
    PROSPECT_A = 13
    PROSPECT_B = 14
    SELL_ORE_A = 15
    SELL_ORE_B = 16
    SELL_COMP_A = 17
    SELL_COMP_B = 18
    SELL_WIDGETS = 19


BUILD_ACTIONS: dict[Action, Kind] = {
    Action.BUILD_MINE_A: Kind.MINE_A,
    Action.BUILD_MINE_B: Kind.MINE_B,
    Action.BUILD_SMELTER_A: Kind.SMELTER_A,
    Action.BUILD_SMELTER_B: Kind.SMELTER_B,
    Action.BUILD_ASSEMBLER: Kind.ASSEMBLER,
    Action.BUILD_GENERATOR: Kind.GENERATOR,
}
REPAIR_ACTIONS: dict[Action, Kind] = {
    Action.REPAIR_MINE_A: Kind.MINE_A,
    Action.REPAIR_MINE_B: Kind.MINE_B,
    Action.REPAIR_SMELTER_A: Kind.SMELTER_A,
    Action.REPAIR_SMELTER_B: Kind.SMELTER_B,
    Action.REPAIR_ASSEMBLER: Kind.ASSEMBLER,
    Action.REPAIR_GENERATOR: Kind.GENERATOR,
}
PROSPECT_ACTIONS: dict[Action, Item] = {
    Action.PROSPECT_A: Item.ORE_A,
    Action.PROSPECT_B: Item.ORE_B,
}
SELL_ACTIONS: dict[Action, Item] = {
    Action.SELL_ORE_A: Item.ORE_A,
    Action.SELL_ORE_B: Item.ORE_B,
    Action.SELL_COMP_A: Item.COMP_A,
    Action.SELL_COMP_B: Item.COMP_B,
    Action.SELL_WIDGETS: Item.WIDGET,
}


@dataclass
class Machine:
    kind: Kind
    condition: float = 1.0

    @property
    def spec(self):
        return MACHINES[self.kind]


@dataclass
class Deposit:
    ore: Item
    remaining: float
    richness: float


@dataclass
class TickReport:
    """What happened during one tick, for reward, logging and rendering."""

    mined: dict[Item, float] = field(default_factory=dict)
    smelted: dict[Item, float] = field(default_factory=dict)
    widgets_built: float = 0.0
    revenue: float = 0.0
    spent: float = 0.0
    upkeep: float = 0.0
    power_supply: float = 0.0
    power_demand: float = 0.0
    power_efficiency: float = 1.0
    events: list[str] = field(default_factory=list)


class World:
    """A tiny Factorio-shaped economy. One `step` is one tick."""

    def __init__(self, config: Config | None = None, seed: int | None = None):
        self.cfg = config or Config()
        self.reset(seed)

    # ---------------------------------------------------------------- setup

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.credits = self.cfg.start_credits
        self.inventory: dict[Item, float] = {item: 0.0 for item in Item}
        self.machines: list[Machine] = [
            Machine(Kind.GENERATOR) for _ in range(self.cfg.start_generators)
        ]
        self.deposits: list[Deposit] = [
            self._new_deposit(ore)
            for ore in (Item.ORE_A, Item.ORE_B)
            for _ in range(self.cfg.start_deposits)
        ]
        self.price_index = 1.0
        self.widgets_built = 0.0
        self.report = TickReport()

    def _new_deposit(self, ore: Item) -> Deposit:
        return Deposit(
            ore=ore,
            remaining=float(self.rng.uniform(*self.cfg.deposit_size)),
            richness=float(self.rng.uniform(*self.cfg.deposit_richness)),
        )

    # -------------------------------------------------------------- queries

    def price(self, item: Item) -> float:
        return BASE_PRICES[item] * self.price_index

    def count(self, kind: Kind) -> int:
        return sum(1 for m in self.machines if m.kind is kind)

    def mean_condition(self, kind: Kind) -> float:
        conditions = [m.condition for m in self.machines if m.kind is kind]
        return float(np.mean(conditions)) if conditions else 0.0

    def reserves(self, ore: Item) -> float:
        return sum(d.remaining for d in self.deposits if d.ore is ore)

    def best_richness(self, ore: Item) -> float:
        live = [d.richness for d in self.deposits if d.ore is ore and d.remaining > 0]
        return max(live) if live else 0.0

    def repair_cost(self, kind: Kind) -> float:
        machine = self._most_damaged(kind)
        if machine is None:
            return 0.0
        damage = 1.0 - machine.condition
        return MACHINES[kind].cost * self.cfg.repair_cost_fraction * damage

    def _most_damaged(self, kind: Kind) -> Machine | None:
        candidates = [m for m in self.machines if m.kind is kind]
        return min(candidates, key=lambda m: m.condition, default=None)

    def legal_actions(self) -> np.ndarray:
        """Boolean mask over Action; illegal actions are no-ops if taken anyway."""
        mask = np.zeros(len(Action), dtype=bool)
        mask[Action.NOOP] = True
        for action, kind in BUILD_ACTIONS.items():
            mask[action] = self.credits >= MACHINES[kind].cost
        for action, kind in REPAIR_ACTIONS.items():
            machine = self._most_damaged(kind)
            mask[action] = (
                machine is not None
                and machine.condition < 1.0
                and self.credits >= self.repair_cost(kind)
            )
        for action in PROSPECT_ACTIONS:
            mask[action] = self.credits >= self.cfg.prospect_cost
        for action, item in SELL_ACTIONS.items():
            mask[action] = self.inventory[item] > 0
        return mask

    # ----------------------------------------------------------------- tick

    def step(self, action: Action | int) -> TickReport:
        self.report = TickReport()
        self._apply_action(Action(action))
        self._produce()
        self._pay_upkeep()
        self._move_market()
        self._roll_disaster()
        self.tick += 1
        return self.report

    def _apply_action(self, action: Action) -> None:
        if action in BUILD_ACTIONS:
            kind = BUILD_ACTIONS[action]
            cost = MACHINES[kind].cost
            if self.credits >= cost:
                self.credits -= cost
                self.report.spent += cost
                self.machines.append(Machine(kind))
        elif action in REPAIR_ACTIONS:
            kind = REPAIR_ACTIONS[action]
            machine = self._most_damaged(kind)
            cost = self.repair_cost(kind)
            if machine is not None and machine.condition < 1.0 and self.credits >= cost:
                self.credits -= cost
                self.report.spent += cost
                machine.condition = 1.0
        elif action in PROSPECT_ACTIONS:
            if self.credits >= self.cfg.prospect_cost:
                self.credits -= self.cfg.prospect_cost
                self.report.spent += self.cfg.prospect_cost
                ore = PROSPECT_ACTIONS[action]
                if self.rng.random() < self.cfg.prospect_success:
                    deposit = self._new_deposit(ore)
                    self.deposits.append(deposit)
                    self.report.events.append(
                        f"survey struck {ore.value}: {deposit.remaining:.0f} units "
                        f"at richness {deposit.richness:.2f}"
                    )
                else:
                    self.report.events.append(f"survey for {ore.value} found nothing")
        elif action in SELL_ACTIONS:
            item = SELL_ACTIONS[action]
            amount = self.inventory[item]
            if amount > 0:
                revenue = amount * self.price(item)
                self.inventory[item] = 0.0
                self.credits += revenue
                self.report.revenue += revenue

    def _running(self, kind: Kind) -> list[Machine]:
        return [
            m for m in self.machines if m.kind is kind and m.condition >= self.cfg.min_condition
        ]

    def _throughput(self, kind: Kind) -> float:
        """Units/tick the healthy machines of a kind could handle at full power."""
        return sum(m.condition * m.spec.rate for m in self._running(kind))

    def _has_work(self, kind: Kind) -> bool:
        if kind in MINE_ORE:
            return self.reserves(MINE_ORE[kind]) > 0
        if kind in SMELTER_IO:
            return self.inventory[SMELTER_IO[kind][0]] > 0
        return all(self.inventory[item] >= qty for item, qty in RECIPE.items())

    def _produce(self) -> None:
        working = [
            k for k in MACHINES if k is not Kind.GENERATOR and self._has_work(k)
        ]
        demand = sum(m.spec.power for k in working for m in self._running(k))
        supply = sum(m.condition * m.spec.rate for m in self._running(Kind.GENERATOR))
        efficiency = 1.0 if demand <= supply else supply / demand if demand else 1.0

        self.report.power_demand = demand
        self.report.power_supply = supply
        self.report.power_efficiency = efficiency

        # Run the chain backwards so goods take a tick to move down it.
        self._assemble(efficiency)
        for kind in (Kind.SMELTER_A, Kind.SMELTER_B):
            self._smelt(kind, efficiency)
        for kind in (Kind.MINE_A, Kind.MINE_B):
            self._mine(kind, efficiency)

        for kind in working:
            for machine in self._running(kind):
                machine.condition = max(0.0, machine.condition - machine.spec.wear)
        for machine in self._running(Kind.GENERATOR):
            machine.condition = max(0.0, machine.condition - machine.spec.wear)

    def _assemble(self, efficiency: float) -> None:
        capacity = self._throughput(Kind.ASSEMBLER) * efficiency
        for item, qty in RECIPE.items():
            capacity = min(capacity, self.inventory[item] / qty)
        if capacity <= 0:
            return
        for item, qty in RECIPE.items():
            self.inventory[item] -= capacity * qty
        self.inventory[Item.WIDGET] += capacity
        self.widgets_built += capacity
        self.report.widgets_built = capacity

    def _smelt(self, kind: Kind, efficiency: float) -> None:
        ore, component = SMELTER_IO[kind]
        amount = min(self._throughput(kind) * efficiency, self.inventory[ore])
        if amount <= 0:
            return
        self.inventory[ore] -= amount
        self.inventory[component] += amount
        self.report.smelted[component] = amount

    def _mine(self, kind: Kind, efficiency: float) -> None:
        ore = MINE_ORE[kind]
        budget = self._throughput(kind) * efficiency  # machine-hours, before richness
        seams = sorted(
            (d for d in self.deposits if d.ore is ore and d.remaining > 0),
            key=lambda d: d.richness,
            reverse=True,
        )
        total = 0.0
        for seam in seams:
            if budget <= 0:
                break
            pulled = min(budget * seam.richness, seam.remaining)
            seam.remaining -= pulled
            budget -= pulled / seam.richness
            total += pulled
            if seam.remaining <= 0:
                self.report.events.append(f"a {ore.value} deposit ran dry")
        if total:
            self.inventory[ore] += total
            self.report.mined[ore] = total

    def _pay_upkeep(self) -> None:
        upkeep = sum(m.spec.upkeep for m in self.machines)
        self.credits -= upkeep
        self.report.upkeep = upkeep

    def _move_market(self) -> None:
        drift = self.cfg.price_reversion * (1.0 - self.price_index)
        shock = self.cfg.price_volatility * self.rng.normal()
        self.price_index = float(
            np.clip(self.price_index + drift + shock, *self.cfg.price_bounds)
        )

    # ------------------------------------------------------------ disasters

    def _roll_disaster(self) -> None:
        if self.rng.random() >= self.cfg.disaster_rate:
            return
        names = list(self.cfg.disaster_weights)
        weights = np.array([self.cfg.disaster_weights[n] for n in names], dtype=float)
        choice = names[int(self.rng.choice(len(names), p=weights / weights.sum()))]
        getattr(self, f"_disaster_{choice}")()

    def _disaster_cave_in(self) -> None:
        live = [d for d in self.deposits if d.remaining > 0]
        if not live:
            return
        seam = live[int(self.rng.integers(len(live)))]
        lost = seam.remaining * float(self.rng.uniform(*self.cfg.cave_in_loss))
        seam.remaining -= lost
        self.report.events.append(f"CAVE-IN: {lost:.0f} units of {seam.ore.value} buried")

    def _disaster_breakdown(self) -> None:
        if not self.machines:
            return
        machine = self.machines[int(self.rng.integers(len(self.machines)))]
        machine.condition = 0.0
        self.report.events.append(f"BREAKDOWN: a {machine.kind.value} is wrecked")

    def _disaster_power_surge(self) -> None:
        generators = [m for m in self.machines if m.kind is Kind.GENERATOR]
        if not generators:
            return
        self.machines.remove(generators[int(self.rng.integers(len(generators)))])
        self.report.events.append("POWER SURGE: a generator burned out")

    def _disaster_market_crash(self) -> None:
        self.price_index *= float(self.rng.uniform(*self.cfg.crash_factor))
        self.price_index = max(self.price_index, self.cfg.price_bounds[0])
        self.report.events.append(f"MARKET CRASH: price index {self.price_index:.2f}")

    def _disaster_quake(self) -> None:
        if not self.machines:
            return
        damage = float(self.rng.uniform(*self.cfg.quake_damage))
        for machine in self.machines:
            machine.condition = max(0.0, machine.condition - damage)
        self.report.events.append(f"QUAKE: every machine lost {damage:.0%} condition")
