"""Rich dashboard for watching a run tick by tick."""

from collections import deque

from rich.columns import Columns
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import MACHINES, Item, Kind
from .env import FactoryEnv
from .world import Action

ACTION_STYLE = {
    "build": "bold green",
    "repair": "yellow",
    "sell": "cyan",
    "prospect": "magenta",
    "noop": "dim",
}
STAGE_COLOR = {
    Kind.MINE_A: "orange3",
    Kind.MINE_B: "orange3",
    Kind.SMELTER_A: "yellow",
    Kind.SMELTER_B: "yellow",
    Kind.ASSEMBLER: "green",
    Kind.GENERATOR: "blue",
}
ITEM_COLOR = {
    Item.ORE_A: "orange3",
    Item.ORE_B: "orange3",
    Item.COMP_A: "yellow",
    Item.COMP_B: "yellow",
    Item.WIDGET: "bold green",
}


def bar(fraction: float, width: int = 12, style: str = "green") -> Text:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return Text.assemble(("█" * filled, style), ("░" * (width - filled), "grey23"))


def condition_style(value: float) -> str:
    return "green" if value > 0.6 else "yellow" if value > 0.25 else "red"


class Dashboard:
    """Builds the live view and keeps a rolling log of what the agent did."""

    def __init__(self, env: FactoryEnv, mode: str, log_lines: int = 8):
        self.env = env
        self.mode = mode
        self.log_lines = log_lines
        self.entries: deque[dict] = deque(maxlen=200)

    # ------------------------------------------------------------- log feed

    def record(self, action: int, info: dict) -> None:
        """Log one decision, folding a repeated action into a single ×N line."""
        name = Action(action).name.lower()
        last = self.entries[-1] if self.entries else None
        if last and not last["event"] and last["name"] == name:
            last["count"] += 1
            last["tick"] = info["tick"]
        else:
            self.entries.append(
                {"tick": info["tick"], "name": name, "count": 1, "event": False}
            )
        for event in info["events"]:
            self.entries.append(
                {"tick": info["tick"], "name": event, "count": 1, "event": True}
            )

    # --------------------------------------------------------------- panels

    def header(self) -> Panel:
        world, report = self.env.world, self.env.world.report
        efficiency = report.power_efficiency
        table = Table.grid(expand=True)
        for _ in range(4):
            table.add_column(justify="left", ratio=1)
        table.add_row(
            Text.assemble(("credits  ", "dim"), (f"{world.credits:,.0f}", "bold white")),
            Text.assemble(
                ("market  ", "dim"),
                (
                    f"x{world.price_index:.2f}",
                    "green" if world.price_index >= 1 else "red",
                ),
            ),
            Text.assemble(("widgets  ", "dim"), (f"{world.widgets_built:,.0f}", "bold green")),
            Text.assemble(("upkeep  ", "dim"), (f"-{report.upkeep:,.1f}/t", "red")),
        )
        table.add_row(
            Text.assemble(
                ("power    ", "dim"),
                (f"{report.power_supply:,.0f}", "blue"),
                ("/", "dim"),
                (f"{report.power_demand:,.0f}", "white"),
            ),
            Text.assemble(
                ("load    ", "dim"),
                bar(
                    report.power_demand / report.power_supply if report.power_supply else 0.0,
                    10,
                    "blue" if efficiency >= 1 else "red",
                ),
                (f" {efficiency:.0%}", "green" if efficiency >= 1 else "bold red"),
            ),
            Text.assemble(("machines  ", "dim"), (f"{len(world.machines)}", "white")),
            Text.assemble(
                ("stock value  ", "dim"),
                (
                    f"{sum(world.inventory[i] * world.price(i) for i in Item):,.0f}",
                    "cyan",
                ),
            ),
        )
        progress = world.tick / self.env.cfg.max_ticks
        return Panel(
            table,
            title=f"[bold]rl_world[/] · [italic]{self.mode}[/] · tick {world.tick}/{self.env.cfg.max_ticks}",
            subtitle=bar(progress, 30, "cyan"),
            border_style="cyan",
        )

    def plant(self) -> Panel:
        world = self.env.world
        table = Table.grid(padding=(0, 1))
        table.add_column("kind", style="white")
        table.add_column("n", justify="right")
        table.add_column("condition")
        table.add_column("pct", justify="right")
        for kind in MACHINES:
            count = world.count(kind)
            condition = world.mean_condition(kind)
            style = condition_style(condition) if count else "dim"
            table.add_row(
                Text(kind.value, style=STAGE_COLOR[kind] if count else "dim"),
                Text(str(count), style="bold" if count else "dim"),
                bar(condition, 8, style),
                Text(f"{condition:.0%}" if count else "-", style=style),
            )
        return Panel(table, title="plant", border_style="grey37")

    def stock(self) -> Panel:
        world = self.env.world
        table = Table.grid(padding=(0, 1))
        table.add_column("item")
        table.add_column("qty", justify="right")
        table.add_column("price", justify="right", style="dim")
        for item in Item:
            quantity = world.inventory[item]
            table.add_row(
                Text(item.value, style=ITEM_COLOR[item]),
                Text(f"{quantity:,.1f}", style="white" if quantity else "dim"),
                Text(f"@{world.price(item):,.2f}"),
            )
        return Panel(table, title="stock", border_style="grey37")

    def seams(self) -> Panel:
        world = self.env.world
        table = Table.grid(padding=(0, 1))
        table.add_column("ore")
        table.add_column("remaining")
        table.add_column("units", justify="right")
        table.add_column("rich", justify="right", style="dim")
        biggest = max(
            (d.remaining for d in world.deposits), default=world.cfg.deposit_size[1]
        )
        for ore in (Item.ORE_A, Item.ORE_B):
            live = sorted(
                (d for d in world.deposits if d.ore is ore and d.remaining > 0),
                key=lambda d: d.remaining,
                reverse=True,
            )
            if not live:
                table.add_row(Text(ore.value, style="red"), Text("exhausted", style="red"), "", "")
                continue
            for index, seam in enumerate(live[:3]):
                share = seam.remaining / biggest if biggest else 0.0
                table.add_row(
                    Text(ore.value if index == 0 else "", style=ITEM_COLOR[ore]),
                    bar(share, 8, "orange3" if share > 0.25 else "red"),
                    Text(f"{seam.remaining:,.0f}"),
                    Text(f"x{seam.richness:.2f}"),
                )
            if len(live) > 3:
                table.add_row("", Text(f"+{len(live) - 3} more", style="dim"), "", "")
        return Panel(table, title="seams", border_style="grey37")

    def log(self) -> Panel:
        lines = []
        for entry in list(self.entries)[-self.log_lines :]:
            tick = (f"{entry['tick']:>4}  ", "dim")
            if entry["event"]:
                lines.append(Text.assemble(tick, (f"! {entry['name']}", "bold red")))
            else:
                style = ACTION_STYLE[entry["name"].split("_")[0]]
                repeat = f" x{entry['count']}" if entry["count"] > 1 else ""
                lines.append(Text.assemble(tick, (entry["name"] + repeat, style)))
        return Panel(
            Group(*lines) if lines else Text("waiting...", style="dim"),
            title="decisions",
            border_style="grey37",
        )

    # ------------------------------------------------------------- assembly

    def layout(self) -> Layout:
        root = Layout()
        root.split_column(
            Layout(self.header(), name="header", size=4),
            Layout(name="body", size=8),
            Layout(self.log(), name="log", ratio=1),
        )
        root["body"].split_row(
            Layout(self.plant(), name="plant", ratio=3),
            Layout(self.stock(), name="stock", ratio=3),
            Layout(self.seams(), name="seams", ratio=4),
        )
        return root

    def static(self) -> Group:
        """Same panels sized to their content, for prompt-driven play."""
        return Group(
            self.header(),
            Columns([self.plant(), self.stock(), self.seams()], expand=True),
            self.log(),
        )


def summary(env: FactoryEnv, total_reward: float, bankrupt: bool) -> Panel:
    world = env.world
    verdict = Text("BANKRUPT", style="bold red") if bankrupt else Text("SURVIVED", style="bold green")
    body = Text.assemble(
        verdict,
        (f"  after {world.tick} ticks\n", "white"),
        ("return  ", "dim"),
        (f"{total_reward:,.1f}\n", "bold cyan"),
        ("widgets ", "dim"),
        (f"{world.widgets_built:,.1f}\n", "bold green"),
        ("credits ", "dim"),
        (f"{world.credits:,.1f}", "white"),
    )
    return Panel(body, title="run complete", border_style="red" if bankrupt else "green")
