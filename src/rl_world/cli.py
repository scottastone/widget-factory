"""Terminal front end: watch a baseline run live, or play the world by hand."""

import argparse
import time

import numpy as np
from rich.console import Console
from rich.live import Live

from .agents import heuristic_policy, random_policy
from .config import Config
from .env import FactoryEnv
from .ui import Dashboard, summary
from .world import Action


def _run_live(env: FactoryEnv, policy, seed: int | None, mode: str, delay: float) -> None:
    console = Console()
    dashboard = Dashboard(env, mode, log_lines=max(4, console.size.height - 14))
    env.reset(seed)
    total = 0.0
    bankrupt = False

    with Live(dashboard.layout(), console=console, refresh_per_second=30) as live:
        while True:
            action = policy(env.world)
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            dashboard.record(action, info)
            live.update(dashboard.layout())
            if delay:
                time.sleep(delay)
            if terminated or truncated:
                bankrupt = terminated
                break

    console.print(summary(env, total, bankrupt))


def _run_plain(env: FactoryEnv, policy, seed: int | None, render_every: int) -> None:
    env.reset(seed)
    total = 0.0
    while True:
        action = policy(env.world)
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if render_every and env.world.tick % render_every == 0:
            print(f"\n>>> {Action(action).name}")
            print(env.render())
        for event in info["events"]:
            print(f"tick {info['tick']:>4}  ! {event}")
        if terminated or truncated:
            break
    print("\n" + env.render())
    outcome = "BANKRUPT" if terminated else "survived"
    print(
        f"\n{outcome} after {env.world.tick} ticks | "
        f"return {total:,.1f} | widgets {env.world.widgets_built:,.1f}"
    )


def _play(env: FactoryEnv, seed: int | None) -> None:
    console = Console()
    dashboard = Dashboard(env, "you")
    env.reset(seed)
    while True:
        console.clear()
        console.print(dashboard.static())
        mask = env.world.legal_actions()
        options = "  ".join(
            f"[bold]{a.value}[/]:{a.name.lower()}" for a in Action if mask[a.value]
        )
        console.print(f"[dim]legal[/] {options}")
        try:
            raw = console.input("[bold cyan]action[/] (number, q to quit)> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if raw in {"q", "quit", "exit"}:
            return
        if not raw.isdigit() or int(raw) not in set(Action):
            continue
        action = int(raw)
        _, reward, terminated, truncated, info = env.step(action)
        dashboard.record(action, info)
        if terminated or truncated:
            console.clear()
            console.print(dashboard.static())
            console.print(summary(env, reward, terminated))
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="A toy resource economy for RL agents.")
    parser.add_argument(
        "mode", choices=("play", "random", "heuristic"), help="who drives the factory"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--reward", choices=("profit", "widgets"), default="profit")
    parser.add_argument(
        "--delay", type=float, default=0.05, help="seconds per tick in the live view"
    )
    parser.add_argument(
        "--plain", action="store_true", help="line-by-line text instead of the live view"
    )
    parser.add_argument(
        "--render-every", type=int, default=50, help="tick cadence for --plain"
    )
    args = parser.parse_args()

    env = FactoryEnv(Config(max_ticks=args.ticks, reward_mode=args.reward), args.seed)
    if args.mode == "play":
        _play(env, args.seed)
        return

    rng = np.random.default_rng(args.seed)
    policy = (
        heuristic_policy if args.mode == "heuristic" else lambda w: random_policy(w, rng)
    )
    if args.plain or not Console().is_terminal:
        _run_plain(env, policy, args.seed, args.render_every)
    else:
        _run_live(env, policy, args.seed, args.mode, args.delay)
