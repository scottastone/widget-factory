"""Terminal front end: watch a baseline run, or play the world by hand."""

import argparse

import numpy as np

from .agents import heuristic_policy, random_policy
from .config import Config
from .env import FactoryEnv
from .world import Action


def _run(env: FactoryEnv, policy, render_every: int, seed: int | None) -> None:
    obs, info = env.reset(seed)
    total = 0.0
    while True:
        action = policy(env.world)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if render_every and env.world.tick % render_every == 0:
            print(f"\n>>> {Action(action).name}")
            print(env.render())
        elif info["events"]:
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


def _play(env: FactoryEnv) -> None:
    print(env.render())
    while True:
        mask = env.world.legal_actions()
        options = "  ".join(
            f"{a.value}:{a.name.lower()}" for a in Action if mask[a.value]
        )
        print(f"\nlegal: {options}")
        try:
            raw = input("action (number, q to quit)> ").strip()
        except EOFError:
            return
        if raw in {"q", "quit", "exit"}:
            return
        if not raw.isdigit() or int(raw) not in set(Action):
            print("not an action")
            continue
        obs, reward, terminated, truncated, info = env.step(int(raw))
        print(f"\nreward {reward:+.2f}")
        print(env.render())
        if terminated or truncated:
            print("\nBANKRUPT" if terminated else "\ntime is up")
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
        "--render-every", type=int, default=50, help="0 to only print disasters"
    )
    args = parser.parse_args()

    env = FactoryEnv(Config(max_ticks=args.ticks, reward_mode=args.reward), args.seed)
    if args.mode == "play":
        _play(env)
        return
    rng = np.random.default_rng(args.seed)
    policy = heuristic_policy if args.mode == "heuristic" else (lambda w: random_policy(w, rng))
    _run(env, policy, args.render_every, args.seed)
