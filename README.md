# rl_world

A toy Factorio-shaped economy for reinforcement-learning agents. You mine two ores,
smelt them into components A and B, and assemble `2A + 1B` into widgets you sell on a
moving market — while power shortages, wear, deposit depletion and disasters try to
bankrupt you.

It is a library with a live terminal dashboard, no GUI.

## Install & run

```bash
uv sync
uv run rl-world heuristic --seed 1        # scripted baseline, live view
uv run rl-world random --seed 1           # random baseline (dies in ~30 ticks)
uv run rl-world play                      # drive it yourself
uv run pytest
```

The `random` and `heuristic` runs open a Rich dashboard that updates in place —
plant condition, stockpiles, remaining seams, a power-load gauge, and a scrolling log of
the agent's decisions with disasters called out in red. Repeated actions fold into one
`x N` line.

| Flag | |
| --- | --- |
| `--delay 0.05` | seconds per tick in the live view; `0` runs flat out |
| `--ticks 500` | episode length |
| `--seed N` | reproducible run |
| `--reward profit\|widgets` | what `step` returns as reward |
| `--plain` | line-by-line text instead of the live view (also used automatically when stdout is not a terminal) |
| `--render-every 50` | dashboard cadence for `--plain`; `0` prints only disasters |

`play` prints the same panels and prompts for an action number each tick, listing only
the legal ones.

## Agent API

Gymnasium's calling convention, without the dependency:

```python
from rl_world import FactoryEnv, Config

env = FactoryEnv(Config(max_ticks=500), seed=0)
obs, info = env.reset(seed=0)                     # obs: float32[27]
obs, reward, terminated, truncated, info = env.step(action)   # action: int in [0, 20)
```

- `terminated` — bankrupt (credits below zero); costs `Config.bankruptcy_penalty`.
- `truncated` — `max_ticks` reached.
- `info["action_mask"]` — bool array over actions; illegal actions are legal to *send*,
  they just do nothing.
- `reward` — change in credits per tick (`reward_mode="profit"`, the default), or widgets
  produced per tick (`reward_mode="widgets"`).
- `env.render()` — ASCII dashboard. `rl_world.OBSERVATION_LABELS` names every obs slot.

`World` (in [world.py](src/rl_world/world.py)) is the simulation and can be driven
directly if you don't want the observation vector.

## The economy

```
deposit --mine--> ore_a --smelter--> comp_a --.
                                               >--assembler--> widget --sell--> credits
deposit --mine--> ore_b --smelter--> comp_b --'
```

Every stage runs once per tick and the chain is processed backwards, so goods take one
tick per stage to move down it. Ore, components and widgets can all be sold, so a partial
factory still earns something — badly.

Machines cost credits to build, draw power, charge upkeep every tick whether or not they
run, and wear out. Generators supply power; when demand exceeds supply *every* machine
runs at `supply / demand` efficiency.

## What makes it hard

| Pressure | Effect |
| --- | --- |
| Finite deposits | Each seam has a size and a richness; mines drain the richest first, then idle. `prospect` costs credits and fails 40% of the time. |
| Power | A brownout throttles the whole plant at once, so expansion has to be paid for twice. |
| Wear | Condition decays while running; output scales with it and below 10% a machine stops. Repairs cost more the longer you wait. |
| Upkeep | Fixed credits per tick per machine — an idle over-built plant bleeds out. |
| Market | Mean-reverting price index on everything; selling into a trough wastes the inventory. |
| Disasters | `cave_in` (buries ore), `breakdown` (wrecks a machine), `power_surge` (destroys a generator), `market_crash` (halves prices), `quake` (damages everything). 5% per tick by default. |

All of it is in [config.py](src/rl_world/config.py) — costs, rates, disaster weights and
rates are dataclass fields, so `Config(disaster_rate=0.0)` gives you a calm world for
debugging.

## Actions (20, discrete)

| | |
| --- | --- |
| 0 | `noop` |
| 1–6 | `build_` mine_a, mine_b, smelter_a, smelter_b, assembler, generator |
| 7–12 | `repair_` same six kinds (repairs that kind's most damaged machine) |
| 13–14 | `prospect_a`, `prospect_b` |
| 15–19 | `sell_` ore_a, ore_b, comp_a, comp_b, widgets (sells the whole stockpile) |

## Baselines

Over 500 ticks, seeds 0–7: the random policy goes bankrupt within ~40 ticks every time;
the heuristic in [agents.py](src/rl_world/agents.py) returns roughly 15k–24k credits and
builds ~1,900 widgets, but still dies to an early disaster run on some seeds.
