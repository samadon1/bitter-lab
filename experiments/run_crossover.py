"""The main experiment: where does "thinking" overtake "knowing"?

Let MCTS play the heuristic over and over, each time giving MCTS more thinking
(more simulations per move). Turn each result into a skill number and plot it as
MCTS thinks harder.

Two things to watch for in the chart:
  1) the CROSSOVER - the point where MCTS starts beating the expert;
  2) the FLATTENING - near the top, each doubling of thinking helps less than the
     last. Skill grows with the size of the effort, not in a straight line.

Run:  python experiments/run_crossover.py
Makes: data/crossover.json  and  crossover.png
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

# make the repo root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # no display needed; write a file
import matplotlib.pyplot as plt  # noqa: E402

from agents import HeuristicAgent, MCTSAgent  # noqa: E402
from elo import elo_diff_from_winrate, games_to_resolve, winrate_standard_error  # noqa: E402
from tournament import match  # noqa: E402

SIM_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_GAMES = 80          # fewer than ideal, but enough to see the shape (more = smoother)
SEED = 12345
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run() -> dict:
    rng = random.Random(SEED)
    rows = []
    print(f"MCTS(sims) vs HeuristicAgent — {N_GAMES} games each, alternating start\n")
    print(f"{'sims':>6} {'winrate':>8} {'elo':>8} {'+/-':>6} {'W/L/D':>12} {'secs':>6}")
    for sims in SIM_COUNTS:
        t = time.time()
        res = match(MCTSAgent(sims), HeuristicAgent(), N_GAMES, rng)  # A = MCTS
        wr = res["winrate_a"]
        elo = elo_diff_from_winrate(wr)
        se = winrate_standard_error(wr, N_GAMES)
        secs = time.time() - t
        rows.append({
            "sims": sims, "winrate": wr, "elo": elo, "winrate_se": se,
            "wins": res["wins_a"], "losses": res["wins_b"], "draws": res["draws"],
            "seconds": secs,
        })
        wld = f"{res['wins_a']}/{res['wins_b']}/{res['draws']}"
        print(f"{sims:>6} {wr:>8.2f} {elo:>8.0f} {400*se:>6.0f} {wld:>12} {secs:>6.1f}")

    data = {
        "config": {"sim_counts": SIM_COUNTS, "n_games": N_GAMES, "seed": SEED,
                   "opponent": "heuristic (anchored at 0 Elo)"},
        "results": rows,
        "note_games_for_70elo": games_to_resolve(70),
    }
    with open(os.path.join(REPO, "data", "crossover.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data


def plot(data: dict) -> None:
    rows = data["results"]
    sims = [r["sims"] for r in rows]
    elo = [r["elo"] for r in rows]
    err = [400 * r["winrate_se"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sims, elo, yerr=err, marker="o", capsize=3, lw=2, label="MCTS (pure compute)")
    ax.axhline(0, ls="--", color="gray", label="HeuristicAgent (human knowledge)")

    # mark the crossover: first sim count where MCTS reaches >= 0 Elo
    cross = next((r["sims"] for r in rows if r["elo"] >= 0), None)
    if cross is not None:
        ax.axvline(cross, ls=":", color="crimson", alpha=0.7)
        ax.annotate(f"crossover ~{cross} sims", xy=(cross, 0),
                    xytext=(cross, min(elo) * 0.4 if min(elo) < 0 else -50),
                    color="crimson", ha="center")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("MCTS simulations per move  (compute, log2 scale)")
    ax.set_ylabel("Elo relative to the heuristic")
    ax.set_title("Bitter lesson on Connect Four: compute overtakes hand-coded knowledge")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(REPO, "figures", "crossover.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    plot(run())
