"""Slice 2: thinking isn't free - measure what it costs.

Five things, building on the crossover from Slice 1:

  1) SPEEDUP    - how much faster is the bitboard than the array board?
  2) WHERE      - profile a move: where does the time actually go?
  3) PER-COST   - the ladder: does each doubling of thinking add as much skill as
                  the last? (It shouldn't - that's diminishing returns.)
  4) THE FLIP   - put both players on a per-move clock. With a tiny budget the
                  instant expert wins; with a generous one the thinker wins.
  5) PARADOX    - the speedup doesn't save time, it buys more thinking in the same
                  time - which is worth extra skill.

Run:  python experiments/run_efficiency.py
Makes: data/efficiency.json + two charts.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents import HeuristicAgent, MCTSAgent  # noqa: E402
from bitboard import BitBoard  # noqa: E402
from elo import elo_diff_from_winrate  # noqa: E402
from engine import GameState  # noqa: E402
from tournament import match  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 7


# 1) ---- raw speed: random playouts per second --------------------------------

def playout_speed(make, n=4000):
    rng = random.Random(1)
    t = time.time()
    for _ in range(n):
        s = make()
        while not s.is_terminal():
            s.play(rng.choice(s.legal_moves()))
    return n / (time.time() - t)


# 2) ---- how many MCTS simulations per second, each backend -------------------

def mcts_speed(make, sims=3000):
    agent = MCTSAgent(simulations=sims, seed=0)
    s = make()
    t = time.time()
    agent.select_move(s)
    return sims / (time.time() - t)


# 3) ---- where does an MCTS move spend its time? ------------------------------

def profile_move(sims=4000):
    agent = MCTSAgent(simulations=sims, seed=0)
    s = GameState()
    pr = cProfile.Profile()
    pr.enable()
    agent.select_move(s)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(6)
    return buf.getvalue()


# 4) ---- the ladder: skill gained per doubling of thinking --------------------

def ladder(rng, n_games=30):
    rungs = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
    out = []
    for weak, strong in rungs:
        res = match(MCTSAgent(strong), MCTSAgent(weak), n_games, rng, make_state=BitBoard)
        p = res["winrate_a"]  # how often the stronger (more sims) beats the weaker
        gap = elo_diff_from_winrate(p)
        out.append({"weak": weak, "strong": strong, "winrate_strong": p, "elo_gain": gap})
        print(f"  {strong:>4} vs {weak:<4}  strong winrate {p:.2f}  -> +{gap:4.0f} Elo per doubling")
    return out


# 5) ---- the budget flip: both players on a per-move clock --------------------

def budget_flip(rng, sims_per_sec, n_games=50):
    budgets = [0.005, 0.01, 0.02, 0.04, 0.08]
    out = []
    for b in budgets:
        # estimate sims/move from the measured rate (steadier than timing one move)
        sims = round(sims_per_sec * b)
        res = match(MCTSAgent(time_budget=b), HeuristicAgent(), n_games, rng)  # A = MCTS
        p = res["winrate_a"]  # MCTS win rate vs the instant expert
        out.append({"budget_ms": b * 1000, "approx_sims": sims, "mcts_winrate": p})
        verdict = "MCTS wins" if p > 0.5 else "expert wins"
        print(f"  {b*1000:>5.0f} ms/move  (~{sims:>4} sims)  MCTS winrate {p:.2f}  -> {verdict}")
    return out


def main():
    rng = random.Random(SEED)

    print("1) Raw board speed (random playouts/sec)")
    arr = playout_speed(GameState)
    bit = playout_speed(BitBoard)
    print(f"   array    {arr:8.0f}/s\n   bitboard {bit:8.0f}/s  -> {bit/arr:.1f}x faster\n")

    print("2) MCTS simulations/sec")
    arr_mcts = mcts_speed(GameState)
    bit_mcts = mcts_speed(BitBoard)
    print(f"   array    {arr_mcts:8.0f}/s\n   bitboard {bit_mcts:8.0f}/s  -> {bit_mcts/arr_mcts:.1f}x faster\n")

    print("3) Where an MCTS move spends its time (top functions):")
    prof = profile_move()
    print(prof)

    print("4) The ladder - skill gained per doubling of thinking:")
    lad = ladder(rng)
    print()

    print("5) The budget flip - both players on a per-move clock:")
    flip = budget_flip(rng, arr_mcts)
    flip_ms = next((r["budget_ms"] for r in flip if r["mcts_winrate"] > 0.5), None)
    print(f"\n   flip happens around {flip_ms} ms/move\n")

    # 6) efficiency paradox: same 10 ms, faster board does more sims -> more Elo
    ten_ms_array = arr_mcts * 0.010
    ten_ms_bit = bit_mcts * 0.010
    paradox = {
        "sims_in_10ms_array": ten_ms_array,
        "sims_in_10ms_bitboard": ten_ms_bit,
        "extra_thinking_factor": bit_mcts / arr_mcts,
    }

    data = {
        "playout_speed": {"array": arr, "bitboard": bit, "speedup": bit / arr},
        "mcts_speed": {"array": arr_mcts, "bitboard": bit_mcts, "speedup": bit_mcts / arr_mcts},
        "ladder": lad,
        "budget_flip": flip,
        "flip_ms": flip_ms,
        "paradox": paradox,
    }
    with open(os.path.join(REPO, "data", "efficiency.json"), "w") as f:
        json.dump(data, f, indent=2)

    _plot_ladder(lad)
    _plot_flip(flip, flip_ms)
    print("saved data/efficiency.json, efficiency_ladder.png, efficiency_budget.png")


def _plot_ladder(lad):
    x = [r["weak"] for r in lad]
    y = [r["elo_gain"] for r in lad]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("simulations before doubling")
    ax.set_ylabel("Elo gained by doubling the thinking")
    ax.set_title("Diminishing returns: each doubling of thinking adds less")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "efficiency_ladder.png"), dpi=130)


def _plot_flip(flip, flip_ms):
    x = [r["budget_ms"] for r in flip]
    y = [r["mcts_winrate"] for r in flip]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", lw=2, label="MCTS (thinker) win rate")
    ax.axhline(0.5, ls="--", color="gray", label="even")
    if flip_ms:
        ax.axvline(flip_ms, ls=":", color="crimson", alpha=0.7, label=f"flip ~{flip_ms:.0f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("time budget per move (ms)")
    ax.set_ylabel("thinker win rate vs instant expert")
    ax.set_title("On a tight clock the cheap expert wins; with time the thinker wins")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "efficiency_budget.png"), dpi=130)


if __name__ == "__main__":
    main()
