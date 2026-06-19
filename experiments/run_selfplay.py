"""Slice 3: learn a value from self-play, and watch strength grow with training.

The agent plays itself, trains a small value network on the outcomes, and we
measure it every round against two opponents:

  - random-rollout MCTS at the SAME number of simulations. This is the key test:
    if value-MCTS wins, a *learned* leaf signal beat random play-outs for equal
    compute - i.e. it lifted the Slice 2 plateau.
  - the Slice 1 hand-coded expert, for an absolute reference.

The headline picture: win rate vs number of self-play games - strength rising as
the agent learns purely from its own play, with no strategy ever programmed in.

Run:  python experiments/run_selfplay.py
Makes: data/selfplay.json, data/valuenet.npz, selfplay_curve.png
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agents import HeuristicAgent, MCTSAgent  # noqa: E402
from bitboard import BitBoard  # noqa: E402
from engine import GameState  # noqa: E402
from selfplay import play_selfplay_game  # noqa: E402
from tournament import match  # noqa: E402
from valuenet import ValueNet  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SELFPLAY_SIMS = 120     # thinking per move during self-play (better outcome labels)
EVAL_SIMS = 80          # equal-compute comparison vs random-rollout MCTS
GAMES_PER_ITER = 100
ITERS = 12
EVAL_GAMES = 60         # more games -> less noisy win rates
BUFFER = 40000
TRAIN_EPOCHS = 5
BATCH = 256
SEED = 0


def evaluate(net, rng, with_heuristic=False) -> dict:
    """Win rate vs random-MCTS (equal sims), and optionally vs the expert.

    vs random-MCTS is measured every round (the key 'did a learned value beat
    random rollouts?' test). vs the expert is slower, so we sample it only at the
    start and end as an absolute reference.
    """
    vs_random = match(
        MCTSAgent(EVAL_SIMS, value_fn=net.predict_value), MCTSAgent(EVAL_SIMS),
        EVAL_GAMES, rng, make_state=BitBoard,
    )["winrate_a"]
    out = {"vs_random_mcts": vs_random, "vs_heuristic": None}
    if with_heuristic:
        out["vs_heuristic"] = match(
            MCTSAgent(EVAL_SIMS, value_fn=net.predict_value), HeuristicAgent(),
            EVAL_GAMES, rng, make_state=GameState,
        )["winrate_a"]
    return out


def main():
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    net = ValueNet(hidden=128, lr=1e-3, seed=SEED)
    selfplay_agent = MCTSAgent(SELFPLAY_SIMS, value_fn=net.predict_value, seed=SEED)
    buffer = deque(maxlen=BUFFER)

    curve = []
    games_played = 0

    def line(games, ev, extra=""):
        h = "  -" if ev["vs_heuristic"] is None else f"{ev['vs_heuristic']:.2f}"
        print(f"games={games:4d}  vs random-MCTS {ev['vs_random_mcts']:.2f}   "
              f"vs expert {h}   {extra}")

    # round 0: untrained baseline
    base = evaluate(net, rng, with_heuristic=True)
    curve.append({"games": 0, **base})
    line(0, base, "(untrained)")

    for it in range(1, ITERS + 1):
        t = time.time()
        # --- self-play: generate games and collect (position, outcome) ---
        for _ in range(GAMES_PER_ITER):
            selfplay_agent.rng.seed(rng.randrange(2**31))
            examples, _ = play_selfplay_game(selfplay_agent, rng)
            buffer.extend(examples)
            games_played += 1

        # --- train the net on everything seen so far ---
        X = np.array([e[0] for e in buffer], dtype=np.float32)
        y = np.array([e[1] for e in buffer], dtype=np.float32)
        loss = net.fit(X, y, epochs=TRAIN_EPOCHS, batch=BATCH, rng=nprng)

        # --- evaluate (expert only on the final round; it's slower) ---
        ev = evaluate(net, rng, with_heuristic=(it == ITERS))
        curve.append({"games": games_played, **ev})
        line(games_played, ev, f"(loss {loss:.3f}, buffer {len(buffer)}, {time.time()-t:.0f}s)")

    # save weights, data, and the chart
    np.savez(os.path.join(REPO, "data", "valuenet.npz"),
             W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2)
    data = {
        "config": {"selfplay_sims": SELFPLAY_SIMS, "eval_sims": EVAL_SIMS,
                   "games_per_iter": GAMES_PER_ITER, "iters": ITERS,
                   "eval_games": EVAL_GAMES, "seed": SEED},
        "curve": curve,
    }
    with open(os.path.join(REPO, "data", "selfplay.json"), "w") as f:
        json.dump(data, f, indent=2)
    _plot(curve)
    print("\nsaved data/selfplay.json, data/valuenet.npz, selfplay_curve.png")


def _plot(curve):
    games = [r["games"] for r in curve]
    vr = [r["vs_random_mcts"] for r in curve]
    hg = [r["games"] for r in curve if r["vs_heuristic"] is not None]
    hv = [r["vs_heuristic"] for r in curve if r["vs_heuristic"] is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(games, vr, marker="o", lw=2, label=f"vs random-rollout MCTS (equal {EVAL_SIMS} sims)")
    ax.scatter(hg, hv, marker="s", color="darkorange", zorder=5, label="vs hand-coded expert (endpoints)")
    ax.axhline(0.5, ls="--", color="gray", label="even")
    ax.set_xlabel("self-play games used for training")
    ax.set_ylabel("win rate")
    ax.set_title("Learning from self-play: strength rises with training (no strategy taught)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "selfplay_curve.png"), dpi=130)


if __name__ == "__main__":
    main()
