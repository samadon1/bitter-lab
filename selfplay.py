"""Make the value network learn from the agent's own games.

The loop, in plain terms:
  - The agent plays games against itself using value-MCTS (the net guides search).
  - For every position, we remember whose turn it was.
  - When the game ends, we label each remembered position: did the player who was
    about to move there end up winning (+1), losing (-1), or drawing (0)?
  - We train the net to predict those labels. Better predictions -> better search ->
    better games -> better labels. Round and round.

No human knowledge goes in - only the rules and the outcomes of its own play. That's
the self-play idea behind AlphaGo/AlphaZero, shrunk to fit a laptop.
"""

from __future__ import annotations

import numpy as np

from bitboard import BitBoard
from valuenet import to_planes


def play_selfplay_game(agent, rng, make_state=BitBoard, explore_moves: int = 10):
    """Play one game of the agent against itself.

    Returns (examples, winner), where examples is a list of (planes, target),
    target being the game result from that position's mover's point of view.

    For the first `explore_moves` moves we pick randomly in proportion to how much
    each move was explored, so games vary and the net sees a wide range of positions.
    After that we just take the best move.
    """
    state = make_state()
    seen = []  # (planes, player_to_move)

    ply = 0
    while not state.is_terminal():
        best, visits = agent.search_visits(state)
        seen.append((to_planes(state), state.current_player))

        if ply < explore_moves and len(visits) > 1:
            moves, counts = zip(*visits)
            probs = np.asarray(counts, dtype=np.float64)
            probs /= probs.sum()
            move = int(rng.choices(moves, weights=probs)[0])
        else:
            move = best

        state.play(move)
        ply += 1

    winner = state.winner
    examples = []
    for planes, player in seen:
        if winner == 0:
            target = 0.0
        elif winner == player:
            target = 1.0
        else:
            target = -1.0
        examples.append((planes, target))
    return examples, winner
