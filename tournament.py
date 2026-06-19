"""Play agents against each other and tally results — Slice 1.

Key fairness detail: Connect Four's first player has a real advantage (with
perfect play, player 1 wins outright). So a match ALTERNATES who moves first
every game. Otherwise we'd be measuring the coin-flip of seating, not skill.
"""

from __future__ import annotations

import random

from engine import GameState


def play_game(player1, player2) -> int:
    """Play one game; player1 moves first. Returns winner: 1, 2, or 0 (draw)."""
    state = GameState()
    agents = {1: player1, 2: player2}
    while not state.is_terminal():
        move = agents[state.current_player].select_move(state)
        state.play(move)
    return state.winner


def _reseed(agent, rng: random.Random) -> None:
    # Give stochastic agents a fresh seed each game so games vary, while the
    # whole experiment stays reproducible from one master rng.
    if hasattr(agent, "rng"):
        agent.rng.seed(rng.randrange(2**31))


def match(agent_a, agent_b, n_games: int, rng: random.Random) -> dict:
    """Play n_games between A and B, alternating the starting player.

    Returns counts from A's perspective plus A's win rate (draws = half a point).
    """
    wins_a = wins_b = draws = 0
    for i in range(n_games):
        _reseed(agent_a, rng)
        _reseed(agent_b, rng)
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b)  # A first
            a_is = 1
        else:
            winner = play_game(agent_b, agent_a)  # B first
            a_is = 2
        if winner == 0:
            draws += 1
        elif winner == a_is:
            wins_a += 1
        else:
            wins_b += 1

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "n_games": n_games,
        "winrate_a": (wins_a + 0.5 * draws) / n_games,
    }
