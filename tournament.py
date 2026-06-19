"""Play two agents against each other and count the results.

One important fairness rule: in Connect Four, going first is a real advantage. So
every match SWITCHES who goes first each game. Otherwise we'd mostly be measuring
who got lucky enough to start, not who actually plays better.
"""

from __future__ import annotations

import random

from engine import GameState


def play_game(player1, player2) -> int:
    """Play one full game. player1 goes first. Returns the winner: 1, 2, or 0 (draw)."""
    state = GameState()
    agents = {1: player1, 2: player2}
    while not state.is_terminal():
        move = agents[state.current_player].select_move(state)
        state.play(move)
    return state.winner


def _reseed(agent, rng: random.Random) -> None:
    # Give each agent a fresh random seed per game so games differ, while the whole
    # run still repeats exactly if you start from the same master seed.
    if hasattr(agent, "rng"):
        agent.rng.seed(rng.randrange(2**31))


def match(agent_a, agent_b, n_games: int, rng: random.Random) -> dict:
    """Play n_games between A and B, swapping who starts each game.

    Counts are from A's point of view. A draw counts as half a win.
    """
    wins_a = wins_b = draws = 0
    for i in range(n_games):
        _reseed(agent_a, rng)
        _reseed(agent_b, rng)
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b)  # A starts
            a_is = 1
        else:
            winner = play_game(agent_b, agent_a)  # B starts
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
