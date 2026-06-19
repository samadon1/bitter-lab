"""Elo: turn win rates into a single skill number — Slice 1.

Elo says the expected score between two players is a logistic function of their
rating difference:

    E_a = 1 / (1 + 10 ** ((R_b - R_a) / 400))

A 400-point gap -> the favorite scores ~0.91; a ~70-point gap -> ~0.60.

For the crossover we anchor the heuristic at 0 Elo and express each MCTS config's
strength as its rating difference *relative to the heuristic*, inferred from the
head-to-head win rate (the inverse of the formula above). The crossover is simply
where an MCTS curve crosses 0 — i.e. where it starts beating the expert.
"""

from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_diff_from_winrate(winrate: float, clamp: float = 0.005) -> float:
    """Invert the logistic: rating advantage implied by a win rate in (0, 1).

    winrate 0.5 -> 0 Elo; 0.6 -> ~+70; 0.91 -> ~+400. Clamped away from 0/1 so a
    clean sweep (or shutout) yields a large-but-finite number instead of +/-inf.
    """
    p = min(max(winrate, clamp), 1.0 - clamp)
    return 400.0 * math.log10(p / (1.0 - p))


def winrate_standard_error(winrate: float, n_games: int) -> float:
    """SE of an observed win rate — how much noise is in this measurement."""
    return math.sqrt(max(winrate * (1.0 - winrate), 1e-9) / n_games)


def games_to_resolve(elo_gap: float) -> int:
    """Rough games needed to call a given Elo gap real (~2-sigma).

    Anchors the chess-literature rule of thumb: ~500 games for a ~70-Elo gap,
    ~1000 for ~40. Returned as guidance for sizing experiments, not gospel.
    """
    p = 1.0 / (1.0 + 10 ** (-elo_gap / 400.0))
    # n such that 2*SE(0.5) ~ (p - 0.5): n ~ 1 / (p - 0.5)**2, with p=0.5 -> inf.
    edge = abs(p - 0.5)
    return 10**9 if edge < 1e-6 else int(round(1.0 / (edge**2)))
