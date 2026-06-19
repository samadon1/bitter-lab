"""Turn "how often did it win?" into a single skill number (Elo).

Elo is the chess rating idea: the gap between two players' numbers predicts how
often the stronger one should win. A 400-point gap means the favourite wins about
90% of the time; a 70-point gap is about 60%.

For our chart we fix the heuristic at 0 and give each MCTS setting a number based
on how often it beats the heuristic. A positive number means it's beating the
expert - so the "crossover" is just where the number goes above 0.
"""

from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """How often player A should beat player B, given their two ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def elo_diff_from_winrate(winrate: float, clamp: float = 0.005) -> float:
    """Go the other way: from an observed win rate to a rating gap.

    A 50% win rate -> 0 (evenly matched). 60% -> about +70. 90% -> about +400.
    We nudge win rates away from exactly 0% or 100% so a clean sweep gives a big
    finite number instead of infinity.
    """
    p = min(max(winrate, clamp), 1.0 - clamp)
    return 400.0 * math.log10(p / (1.0 - p))


def winrate_standard_error(winrate: float, n_games: int) -> float:
    """How shaky a win rate is: more games -> smaller number -> more trustworthy."""
    return math.sqrt(max(winrate * (1.0 - winrate), 1e-9) / n_games)


def games_to_resolve(elo_gap: float) -> int:
    """Rough number of games needed before you can trust a gap of this size.

    Matches the chess rule of thumb: ~500 games to be sure of a 70-point gap,
    ~1000 for a 40-point gap. A guideline for sizing experiments, not a hard rule.
    """
    p = 1.0 / (1.0 + 10 ** (-elo_gap / 400.0))
    edge = abs(p - 0.5)
    return 10**9 if edge < 1e-6 else int(round(1.0 / (edge**2)))
