"""Check the fast bitboard against the plain engine.

If the two ever disagree about legal moves, who won, or when the game ends, the
bitboard has a bug. We play many identical random games on both and compare every
step. This is the "trust the fast version by checking it against the simple one"
pattern from Slice 0.
"""

import random

from bitboard import BitBoard
from engine import GameState


def test_bitboard_matches_engine_on_random_games():
    rng = random.Random(2024)
    for game in range(300):
        slow = GameState()
        fast = BitBoard()
        while True:
            slow_moves = slow.legal_moves()
            fast_moves = fast.legal_moves()
            assert slow_moves == fast_moves, f"legal moves differ (game {game})"
            assert slow.is_terminal() == fast.is_terminal()
            assert slow.winner == fast.winner
            assert slow.current_player == fast.current_player
            if slow.is_terminal():
                break
            move = rng.choice(slow_moves)
            slow.play(move)
            fast.play(move)


def test_bitboard_detects_each_win_direction():
    # vertical
    b = BitBoard()
    for col in (0, 1, 0, 1, 0, 1, 0):
        b.play(col)
    assert b.winner == 1

    # horizontal
    b = BitBoard()
    for col in (0, 0, 1, 1, 2, 2, 3):
        b.play(col)
    assert b.winner == 1

    # "/" diagonal for player 1
    b = BitBoard()
    for col in (0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3):
        b.play(col)
    assert b.winner == 1


def test_bitboard_full_column_raises():
    import pytest

    b = BitBoard()
    for _ in range(6):
        b.play(3)
    with pytest.raises(ValueError):
        b.play(3)
