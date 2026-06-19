"""Tests for the Connect Four engine.

The game always behaves the same way, so we can check its rules completely here.
That's the point of testing this part hard: it's the one piece we can fully trust,
which makes every later bug easier to track down.
"""

import numpy as np
import pytest

from engine import COLS, ROWS, GameState, has_won


def fresh() -> GameState:
    return GameState()


# -- win detection: all four directions ------------------------------------


def test_horizontal_win():
    g = fresh()
    # X plays cols 0,1,2,3 ; O plays col 6 in between
    for col in (0, 6, 1, 6, 2, 6, 3):
        g.play(col)
    assert g.winner == 1
    assert g.is_terminal()


def test_vertical_win():
    g = fresh()
    for col in (3, 4, 3, 4, 3, 4, 3):
        g.play(col)
    assert g.winner == 1
    assert g.is_terminal()


def test_diagonal_up_right_win():
    # Build a "/" diagonal for player 1.
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    coords = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for r, c in coords:
        board[r, c] = 1
    assert has_won(board, 1)
    assert not has_won(board, 2)


def test_diagonal_up_left_win():
    # Build a "\" diagonal for player 2.
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    coords = [(0, 3), (1, 2), (2, 1), (3, 0)]
    for r, c in coords:
        board[r, c] = 2
    assert has_won(board, 2)
    assert not has_won(board, 1)


def test_no_win_on_empty_board():
    g = fresh()
    assert g.winner == 0
    assert not g.is_terminal()
    assert not has_won(g.board, 1)
    assert not has_won(g.board, 2)


def test_three_in_a_row_is_not_a_win():
    g = fresh()
    for col in (0, 6, 1, 6, 2):  # X has only three in a row
        g.play(col)
    assert g.winner == 0


# -- move legality ----------------------------------------------------------


def test_legal_moves_full_on_empty_board():
    g = fresh()
    assert g.legal_moves() == list(range(COLS))


def test_legal_moves_excludes_full_column():
    g = fresh()
    # Fill column 0 with 6 pieces (alternating players).
    for _ in range(ROWS):
        g.play(0)
        if not g.is_terminal():
            continue
    # Column 0 should now be full and excluded.
    assert 0 not in g.legal_moves()
    assert len(g.legal_moves()) == COLS - 1


def test_play_on_full_column_raises():
    g = fresh()
    for _ in range(ROWS):
        # play col 0, but keep the other player dumping into col 6 so no win on col 0
        g.play(0)
    with pytest.raises(ValueError):
        g.play(0)


def test_play_out_of_range_raises():
    g = fresh()
    with pytest.raises(ValueError):
        g.play(COLS)
    with pytest.raises(ValueError):
        g.play(-1)


# -- gravity / stacking -----------------------------------------------------


def test_pieces_stack_from_bottom():
    g = fresh()
    g.play(3)  # player 1 -> row 0
    g.play(3)  # player 2 -> row 1
    assert g.board[0, 3] == 1
    assert g.board[1, 3] == 2


# -- draw -------------------------------------------------------------------


def test_full_board_draw_has_no_winner():
    # A full board verified (by exhaustive search) to contain no four-in-a-row.
    draw = [
        [1, 1, 1, 2, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1],
        [1, 1, 2, 1, 2, 1, 1],
        [2, 2, 2, 1, 2, 2, 2],
        [1, 1, 1, 2, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1],
    ]
    g = fresh()
    g.board = np.array(draw, dtype=np.int8)
    assert g.is_full()
    assert g.is_terminal()  # full board is terminal even with no winner
    assert not has_won(g.board, 1)
    assert not has_won(g.board, 2)
    assert g.result(1) == 0.0 and g.result(2) == 0.0


# -- clone independence -----------------------------------------------------


def test_clone_is_independent():
    g = fresh()
    g.play(3)
    h = g.clone()
    h.play(3)
    # Mutating the clone must not touch the original.
    assert g.board[1, 3] == 0
    assert h.board[1, 3] != 0


# -- center opening sanity (solved-game optimal first move) -----------------


def test_center_is_playable_first_move():
    g = fresh()
    assert 3 in g.legal_moves()  # center column, the proven optimal opening
    g.play(3)
    assert g.board[0, 3] == 1
