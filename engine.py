"""Connect Four engine — Slice 0.

This is deliberately *Software 1.0*: explicit, deterministic logic with a small,
enumerable input space. That is exactly why we can test it exhaustively (see
tests/test_engine.py) — something we can never do for a learned model. It is the
honest control case for the whole Bitter Lab project.

Board convention:
  - shape (ROWS, COLS) = (6, 7), dtype int8
  - row 0 is the BOTTOM of the board; pieces fall to the lowest empty row
  - 0 = empty, 1 = player one, 2 = player two
"""

from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
CONNECT = 4

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # horizontal, vertical, two diagonals


class GameState:
    """Mutable Connect Four position with a clone() for search."""

    __slots__ = ("board", "current_player", "winner", "last_move")

    def __init__(self, board: np.ndarray | None = None, current_player: int = 1):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8) if board is None else board
        self.current_player = current_player
        self.winner = 0  # 0 = no winner yet, otherwise the winning player
        self.last_move: tuple[int, int] | None = None

    # -- core API -----------------------------------------------------------

    def clone(self) -> "GameState":
        g = GameState(self.board.copy(), self.current_player)
        g.winner = self.winner
        g.last_move = self.last_move
        return g

    def legal_moves(self) -> list[int]:
        """Columns that are not yet full."""
        return [c for c in range(COLS) if self.board[ROWS - 1, c] == 0]

    def is_full(self) -> bool:
        return not bool((self.board[ROWS - 1] == 0).any())

    def play(self, col: int) -> "GameState":
        """Drop the current player's piece into `col`, mutating self.

        Sets self.winner if this move completes four-in-a-row, then switches
        the player to move. Returns self for chaining.
        """
        if not (0 <= col < COLS):
            raise ValueError(f"Column {col} out of range 0..{COLS - 1}")
        if self.board[ROWS - 1, col] != 0:
            raise ValueError(f"Column {col} is full")

        row = self._drop_row(col)
        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        if self._completes_win(row, col, self.current_player):
            self.winner = self.current_player

        self.current_player = 3 - self.current_player  # 1 <-> 2
        return self

    def is_terminal(self) -> bool:
        return self.winner != 0 or self.is_full()

    def result(self, player: int) -> float:
        """+1 if `player` won, -1 if they lost, 0 for draw / non-terminal draw."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # -- internals ----------------------------------------------------------

    def _drop_row(self, col: int) -> int:
        col_vals = self.board[:, col]
        # lowest empty row
        for r in range(ROWS):
            if col_vals[r] == 0:
                return r
        raise ValueError(f"Column {col} is full")  # unreachable given play() guard

    def _completes_win(self, r: int, c: int, player: int) -> bool:
        """Check only the lines through the just-played cell (fast path)."""
        for dr, dc in _DIRECTIONS:
            count = 1
            count += self._run(r, c, dr, dc, player)
            count += self._run(r, c, -dr, -dc, player)
            if count >= CONNECT:
                return True
        return False

    def _run(self, r: int, c: int, dr: int, dc: int, player: int) -> int:
        n = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == player:
            n += 1
            rr += dr
            cc += dc
        return n

    def __str__(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [
            " ".join(symbols[int(v)] for v in self.board[r]) for r in reversed(range(ROWS))
        ]
        return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(COLS))


def has_won(board: np.ndarray, player: int) -> bool:
    """Full-board scan for any four-in-a-row by `player`.

    Slower than GameState's incremental check, but simple and obviously correct —
    used by tests to verify arbitrary positions.
    """
    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != player:
                continue
            for dr, dc in _DIRECTIONS:
                if _is_line(board, r, c, dr, dc, player):
                    return True
    return False


def _is_line(board: np.ndarray, r: int, c: int, dr: int, dc: int, player: int) -> bool:
    for i in range(CONNECT):
        rr, cc = r + dr * i, c + dc * i
        if not (0 <= rr < ROWS and 0 <= cc < COLS) or board[rr, cc] != player:
            return False
    return True
