"""The Connect Four game: the board and its rules.

No AI here, just the plain rules of the game. Because the game always behaves the
same way, we can test every rule completely (see tests/test_engine.py). That makes
this the one part of the project we can fully trust.

The board:
  - 6 rows, 7 columns
  - row 0 is the BOTTOM; a dropped piece falls to the lowest empty row
  - 0 = empty cell, 1 = player one, 2 = player two
"""


from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
CONNECT = 4  # how many in a row you need to win

# The four directions a winning line can run: right, up, and the two diagonals.
# Each pair is (row step, column step). We only need four because a line and its
# reverse are the same line; we just look both ways from a cell.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class GameState:
    """A single Connect Four position: the board plus whose turn it is."""

    __slots__ = ("board", "current_player", "winner", "last_move")

    def __init__(self, board: np.ndarray | None = None, current_player: int = 1):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8) if board is None else board
        self.current_player = current_player
        self.winner = 0  # 0 means nobody has won yet
        self.last_move: tuple[int, int] | None = None

    # -- the main things you can do with a position -------------------------

    def clone(self) -> "GameState":
        """Make a separate copy you can experiment on without changing this one.

        Search agents rely on this: they try moves on a copy, then throw it away.
        """
        g = GameState(self.board.copy(), self.current_player)
        g.winner = self.winner
        g.last_move = self.last_move
        return g

    def legal_moves(self) -> list[int]:
        """Which columns still have room. A column is full when its top cell is taken."""
        return [c for c in range(COLS) if self.board[ROWS - 1, c] == 0]

    def is_full(self) -> bool:
        """True when there are no empty cells left in the top row (board is full)."""
        return not bool((self.board[ROWS - 1] == 0).any())

    def play(self, col: int) -> "GameState":
        """Drop the current player's piece into a column.

        Updates the board, records a win if this move makes four in a row, then
        hands the turn to the other player. Returns itself so calls can chain.
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

        self.current_player = 3 - self.current_player  # flips 1->2 and 2->1
        return self

    def is_terminal(self) -> bool:
        """True when the game is over: someone won, or the board filled up."""
        return self.winner != 0 or self.is_full()

    def result(self, player: int) -> float:
        """Outcome from one player's point of view: +1 win, -1 loss, 0 draw."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # -- helpers (used internally) -----------------------------------------

    def _drop_row(self, col: int) -> int:
        """Find the lowest empty row in a column. That's where gravity puts the piece."""
        col_vals = self.board[:, col]
        for r in range(ROWS):
            if col_vals[r] == 0:
                return r
        raise ValueError(f"Column {col} is full")  # play() already prevents this

    def _completes_win(self, r: int, c: int, player: int) -> bool:
        """Did the piece just placed at (r, c) make four in a row?

        We only need to look at lines passing through that one cell, because it's
        the only thing on the board that changed.
        """
        for dr, dc in _DIRECTIONS:
            count = 1  # the piece we just placed
            count += self._run(r, c, dr, dc, player)    # same colour one way
            count += self._run(r, c, -dr, -dc, player)  # ...and the other way
            if count >= CONNECT:
                return True
        return False

    def _run(self, r: int, c: int, dr: int, dc: int, player: int) -> int:
        """Count how many of `player`'s pieces sit in a row starting next to (r, c)."""
        n = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == player:
            n += 1
            rr += dr
            cc += dc
        return n

    def __str__(self) -> str:
        # Print with the top row first so it looks like a real upright board.
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [
            " ".join(symbols[int(v)] for v in self.board[r]) for r in reversed(range(ROWS))
        ]
        return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(COLS))


def has_won(board: np.ndarray, player: int) -> bool:
    """Check the whole board for any four-in-a-row by `player`.

    This is the slow, obvious version. The tests use it to double-check the fast
    version above: if they ever disagree, the fast one has a bug.
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
    """Are there four of `player`'s pieces in a row starting at (r, c)?"""
    for i in range(CONNECT):
        rr, cc = r + dr * i, c + dc * i
        if not (0 <= rr < ROWS and 0 <= cc < COLS) or board[rr, cc] != player:
            return False
    return True
