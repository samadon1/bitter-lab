"""A faster Connect Four board, stored as numbers instead of a grid.

Same game, same rules, same public methods as engine.GameState - so the players
can use it without changing. The difference is purely speed:

  - The whole board is two integers (whose pieces are where), not a 42-cell array.
  - Copying a position is copying two numbers, which is what search does constantly.
  - "Did that move win?" is a handful of bit shifts instead of scanning the board.

How the bits are laid out:
  - Each column gets 7 bits: 6 for the real cells, plus 1 spare "gap" bit on top.
  - The gap bit is always left empty. It stops a line in one column from looking
    like it continues into the next column when we shift bits around.
  - Bit number = column * 7 + row, with row 0 at the bottom.

This bit trick is the standard, well-known way to store Connect Four. We check it
against the plain engine in tests/test_bitboard.py.
"""

from __future__ import annotations

ROWS = 6
COLS = 7
_H = ROWS + 1          # 7 bits per column (6 real rows + 1 gap)
_FULL_PLY = ROWS * COLS  # 42 moves fills the board


def _col_mask(col: int) -> int:
    """The 6 real-cell bits for one column."""
    return ((1 << ROWS) - 1) << (col * _H)


def _alignment(pos: int) -> bool:
    """True if `pos` (one player's pieces) contains any four in a row.

    Each check shifts the bits by the spacing between neighbours in a direction,
    ANDs to find adjacent pairs, then shifts/ANDs again to find four in a row.
      step 1 = up (within a column)   step 7 = across (to the next column)
      step 8 = the "/" diagonal        step 6 = the "\" diagonal
    """
    for step in (1, 7, 8, 6):
        m = pos & (pos >> step)
        if m & (m >> (2 * step)):
            return True
    return False


class BitBoard:
    """A Connect Four position with the same interface as engine.GameState."""

    __slots__ = ("current_position", "mask", "moves", "winner", "current_player")

    def __init__(self):
        self.current_position = 0   # pieces of the player whose turn it is
        self.mask = 0               # every occupied cell (both players)
        self.moves = 0              # how many pieces have been played
        self.winner = 0
        self.current_player = 1

    def clone(self) -> "BitBoard":
        b = BitBoard()
        b.current_position = self.current_position
        b.mask = self.mask
        b.moves = self.moves
        b.winner = self.winner
        b.current_player = self.current_player
        return b

    def legal_moves(self) -> list[int]:
        # A column has room unless its top real cell is taken.
        top = lambda c: 1 << (c * _H + ROWS - 1)
        return [c for c in range(COLS) if not (self.mask & top(c))]

    def is_full(self) -> bool:
        return self.moves == _FULL_PLY

    def is_terminal(self) -> bool:
        return self.winner != 0 or self.moves == _FULL_PLY

    def result(self, player: int) -> float:
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def play(self, col: int) -> "BitBoard":
        """Drop the current player's piece into a column."""
        mover = self.current_player

        # The single bit for the landing cell: adding 1 at the column bottom
        # carries up through any filled cells to the lowest empty one.
        move_bit = (self.mask + (1 << (col * _H))) & _col_mask(col)
        if move_bit == 0:
            raise ValueError(f"Column {col} is full")

        # Check a win for the mover before we flip whose turn it is.
        if _alignment(self.current_position | move_bit):
            self.winner = mover

        # Switch perspective to the other player, then record the new piece.
        self.current_position ^= self.mask
        self.mask |= move_bit
        self.moves += 1
        self.current_player = 2 if mover == 1 else 1
        return self
