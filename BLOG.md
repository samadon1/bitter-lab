# Bitter Lab — build log

A running log of building Connect Four to *feel* the bitter lesson and its efficiency
counterweight, the two ideas at the heart of the Harvard ML Systems book. These are raw
notes written as I go — what I built, what I actually learned, and what surprised me — not
a polished post. The clean writeup gets distilled from this at the end. Keeping it honest
and complete here is the point; the gotchas are the part worth remembering.

**The thesis in one line:** a knowledge-free searcher should overtake my hand-coded expert
as I give it more compute (the bitter lesson) — and then lose again the moment I put it on
a tight time budget (the efficiency counterweight). Both shown on one small board.

---

## What is Connect Four, and why build the whole project on it?

Connect Four is a two-player game on a vertical grid, **7 columns wide and 6 rows tall** —
42 slots. It stands upright like a little plastic rack. The rules are complete in four
lines:

- Players alternate turns. One is "X", the other "O" (player 1 and player 2 in the code).
- On your turn you choose a **column** and drop a disc in. **Gravity** pulls it to the
  lowest empty slot in that column — you pick the column, the stack decides the row.
- You **win** the instant you get **four of your discs in a line**: horizontal, vertical,
  or either diagonal.
- If all 42 slots fill with no line of four, it's a **draw**.

A child learns it in thirty seconds. That simplicity is exactly why it's the right vehicle
for this project, for three concrete reasons:

1. **Small but not trivial.** At most 7 moves are available per turn (the "branching
   factor"). That keeps the tree of possible futures small enough to search on a laptop,
   yet the game is deep enough that thinking further ahead genuinely wins more. That's the
   precondition for showing "more compute → stronger play" as a clean, measurable curve.
2. **It's a *solved* game.** In 1988 it was proven that the first player can force a win
   with perfect play, and the optimal opening move is the **center column**. This matters
   twice. First, my hand-coded heuristic's "prefer the center" rule isn't a cheap hack —
   it's literally optimal opening theory, so the knowledge agent is legitimately strong,
   not a strawman I set up to lose. Second, "perfect play" exists as a north star to
   reason against.
3. **No hidden information, no luck.** Both players see the whole board; nothing is random.
   So when one agent beats another, it's purely decision quality — exactly the thing I want
   to isolate when I ask "did more compute make it play better?"

In the code, that upright grid is a 6×7 NumPy array in `engine.py`; "drop a disc" is
`play(col)`; gravity is `_drop_row`, which finds the lowest empty slot; and "four in a row"
is the win check.

---

## 2026-06-19 — Slice 0: the engine

**Goal:** a correct, deterministic Connect Four with no AI in it at all. This is the
baseline that everything else gets measured against, so it has to be bulletproof first.

**What I built:**

- `engine.py` — a `GameState` holding the board and whose turn it is, with: `legal_moves()`
  (which columns aren't full), `play(col)` (drop a disc with gravity, detect a win, switch
  player), `is_terminal()` / `result()` (win, loss, or draw), and `clone()` (a deep copy the
  search agent will lean on heavily next slice, because MCTS needs to explore hypothetical
  futures without wrecking the real board).
- `tests/test_engine.py` — 14 tests covering wins in all four directions, illegal moves
  (full column, out of range), pieces stacking from the bottom, a verified draw board, and
  that a clone is truly independent of its parent. All green.

**Self-test — why can I write exhaustive tests for this engine, but not for a trained
model?**

Because this engine is *Software 1.0*: explicit, deterministic logic over a small,
enumerable input space. The same input always produces the same output, so a finite test
suite can pin its behavior down completely. A learned model is *Software 2.0* — the
opposite. Its behavior is "compiled" from data rather than written by hand, the input space
is astronomically large (think every possible image, or every possible board history), and
you can only ever *sample* it. You can never test it exhaustively. That unfixable distance
between what you can test and what the model will actually encounter is what the book calls
the **verification gap**, and it's why ML systems rely on *monitoring in production* instead
of proof before deployment.

This is the reason the engine is the honest **control case** for the entire project. It's
the one component I can actually prove correct. So when MCTS or a self-play network starts
behaving strangely in a later slice — and it will — I'll know with certainty that the bug
is in the agent, not in the rules of the game underneath it. You want at least one part of
any ML system you can fully trust; here, this is it.

**A concrete detail in the code worth noticing:** I have two different win-checkers. The
fast one, `is_win`, runs during every move and only checks the four lines passing through
the cell that was *just played* — because that's the only cell that could have created a
new win, it's cheap. The slow one, `has_won`, scans the entire board and is obviously,
boringly correct. The tests use the slow-but-clear one to validate the fast-but-clever one.
"Verify the optimized path against the simple reference path" is a habit that runs through
all of ML systems work, and it comes back almost immediately: in Slice 2 the bitboard
replaces the array engine for speed, and I'll validate that exactly the same way — fast
implementation checked against the simple one.

**Surprise / gotcha — the bitter lesson showed up before I'd written a single agent.**

My first attempt at a "full board with no winner" test used a hand-crafted pattern of 2×2
colour blocks, which I was sure contained no four-in-a-row. The test failed: 2×2 blocks
*always* produce a diagonal four (step diagonally and you stay inside the same coloured
block long enough to make a line). I genuinely could not eyeball a valid draw position on a
trivial 6×7 grid. The fix was to write a six-line backtracking search that simply *tries*
colours cell by cell until it finds a full board with no win, then I hard-coded that
verified board into the test.

The lesson is funny given the subject: even on the smallest possible problem, my human
intuition was confidently wrong, and a dumb exhaustive search was right. That is the bitter
lesson in a nutshell — and I hit it before the project's actual agents even existed.

**Artifact:** `pytest tests/` → 14 passed. Committed as Slice 0.

### Engine walkthrough (`engine.py`, line by line)

Reading my own engine back, section by section, to make sure the foundation is actually
understood and not just passing tests. The design *why* matters more than the syntax —
especially the parts that look like over-engineering on a toy but are really early
rehearsals of the iron law.

**Header + constants (`engine.py:1`, `:18`, `:22`).** Three named constants — `ROWS=6`,
`COLS=7`, `CONNECT=4` — instead of magic numbers, so the whole engine retargets to another
board size by editing three lines, and every loop bound reads as what it means.
`_DIRECTIONS = ((0,1),(1,0),(1,1),(1,-1))` is `(row-delta, col-delta)` steps: right, up,
up-right, up-left. Only *four* directions, not eight, because a line and its reverse are
the same line — I walk both ways from a cell, so I never need the negatives. This one tuple
is the single source of truth for "what a line looks like," shared by both win-checkers.

**The class + `__slots__` (`engine.py:25`, `:28`).** `GameState` holds the board, whose turn
it is, the winner (0 = nobody yet), and the last move. `__slots__` is the first deliberate
systems choice: normally every Python object carries a `__dict__` hash map so you can attach
arbitrary attributes; `__slots__` declares the only four fields and drops that dict — less
memory, faster attribute access. Pointless for one object, decisive for the hundreds of
thousands of `GameState`s that MCTS will spawn per move in Slice 1. That's the iron law
arriving early: when the arithmetic is trivial, *overhead* dominates, and overhead is the
thing you optimize.

**`clone()` (`engine.py:38`).** Makes an independent copy — `self.board.copy()` plus the
scalars. This is the method search lives on: MCTS explores imagined futures by cloning the
state, playing into the copy, and discarding it, never touching the real game. `board.copy()`
is also the single most-run expensive line in the file. Foreshadow for Slice 2: the bitboard
makes a board one 64-bit integer, so "clone" becomes copying *one number* instead of a
42-element array — a concrete piece of the speedup I'll measure.

**`legal_moves()` + `is_full()` (`engine.py:44`).** Both ride one trick: a column is full iff
its *top* cell (`board[ROWS-1, c]`) is occupied. So legal moves keep every column whose top
is still empty, and `is_full` asks whether the top row has any empty cell left, using NumPy
vectorization (`(board[ROWS-1] == 0).any()`). No need to scan whole columns.

**`play()` — the heart (`engine.py:51`).** Four steps: (1) *guards* reject out-of-range and
full columns with a loud `ValueError` — the deterministic-failure luxury of Software 1.0;
(2) *drop* finds the landing row via gravity and writes the piece; (3) *win check* runs only
around the cell just placed; (4) *switch player* with `3 - current_player`, a branchless
toggle (3−1=2, 3−2=1, no `if`). It mutates and returns `self`, so moves chain.

**`is_terminal()` + `result()` (`engine.py:72`).** `result(player)` is *perspective-relative*:
+1 if that player won, −1 if they lost, 0 for a draw. One sign convention keeps search code
honest — every agent reasons "good *for me*."

**`_drop_row()` (`engine.py:83`).** Returns the lowest empty row in a column. That's gravity.
The trailing unreachable `raise` is a defensive assertion so the function is correct even if
called directly, bypassing `play`'s guard.

**`_completes_win()` + `_run()` — the fast win check (`engine.py:91`).** The one real insight
in the file: placing a piece can only create a four-in-a-row that *passes through that piece*.
So don't scan the board — only inspect lines through `(r,c)`. For each direction, start
`count = 1` (the new piece), then `_run` forward and backward counting same-colored pieces;
≥ 4 means a win. At most ~3 steps each way, versus 42 cells for a full scan — a locality
optimization that pays off precisely because it runs inside a deep search.

**`__str__()` (`engine.py:110`).** Prints with `reversed(range(ROWS))` so row 5 is on top and
row 0 on the bottom, matching the physical board (gravity falls down). Pure debugging
affordance, but I'll be glad it's there in Slice 1.

**`has_won()` + `_is_line()` — the reference checker (`engine.py:118`).** The deliberately
dumb one: scan every cell, and from each, check four-in-a-row in any direction. Slower but
obviously correct. Its whole job is to *validate* the clever incremental check in the tests.
This is the "trust an optimization by checking it against a simple reference" pattern — and
the exact template for how I'll validate the Slice 2 bitboard against this array engine.

**Three things to carry forward:** `__slots__` and `clone()` exist because search allocates
state at massive volume (iron-law overhead); the incremental win check is a locality
optimization; and the two-checker pattern is *how you earn trust in an optimization*. All
three come back in Slice 2.
