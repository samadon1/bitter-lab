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
