# Bitter Lab — build log

A running log of building Connect Four to *feel* the bitter lesson and its efficiency
counterweight (from the Harvard ML Systems book). Raw notes, not a polished post —
what I built, what I learned, and what surprised me. The clean writeup gets distilled
from this at the end.

The thesis in one line: a knowledge-free searcher should overtake my hand-coded expert
as I give it more compute (the bitter lesson) — and then lose again the moment I put it
on a tight time budget (the efficiency counterweight). Both on one board.

---

## 2026-06-19 — Slice 0: the engine

**Goal:** a correct, deterministic Connect Four. No AI yet. This is the baseline
everything else is measured against.

**Built:**
- `engine.py` — board with gravity, win detection in all 4 directions, terminal/draw
  detection, `clone()` for search.
- `tests/test_engine.py` — 14 tests. All green.

**Self-test — why can I write exhaustive tests for this engine, but not for a trained model?**

Because this engine is *Software 1.0* — explicit, deterministic logic over a small,
enumerable input space. Same input always gives the same output, so a finite test suite
can pin down its behavior completely. A learned model (Software 2.0) is the opposite: its
behavior is "compiled" from data, the input space is astronomically large, and you can
only ever *sample* it — that's the book's **verification gap**. This is exactly why the
engine is the honest *control case* for the whole project: it's the one component I can
prove correct, so when MCTS or a self-play net behaves weirdly later, I know the bug isn't
here.

**One concrete thing in the code worth noticing:** `is_win` during play only checks the
four lines through the *just-played cell* (fast), while `has_won` does a full-board scan
(obviously correct). The tests use the slow-but-clear one to verify the fast one — a small
instance of "test the optimized path against the simple path" that recurs all through ML
systems work. It comes back in Slice 2 when the bitboard replaces the array engine: same
trick, validate fast against simple.

**Surprise / gotcha:**
- My first "full board with no winner" test board was hand-crafted with 2×2 colour blocks.
  It failed — turns out 2×2 blocks *always* produce a diagonal four-in-a-row. Couldn't
  eyeball a valid draw position. Fix: wrote a 6-line backtracking search to *find* a
  verified draw board, then hard-coded that. Lesson that's funny in context — even on a
  trivial 6×7 grid, my human intuition was wrong and a dumb exhaustive search was right.
  The bitter lesson, in miniature, before I'd even written an agent.

**Artifact:** `pytest tests/` → 14 passed.
