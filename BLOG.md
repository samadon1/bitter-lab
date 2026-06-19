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

**What I learned:**
- This engine is *Software 1.0*: explicit rules, deterministic, small input space.
  That's the whole reason I can test it exhaustively. A trained model is Software 2.0 —
  behavior compiled from data, input space too big to enumerate, so you can only sample
  it. That's the book's "verification gap." My engine is the one piece of this project I
  can actually *prove* correct, which makes it the honest control case: if a later agent
  misbehaves, the bug isn't in here.
- Small but real systems habit: I have two win-checkers. `is_win` (fast) only checks the
  lines through the cell just played; `has_won` (slow) scans the whole board. The tests
  use the obviously-correct slow one to validate the fast one. "Verify the optimized path
  against the simple path" — that pattern is going to come back in Slice 2 with bitboards.

**Surprise / gotcha:**
- My first "full board with no winner" test board was hand-crafted with 2×2 colour blocks.
  It failed — turns out 2×2 blocks *always* produce a diagonal four-in-a-row. Couldn't
  eyeball a valid draw position. Fix: wrote a 6-line backtracking search to *find* a
  verified draw board, then hard-coded that. Lesson that's funny in context — even on a
  trivial 6×7 grid, my human intuition was wrong and a dumb exhaustive search was right.
  The bitter lesson, in miniature, before I'd even written an agent.

**Artifact:** `pytest tests/` → 14 passed.
