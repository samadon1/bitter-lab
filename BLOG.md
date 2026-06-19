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

---

## 2026-06-19 — Slice 1: two agents and the crossover (the keystone)

**Goal:** put human knowledge and pure compute on the same board and find the point where
compute wins. This is the whole project in one chart.

**What I built:**

- `agents.py` — three players. `RandomAgent` (no knowledge, no compute — the floor, and the
  policy MCTS rolls out with). `HeuristicAgent` — human knowledge, ~1 ply only: take a win,
  block a loss, else pick the move with the best positional score (center control + scoring
  every length-4 window). `MCTSAgent` — pure compute: Monte Carlo Tree Search with *random*
  rollouts, knowing only the rules, with one knob — simulations per move.
- `tournament.py` — plays matches and, crucially, **alternates who moves first** every game.
  Connect Four's first player has a real edge, so without alternating I'd be measuring
  seating luck, not skill.
- `elo.py` — turns win rate into an Elo gap. I anchor the heuristic at 0 and read each MCTS
  config's strength as its rating relative to that. Crossover = where MCTS crosses 0.
- `experiments/run_crossover.py` — sweeps sims = 1…1024, 80 games each, writes
  `data/crossover.json` and `crossover.png`.

**MCTS in four phases (what I actually learned building it):** every "simulation" is one
loop of SELECT (descend the tree from the root, at each step picking the child with the best
UCB1 score — a formula that balances "this move has done well" against "this move is
under-explored"), EXPAND (add one new child for a move not yet tried), SIMULATE (from there,
play *uniformly random* moves to the end of the game), and BACKPROPAGATE (push the win/loss
back up the path, crediting each move). Do that `simulations` times, then play the
most-*visited* child — visits are a calmer signal of what the search trusts than the raw
average. The only knowledge in the whole thing is the rules. Strength comes entirely from
doing more loops. That loop count *is* the compute axis of the bitter lesson.

The single subtle part was bookkeeping the *perspective*: a node stores reward for the
player who moved *into* it, so when UCB1 at a parent maximizes a child's mean, it's
maximizing the parent-mover's own win rate. Get the sign wrong and the agent confidently
plays the opponent's best move. (It passed the "MCTS-100 beats random 40–0" check, which is
how I knew the signs were right.)

**The result:**

```
sims    winrate   Elo vs heuristic
   1       0.01      -759
   8       0.00      -920   (loses every game)
  64       0.28      -168
 128       0.36      -103
 256       0.61       +75   <- crossover: now beating the expert
 512       0.89      +359
1024       0.94      +470
```

A knowledge-free searcher that loses *every* game at 8 simulations becomes a 94%-winrate
monster at 1024, overtaking my genuinely-strong hand-coded expert somewhere around 256
simulations. Nothing about Connect Four was taught to it — I just let it think more. That's
the bitter lesson, on my laptop, in one curve.

**Self-test — what is the "compute" on my x-axis, and why does each doubling buy less?**
The x-axis is MCTS simulations per move: one unit = one full SELECT/EXPAND/SIMULATE/BACKPROP
loop, ending in a random playout to the end of the game. Doubling it doubles the imagined
futures the agent samples before committing. It buys less near the top (512→1024 added +111
Elo vs the +284 from 256→512) because skill rises with the *log* of compute, not linearly —
the honest, bitter shape of the lesson.

**Honest caveat I hit (and it's a real lesson):** measuring Elo against a *single fixed
opponent* saturates. Once MCTS wins ~94%, beating the heuristic harder barely moves the
number — not because more compute stopped helping, but because I've run out of headroom
against *this* yardstick. The clean way to see diminishing returns at the top is a *ladder*:
MCTS-N vs MCTS-2N, so the opponent scales with you. I'm noting it here and will add the
ladder in Slice 2 rather than overclaim the bend from one saturating curve. (Also why the
1→8 region looks flat-then-noisy: those configs lose ~every game, so the win rate floors and
the Elo clamps — same saturation, other end.)

**The cost, which sets up Slice 2:** the 1024-sim config took ~65 s for 80 games, roughly
0.8 s per move, almost all of it in `clone()` and random rollouts over the NumPy board. The
agent is strong because it computes a lot — and right now each unit of compute is expensive.
That's the perfect setup for the efficiency half: make each simulation cheaper (bitboard),
then ask whether the strength was worth the price (Return on Compute), and what happens when
I cap the clock (the budget flip).

**Artifacts:** `data/crossover.json`, `crossover.png`. Crossover ≈ 256 simulations.

---

## 2026-06-19 — Slice 2: thinking isn't free

**Goal:** Slice 1 proved more thinking wins. This slice asks the real-world follow-up: what
does that thinking *cost*, and when is it worth it? Four measurements.

**1) Where does the time go? (measure before optimizing.)** Profiled one MCTS move. Almost all
the time is inside the engine: the win-check (`_run`, `_completes_win`), `is_full`,
`legal_moves`, `play`. No surprise data-loading or overhead — it's pure per-position work. So
the thing to make cheaper is the board itself. That's the diagnostic habit: don't guess where
the cost is, look first, then optimize the part that actually dominates.

**2) The bitboard — make each unit of thinking cheaper.** Rewrote the board as two integers
with bit-shift win detection (`bitboard.py`), kept the exact same interface so the agents
didn't change. Validated it against the plain engine by playing 300 identical random games and
checking every step agrees — the "trust the fast version against the simple one" pattern,
exactly as promised in Slice 0. Result: **~2.6× faster**, both on raw random playouts and on
MCTS simulations per second. The win comes from cheap cloning (copy two numbers, not a 42-cell
array) and shift-based win checks; pure-Python loop overhead is what keeps it from being more.

**3) Diminishing returns — the ladder.** To see the bend honestly (Slice 1's curve saturated
against the fixed heuristic), I laddered MCTS against *itself*: N sims vs 2N sims, so the
opponent scales with me. Elo gained per doubling:

```
  64 vs   32   +120
 128 vs   64   +280
 256 vs  128   +83
 512 vs  256   +23
1024 vs  512   +0     <- doubling stops helping
```

After the noisy low end it's a clean decline, all the way to a *plateau*: going 512→1024 wins
exactly half its games, i.e. buys nothing. Interesting why: random rollouts are a weak signal,
and past a point more of them doesn't sharpen the estimate. The cap isn't compute, it's the
*quality* of each thought. (That's the door into Slice 3 — replacing random rollouts with a
learned value would push the plateau out.) This is Return on Compute made literal: the same
compute eventually returns zero, so "just scale it" has an end even here.

**4) The budget flip — when cheap knowledge wins again.** Put both players on a per-move clock
and varied the budget. MCTS win rate vs the instant 1-ply expert:

```
   5 ms (~33 sims)   0.16   expert wins
  10 ms (~65 sims)   0.18   expert wins
  20 ms (~131 sims)  0.56   thinker wins   <- the flip
  40 ms (~262 sims)  0.71   thinker wins
  80 ms (~523 sims)  0.86   thinker wins
```

Below ~20ms/move the cheap hand-coded expert wins; above it, the thinker takes over. Same two
players, opposite outcomes — the deciding variable is the compute budget. This is the book's
"a better algorithm is not automatically a better system": on a tight budget (think edge
device, milliwatts) cheap knowledge is the right call; with compute to spare (cloud) scale
wins. One board shows both halves.

**5) The efficiency paradox.** The 2.6× faster board doesn't *save* time — you spend it on more
thinking. At a 10ms budget the array board manages ~65 sims (loses); the bitboard would manage
~170 (much closer to the flip). So the speedup doesn't bank a cheaper status quo, it shifts the
flip leftward — efficiency lets scale win at a *tighter* budget. Efficiency enables scale;
scale then demands more efficiency.

**Self-test — when does scale win, when does cheap knowledge win, and what flips it?**
The per-move compute budget flips it. Plenty of compute → thinking (scale) wins; tight compute
→ cheap knowledge wins. And making compute more *efficient* moves the flip point, so the same
hardware buys you into the winning regime sooner. Slice 1 was "scale wins." Slice 2 is "...but
only if you can afford it, and affording it is an engineering problem."

**Artifacts:** `data/efficiency.json`, `efficiency_ladder.png`, `efficiency_budget.png`.
Bitboard ≈ 2.6× faster; budget flip ≈ 20 ms/move.

---

## 2026-06-19 — Slice 3: let it learn (self-play)

**Goal:** Slice 2 found a plateau — more random rollouts stopped helping because each
rollout is a weak, coin-flip signal. The fix Sutton would point at: don't hand the agent a
better signal, let it *learn* one from its own games. So: replace the random rollout with a
small value network trained purely on self-play outcomes, and watch strength grow with the
number of training games. No strategy is ever programmed in — only the rules and the results
of its own play.

**What I built (all additive — Slices 1 and 2 untouched):**
- `valuenet.py` — a tiny two-layer NumPy network (84 → 128 → 1, tanh output), backprop and
  Adam by hand. It answers one question about a position: "how good is this for the player to
  move?", from −1 to +1. I used NumPy, not a big library, on purpose: the net is called one
  position at a time deep inside the search loop, and at batch-size-1 a heavyweight library's
  per-call overhead would cost more than the maths. (That's the iron law again.)
- `agents.py` — `MCTSAgent` gained an optional `value_fn`. With it, the leaf is scored by the
  net instead of a random play-out. Without it, the old random-rollout agent is unchanged —
  and it's exactly the baseline I measure against.
- `selfplay.py` + `experiments/run_selfplay.py` — the agent plays itself, every position is
  labelled by who eventually won, the net trains on those labels, and the better net makes
  better games. Round and round.

**The honest journey (this is the real lesson):**

First runs were a mess. At ~600 self-play games and 80 sims, value-MCTS hovered around 0.35
win rate vs random-rollout MCTS at equal sims — i.e. the trained net was *losing* to plain
random play-outs, even though its prediction loss was clearly falling. Confusing: the net is
learning to predict outcomes, but it's not helping the search?

Rather than thrash, I ran one diagnostic: feed the search a *known-good* value (the Slice 1
heuristic's own positional score, as an oracle) and see if the machinery can win at all.
Result: oracle-value-MCTS beat random-MCTS **0.82** at equal 80 sims. So the plumbing was
correct — a good value clearly lifts the plateau. The problem was purely that my net wasn't
*accurate* enough yet. Random rollouts turn out to be a surprisingly strong baseline (a
play-out averages over many real future positions), and a weak value can be worse than that.

The fix was scale, which is the whole point. Bumping to 1200 self-play games, 120 sims for
cleaner outcome labels, and a bigger net:

```
self-play games   vs random-MCTS (equal 80 sims)   vs expert
        0 (raw)            0.20                        0.07
      100                  0.52
      400                  0.69
      600                  0.72
     1200                  0.69                        0.60
```

Now it works. The learned value starts *below* random rollouts (an untrained net is worse
than coin-flips), climbs past them by ~100 games, and settles around 0.65–0.72 — it beats
random rollouts at equal compute, lifting the Slice 2 plateau. And at 1200 games it beats the
Slice 1 hand-coded expert 0.60, at only 80 sims — where random-rollout MCTS needed ~256 sims
just to *tie* that expert. The agent taught itself, from nothing but its own games, to be
stronger per-unit-compute than both the random searcher and the hand-coded human knowledge.

**Self-test — why does the learning curve bend like the search curve, and where does the
compute go?** It bends because skill again rises with the *log* of effort: the first hundred
games take it from hopeless to even, and each later batch adds less (it flattens around 0.7).
Same diminishing-returns shape as Slice 1's search curve, reached by a different road —
*learning* instead of *searching*. The compute goes almost entirely into self-play game
generation (each move is a full MCTS search, and each search does many net evaluations), which
is why a faster board and a lean batch-1 net mattered. CPU only; ~7–11s per 100-game round.

**The meta-lesson, and why it's the most faithful slice to Sutton.** Slice 1: scale of search
beats knowledge. Slice 3: scale of *experience* beats both — but only at scale. The weak early
runs weren't a bug; they were the bitter lesson pointing at *learning itself*. A learned
component is only as good as its data and compute, and beating a strong baseline (random
rollouts) took real volume. The honest negative result plus the oracle control taught more
than a lucky clean win would have — including *why* full AlphaZero adds a policy head and
trains on millions of games (the value-only, few-hundred-games version is genuinely fiddly,
exactly as the plan warned).

**Artifacts:** `data/selfplay.json`, `data/valuenet.npz`, `selfplay_curve.png`.
Self-play value-MCTS ≈ 0.69 vs random-MCTS and 0.60 vs the expert at equal/low sims.
