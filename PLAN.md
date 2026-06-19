# Bitter Lab — Connect Four as a demonstration of the Bitter Lesson + its efficiency counterweight

## Context

I'm learning ML systems from Harvard's *Machine Learning Systems* book (mlsysbook.ai) and I learn by building. Chapter 1 ("Introduction") has two spines:

1. **The bitter lesson** — general methods that scale with compute beat hand-engineered knowledge.
2. **The efficiency counterweight** — scale is not free. The *iron law* decomposes cost, *Return on Compute (RoC)* decides if an optimization is worth it, and "a better algorithm is not automatically a better system."

This project builds a Connect Four lab that makes both spines **measurable on a laptop**: a knowledge-free search agent overtakes a hand-coded expert as compute rises (bitter lesson), and then — under a tight compute budget — the cheap expert wins again (efficiency / RoC).

**Scope decision (locked):** Connect Four only. It is the best vehicle for the *scale + efficiency* half of the chapter. It structurally **cannot** show the data-centric / silent-degradation half (no shifting world, no dataset, compute-bound so no memory-wall term). The writeup will claim the first and not pretend to the second.

**Goal is learning, not shipping.** Every slice produces running code and a number, and each number is the hook for understanding a specific chapter concept. No more design docs — from here it's code + measurement.

## Working method

- Build in small vertical slices, smallest first. Finish and commit each slice before the next.
- Each slice ends with a **learning checkpoint**: a concept I should be able to explain in my own words, backed by a number I measured.
- Honesty rule for the demo: keep "search-efficiency tricks" (pruning, move ordering, bitboards) on the *compute* side of the ledger, and keep hand-tuned evaluation on the *knowledge* side. Never let knowledge leak into the "general method" agent.

## Stack

- Python 3.11+, NumPy, matplotlib, pytest. No ML frameworks until the optional RL slice.
- Engine and agents as importable modules (testable, reusable); experiments as scripts that emit plots + JSON.

## Repo layout

```
mlsys/
  engine.py            # board, move gen, win detection, terminal check (2D array first)
  agents.py            # RandomAgent, HeuristicAgent (knowledge), MCTSAgent (general method)
  elo.py               # win-rate -> Elo, with enough-games guidance
  tournament.py        # round-robin harness, returns results as data
  bitboard.py          # Slice 2: 64-bit board representation + bitwise win detection
  experiments/
    run_crossover.py   # Slice 1: Elo vs log2(compute), the crossover chart
    run_efficiency.py  # Slice 2: RoC curve, profiling, bitboard speedup, budget-flip
  data/                # precomputed JSON results (so plots are reproducible)
  tests/
    test_engine.py     # correctness tests for the engine
  README.md            # builder's quickstart (trimmed, written last)
```

---

## Slice 0 — The engine (no AI)

**Build:** Connect Four core in `engine.py`:
- 6x7 board as a NumPy array (start simple, not bitboard).
- `legal_moves(board)`, `play(board, col, player)`, `is_win(board, player)`, `is_terminal(board)`.
- A clean `GameState` API the agents will share.

**Test:** `tests/test_engine.py` — known wins (horizontal/vertical/both diagonals), full-column rejection, draw on full board, first-player center fact as a sanity anchor.

**Learning checkpoint:** This is **Software 1.0** — explicit, deterministic, exhaustively testable because the API-level input space is small. It is the honest baseline everything else is measured against. Be able to explain *why* this differs from the Software 2.0 systems the book describes (learned behavior, silent failure) — Connect Four is the control case.

**Done when:** all engine tests pass.

---

## Slice 1 — Two agents + the crossover (the keystone)

**Build in `agents.py`:**
- `HeuristicAgent` — genuine human knowledge: center preference (literally the solved-game optimal opening), reward own 2/3-in-a-rows, block opponent threats. Tune it until it is actually strong. A strawman here invalidates the whole demo.
- `MCTSAgent` — Monte Carlo Tree Search with **random rollouts**. Zero domain knowledge. Single knob: number of simulations per move.
- (Use MCTS-with-random-rollouts, *not* depth-limited minimax: a depth-limited minimax needs a leaf evaluator, and that evaluator is smuggled human knowledge that would contaminate the crossover.)

**Build `elo.py` + `tournament.py`:** round-robin between MCTS at increasing simulation counts and the heuristic; convert win rates to Elo. Use thousands of games per pairing (resolving a ~70 Elo gap needs ~500+ games).

**Experiment `run_crossover.py`:** plot Elo vs log2(simulations). Expect: at low compute the heuristic wins; past some threshold MCTS crosses over and pulls away.

**Learning checkpoint:** The **bitter lesson**, live. Be able to explain: search/simulations as a compute axis; why the curve *bends* (diminishing returns); Elo ↔ win-rate logistic; and why measurement needs many games (statistical noise). This chart is also the eventual LinkedIn artifact.

**Done when:** the crossover is visible and reproducible from saved JSON.

---

## Slice 2 — The efficiency layer (the book's second half)

**Measure, don't just count** (`run_efficiency.py`):
- Time/move and nodes/move per agent.
- **Return on Compute:** Elo gained per additional node. Plot it — watch it collapse toward zero. This is the chapter's "1% gain for 10x compute fails the RoC test," derived by me.
- **Profile** where time goes (tree search vs win-check). This is the **D·A·M diagnostic habit**: "which term dominates?" Honest framing — Connect Four is compute-bound, so the answer is the `Ops/(Peak·eta)` term, and I *show* that by profiling rather than assert it.

**Bitboard upgrade (`bitboard.py`):**
- Reimplement board as a 64-bit int (7-bits-per-column + sentinel row); win detection via bit shifts (1, 7, 6, 8).
- **Measure the speedup** vs the array engine. That delta is the `eta` (utilization) term and compute-efficiency made physical.
- **The efficiency paradox:** don't bank the speedup as savings — spend it on more simulations at the same wall-clock, and show the stronger play. ("Efficiency enables scale; scale demands efficiency.")

**The budget-flip (the synthesis):**
- Cap both agents at a fixed budget (e.g. ~10 ms/move or N nodes). Re-run the match.
- Expect the cheap `HeuristicAgent` to win again, because knowledge is compute-cheap and MCTS can't search deep enough in time.
- **Learning checkpoint:** this is the **deployment spectrum / TinyML / RoC** lesson and the **"better algorithm != better system"** fallacy, both on one board. Be able to explain: scale wins when compute is free (Sutton); knowledge can be the RoC-optimal choice at the edge (the book).

**Done when:** RoC curve, measured bitboard speedup, and the budget-flip result are all saved and plotted.

---

## Slice 3 — Self-play RL (AlphaZero-lite). OPTIONAL, hardest, most faithful to Sutton.

**Build:** a tiny policy+value network, MCTS-guided self-play on the same board. Knob = number of self-play games. Plot Elo vs log(training compute) — same diminishing-returns bend as Slice 1, reached by a second road (learning, not search). Cross-check: pit the trained agent against the Slice 1 heuristic.

**Learning checkpoint:** RL, self-play, the `r proportional to log(C)` scaling law — and the **first real Machine-axis story**: net inference dominates MCTS, batch size matters (the book's "batch < 32 wastes the accelerator" note), and CPU vs GPU timing finally becomes meaningful (a second iron-law term to play with).

**Note:** genuinely fiddly (self-play instability, catastrophic forgetting, replay buffer, MCTS sims >= 128). Only start after Slices 0-2 are done. Treat as a stretch goal.

---

## Honesty calibration (so claims stay defensible)

- Through Slice 2, Connect Four exercises **only the compute term** of the iron law. No data-movement/memory-bandwidth story. Claim the diagnostic *habit* (profile before optimizing), RoC, and the efficiency paradox — not the full three-term decomposition.
- The Machine axis and a second iron-law term only become real in the optional Slice 3.
- The data-drift / silent-failure spine of the chapter is explicitly out of scope.

## Verification

- `pytest tests/` — engine correctness (Slice 0 gate).
- `python experiments/run_crossover.py` — produces the Elo-vs-compute crossover plot + `data/crossover.json`; eyeball that the curve bends and MCTS overtakes the heuristic.
- `python experiments/run_efficiency.py` — produces RoC curve, prints the bitboard speedup factor, and reports the budget-flip winner; confirm the heuristic wins under the tight budget.
- Each experiment reads/writes JSON in `data/` so results are reproducible without re-running long tournaments.
