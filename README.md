# Bitter Lab

A small Connect Four project for learning how machine-learning systems really work,
by building one and measuring it.

## The idea, in plain words

There are two ways to make a computer good at a game:

1. **Teach it what you know.** Write down good moves and strategy by hand.
2. **Let it think.** Give it only the rules and let it try lots of moves in its head.

A famous observation in AI (the "bitter lesson") is that, given enough computing power,
the second way wins - the thinker beats the expert. This project shows that happening on
a board you can watch, then asks the follow-up question that matters in the real world:
**thinking isn't free - is it worth the cost?** When you put both players on a strict
timer, the cheap expert can win again.

## What's here

| File | What it does |
|------|--------------|
| `engine.py` | The Connect Four game and its rules. No AI. |
| `agents.py` | The players: random, a hand-coded expert, and a "thinker" (MCTS). |
| `tournament.py` | Plays the players against each other and counts wins. |
| `elo.py` | Turns win rates into a single skill number. |
| `experiments/run_crossover.py` | The main experiment + chart. |
| `tests/` | Checks the game rules are correct. |
| `BLOG.md` | A running build log - what I built, learned, and got surprised by. |
| `PLAN.md` | The step-by-step plan for the whole project. |

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pytest

pytest tests/                       # check the game rules
python experiments/run_crossover.py # run the experiment, make the chart
```

## What it shows so far

Give the thinker more thinking time and it goes from losing every game to winning almost
every game, overtaking the hand-coded expert. That's the chart in `crossover.png`.

Still to come: make each unit of thinking cheaper and measure the speedup, then put both
players on a timer and watch the cheap expert win again.
