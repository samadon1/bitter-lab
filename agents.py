"""Connect Four agents — Slice 1.

Three players, sorted onto the two sides of the bitter-lesson ledger:

  - RandomAgent   : no knowledge, no compute. Baseline floor + the rollout policy MCTS uses.
  - HeuristicAgent: HUMAN KNOWLEDGE. Hand-coded tactics + positional scoring, ~1 ply only,
                    so it is strong but compute-cheap. This is the "expert".
  - MCTSAgent     : PURE COMPUTE. Monte Carlo Tree Search with *random* rollouts. Knows only
                    the rules. Its single knob is the simulation count. This is the "method".

Honesty rule (see PLAN.md): the heuristic's evaluation is knowledge and stays out of MCTS;
MCTS's only lever is how much it searches. Nothing about Connect Four strategy leaks into it.
"""

from __future__ import annotations

import math
import random

from engine import COLS, GameState, has_won

CENTER = COLS // 2


# ---------------------------------------------------------------------------
# Agent 0 — Random (the floor, and MCTS's rollout policy)
# ---------------------------------------------------------------------------

class RandomAgent:
    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = random.Random(seed)
        self.name = name

    def select_move(self, state: GameState) -> int:
        return self.rng.choice(state.legal_moves())


# ---------------------------------------------------------------------------
# Agent 1 — Heuristic (human knowledge, compute-cheap)
# ---------------------------------------------------------------------------

def _score_window(window: list[int], me: int) -> int:
    """Score one length-4 line for `me`. Classic Connect Four heuristic."""
    opp = 3 - me
    mine = window.count(me)
    theirs = window.count(opp)
    empty = window.count(0)

    if mine == 4:
        return 100
    if mine == 3 and empty == 1:
        return 5
    if mine == 2 and empty == 2:
        return 2
    if theirs == 3 and empty == 1:
        return -4  # an open enemy threat is bad
    return 0


def evaluate(board, me: int) -> int:
    """Positional value of `board` for `me`: center control + all 4-windows."""
    score = 0
    # Center column control — central pieces touch the most lines (and the
    # center opening is the solved-game optimal first move).
    center_col = [int(board[r, CENTER]) for r in range(board.shape[0])]
    score += center_col.count(me) * 3

    rows, cols = board.shape
    # Horizontal, vertical, and both diagonal windows of length 4.
    for r in range(rows):
        for c in range(cols):
            if c + 3 < cols:
                score += _score_window([int(board[r, c + i]) for i in range(4)], me)
            if r + 3 < rows:
                score += _score_window([int(board[r + i, c]) for i in range(4)], me)
            if r + 3 < rows and c + 3 < cols:
                score += _score_window([int(board[r + i, c + i]) for i in range(4)], me)
            if r + 3 < rows and c - 3 >= 0:
                score += _score_window([int(board[r + i, c - i]) for i in range(4)], me)
    return score


class HeuristicAgent:
    """Human expertise, ~1 ply of search.

    Decision order: take an immediate win; else block an immediate enemy win;
    else play the move that maximizes the positional score. It is deliberately
    shallow — knowledge is meant to be *cheap* (that matters for Slice 2's budget
    flip), so it never searches deep.
    """

    def __init__(self, seed: int | None = None, name: str = "heuristic"):
        self.rng = random.Random(seed)
        self.name = name

    def select_move(self, state: GameState) -> int:
        me = state.current_player
        opp = 3 - me
        legal = state.legal_moves()

        # 1) Win now if we can.
        for m in legal:
            nxt = state.clone()
            nxt.play(m)
            if nxt.winner == me:
                return m

        # 2) Block an immediate enemy win: a column where, if the opponent
        #    dropped, they'd make four. Playing it ourselves denies that cell.
        for m in legal:
            probe = state.clone()
            row = probe._drop_row(m)
            probe.board[row, m] = opp
            if has_won(probe.board, opp):
                return m

        # 3) Otherwise, maximize positional score one ply ahead.
        best_score = -math.inf
        best_moves: list[int] = []
        for m in legal:
            nxt = state.clone()
            nxt.play(m)
            s = evaluate(nxt.board, me)
            if s > best_score:
                best_score = s
                best_moves = [m]
            elif s == best_score:
                best_moves.append(m)
        return self.rng.choice(best_moves)


# ---------------------------------------------------------------------------
# Agent 2 — MCTS (pure compute, zero knowledge)
# ---------------------------------------------------------------------------

class _Node:
    """One node in the search tree = one game position."""

    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "value")

    def __init__(self, state: GameState, parent: "_Node | None", move: int | None):
        self.state = state
        self.parent = parent
        self.move = move  # the move (from parent) that produced this position
        self.children: list[_Node] = []
        self.untried: list[int] = state.legal_moves() if not state.is_terminal() else []
        self.visits = 0
        self.value = 0.0  # summed reward for the player who MOVED INTO this node


def _reward(winner: int, mover: int) -> float:
    """Rollout reward from `mover`'s perspective: win=1, draw=0.5, loss=0."""
    if winner == 0:
        return 0.5
    return 1.0 if winner == mover else 0.0


class MCTSAgent:
    """Monte Carlo Tree Search with random rollouts.

    Four phases per simulation:
      SELECT     — from the root, descend by UCB1 to a not-fully-expanded node.
      EXPAND     — add one new child for an untried move.
      SIMULATE   — play uniformly random moves from there until the game ends.
      BACKPROP   — push the result back up, crediting each move's mover.

    The only knowledge it has is the rules of the game. Its only knob is
    `simulations`: how many SELECT/EXPAND/SIMULATE/BACKPROP loops it runs per move.
    That count IS the compute axis of the bitter lesson.
    """

    def __init__(self, simulations: int, c: float = math.sqrt(2),
                 seed: int | None = None, name: str | None = None):
        self.simulations = simulations
        self.c = c  # UCB1 exploration constant
        self.rng = random.Random(seed)
        self.name = name or f"mcts-{simulations}"

    def select_move(self, state: GameState) -> int:
        root = _Node(state.clone(), parent=None, move=None)

        for _ in range(self.simulations):
            node = self._select(root)
            node = self._expand(node)
            winner = self._simulate(node.state)
            self._backprop(node, winner)

        # Robust choice: the most-visited child (not the highest mean — visits
        # are a less noisy signal of which move the search actually trusts).
        best = max(root.children, key=lambda ch: ch.visits)
        return best.move

    # -- phase 1: selection (UCB1) -----------------------------------------
    def _select(self, node: _Node) -> _Node:
        while not node.state.is_terminal():
            if node.untried:
                return node  # not fully expanded -> expand here
            node = self._best_uct_child(node)
        return node

    def _best_uct_child(self, node: _Node) -> _Node:
        log_n = math.log(node.visits)
        best, best_score = None, -math.inf
        for ch in node.children:
            exploit = ch.value / ch.visits          # mean reward for this move
            explore = self.c * math.sqrt(log_n / ch.visits)
            score = exploit + explore
            if score > best_score:
                best, best_score = ch, score
        return best

    # -- phase 2: expansion -------------------------------------------------
    def _expand(self, node: _Node) -> _Node:
        if not node.untried:
            return node  # terminal node; nothing to expand
        move = node.untried.pop(self.rng.randrange(len(node.untried)))
        child_state = node.state.clone()
        child_state.play(move)
        child = _Node(child_state, parent=node, move=move)
        node.children.append(child)
        return child

    # -- phase 3: simulation (random rollout) ------------------------------
    def _simulate(self, state: GameState) -> int:
        rollout = state.clone()
        while not rollout.is_terminal():
            rollout.play(self.rng.choice(rollout.legal_moves()))
        return rollout.winner

    # -- phase 4: backpropagation ------------------------------------------
    def _backprop(self, node: _Node | None, winner: int) -> None:
        while node is not None:
            node.visits += 1
            if node.move is not None:
                # the player who moved INTO this node is the one to move at its parent
                mover = node.parent.state.current_player
                node.value += _reward(winner, mover)
            node = node.parent
