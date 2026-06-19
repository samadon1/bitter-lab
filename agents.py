"""The players.

Three ways to play Connect Four:

  - RandomAgent    : just picks a legal column at random. The weakest baseline, and
                     also the "coin-flip" policy MCTS uses when it plays out a game.
  - HeuristicAgent : plays well because it KNOWS good moves. Hand-written rules:
                     win if you can, block the opponent, otherwise favour strong
                     squares. It barely looks ahead, so it's quick.
  - MCTSAgent      : plays well because it THINKS A LOT. It knows only the rules,
                     not any strategy, and gets better purely by trying more moves
                     in its head. The "think a lot" amount is its only setting.

The whole experiment compares these two ways of being good: knowing vs. thinking.
We keep them separate on purpose - the heuristic's know-how never leaks into MCTS,
so MCTS's only advantage is raw effort.
"""

from __future__ import annotations

import math
import random
import time

from engine import COLS, GameState, has_won

CENTER = COLS // 2


# ---------------------------------------------------------------------------
# Random: the weakest player (and MCTS's playout policy)
# ---------------------------------------------------------------------------

class RandomAgent:
    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = random.Random(seed)
        self.name = name

    def select_move(self, state: GameState) -> int:
        return self.rng.choice(state.legal_moves())


# ---------------------------------------------------------------------------
# Heuristic: plays well because it knows good moves (and it's cheap)
# ---------------------------------------------------------------------------

def _score_window(window: list[int], me: int) -> int:
    """Give a score to one group of 4 cells, from `me`'s point of view.

    More of my pieces (with room to finish the line) is good; an almost-complete
    enemy line is bad.
    """
    opp = 3 - me
    mine = window.count(me)
    theirs = window.count(opp)
    empty = window.count(0)

    if mine == 4:
        return 100          # already a win
    if mine == 3 and empty == 1:
        return 5            # one away from winning
    if mine == 2 and empty == 2:
        return 2            # building
    if theirs == 3 and empty == 1:
        return -4           # enemy is one away - dangerous
    return 0


def evaluate(board, me: int) -> int:
    """Rate a whole board for `me`: reward central pieces and strong lines."""
    score = 0
    # Central pieces touch more possible lines, so they're worth more.
    center_col = [int(board[r, CENTER]) for r in range(board.shape[0])]
    score += center_col.count(me) * 3

    rows, cols = board.shape
    # Look at every group of 4 cells in a row: across, up, and both diagonals.
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
    """Picks a move using simple know-how, looking only one move ahead.

    Order of decisions: win right now if possible; otherwise stop the opponent
    from winning next turn; otherwise play the move that leaves the best-looking
    board. It stays shallow on purpose - know-how is supposed to be cheap, which
    matters later when we put both players on a time limit.
    """

    def __init__(self, seed: int | None = None, name: str = "heuristic"):
        self.rng = random.Random(seed)
        self.name = name

    def select_move(self, state: GameState) -> int:
        me = state.current_player
        opp = 3 - me
        legal = state.legal_moves()

        # 1) If a move wins immediately, take it.
        for m in legal:
            nxt = state.clone()
            nxt.play(m)
            if nxt.winner == me:
                return m

        # 2) If the opponent could win next turn in some column, play there to
        #    block it (our piece takes the cell they needed).
        for m in legal:
            probe = state.clone()
            row = probe._drop_row(m)
            probe.board[row, m] = opp
            if has_won(probe.board, opp):
                return m

        # 3) Otherwise pick the move that leaves the highest-scoring board.
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
        return self.rng.choice(best_moves)  # break ties randomly


# ---------------------------------------------------------------------------
# MCTS: plays well because it thinks a lot (knows only the rules)
# ---------------------------------------------------------------------------

class _Node:
    """One spot in the "what if" tree - a single imagined position."""

    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "value")

    def __init__(self, state: GameState, parent: "_Node | None", move: int | None):
        self.state = state
        self.parent = parent
        self.move = move  # the move that led here from the parent position
        self.children: list[_Node] = []
        self.untried: list[int] = state.legal_moves() if not state.is_terminal() else []
        self.visits = 0      # how many times we've explored through here
        self.value = 0.0     # total score for the player who made the move into here


def _reward(winner: int, mover: int) -> float:
    """Score a finished game for one player: win = 1, draw = 0.5, loss = 0."""
    if winner == 0:
        return 0.5
    return 1.0 if winner == mover else 0.0


class MCTSAgent:
    """Monte Carlo Tree Search: get good by trying lots of moves in your head.

    To choose a move, repeat this many times (that count is the only setting):

      1. WALK DOWN  - from the current position, keep choosing the most promising
                      next move until you reach one you haven't tried before.
      2. TRY IT     - add that new move to the tree.
      3. PLAY OUT   - from there, finish the game with random moves and see who wins.
      4. WALK BACK  - send that result back up, updating the score of every move
                      you took along the way.

    "Most promising" balances two things (a formula called UCB1): moves that have
    been winning, and moves you haven't tried much yet. After all the repeats, play
    the move you explored the most - that's the one the search trusts most.

    It's told nothing about Connect Four strategy. It only gets stronger by doing
    more repeats. That number of repeats is the "compute" we turn up.
    """

    def __init__(self, simulations: int = 0, c: float = math.sqrt(2),
                 time_budget: float | None = None,
                 seed: int | None = None, name: str | None = None):
        # Set EITHER a fixed number of simulations, OR a time budget (seconds)
        # to think as much as fits in that time. Time budget is for Slice 2's
        # "put both players on a clock" experiment.
        self.simulations = simulations
        self.time_budget = time_budget
        self.c = c                       # how much to favour unexplored moves
        self.rng = random.Random(seed)
        self.last_sims = 0               # how many sims the last move actually ran
        if name:
            self.name = name
        elif time_budget is not None:
            self.name = f"mcts-{time_budget * 1000:.0f}ms"
        else:
            self.name = f"mcts-{simulations}"

    def select_move(self, state: GameState) -> int:
        root = _Node(state.clone(), parent=None, move=None)

        def one_simulation():
            node = self._select(root)            # 1. walk down
            node = self._expand(node)            # 2. try a new move
            winner = self._simulate(node.state)  # 3. play out randomly
            self._backprop(node, winner)         # 4. walk back, update scores

        if self.time_budget is not None:
            end = time.perf_counter() + self.time_budget
            n = 0
            while time.perf_counter() < end:
                one_simulation()
                n += 1
            self.last_sims = n
        else:
            for _ in range(self.simulations):
                one_simulation()
            self.last_sims = self.simulations

        # Play the move we explored most often.
        best = max(root.children, key=lambda ch: ch.visits)
        return best.move

    # 1. walk down to a position with an untried move
    def _select(self, node: _Node) -> _Node:
        while not node.state.is_terminal():
            if node.untried:
                return node
            node = self._best_uct_child(node)
        return node

    def _best_uct_child(self, node: _Node) -> _Node:
        """Pick the child that best balances "winning so far" vs "rarely tried"."""
        log_n = math.log(node.visits)
        best, best_score = None, -math.inf
        for ch in node.children:
            winning = ch.value / ch.visits                  # how well it's done
            unexplored = self.c * math.sqrt(log_n / ch.visits)  # bonus for being rare
            score = winning + unexplored
            if score > best_score:
                best, best_score = ch, score
        return best

    # 2. add one new move to the tree
    def _expand(self, node: _Node) -> _Node:
        if not node.untried:
            return node  # game already over here; nothing to add
        move = node.untried.pop(self.rng.randrange(len(node.untried)))
        child_state = node.state.clone()
        child_state.play(move)
        child = _Node(child_state, parent=node, move=move)
        node.children.append(child)
        return child

    # 3. finish the game with random moves and report the winner
    def _simulate(self, state: GameState) -> int:
        rollout = state.clone()
        while not rollout.is_terminal():
            rollout.play(self.rng.choice(rollout.legal_moves()))
        return rollout.winner

    # 4. send the result back up, updating each move we took
    def _backprop(self, node: _Node | None, winner: int) -> None:
        while node is not None:
            node.visits += 1
            if node.move is not None:
                # credit the player who made the move into this node
                mover = node.parent.state.current_player
                node.value += _reward(winner, mover)
            node = node.parent
