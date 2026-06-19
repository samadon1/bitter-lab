"""Make the explanatory pictures for the tutorial.

Everything here is drawn from code so it stays reproducible and consistent. Run:
    python viz.py
and it writes PNGs into figures/.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from engine import GameState  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

BOARD_BLUE = "#2f54c9"
EDGE = "#1c357f"
EMPTY = "#eaf0ff"
RED = "#e8443b"
GOLD = "#f6c52b"
INK = "#16203a"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------

def draw_board(ax, grid, highlight=None):
    """grid: 6x7 with 0 empty, 1 red, 2 gold. Row 0 is the bottom."""
    rows, cols = 6, 7
    ax.add_patch(FancyBboxPatch(
        (-0.6, -0.6), cols - 1 + 1.2, rows - 1 + 1.2,
        boxstyle="round,pad=0,rounding_size=0.35", fc=BOARD_BLUE, ec="none", zorder=0))
    for r in range(rows):
        for c in range(cols):
            v = int(grid[r][c])
            fc = EMPTY if v == 0 else (RED if v == 1 else GOLD)
            ax.add_patch(Circle((c, r), 0.40, fc=fc, ec=EDGE, lw=1.6, zorder=2))
    for (r, c) in (highlight or []):
        ax.add_patch(Circle((c, r), 0.40, fc="none", ec=INK, lw=4, zorder=3))
    ax.set_xlim(-0.75, cols - 1 + 0.75)
    ax.set_ylim(-0.95, rows - 1 + 0.75)
    ax.set_aspect("equal")
    ax.axis("off")


def _play(seq):
    g = GameState()
    for col in seq:
        g.play(col)
    return g.board


def fig_board_empty():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw_board(ax, np.zeros((6, 7), int))
    for c in range(7):
        ax.text(c, -0.9, str(c), ha="center", va="center", color="#5b6b8c", fontsize=10)
    ax.set_title("Connect Four: 7 columns, 6 rows", color=INK, fontsize=13)
    save(fig, "board_empty.png")


def fig_board_win():
    # red makes four along the bottom row (columns 0..3)
    board = _play([0, 0, 1, 1, 2, 2, 3])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw_board(ax, board, highlight=[(0, 0), (0, 1), (0, 2), (0, 3)])
    ax.set_title("Four in a row wins (here: red, along the bottom)", color=INK, fontsize=12)
    save(fig, "board_win.png")


def fig_gravity():
    board = _play([3, 3, 2])  # a few pieces so column 3 is partly filled
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    draw_board(ax, board)
    # arrow dropping into column 5
    ax.add_patch(Circle((5, 5.0), 0.40, fc=RED, ec=EDGE, lw=1.6, zorder=4))
    ax.add_patch(FancyArrowPatch((5, 4.5), (5, 0.6), arrowstyle="-|>", mutation_scale=18,
                                 color=INK, lw=2, zorder=4))
    ax.text(5, 5.7, "you pick the column", ha="center", color=INK, fontsize=10)
    ax.text(5.0, -0.9, "it falls to the bottom", ha="center", color=INK, fontsize=10)
    ax.set_title("You choose a column. Gravity chooses the row.", color=INK, fontsize=12)
    save(fig, "gravity.png")


# ---------------------------------------------------------------------------
# the futures explode (branching)
# ---------------------------------------------------------------------------

def fig_futures():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    levels = [(0, 1), (-1, 7), (-2, 49)]
    pos = {0: [0.0]}
    # level 1 positions
    pos[1] = list(np.linspace(-9, 9, 7))
    # level 2 positions grouped under each level-1 node
    l2 = []
    for x in pos[1]:
        l2 += list(np.linspace(x - 1.0, x + 1.0, 7))
    pos[2] = l2

    def y(level):
        return -level * 2.2

    # edges 0->1
    for x1 in pos[1]:
        ax.plot([0, x1], [y(0), y(1)], color="#c4ccdf", lw=0.8, zorder=1)
    # edges 1->2
    for i, x1 in enumerate(pos[1]):
        for x2 in pos[2][i * 7:(i + 1) * 7]:
            ax.plot([x1, x2], [y(1), y(2)], color="#dfe4ef", lw=0.4, zorder=1)
    # nodes
    for level, xs in [(0, pos[0]), (1, pos[1]), (2, pos[2])]:
        s = {0: 220, 1: 90, 2: 18}[level]
        ax.scatter(xs, [y(level)] * len(xs), s=s, color=BOARD_BLUE, zorder=2, edgecolors="white")
    labels = ["now: 1 position", "after 1 move: 7", "after 2 moves: 49"]
    for level, lab in enumerate(labels):
        ax.text(10.6, y(level), lab, va="center", ha="left", color=INK, fontsize=11)
    ax.text(10.6, y(2) - 1.4, "after 5 moves: ~16,800\nafter 10 moves: ~280,000,000",
            va="top", ha="left", color="#5b6b8c", fontsize=10)
    ax.set_xlim(-11, 22)
    ax.set_ylim(y(2) - 2.4, 1.2)
    ax.axis("off")
    ax.set_title("You can't look all the way ahead: the futures explode", color=INK, fontsize=13)
    save(fig, "futures.png")


# ---------------------------------------------------------------------------
# MCTS in four steps
# ---------------------------------------------------------------------------

def _base_tree(ax):
    nodes = {"R": (0, 3), "A": (-1.5, 2), "B": (1.5, 2), "C": (-1.5, 1), "D": (-1.5, 0)}
    edges = [("R", "A"), ("R", "B"), ("A", "C"), ("C", "D")]
    return nodes, edges


def _draw_tree(ax, nodes, edges, bold_path=None, extra_nodes=None, color_map=None):
    bold = set()
    if bold_path:
        for a, b in zip(bold_path, bold_path[1:]):
            bold.add((a, b))
            bold.add((b, a))
    for a, b in edges:
        lw = 3.2 if (a, b) in bold else 1.0
        col = INK if (a, b) in bold else "#aab4cc"
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=col, lw=lw, zorder=1)
    color_map = color_map or {}
    for n, (x, yy) in nodes.items():
        ax.add_patch(Circle((x, yy), 0.30, fc=color_map.get(n, "white"),
                             ec=INK, lw=1.6, zorder=2))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.4, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_mcts_phases():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))

    # 1. walk down
    nodes, edges = _base_tree(axes[0])
    _draw_tree(axes[0], nodes, edges, bold_path=["R", "A", "C"])
    axes[0].set_title("1. Walk down\nfollow the best moves", color=INK, fontsize=11)

    # 2. try a move (add E under C)
    nodes, edges = _base_tree(axes[1])
    nodes["E"] = (-0.3, 0)
    edges = edges + [("C", "E")]
    _draw_tree(axes[1], nodes, edges, bold_path=["R", "A", "C"],
               color_map={"E": "#a6e3a1"})
    axes[1].text(-0.3, -0.7, "new", ha="center", color="#2f7d32", fontsize=9)
    axes[1].set_title("2. Try a move\nadd one new position", color=INK, fontsize=11)

    # 3. play out randomly
    nodes, edges = _base_tree(axes[2])
    nodes["E"] = (-0.3, 0)
    edges = edges + [("C", "E")]
    _draw_tree(axes[2], nodes, edges, color_map={"E": "#a6e3a1"})
    xs = np.linspace(-0.3, 1.6, 30)
    ys = np.linspace(0, -1.1, 30) + 0.12 * np.sin(np.linspace(0, 9, 30))
    axes[2].plot(xs, ys, ls=":", color="#888", lw=1.6)
    axes[2].text(1.7, -1.15, "LOSS", color=RED, fontsize=10, va="center")
    axes[2].set_title("3. Play out\nfinish with random moves", color=INK, fontsize=11)

    # 4. walk back
    nodes, edges = _base_tree(axes[3])
    nodes["E"] = (-0.3, 0)
    edges = edges + [("C", "E")]
    _draw_tree(axes[3], nodes, edges, color_map={"E": "#a6e3a1"})
    for a, b in [("E", "C"), ("C", "A"), ("A", "R")]:
        axes[3].add_patch(FancyArrowPatch(nodes[a], nodes[b], arrowstyle="-|>",
                          mutation_scale=14, color=INK, lw=2, zorder=3,
                          shrinkA=12, shrinkB=12))
    axes[3].set_title("4. Walk back\nupdate every move taken", color=INK, fontsize=11)

    fig.suptitle("How the thinker thinks: one simulation = these four steps",
                 color=INK, fontsize=13, y=1.02)
    save(fig, "mcts_phases.png")


# ---------------------------------------------------------------------------
# the expert's "windows of four"
# ---------------------------------------------------------------------------

def fig_windows():
    board = _play([3, 2, 3, 4, 2])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    draw_board(ax, board)

    def win_line(cells, color):
        # connect the four cell centres with a thick translucent line
        xs = [c[1] for c in cells]
        ys = [c[0] for c in cells]
        ax.plot(xs, ys, color=color, lw=9, alpha=0.55, solid_capstyle="round", zorder=5)

    win_line([(0, 0), (0, 1), (0, 2), (0, 3)], "#27c93f")      # horizontal
    win_line([(0, 6), (1, 6), (2, 6), (3, 6)], "#ff5fb0")      # vertical
    win_line([(0, 1), (1, 2), (2, 3), (3, 4)], "#ffd23f")      # diagonal
    ax.set_title("The expert scores every window of four\n(across, up, and diagonal)",
                 color=INK, fontsize=12)
    save(fig, "windows.png")


# ---------------------------------------------------------------------------
# the bitboard layout
# ---------------------------------------------------------------------------

def fig_bitboard():
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for c in range(7):
        for r in range(7):
            idx = c * 7 + r
            sentinel = (r == 6)
            fc = "#dfe4ef" if sentinel else "white"
            ax.add_patch(Rectangle((c, r), 0.9, 0.9, fc=fc, ec=INK, lw=1.2))
            label = f"{idx}" if not sentinel else "gap"
            col = "#9aa3b8" if sentinel else INK
            ax.text(c + 0.45, r + 0.45, label, ha="center", va="center", color=col, fontsize=9)
    ax.text(3.5, 7.4, "each column = 7 bits (6 cells + 1 gap on top)",
            ha="center", color=INK, fontsize=11)
    ax.set_xlim(-0.3, 7.2)
    ax.set_ylim(-0.3, 7.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The whole board as bit positions (number = column*7 + row)",
                 color=INK, fontsize=12, y=-0.06)
    save(fig, "bitboard.png")


# ---------------------------------------------------------------------------
# the value network
# ---------------------------------------------------------------------------

def fig_valuenet():
    fig, ax = plt.subplots(figsize=(10, 4.2))

    # a mini board on the left
    board = _play([3, 2, 4, 3])
    for r in range(6):
        for c in range(7):
            v = int(board[r][c])
            fc = EMPTY if v == 0 else (RED if v == 1 else GOLD)
            ax.add_patch(Circle((c * 0.32, r * 0.32 + 0.6), 0.13, fc=fc, ec=EDGE, lw=0.8))
    ax.text(0.96, 2.7, "the board", ha="center", color=INK, fontsize=10)

    ax.text(3.2, 1.6, "84\nnumbers", ha="center", va="center", color=INK, fontsize=10)
    ax.add_patch(FancyArrowPatch((2.4, 1.6), (3.7, 1.6), arrowstyle="-|>",
                                 mutation_scale=16, color=INK, lw=1.6))

    # hidden layer
    hx = 5.0
    for i, yy in enumerate(np.linspace(0.2, 3.0, 6)):
        ax.add_patch(Circle((hx, yy), 0.16, fc="#cdd8ff", ec=INK, lw=1))
    ax.text(hx, -0.25, "hidden layer", ha="center", color=INK, fontsize=10)
    ax.add_patch(FancyArrowPatch((3.8, 1.6), (4.7, 1.6), arrowstyle="-|>",
                                 mutation_scale=16, color=INK, lw=1.6))

    # output
    ox = 6.8
    ax.add_patch(Circle((ox, 1.6), 0.22, fc="#a6e3a1", ec=INK, lw=1.2))
    ax.add_patch(FancyArrowPatch((5.25, 1.6), (6.5, 1.6), arrowstyle="-|>",
                                 mutation_scale=16, color=INK, lw=1.6))

    # the value dial
    ax.annotate("", xy=(9.4, 1.6), xytext=(7.2, 1.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.plot([7.6, 9.2], [0.9, 0.9], color="#aab4cc", lw=3)
    ax.text(7.6, 0.55, "-1\nlosing", ha="center", color=RED, fontsize=9)
    ax.text(9.2, 0.55, "+1\nwinning", ha="center", color="#2f7d32", fontsize=9)
    ax.text(8.4, 0.55, "0", ha="center", color="#5b6b8c", fontsize=9)
    ax.scatter([8.9], [0.9], s=80, color=INK, zorder=5)
    ax.text(8.4, 2.2, "how good for the\nplayer to move?", ha="center", color=INK, fontsize=10)

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.6, 3.2)
    ax.axis("off")
    ax.set_title("The value network: board in, one number out", color=INK, fontsize=13)
    save(fig, "valuenet.png")


def main():
    fig_board_empty()
    fig_board_win()
    fig_gravity()
    fig_futures()
    fig_mcts_phases()
    fig_windows()
    fig_bitboard()
    fig_valuenet()
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
