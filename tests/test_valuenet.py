"""Checks for the value network before we trust it inside search."""

import random

import numpy as np

from bitboard import BitBoard
from engine import GameState
from valuenet import N_IN, ValueNet, to_planes


def test_to_planes_matches_across_board_types():
    # The same moves on both board types must give identical network input,
    # or the net would see different numbers for training vs evaluation.
    rng = random.Random(0)
    for _ in range(50):
        g, b = GameState(), BitBoard()
        for _ in range(rng.randint(0, 20)):
            if g.is_terminal():
                break
            m = rng.choice(g.legal_moves())
            g.play(m)
            b.play(m)
        assert np.array_equal(to_planes(g), to_planes(b))


def test_planes_have_right_shape_and_values():
    g = GameState()
    g.play(3)  # one piece on the board
    planes = to_planes(g)
    assert planes.shape == (N_IN,)
    # exactly one cell filled, and it's in the opponent plane (we just switched turn)
    assert planes.sum() == 1.0


def test_network_can_learn():
    # Make a small fixed dataset and check the net drives its error down.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, N_IN)).astype(np.float32)
    true_w = rng.standard_normal((N_IN, 1)).astype(np.float32)
    y = np.tanh(X @ true_w).reshape(-1)  # a learnable target in [-1, 1]

    net = ValueNet(hidden=64, lr=1e-2, seed=1)
    start = net.train_step(X, y)
    net.fit(X, y, epochs=200, batch=64, rng=rng)
    end = net.train_step(X, y)

    assert end < start * 0.5, f"loss did not fall enough: {start:.3f} -> {end:.3f}"
