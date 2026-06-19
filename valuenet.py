"""A tiny value network: "how good is this position for the player to move?"

This replaces MCTS's random playout. Instead of finishing the game with coin-flip
moves to guess who wins, we ask a small learned function for its opinion. The
function learns from the agent's own games (see selfplay.py).

It's a plain two-layer network written in NumPy:
    84 inputs  ->  hidden layer (ReLU)  ->  1 output (tanh, between -1 and +1)

Inputs: two 6x7 "pictures" of the board flattened together - one marking the
current player's pieces, one marking the opponent's (84 numbers total).
Output: +1 means "great for the player to move", -1 means "losing", 0 "even".

Why NumPy and not a big library: the network is called one position at a time,
deep inside the search loop, millions of times. At that size the overhead of a
heavyweight library would cost more than the actual maths. A bare NumPy net is
leaner here - and we get to see every gradient.
"""

from __future__ import annotations

import numpy as np

N_CELLS = 42        # 6 rows x 7 cols
N_IN = 2 * N_CELLS  # two planes: my pieces, opponent's pieces
ROWS, COLS = 6, 7


def to_planes(state) -> np.ndarray:
    """Turn a board position into the 84-number input, from the mover's view.

    Works for both board types (array engine and bitboard) and - importantly -
    produces the SAME numbers for the same position, so the net sees consistent
    input whether a game was played on the fast or the slow board. Cell order is
    fixed as row*7 + col (row 0 at the bottom).
    """
    planes = np.zeros(N_IN, dtype=np.float32)

    if hasattr(state, "current_position"):  # BitBoard
        me_bits = state.current_position
        opp_bits = state.mask ^ state.current_position
        for r in range(ROWS):
            for c in range(COLS):
                bit = 1 << (c * 7 + r)
                idx = r * COLS + c
                if me_bits & bit:
                    planes[idx] = 1.0
                elif opp_bits & bit:
                    planes[N_CELLS + idx] = 1.0
    else:  # GameState (NumPy board)
        me = state.current_player
        board = state.board
        planes[:N_CELLS] = (board == me).reshape(-1).astype(np.float32)
        planes[N_CELLS:] = (board == 3 - me).reshape(-1).astype(np.float32)

    return planes


def _relu(x):
    return np.maximum(x, 0.0)


class ValueNet:
    """Two-layer network trained to predict the game result for the mover."""

    def __init__(self, hidden: int = 64, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He-style init for the ReLU layer, small init for the output layer.
        self.W1 = (rng.standard_normal((N_IN, hidden)) * np.sqrt(2.0 / N_IN)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr
        self._init_adam()

    # -- Adam optimizer state ----------------------------------------------
    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    # -- forward pass ------------------------------------------------------
    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        v = np.tanh(z2)
        return v, (X, z1, a1, z2, v)

    def predict_value(self, state) -> float:
        """The net's opinion of `state` for the player to move, in [-1, 1]."""
        x = to_planes(state)[None, :]
        v, _ = self._forward(x)
        return float(v[0, 0])

    # -- one training step on a batch (mean-squared error) -----------------
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        n = X.shape[0]
        v, (X, z1, a1, z2, _) = self._forward(X)
        y = y.reshape(-1, 1)

        loss = float(np.mean((v - y) ** 2))

        # backprop
        dv = 2.0 * (v - y) / n
        dz2 = dv * (1.0 - v ** 2)          # through tanh
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)               # through ReLU
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        self._adam_update({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})
        return loss

    def fit(self, X, y, epochs=5, batch=128, rng=None):
        """Train for a few passes over the data, shuffling each pass."""
        rng = rng or np.random.default_rng(0)
        last = 0.0
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                idx = order[i:i + batch]
                last = self.train_step(X[idx], y[idx])
        return last

    def _adam_update(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            self._params()[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)
