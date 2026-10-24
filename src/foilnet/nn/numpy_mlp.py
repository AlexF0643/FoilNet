"""A multi-layer perceptron written from scratch in NumPy, with hand-derived backprop.

This exists to understand what a framework does for you. PyTorch (Phase 5) is used for
the real experiments; this one is checked against finite differences and then fitted to
a small piece of the dataset.

The network
-----------
Layer l computes ``z_l = a_{l-1} @ W_l + b_l`` and ``a_l = phi(z_l)``, with ``a_0 = x``.
Every layer except the last applies the activation ``phi``; the last is linear, so the
network can output any real number (Cl and Cm are not bounded to a range).

The loss is the mean squared error over all samples and outputs,

    L = mean((a_L - y)^2)  +  0.5 * l2 * sum(W^2)

and backprop is the chain rule applied layer by layer, from the output backwards.
Write ``delta_l = dL/dz_l`` for the error signal at layer l. For the linear output
layer ``z_L = a_L``, so

    delta_L = 2 * (a_L - y) / (number of elements in y)

and for each layer, with N samples in the batch,

    dL/dW_l = a_{l-1}^T @ delta_l  +  l2 * W_l     (each weight is shared by all N samples)
    dL/db_l = sum over samples of delta_l
    delta_{l-1} = (delta_l @ W_l^T) * phi'(z_{l-1})

The last line is the heart of it: the error at layer l-1 is the error at layer l sent
back through the weights, then scaled by how steeply the activation was sloping there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# name -> (phi, phi') where phi' takes (z, a) with a = phi(z), so tanh can reuse a.
_ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
    "tanh": (np.tanh, lambda z, a: 1.0 - a**2),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0.0).astype(float)),
}


class NumpyMLP:
    """Fully connected network ``sizes[0] -> ... -> sizes[-1]`` with a linear output layer."""

    def __init__(self, sizes: Sequence[int], activation: str = "tanh", seed: int = 0):
        if len(sizes) < 2 or min(sizes) < 1:
            raise ValueError(f"sizes needs at least two positive entries, got {list(sizes)}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(_ACTIVATIONS)}")
        self.sizes = tuple(int(s) for s in sizes)
        self.activation = activation
        rng = np.random.default_rng(seed)
        # Glorot/Xavier uniform: keeps the signal's scale roughly constant from layer to layer.
        self.weights = [
            rng.uniform(-lim, lim, size=(n_in, n_out))
            for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:], strict=True)
            for lim in [np.sqrt(6.0 / (n_in + n_out))]
        ]
        self.biases = [np.zeros(n_out) for n_out in self.sizes[1:]]

    @property
    def parameters(self) -> list[np.ndarray]:
        """``[W_1, b_1, W_2, b_2, ...]``: the arrays themselves, so they can be updated in place."""
        return [p for pair in zip(self.weights, self.biases, strict=True) for p in pair]

    @property
    def n_parameters(self) -> int:
        return sum(p.size for p in self.parameters)

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.sizes[0]:
            raise ValueError(f"x must have shape (n, {self.sizes[0]}), got {x.shape}")
        return x

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        phi, _ = _ACTIVATIONS[self.activation]
        inputs, pre = [x], []  # a_{l-1} and z_l for every layer, kept for backprop
        a = x
        last = len(self.weights) - 1
        for layer, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = a @ w + b
            pre.append(z)
            a = z if layer == last else phi(z)
            inputs.append(a)
        return a, inputs, pre

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(self._check_x(x))[0]

    def loss(self, x: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
        out = self.predict(x)
        value = float(np.mean((out - y) ** 2))
        if l2:
            value += 0.5 * l2 * sum(float(np.sum(w**2)) for w in self.weights)
        return value

    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, l2: float = 0.0
    ) -> tuple[float, list[np.ndarray]]:
        """Loss and its gradient with respect to :attr:`parameters` (same order)."""
        x = self._check_x(x)
        y = np.asarray(y, dtype=float)
        out, inputs, pre = self._forward(x)
        if y.shape != out.shape:
            raise ValueError(f"y has shape {y.shape}, the network outputs {out.shape}")

        value = float(np.mean((out - y) ** 2))
        if l2:
            value += 0.5 * l2 * sum(float(np.sum(w**2)) for w in self.weights)

        _, dphi = _ACTIVATIONS[self.activation]
        delta = 2.0 * (out - y) / out.size  # delta_L: output layer is linear
        grads: list[np.ndarray] = [np.empty(0)] * (2 * len(self.weights))
        for layer in reversed(range(len(self.weights))):
            grads[2 * layer] = inputs[layer].T @ delta + l2 * self.weights[layer]
            grads[2 * layer + 1] = delta.sum(axis=0)
            if layer > 0:
                # send the error back through the weights, then through the activation slope
                delta = (delta @ self.weights[layer].T) * dphi(pre[layer - 1], inputs[layer])
        return value, grads


class Adam:
    """The Adam optimiser: per-parameter step sizes from running averages of the gradient
    (first moment) and its square (second moment), with bias correction for early steps."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params, self.lr, self.beta1, self.beta2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        """Update the parameters in place."""
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g**2
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gradient_check(
    model: NumpyMLP, x: np.ndarray, y: np.ndarray, l2: float = 0.0, h: float = 1e-6
) -> float:
    """Relative difference between backprop and central finite-difference gradients.

    Each parameter is nudged by +/- h and the loss re-evaluated:
    ``dL/dp ~ (L(p + h) - L(p - h)) / (2 h)``. Returns
    ``||g_backprop - g_numeric|| / (||g_backprop|| + ||g_numeric||)``; about 1e-8 or
    smaller means backprop is right, anything near 1e-3 or larger means it is not.
    """
    _, analytic = model.loss_and_grads(x, y, l2)
    numeric = []
    for p in model.parameters:
        grad = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            idx = it.multi_index
            original = p[idx]
            p[idx] = original + h
            plus = model.loss(x, y, l2)
            p[idx] = original - h
            minus = model.loss(x, y, l2)
            p[idx] = original
            grad[idx] = (plus - minus) / (2.0 * h)
        numeric.append(grad)
    a = np.concatenate([g.ravel() for g in analytic])
    n = np.concatenate([g.ravel() for g in numeric])
    return float(np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n)))


def fit(
    model: NumpyMLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2: float = 0.0,
    seed: int = 0,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Train with mini-batch Adam. Returns per-epoch ``train_loss`` (and ``val_loss``).

    Inputs and targets should already be standardised. Reported losses are plain MSE,
    without the weight-decay term, so training and validation are comparable.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1")
    x, y = model._check_x(x), np.asarray(y, dtype=float)
    optimiser = Adam(model.parameters, lr=lr)
    rng = np.random.default_rng(seed)
    history: dict[str, list[float]] = {"train_loss": []}
    if x_val is not None:
        history["val_loss"] = []

    n = len(x)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            _, grads = model.loss_and_grads(x[batch], y[batch], l2)
            optimiser.step(grads)
        history["train_loss"].append(model.loss(x, y))
        if x_val is not None:
            history["val_loss"].append(model.loss(x_val, y_val))
    return history
