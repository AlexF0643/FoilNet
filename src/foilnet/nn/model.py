"""The PyTorch network. Same architecture as :class:`foilnet.nn.numpy_mlp.NumpyMLP`."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}


class MLP(nn.Module):
    """Fully connected network with a linear output layer.

    Weights use Glorot-uniform initialisation and biases start at zero, the same as the
    NumPy version, so the two can be compared directly.
    """

    def __init__(
        self, n_in: int, hidden: Sequence[int], n_out: int, activation: str = "tanh"
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(ACTIVATIONS)}")
        if n_in < 1 or n_out < 1 or any(h < 1 for h in hidden):
            raise ValueError("layer sizes must be positive")
        self.sizes = (n_in, *map(int, hidden), n_out)
        self.activation = activation
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:], strict=True)):
            linear = nn.Linear(a, b)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(self.sizes) - 2:
                layers.append(ACTIVATIONS[activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
