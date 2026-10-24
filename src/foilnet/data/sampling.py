"""Latin hypercube sampling over a box of named parameters."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import qmc


def latin_hypercube(
    ranges: Mapping[str, tuple[float, float]], n: int, seed: int
) -> np.ndarray:
    """``(n, len(ranges))`` samples, columns in the order of ``ranges``.

    Each column has exactly one sample in each of ``n`` equal-width bins, so every
    parameter is covered evenly even for small ``n``. The same ``seed`` always gives the
    same samples.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not ranges:
        raise ValueError("ranges is empty")
    for name, (low, high) in ranges.items():
        if not low < high:
            raise ValueError(f"range for {name!r} must have low < high, got ({low}, {high})")
    sampler = qmc.LatinHypercube(d=len(ranges), seed=seed)
    low = [lo for lo, _ in ranges.values()]
    high = [hi for _, hi in ranges.values()]
    return qmc.scale(sampler.random(n), low, high)
