"""How much faster is the surrogate than the solver it copies?

Three solver timings are reported, because the answer depends on what changes between calls:

* **new shape each time**: build the contour, assemble and solve the panel system. This is
  what a design loop that varies the shape pays, and what the surrogate replaces.
* **same shape, new angle**: ``PanelSolver`` factorises once and each further angle is a
  couple of vector operations, so the solver is already fast here. The surrogate has no
  advantage for a pure angle sweep of one shape.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from foilnet.geometry import naca4
from foilnet.solver import PanelSolver


def _median_seconds(fn: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


@dataclass(frozen=True)
class TimingResult:
    n_panels: int
    solver_new_shape: float  # seconds per (shape, angle) -> (Cl, Cm)
    solver_same_shape: float  # seconds per extra angle on an already-built shape
    surrogate_batch: float  # seconds per sample, large batch
    surrogate_single: float  # seconds per call, one sample at a time

    @property
    def speedup_batch(self) -> float:
        return self.solver_new_shape / self.surrogate_batch

    @property
    def speedup_single(self) -> float:
        return self.solver_new_shape / self.surrogate_single


def measure(
    predict: Callable[[np.ndarray], object],
    n_panels: int = 160,
    batch: int = 10_000,
    repeats: int = 15,
    seed: int = 0,
) -> TimingResult:
    """Time the solver and ``predict`` (any callable from an ``(n, 4)`` array of
    camber, camber position, thickness, alpha). Medians over ``repeats`` runs."""
    rng = np.random.default_rng(seed)

    def sample(n: int) -> np.ndarray:
        return np.column_stack(
            [
                rng.uniform(0.0, 0.06, n),
                rng.uniform(0.2, 0.6, n),
                rng.uniform(0.06, 0.18, n),
                rng.uniform(-5.0, 12.0, n),
            ]
        )

    rows = sample(repeats)
    index = iter(range(10**9))

    def solver_new() -> None:
        m, p, t, alpha = rows[next(index) % repeats]
        PanelSolver(naca4(m, p, t, n_panels)).solve(alpha)

    solver_built = PanelSolver(naca4(0.02, 0.4, 0.12, n_panels))
    alphas = rng.uniform(-5.0, 12.0, 200)
    counter = iter(range(10**9))

    def solver_alpha() -> None:
        solver_built.solve(alphas[next(counter) % len(alphas)])

    big, one = sample(batch), sample(1)
    predict(one)  # warm up
    return TimingResult(
        n_panels=n_panels,
        solver_new_shape=_median_seconds(solver_new, repeats),
        solver_same_shape=_median_seconds(solver_alpha, 200),
        surrogate_batch=_median_seconds(lambda: predict(big), max(3, repeats // 3)) / batch,
        surrogate_single=_median_seconds(lambda: predict(one), 100),
    )
