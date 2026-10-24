"""Cosine spacing and panel construction from a closed contour."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_spacing(n: int) -> np.ndarray:
    """``n + 1`` points from 0 to 1, clustered at both ends (x = (1 - cos(pi i / n)) / 2)."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def signed_area(coords: np.ndarray) -> float:
    """Shoelace area of a closed contour; positive when traversed counter-clockwise."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


@dataclass(frozen=True)
class Panels:
    """Straight panels joining consecutive contour points.

    ``tangent`` points along the direction of travel. ``normal`` points out of the
    body (to the right of travel for a counter-clockwise contour).
    """

    start: np.ndarray  # (n, 2)
    end: np.ndarray  # (n, 2)
    midpoint: np.ndarray  # (n, 2) control points
    length: np.ndarray  # (n,)
    tangent: np.ndarray  # (n, 2) unit vectors
    normal: np.ndarray  # (n, 2) unit vectors, outward

    @property
    def n(self) -> int:
        return len(self.length)


def build_panels(coords: np.ndarray, closure_tol: float = 1e-9) -> Panels:
    """Build panels from a closed, counter-clockwise ``(n + 1, 2)`` contour."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 4:
        raise ValueError(f"coords must have shape (n + 1, 2) with n >= 3, got {coords.shape}")
    gap = float(np.linalg.norm(coords[0] - coords[-1]))
    if gap > closure_tol:
        raise ValueError(f"contour is not closed: first and last points are {gap:.3g} apart")
    if signed_area(coords) <= 0.0:
        raise ValueError("contour must be counter-clockwise (trailing edge, upper surface first)")

    start, end = coords[:-1], coords[1:]
    delta = end - start
    length = np.hypot(delta[:, 0], delta[:, 1])
    if np.any(length <= 0.0):
        raise ValueError("contour has a zero-length panel (repeated point)")
    tangent = delta / length[:, None]
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    return Panels(start, end, 0.5 * (start + end), length, tangent, normal)
