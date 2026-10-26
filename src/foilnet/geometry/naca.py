"""NACA 4-digit aerofoil geometry.

Coordinates are in chord lengths (chord = 1, leading edge at the origin) and are
ordered the way the panel solver expects: starting at the trailing edge, along the
upper surface to the leading edge, then back along the lower surface to the
trailing edge. That traversal is counter-clockwise, and the contour is closed (the
last point is the first point).
"""

from __future__ import annotations

import numpy as np

from foilnet.geometry.paneling import cosine_spacing

# Thickness polynomial coefficients. The last one is -0.1036 rather than the
# original -0.1015 so that the thickness is zero at x/c = 1 (a closed trailing edge).
_A0, _A1, _A2, _A3, _A4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1036


def parse_naca4(code: str) -> tuple[float, float, float]:
    """Turn a code like ``"2412"`` into ``(m, p, t)``: camber, camber position, thickness."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"not a NACA 4-digit code: {code!r}")
    return int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0


def naca4_thickness(x: np.ndarray | float, t: float) -> np.ndarray:
    """Half-thickness distribution y_t(x) of a closed-trailing-edge NACA 4-digit section."""
    x = np.asarray(x, dtype=float)
    return 5.0 * t * (
        _A0 * np.sqrt(x) + _A1 * x + _A2 * x**2 + _A3 * x**3 + _A4 * x**4
    )


def naca4_camber(x: np.ndarray | float, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line ``(y_c, dy_c/dx)`` for max camber ``m`` at chordwise position ``p``."""
    x = np.asarray(x, dtype=float)
    if m == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    if not 0.0 < p < 1.0:
        raise ValueError(f"camber position must be in (0, 1) for a cambered section, got {p}")
    front = x < p
    yc = np.where(
        front,
        m / p**2 * (2.0 * p * x - x**2),
        m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
    )
    dyc = np.where(
        front,
        2.0 * m / p**2 * (p - x),
        2.0 * m / (1.0 - p) ** 2 * (p - x),
    )
    return yc, dyc


def naca4(m: float, p: float, t: float, n_panels: int = 160) -> np.ndarray:
    """Closed contour of a NACA 4-digit aerofoil as an ``(n_panels + 1, 2)`` array.

    ``m`` is the maximum camber, ``p`` its chordwise position and ``t`` the maximum
    thickness, all as fractions of chord. Points are cosine-spaced in x on each
    surface, which clusters them at the leading and trailing edges. The surfaces are
    offset perpendicular to the camber line, so ``2 * y_t`` is the true local thickness.
    """
    if n_panels < 4 or n_panels % 2:
        raise ValueError(f"n_panels must be even and at least 4, got {n_panels}")
    if not 0.0 < t < 1.0:
        raise ValueError(f"thickness must be in (0, 1), got {t}")
    n_half = n_panels // 2

    x = cosine_spacing(n_half)  # 0 -> 1, n_half + 1 points
    yt = naca4_thickness(x, t)
    yc, dyc = naca4_camber(x, m, p)
    theta = np.arctan(dyc)

    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    # Upper surface TE -> LE, then lower surface LE -> TE (leading edge not repeated).
    xs = np.concatenate([xu[::-1], xl[1:]])
    ys = np.concatenate([yu[::-1], yl[1:]])
    coords = np.column_stack([xs, ys])
    coords[-1] = coords[0]  # the trailing edge is closed; remove rounding-error gap
    return coords


def naca4_from_code(code: str, n_panels: int = 160) -> np.ndarray:
    """Same as :func:`naca4` but takes a code such as ``"2412"``."""
    return naca4(*parse_naca4(code), n_panels=n_panels)
