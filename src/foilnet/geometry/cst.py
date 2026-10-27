"""CST (class/shape transformation) aerofoil parameterisation, after Kulfan (2008).

Each surface is ``y(x) = C(x) * S(x)`` on a unit chord, with

* the class function ``C(x) = x^0.5 * (1 - x)``: a round leading edge and a sharp trailing
  edge (the exponents ``N1 = 0.5``, ``N2 = 1`` are fixed), and
* a shape function ``S(x) = sum_i A_i * B_i(x)``, a weighted sum of Bernstein polynomials of
  degree ``n = len(A) - 1``. The weights ``A_i`` are the shape parameters.

An aerofoil is an upper set ``au`` and a lower set ``al`` of coefficients, ``n + 1`` each.
Every shape is linear in its coefficients, so fitting a given surface is a least-squares
problem. Because ``S(1) = A_n``, the trailing edge leaves each surface at slope ``A_n``, so
the *included trailing-edge angle* is ``atan(au[-1]) + atan(-al[-1])``. That angle matters
for accuracy: the panel solver degrades on near-cusped trailing edges (see
``docs/cst-validity.md``).

Contours use the same conventions as :mod:`foilnet.geometry.naca`: chord 1, leading edge at
the origin, closed, counter-clockwise from the upper trailing edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from foilnet.geometry.paneling import cosine_spacing

N1, N2 = 0.5, 1.0


def class_function(x: np.ndarray) -> np.ndarray:
    return x**N1 * (1.0 - x) ** N2


def bernstein_basis(x: np.ndarray, n_coefficients: int) -> np.ndarray:
    """``(len(x), n_coefficients)`` Bernstein polynomials of degree ``n_coefficients - 1``."""
    if n_coefficients < 1:
        raise ValueError(f"n_coefficients must be at least 1, got {n_coefficients}")
    x = np.asarray(x, dtype=float)
    n = n_coefficients - 1
    return np.stack([comb(n, i) * x**i * (1.0 - x) ** (n - i) for i in range(n + 1)], axis=1)


def cst_surface(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Height of one surface at chord positions ``x``."""
    coefficients = np.asarray(coefficients, dtype=float)
    return class_function(np.asarray(x, dtype=float)) * (
        bernstein_basis(x, len(coefficients)) @ coefficients
    )


def cst_coordinates(upper: np.ndarray, lower: np.ndarray, n_panels: int = 160) -> np.ndarray:
    """Closed contour ``(n_panels + 1, 2)`` of the aerofoil with the given coefficients.

    Points are cosine-spaced in x on each surface, as for the NACA sections.
    """
    if n_panels < 4 or n_panels % 2:
        raise ValueError(f"n_panels must be even and at least 4, got {n_panels}")
    x = cosine_spacing(n_panels // 2)
    yu, yl = cst_surface(x, upper), cst_surface(x, lower)
    xs = np.concatenate([x[::-1], x[1:]])
    ys = np.concatenate([yu[::-1], yl[1:]])
    coords = np.column_stack([xs, ys])
    coords[-1] = coords[0]  # the trailing edge is sharp, so the contour closes exactly
    return coords


def fit_cst_surface(x: np.ndarray, y: np.ndarray, n_coefficients: int) -> np.ndarray:
    """Least-squares CST coefficients for a surface given as points ``(x, y)``."""
    x = np.asarray(x, dtype=float)
    inside = (x > 0.0) & (x < 1.0)  # the class function is zero at both ends
    design = class_function(x[inside])[:, None] * bernstein_basis(x[inside], n_coefficients)
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float)[inside], rcond=None)
    return coefficients


def fit_cst(coords: np.ndarray, n_coefficients: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``(upper, lower)`` coefficients to a closed contour in the package's ordering
    (upper surface from the trailing edge to the leading edge, then the lower surface)."""
    coords = np.asarray(coords, dtype=float)
    i_le = int(np.argmin(coords[:, 0]))
    upper = coords[: i_le + 1][::-1]  # leading edge to trailing edge
    lower = coords[i_le:]
    return (
        fit_cst_surface(upper[:, 0], upper[:, 1], n_coefficients),
        fit_cst_surface(lower[:, 0], lower[:, 1], n_coefficients),
    )


def te_included_angle_deg(upper: np.ndarray, lower: np.ndarray) -> float:
    """Angle in degrees between the two surfaces where they meet at the trailing edge."""
    return float(np.degrees(np.arctan(upper[-1]) + np.arctan(-lower[-1])))


def thickness_and_camber(
    upper: np.ndarray, lower: np.ndarray, n_points: int = 801
) -> tuple[float, float]:
    """``(max thickness, max |camber|)`` as fractions of chord, from the vertical distance
    between the surfaces and their midline."""
    x = cosine_spacing(n_points - 1)
    yu, yl = cst_surface(x, upper), cst_surface(x, lower)
    return float(np.max(yu - yl)), float(np.max(np.abs(0.5 * (yu + yl))))


@dataclass(frozen=True)
class ShapeLimits:
    """What counts as a usable shape.

    * ``min_te_angle_deg``: below a few degrees the panel solver's accuracy degrades.
    * ``min_le_coefficient`` and ``min_le_ratio``: a round nose. Near the leading edge each
      surface is ``a * sqrt(x)``, a parabola with nose radius ``a**2 / 2``. If the lower
      coefficient ``-al[0]`` is near zero the lower surface leaves the nose flat, the nose is a
      knife edge, and potential flow has infinite suction there: the solver's answer does not
      converge as panels are added, so the sample is not a usable label. Both ``au[0]`` and
      ``-al[0]`` must be at least ``min_le_coefficient`` (a NACA 0006 has about 0.09) and the
      smaller must be at least ``min_le_ratio`` times the larger.
    * the thickness and camber limits keep samples recognisably aerofoil-like.
    """

    min_te_angle_deg: float = 5.0
    min_thickness: float = 0.04
    max_thickness: float = 0.22
    max_camber: float = 0.08
    min_le_coefficient: float = 0.10
    min_le_ratio: float = 0.25


def check_shape(upper: np.ndarray, lower: np.ndarray, limits: ShapeLimits) -> list[str]:
    """Reasons the shape is not acceptable; an empty list means it is."""
    x = cosine_spacing(400)
    gap = cst_surface(x, upper) - cst_surface(x, lower)
    problems = []
    if np.any(gap < -1e-9):
        problems.append("the surfaces cross")
    thickness, camber = thickness_and_camber(upper, lower)
    if thickness < limits.min_thickness:
        problems.append(f"thickness {thickness:.3f} is below {limits.min_thickness}")
    if thickness > limits.max_thickness:
        problems.append(f"thickness {thickness:.3f} is above {limits.max_thickness}")
    if camber > limits.max_camber:
        problems.append(f"camber {camber:.3f} is above {limits.max_camber}")
    angle = te_included_angle_deg(upper, lower)
    if angle < limits.min_te_angle_deg:
        problems.append(f"trailing-edge angle {angle:.1f} deg is below {limits.min_te_angle_deg}")
    nose_up, nose_low = float(upper[0]), float(-lower[0])
    if min(nose_up, nose_low) < limits.min_le_coefficient:
        problems.append(
            f"leading edge too sharp: au0={nose_up:.3f}, -al0={nose_low:.3f} "
            f"(each must be at least {limits.min_le_coefficient})"
        )
    elif min(nose_up, nose_low) < limits.min_le_ratio * max(nose_up, nose_low):
        problems.append(
            f"leading edge lopsided: au0={nose_up:.3f}, -al0={nose_low:.3f} "
            f"(smaller must be at least {limits.min_le_ratio} of the larger)"
        )
    return problems
