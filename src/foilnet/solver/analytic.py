"""Exact and closed-form results the panel solver is validated against.

* Thin-aerofoil theory for NACA 4-digit camber lines (Anderson, *Fundamentals of
  Aerodynamics*, ch. 4).
* The Karman-Trefftz aerofoil: the exact potential-flow solution, with the Kutta
  condition, obtained by conformally mapping the flow past a circle. With ``n = 2`` it is
  the Joukowski aerofoil (cusped trailing edge); ``n < 2`` gives a finite trailing-edge
  angle of ``(2 - n) * pi``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from foilnet.geometry.naca import naca4_camber


def thin_airfoil_theory(
    camber_slope: Callable[[float], float], alpha_deg: float, kinks: Sequence[float] = ()
) -> dict[str, float]:
    """Thin-aerofoil-theory ``cl``, ``cm_c4`` (nose-up positive) and ``alpha_l0_deg``.

    ``camber_slope(x)`` is dy_c/dx on a unit chord; ``kinks`` lists x positions where it
    is not smooth, so the integrator can step over them.
    """

    def slope(theta: float) -> float:
        return camber_slope(0.5 * (1.0 - np.cos(theta)))

    points = [float(np.arccos(1.0 - 2.0 * x)) for x in kinks] or None

    def integral(weight: Callable[[float], float]) -> float:
        value, _ = quad(lambda th: slope(th) * weight(th), 0.0, np.pi, points=points, limit=200)
        return value

    a_rad = np.deg2rad(alpha_deg)
    a0 = a_rad - integral(lambda th: 1.0) / np.pi
    a1 = 2.0 / np.pi * integral(np.cos)
    a2 = 2.0 / np.pi * integral(lambda th: np.cos(2.0 * th))
    alpha_l0 = integral(lambda th: 1.0 - np.cos(th)) / np.pi
    return {
        "cl": float(np.pi * (2.0 * a0 + a1)),
        "cm_c4": float(0.25 * np.pi * (a2 - a1)),
        "alpha_l0_deg": float(np.rad2deg(alpha_l0)),
    }


def thin_airfoil_naca4(m: float, p: float, alpha_deg: float) -> dict[str, float]:
    """Thin-aerofoil theory for the camber line of a NACA 4-digit section."""
    return thin_airfoil_theory(
        lambda x: float(naca4_camber(x, m, p)[1]), alpha_deg, kinks=(p,) if m != 0.0 else ()
    )


@dataclass(frozen=True)
class KarmanTrefftz:
    """Karman-Trefftz aerofoil from a circle through zeta = a, centred at ``(xc, yc)``.

    The circle must enclose zeta = -a, i.e. ``(xc + a)**2 + yc**2 < R**2``.
    """

    n: float = 1.9
    a: float = 1.0
    xc: float = -0.1
    yc: float = 0.1

    def __post_init__(self) -> None:
        if not 1.0 < self.n <= 2.0:
            raise ValueError(f"n must be in (1, 2], got {self.n}")
        if (self.xc + self.a) ** 2 + self.yc**2 >= self.radius**2:
            raise ValueError("circle must enclose zeta = -a")

    @property
    def radius(self) -> float:
        return float(np.hypot(self.a - self.xc, self.yc))

    @property
    def phi_te(self) -> float:
        """Angle on the circle of the trailing-edge point zeta = a."""
        return float(np.arctan2(-self.yc, self.a - self.xc))

    def _zeta(self, phi: np.ndarray) -> np.ndarray:
        return complex(self.xc, self.yc) + self.radius * np.exp(1j * phi)

    def _u(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Modulus and a *continuous* argument of u = (zeta - a) / (zeta + a).

        phi must lie in [phi_te, phi_te + 2 pi]. The argument is continuous so that the
        non-integer power u**n is single valued along the whole circle.
        """
        zeta = self._zeta(phi)
        modulus = np.abs(zeta - self.a) / np.abs(zeta + self.a)
        arg_minus = 0.5 * (phi + self.phi_te) + 0.5 * np.pi  # inscribed-angle theorem
        arg_plus = np.mod(np.angle(zeta + self.a), 2.0 * np.pi)  # grows from 0 to 2 pi
        return modulus, arg_minus - arg_plus

    def surface(self, phi: np.ndarray) -> np.ndarray:
        """Aerofoil points ``(len(phi), 2)`` for circle angles ``phi``."""
        phi = np.asarray(phi, dtype=float)
        modulus, arg = self._u(phi)
        w = modulus**self.n * np.exp(1j * self.n * arg)
        z = self.n * self.a * (1.0 + w) / (1.0 - w)
        return np.column_stack([z.real, z.imag])

    def circulation(self, alpha_deg: float) -> float:
        """Clockwise-positive circulation fixed by the Kutta condition (freestream speed 1)."""
        beta = self.phi_te
        return float(4.0 * np.pi * self.radius * np.sin(np.deg2rad(alpha_deg) - beta))

    def cp(self, phi: np.ndarray, alpha_deg: float) -> np.ndarray:
        """Exact pressure coefficient at circle angles ``phi`` (not at the trailing edge itself)."""
        phi = np.asarray(phi, dtype=float)
        al = np.deg2rad(alpha_deg)
        dw = (
            np.exp(-1j * al)
            - np.exp(1j * (al - 2.0 * phi))
            + 1j * self.circulation(alpha_deg) / (2.0 * np.pi * self.radius) * np.exp(-1j * phi)
        )
        zeta = self._zeta(phi)
        modulus, arg = self._u(phi)
        u_pow = modulus ** (self.n - 1.0) * np.exp(1j * (self.n - 1.0) * arg)
        w = u_pow * modulus * np.exp(1j * arg)
        dz = 4.0 * self.n**2 * self.a**2 * u_pow / ((zeta + self.a) ** 2 * (1.0 - w) ** 2)
        return 1.0 - (np.abs(dw) / np.abs(dz)) ** 2

    def contour(self, n_panels: int) -> tuple[np.ndarray, np.ndarray]:
        """Closed counter-clockwise contour from the trailing edge, and its circle angles."""
        phi = self.phi_te + 2.0 * np.pi * np.arange(n_panels + 1) / n_panels
        coords = self.surface(phi)
        coords[-1] = coords[0]
        return coords, phi

    def cl(self, alpha_deg: float, chord: float) -> float:
        """Exact lift coefficient (Kutta-Joukowski) for a given reference chord."""
        return 2.0 * self.circulation(alpha_deg) / chord
