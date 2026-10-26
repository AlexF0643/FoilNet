"""Linear-vortex panel method with a Kutta condition (inviscid, incompressible).

Method (after Katz & Plotkin, *Low-Speed Aerodynamics*, ch. 11): the body contour is
replaced by straight panels carrying a vortex sheet whose strength varies linearly
along each panel and is continuous from panel to panel. The unknowns are the sheet
strengths at the contour nodes. Zero normal velocity is imposed at every panel
midpoint, and a Kutta condition closes the system at the trailing edge.

Conventions
-----------
* The contour is closed and counter-clockwise, as produced by ``foilnet.geometry``.
* Sheet strength ``gamma`` is counter-clockwise positive, so just outside the body
  the flow speed along the direction of travel *is* ``gamma``. Cp = 1 - gamma^2.
* Lift follows from Kutta-Joukowski: Cl = -2 * (sum of gamma * ds) / (V * chord).
* Cm is about the quarter-chord point, nose-up positive.
* Freestream speed is 1, angle of attack ``alpha`` is measured from the x axis.

Limits: no viscosity, so no stall and no drag. Pressure drag from integrating Cp is
reported only as a numerical-error diagnostic (d'Alembert says it should be zero).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from foilnet.geometry.paneling import Panels, build_panels

_INV_2PI = 1.0 / (2.0 * np.pi)

# Recorded in every dataset's metadata. Change it whenever a code change alters results.
SOLVER_ID = "linear-vortex-kutta-v1"


@dataclass(frozen=True)
class PanelSolution:
    """Result of one solve."""

    alpha_deg: float
    cl: float  # Kutta-Joukowski
    cm: float  # about quarter chord, nose-up positive
    cl_pressure: float  # from integrating Cp, cross-check on ``cl``
    cd_pressure: float  # should be ~0 (d'Alembert); a measure of numerical error
    cp: np.ndarray  # (n,) at panel midpoints
    gamma: np.ndarray  # (n + 1,) vortex strength at nodes
    panels: Panels
    chord: float


def _influence_matrix(panels: Panels) -> np.ndarray:
    """(n + 1, n + 1) system: n zero-normal-velocity rows plus the Kutta row."""
    n = panels.n
    d = panels.midpoint[:, None, :] - panels.start[None, :, :]  # control point i, panel j
    t = panels.tangent[None, :, :]
    left = np.stack([-t[..., 1], t[..., 0]], axis=-1)  # left-hand normal of panel j
    length = panels.length[None, :]

    xi = np.sum(d * t, axis=-1)
    eta = np.sum(d * left, axis=-1)
    r1sq = xi**2 + eta**2
    r2sq = (xi - length) ** 2 + eta**2
    dtheta = np.arctan2(eta, xi - length) - np.arctan2(eta, xi)  # angle subtended by the panel
    ln_r = 0.5 * np.log(r1sq / r2sq)

    # Integrals of the sheet kernel against 1 and s along the panel (s = distance from start).
    a1 = xi * dtheta - eta * ln_r
    b1 = xi * ln_r - length + eta * dtheta

    # Velocity in panel-j axes (u along tangent, v along left normal) per unit gamma at the
    # panel's start node (a) and end node (b).
    u_a, u_b = -_INV_2PI * (dtheta - a1 / length), -_INV_2PI * a1 / length
    v_a, v_b = _INV_2PI * (ln_r - b1 / length), _INV_2PI * b1 / length

    # Project onto the outward normal of control point i.
    n_i = panels.normal[:, None, :]
    t_dot_n = np.sum(t * n_i, axis=-1)
    left_dot_n = np.sum(left * n_i, axis=-1)
    coef_a = u_a * t_dot_n + v_a * left_dot_n
    coef_b = u_b * t_dot_n + v_b * left_dot_n

    matrix = np.zeros((n + 1, n + 1))
    matrix[:n, :n] += coef_a
    matrix[:n, 1:] += coef_b
    matrix[n, 0] = matrix[n, n] = 1.0  # Kutta: gamma at the two trailing-edge nodes cancel
    return matrix


class PanelSolver:
    """Panel solver for one contour; solving a new angle of attack is almost free.

    The influence matrix does not depend on alpha, and the right-hand side is linear in
    (cos alpha, sin alpha), so two base solutions are computed once and combined.
    """

    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        self.panels = build_panels(self.coords)
        x = self.coords[:, 0]
        self.chord = float(x.max() - x.min())
        i_le = int(np.argmin(x))
        self._ref = np.array([x.min() + 0.25 * self.chord, self.coords[i_le, 1]])

        n = self.panels.n
        lu = lu_factor(_influence_matrix(self.panels))
        rhs = np.zeros((n + 1, 2))
        rhs[:n] = -self.panels.normal  # columns: freestream along x, along y
        base = lu_solve(lu, rhs)
        self._gamma_x, self._gamma_y = base[:, 0], base[:, 1]

    def solve(self, alpha_deg: float) -> PanelSolution:
        a = np.deg2rad(alpha_deg)
        gamma = np.cos(a) * self._gamma_x + np.sin(a) * self._gamma_y
        panels = self.panels

        gamma_mid = 0.5 * (gamma[:-1] + gamma[1:])
        cp = 1.0 - gamma_mid**2

        circulation = float(np.sum(panels.length * gamma_mid))
        cl = -2.0 * circulation / self.chord

        # Pressure forces per unit dynamic pressure: dF = -Cp * n * ds.
        force = -(cp * panels.length)[:, None] * panels.normal
        total = force.sum(axis=0) / self.chord
        drag_dir = np.array([np.cos(a), np.sin(a)])
        lift_dir = np.array([-np.sin(a), np.cos(a)])
        arm = panels.midpoint - self._ref
        moment_z = np.sum(arm[:, 0] * force[:, 1] - arm[:, 1] * force[:, 0])
        cm = -float(moment_z) / self.chord**2  # nose-up positive

        return PanelSolution(
            alpha_deg=float(alpha_deg),
            cl=cl,
            cm=cm,
            cl_pressure=float(total @ lift_dir),
            cd_pressure=float(total @ drag_dir),
            cp=cp,
            gamma=gamma,
            panels=panels,
            chord=self.chord,
        )


def solve(coords: np.ndarray, alpha_deg: float) -> PanelSolution:
    """Solve one contour at one angle of attack."""
    return PanelSolver(coords).solve(alpha_deg)
