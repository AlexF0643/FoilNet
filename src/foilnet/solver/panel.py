"""Linear-vortex panel method with a Kutta condition (inviscid, incompressible).

Method (after Katz & Plotkin, *Low-Speed Aerodynamics*, ch. 11): the body contour is
replaced by straight panels carrying a vortex sheet whose strength varies linearly
along each panel and is continuous from panel to panel. The unknowns are the sheet
strengths at the contour nodes. Zero normal velocity is imposed at every panel
midpoint, and a Kutta condition closes the system at the trailing edge.

Trailing edge
-------------
Nodes 1 and n + 1 of a closed contour are the same point. The Kutta condition
gamma_1 = -gamma_(n+1) alone leaves the *size* of that pair almost undetermined: equal and
opposite strengths on two tiny, nearly parallel panels meeting at one point induce almost
no velocity anywhere, so the system has a mode with a singular value about 1e-5 of the
largest. The solution then carries an arbitrary multiple of that mode, which shows as a
spurious spike in Cp on the single trailing-edge panel (Cp below -20 on some shapes) that
does not shrink with more panels. The ``"smooth"`` treatment (the default) also ties the
trailing-edge strength to its neighbours, gamma_1 = (gamma_2 - gamma_n) / 2, which removes
the mode; the n equations then have n - 1 unknowns and are solved in the least-squares sense
(they are consistent to discretisation error). ``"legacy"`` is the original Kutta-only
system (solver id ``linear-vortex-kutta-v1``), kept so the difference can be reproduced.

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
from scipy.linalg import lstsq, lu_factor, lu_solve

from foilnet.geometry.paneling import Panels, build_panels

_INV_2PI = 1.0 / (2.0 * np.pi)

# Recorded in every dataset's metadata. Change it whenever a code change alters results.
SOLVER_ID = "linear-vortex-kutta-v2"
LEGACY_SOLVER_ID = "linear-vortex-kutta-v1"
TE_TREATMENTS = ("smooth", "legacy")


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
    """(n, n + 1): normal velocity at each control point per unit node strength."""
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

    matrix = np.zeros((n, n + 1))
    matrix[:, :n] += coef_a
    matrix[:, 1:] += coef_b
    return matrix


def _trailing_edge_map(n: int, smooth: bool) -> np.ndarray:
    """(n + 1, k): node strengths as a linear function of the free unknowns.

    Legacy: k = n, the two trailing-edge nodes are one unknown with opposite signs (Kutta).
    Smooth: k = n - 1, the interior nodes are free and the trailing-edge pair follows from
    gamma_1 = (gamma_2 - gamma_n) / 2 and gamma_(n+1) = -gamma_1.
    """
    if smooth:
        t = np.zeros((n + 1, n - 1))
        t[np.arange(1, n), np.arange(n - 1)] = 1.0
        t[0, 0], t[0, n - 2] = 0.5, -0.5
        t[n] = -t[0]
        return t
    t = np.zeros((n + 1, n))
    t[np.arange(n), np.arange(n)] = 1.0
    t[n, 0] = -1.0  # gamma_(n+1) = -gamma_1 (Kutta)
    return t


class PanelSolver:
    """Panel solver for one contour; solving a new angle of attack is almost free.

    The influence matrix does not depend on alpha, and the right-hand side is linear in
    (cos alpha, sin alpha), so two base solutions are computed once and combined.
    """

    def __init__(self, coords: np.ndarray, te_treatment: str = "smooth"):
        if te_treatment not in TE_TREATMENTS:
            raise ValueError(f"te_treatment must be one of {TE_TREATMENTS}, got {te_treatment!r}")
        self.te_treatment = te_treatment
        self.coords = np.asarray(coords, dtype=float)
        self.panels = build_panels(self.coords)
        x = self.coords[:, 0]
        self.chord = float(x.max() - x.min())
        i_le = int(np.argmin(x))
        self._ref = np.array([x.min() + 0.25 * self.chord, self.coords[i_le, 1]])

        n = self.panels.n
        rhs = -self.panels.normal  # (n, 2): freestream along x, along y
        t_map = _trailing_edge_map(n, te_treatment == "smooth")
        system = _influence_matrix(self.panels) @ t_map
        if te_treatment == "smooth":
            unknowns, *_ = lstsq(system, rhs, lapack_driver="gelsy")
        else:
            unknowns = lu_solve(lu_factor(system), rhs)
        base = t_map @ unknowns
        self._gamma_x, self._gamma_y = base[:, 0], base[:, 1]
        # largest normal velocity left over at a control point, per unit freestream
        self.max_residual = float(np.abs(system @ unknowns - rhs).max())

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


def solve(coords: np.ndarray, alpha_deg: float, te_treatment: str = "smooth") -> PanelSolution:
    """Solve one contour at one angle of attack."""
    return PanelSolver(coords, te_treatment).solve(alpha_deg)
