"""Panel solver versus exact and closed-form solutions.

Stated tolerances (160 panels): Cl within 0.1% of max(1, |Cl|) of the exact
Karman-Trefftz result, and Cp within 0.02 + 2% of the local |Cp| away from the
trailing edge. The error is largest at the leading-edge suction peak and falls about
4x per doubling of panels (see ``test_cp_error_converges_at_the_suction_peak``).
"""

import numpy as np
import pytest

from foilnet.geometry import build_panels, naca4_from_code
from foilnet.solver import PanelSolver, solve
from foilnet.solver.analytic import (
    KarmanTrefftz,
    thin_airfoil_naca4,
    thin_airfoil_theory,
)

# Finite trailing-edge angles (n < 2), cambered and not, circle centre moved around.
KT_SHAPES = [
    KarmanTrefftz(n=1.9, xc=-0.1, yc=0.1),
    KarmanTrefftz(n=1.8, xc=-0.15, yc=0.05),
    KarmanTrefftz(n=1.85, xc=-0.1, yc=-0.1),
    KarmanTrefftz(n=1.9, xc=-0.08, yc=0.0),
]
ALPHAS = [-4.0, 0.0, 3.0, 10.0]


# --- the exact solution itself, checked without the solver -----------------------------


def test_kt_with_n2_is_the_joukowski_map():
    kt = KarmanTrefftz(n=2.0, a=1.0, xc=-0.1, yc=0.1)
    coords, phi = kt.contour(400)
    zeta = kt._zeta(phi)
    joukowski = zeta + 1.0 / zeta
    assert np.abs(joukowski - (coords[:, 0] + 1j * coords[:, 1])).max() < 1e-12


@pytest.mark.parametrize("kt", KT_SHAPES)
@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_kt_exact_cp_integrates_to_exact_lift_and_zero_drag(kt, alpha):
    coords, phi = kt.contour(3000)
    panels = build_panels(coords)
    chord = coords[:, 0].max() - coords[:, 0].min()
    cp = kt.cp(0.5 * (phi[:-1] + phi[1:]), alpha)
    force = (-(cp * panels.length)[:, None] * panels.normal).sum(axis=0) / chord
    a = np.deg2rad(alpha)
    assert force @ [-np.sin(a), np.cos(a)] == pytest.approx(kt.cl(alpha, chord), rel=1e-5)
    assert abs(force @ [np.cos(a), np.sin(a)]) < 1e-5


@pytest.mark.parametrize("n", [1.8, 1.9])
def test_kt_trailing_edge_angle(n):
    kt = KarmanTrefftz(n=n)
    eps = 1e-3
    te = kt.surface(np.array([kt.phi_te + 2 * np.pi]))[0]  # w = 0 exactly: z = n a
    upper = kt.surface(np.array([kt.phi_te + eps]))[0] - te
    lower = kt.surface(np.array([kt.phi_te + 2 * np.pi - eps]))[0] - te
    cos_angle = upper @ lower / (np.linalg.norm(upper) * np.linalg.norm(lower))
    assert np.degrees(np.arccos(cos_angle)) == pytest.approx(np.degrees((2 - n) * np.pi), abs=0.5)


@pytest.mark.parametrize("kwargs", [dict(n=1.0), dict(n=2.5), dict(n=1.9, xc=0.5, yc=0.0)])
def test_kt_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        KarmanTrefftz(**kwargs)


# --- thin-aerofoil theory ------------------------------------------------------------


def test_thin_theory_flat_plate_is_two_pi_alpha():
    out = thin_airfoil_theory(lambda x: 0.0, 3.0)
    assert out["cl"] == pytest.approx(2 * np.pi * np.deg2rad(3.0))
    assert out["cm_c4"] == pytest.approx(0.0, abs=1e-12)
    assert out["alpha_l0_deg"] == pytest.approx(0.0, abs=1e-12)


def test_thin_theory_parabolic_arc_closed_form():
    # y = 4 h x (1 - x): cl = 2 pi alpha + 4 pi h, cm_c/4 = -pi h (hand derivation)
    h, alpha = 0.03, 2.0
    out = thin_airfoil_theory(lambda x: 4 * h * (1 - 2 * x), alpha)
    assert out["cl"] == pytest.approx(2 * np.pi * np.deg2rad(alpha) + 4 * np.pi * h, rel=1e-9)
    assert out["cm_c4"] == pytest.approx(-np.pi * h, rel=1e-9)


def test_thin_theory_naca4_is_self_consistent():
    out = thin_airfoil_naca4(0.04, 0.4, 3.0)
    expected = 2 * np.pi * (np.deg2rad(3.0) - np.deg2rad(out["alpha_l0_deg"]))
    assert out["cl"] == pytest.approx(expected, rel=1e-9)
    assert out["alpha_l0_deg"] < 0  # positive camber lowers the zero-lift angle


# --- solver against the exact Karman-Trefftz solution --------------------------------


@pytest.mark.parametrize("kt", KT_SHAPES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_solver_lift_matches_exact(kt, alpha):
    coords, _ = kt.contour(160)
    sol = solve(coords, alpha)
    exact = kt.cl(alpha, sol.chord)
    assert abs(sol.cl - exact) < 1e-3 * max(1.0, abs(exact))


@pytest.mark.parametrize("kt", KT_SHAPES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_solver_cp_matches_exact_away_from_trailing_edge(kt, alpha):
    coords, phi = kt.contour(160)
    sol = solve(coords, alpha)
    exact = kt.cp(0.5 * (phi[:-1] + phi[1:]), alpha)
    error = np.abs(sol.cp - exact)[3:-3]  # the wedge at the trailing edge converges slowly
    assert np.all(error < 0.02 + 0.02 * np.abs(exact[3:-3]))


def test_cp_error_converges_at_the_suction_peak():
    kt = KarmanTrefftz(n=1.9, xc=-0.08, yc=0.0)
    errors = []
    for n_panels in (160, 320, 640):
        coords, phi = kt.contour(n_panels)
        exact = kt.cp(0.5 * (phi[:-1] + phi[1:]), 10.0)
        errors.append(np.abs(solve(coords, 10.0).cp - exact)[3:-3].max())
    assert errors[0] > 2.0 * errors[1]
    assert errors[1] > 2.0 * errors[2]
    assert errors[2] < 0.01


# --- other physical checks -----------------------------------------------------------


def test_symmetric_section_has_no_lift_or_moment_at_zero_alpha():
    sol = solve(naca4_from_code("0012", 160), 0.0)
    assert sol.cl == pytest.approx(0.0, abs=1e-10)
    assert sol.cm == pytest.approx(0.0, abs=1e-10)
    assert np.abs(sol.cp - sol.cp[::-1]).max() < 1e-10


def test_symmetric_section_lift_is_odd_in_alpha():
    solver = PanelSolver(naca4_from_code("0012", 160))
    assert solver.solve(-4.0).cl == pytest.approx(-solver.solve(4.0).cl, abs=1e-10)


@pytest.mark.parametrize("code", ["0006", "0012", "0018"])
def test_lift_slope_follows_thickness_rule(code):
    # d(Cl)/d(alpha) ~ 2 pi (1 + 0.77 t/c) for NACA sections: the thickness correction
    # that thin-aerofoil theory leaves out.
    t = int(code[2:]) / 100
    solver = PanelSolver(naca4_from_code(code, 320))
    slope = (solver.solve(1.0).cl - solver.solve(0.0).cl) / np.deg2rad(1.0)
    assert slope / (2 * np.pi) == pytest.approx(1 + 0.77 * t, rel=0.015)


@pytest.mark.parametrize("code", ["2406", "4406", "2306"])
def test_thin_cambered_sections_agree_with_thin_theory(code):
    m, p = int(code[0]) / 100, int(code[1]) / 10
    solver = PanelSolver(naca4_from_code(code, 320))
    cl0 = solver.solve(0.0).cl
    alpha_l0 = -cl0 / (solver.solve(1.0).cl - cl0)  # degrees
    thin = thin_airfoil_naca4(m, p, 2.0)
    assert alpha_l0 == pytest.approx(thin["alpha_l0_deg"], abs=0.1)
    assert solver.solve(2.0).cm == pytest.approx(thin["cm_c4"], abs=0.005)


@pytest.mark.parametrize("code,alpha", [("0012", 5.0), ("2412", 4.0), ("0006", 3.0)])
def test_pressure_drag_is_near_zero_and_lifts_agree(code, alpha):
    sol = solve(naca4_from_code(code, 160), alpha)
    assert abs(sol.cd_pressure) < 1e-3  # d'Alembert's paradox, up to discretisation error
    assert sol.cl_pressure == pytest.approx(sol.cl, abs=5e-4)


def test_stagnation_point_pressure_approaches_one():
    # Cp is sampled at panel midpoints, so the peak only reaches 1 as panels are refined.
    coarse = solve(naca4_from_code("0012", 160), 0.0).cp.max()
    fine = solve(naca4_from_code("0012", 640), 0.0).cp.max()
    assert 0.97 < coarse <= 1.0 + 1e-9
    assert fine > 0.995


def test_kutta_condition_holds_at_trailing_edge():
    sol = solve(naca4_from_code("2412", 160), 4.0)
    assert sol.gamma[0] + sol.gamma[-1] == pytest.approx(0.0, abs=1e-12)


def test_flow_inside_the_body_is_near_zero():
    """Independent check of the influence formulas, by numerical quadrature of the sheet."""
    alpha = 5.0
    sol = solve(naca4_from_code("0012", 200), alpha)
    panels = sol.panels
    nodes, weights = np.polynomial.legendre.leggauss(24)
    s = 0.5 * (nodes + 1.0)  # fraction along each panel
    w = 0.5 * weights

    def velocity(point):
        pos = panels.start[:, None, :] + s[None, :, None] * (panels.end - panels.start)[:, None, :]
        gam = sol.gamma[:-1, None] * (1 - s)[None, :] + sol.gamma[1:, None] * s[None, :]
        d = point - pos  # (panel, quad point, 2)
        r2 = np.sum(d**2, axis=-1)
        strength = gam * w[None, :] * panels.length[:, None] / (2 * np.pi * r2)
        # counter-clockwise vortex: velocity = gamma / (2 pi r^2) * (-dy, dx)
        return np.array([np.sum(strength * -d[..., 1]), np.sum(strength * d[..., 0])])

    free = np.array([np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))])
    for point in ([0.5, 0.0], [0.3, 0.01], [0.7, -0.01]):
        assert np.linalg.norm(free + velocity(np.array(point))) < 0.02


def test_solver_object_matches_convenience_function():
    coords = naca4_from_code("4412", 120)
    solver = PanelSolver(coords)
    for alpha in (-3.0, 0.0, 7.5):
        assert solver.solve(alpha).cl == pytest.approx(solve(coords, alpha).cl, abs=1e-12)
