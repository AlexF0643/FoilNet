import numpy as np
import pytest
from scipy.integrate import quad

from foilnet.geometry import (
    ShapeLimits,
    bernstein_basis,
    build_panels,
    check_shape,
    class_function,
    cst_coordinates,
    cst_surface,
    fit_cst,
    naca4_from_code,
    signed_area,
    te_included_angle_deg,
    thickness_and_camber,
)
from foilnet.solver import solve

# a plain, valid shape: roughly a 12% section with a little camber
UPPER = np.array([0.20, 0.17, 0.18, 0.12])
LOWER = np.array([-0.16, -0.12, -0.10, -0.02])


def test_bernstein_basis_is_a_partition_of_unity():
    x = np.linspace(0, 1, 11)
    for n in (1, 2, 4, 7):
        basis = bernstein_basis(x, n)
        assert basis.shape == (11, n)
        assert np.allclose(basis.sum(axis=1), 1.0)
        assert np.all(basis >= 0)
    assert np.array_equal(bernstein_basis(x, 1), np.ones((11, 1)))


def test_bernstein_rejects_zero_coefficients():
    with pytest.raises(ValueError):
        bernstein_basis(np.array([0.5]), 0)


def test_class_function_vanishes_at_both_ends_with_a_round_nose():
    assert class_function(np.array([0.0, 1.0])) == pytest.approx([0.0, 0.0])
    x = 1e-6
    assert float(class_function(np.array(x))) == pytest.approx(np.sqrt(x), rel=1e-3)


def test_surface_is_linear_in_its_coefficients():
    x = np.linspace(0.01, 0.99, 25)
    a, b = np.array([0.1, 0.2, 0.1]), np.array([0.3, -0.1, 0.2])
    assert np.allclose(cst_surface(x, 2 * a + b), 2 * cst_surface(x, a) + cst_surface(x, b))
    assert np.all(cst_surface(x, np.zeros(3)) == 0.0)


@pytest.mark.parametrize("n_panels", [40, 160])
def test_contour_is_closed_ccw_and_anchored(n_panels):
    coords = cst_coordinates(UPPER, LOWER, n_panels)
    assert coords.shape == (n_panels + 1, 2)
    assert np.array_equal(coords[0], coords[-1])
    assert coords[0] == pytest.approx([1.0, 0.0], abs=1e-12)
    assert coords[n_panels // 2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert signed_area(coords) > 0
    assert build_panels(coords).n == n_panels


def test_mirror_symmetric_coefficients_give_a_symmetric_contour():
    coords = cst_coordinates(UPPER, -UPPER, 80)
    mirrored = coords[::-1] * np.array([1.0, -1.0])
    assert np.array_equal(coords[:, 0], mirrored[:, 0])
    assert np.allclose(coords[:, 1], mirrored[:, 1], atol=1e-15)


def test_contour_area_matches_integrated_thickness():
    exact, _ = quad(lambda x: float(cst_surface(np.array([x]), UPPER - LOWER)[0]), 0, 1)
    assert signed_area(cst_coordinates(UPPER, LOWER, 600)) == pytest.approx(exact, rel=2e-3)


def test_invalid_panel_counts_rejected():
    for bad in (2, 7):
        with pytest.raises(ValueError):
            cst_coordinates(UPPER, LOWER, bad)


def test_trailing_edge_angle_matches_the_surfaces_near_the_edge():
    eps = 1e-5
    near, edge = np.array([1.0 - eps]), np.array([1.0])
    slope_up = (cst_surface(edge, UPPER)[0] - cst_surface(near, UPPER)[0]) / eps
    slope_lo = (cst_surface(edge, LOWER)[0] - cst_surface(near, LOWER)[0]) / eps
    measured = np.degrees(np.arctan(-slope_up) - np.arctan(-slope_lo))
    assert te_included_angle_deg(UPPER, LOWER) == pytest.approx(measured, abs=0.05)
    assert te_included_angle_deg(UPPER, LOWER) == pytest.approx(
        np.degrees(np.arctan(0.12) + np.arctan(0.02))
    )


def test_fitting_recovers_known_coefficients_exactly():
    coords = cst_coordinates(UPPER, LOWER, 400)
    upper, lower = fit_cst(coords, 4)
    assert np.allclose(upper, UPPER, atol=1e-9)
    assert np.allclose(lower, LOWER, atol=1e-9)


def test_fit_of_a_naca_section_reproduces_its_thickness_and_trailing_edge():
    upper, lower = fit_cst(naca4_from_code("0012", 320), 6)
    thickness, camber = thickness_and_camber(upper, lower)
    assert thickness == pytest.approx(0.12, abs=2e-3)
    assert camber == pytest.approx(0.0, abs=2e-4)
    assert te_included_angle_deg(upper, lower) == pytest.approx(16.5, abs=1.0)


@pytest.mark.parametrize("code,tolerance", [("0012", 1e-3), ("2412", 0.008), ("1306", 1e-3)])
def test_cst_fit_of_a_naca_section_gives_nearly_the_same_lift(code, tolerance):
    # Four to six coefficients cannot reproduce every NACA leading edge exactly, so this is a
    # loose check (under 1%): the point is that the CST geometry and the solver agree.
    reference = naca4_from_code(code, 320)
    upper, lower = fit_cst(reference, 6)
    cl_naca = solve(reference, 4.0).cl
    cl_cst = solve(cst_coordinates(upper, lower, 320), 4.0).cl
    assert abs(cl_cst - cl_naca) / cl_naca < tolerance


def test_symmetric_cst_shape_has_no_lift_at_zero_alpha_and_odd_lift():
    coords = cst_coordinates(UPPER, -UPPER, 160)
    assert solve(coords, 0.0).cl == pytest.approx(0.0, abs=1e-10)
    assert solve(coords, -4.0).cl == pytest.approx(-solve(coords, 4.0).cl, abs=1e-10)


def test_solver_is_accurate_at_the_minimum_trailing_edge_angle():
    # included angle just above the 5 degree limit: 160 panels must agree with 640 panels
    upper = np.array([0.20, 0.17, 0.18, 0.047])
    lower = np.array([-0.16, -0.12, -0.10, -0.047])
    assert te_included_angle_deg(upper, lower) == pytest.approx(5.4, abs=0.1)
    coarse = solve(cst_coordinates(upper, lower, 160), 8.0).cl
    fine = solve(cst_coordinates(upper, lower, 640), 8.0).cl
    assert abs(coarse - fine) < 1e-3 * max(1.0, abs(fine))


# --- validity checks ----------------------------------------------------------------------


def test_a_plain_shape_passes_every_check():
    assert check_shape(UPPER, LOWER, ShapeLimits()) == []


@pytest.mark.parametrize(
    "upper,lower,expected",
    [
        (-UPPER, -LOWER, "cross"),  # upper surface below the lower
        (UPPER * 0.2, LOWER * 0.2, "below"),  # too thin
        (UPPER * 3, LOWER * 3, "above"),  # too thick
        (UPPER + 0.3, LOWER + 0.3, "camber"),  # shifted upward: too much camber
        (np.array([0.2, 0.17, 0.18, 0.01]), np.array([-0.16, -0.12, -0.1, -0.01]), "trailing-edge"),
        (UPPER, np.array([0.02, -0.12, -0.10, -0.02]), "too sharp"),  # lower surface leaves flat
        (np.array([0.06, 0.17, 0.18, 0.12]), LOWER, "too sharp"),  # upper nose too small
        (np.array([0.50, 0.17, 0.18, 0.12]), np.array([-0.11, -0.12, -0.10, -0.02]), "lopsided"),
    ],
)
def test_bad_shapes_are_reported(upper, lower, expected):
    problems = check_shape(upper, lower, ShapeLimits())
    assert problems and any(expected in p for p in problems)


def test_thickness_and_camber_of_a_known_shape():
    thickness, camber = thickness_and_camber(UPPER, -UPPER)
    assert camber == pytest.approx(0.0, abs=1e-12)
    grid = np.linspace(0, 1, 2001)
    assert thickness == pytest.approx(2 * float(np.max(cst_surface(grid, UPPER))), rel=1e-4)
