import numpy as np
import pytest
from scipy.integrate import quad

from foilnet.geometry import (
    build_panels,
    cosine_spacing,
    naca4,
    naca4_camber,
    naca4_from_code,
    naca4_thickness,
    parse_naca4,
    signed_area,
)


def test_parse_code():
    assert parse_naca4("2412") == pytest.approx((0.02, 0.4, 0.12))
    assert parse_naca4("0012") == pytest.approx((0.0, 0.0, 0.12))


@pytest.mark.parametrize("bad", ["241", "24120", "abcd", ""])
def test_parse_rejects_bad_codes(bad):
    with pytest.raises(ValueError):
        parse_naca4(bad)


def test_cosine_spacing_endpoints_and_clustering():
    x = cosine_spacing(40)
    assert x[0] == 0.0 and x[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x) > 0)
    assert np.diff(x)[0] < np.diff(x)[20] > np.diff(x)[-1]
    assert np.allclose(x + x[::-1], 1.0)


@pytest.mark.parametrize("code", ["0012", "2412", "4415", "0006"])
def test_trailing_edge_is_closed(code):
    assert naca4_thickness(1.0, 0.12) == pytest.approx(0.0, abs=1e-12)
    coords = naca4_from_code(code, n_panels=100)
    assert coords.shape == (101, 2)
    assert np.array_equal(coords[0], coords[-1])
    assert coords[0] == pytest.approx([1.0, 0.0], abs=1e-12)


def test_leading_edge_at_origin():
    coords = naca4_from_code("2412", n_panels=100)
    assert coords[50] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert coords[:, 0].min() == pytest.approx(0.0, abs=1e-12)


def test_thickness_at_30_percent_chord():
    # NACA 0012 half-thickness at x/c = 0.30, worked by hand from the closed-TE polynomial
    assert float(naca4_thickness(0.3, 0.12)) == pytest.approx(0.060007, abs=1e-6)
    # and the same through the full contour (upper surface runs TE -> LE, so reverse for interp)
    coords = naca4(0.0, 0.0, 0.12, n_panels=400)
    upper = coords[:201][::-1]
    assert np.interp(0.3, upper[:, 0], upper[:, 1]) == pytest.approx(0.060007, abs=1e-4)


def test_maximum_thickness_matches_parameter():
    for t in (0.06, 0.12, 0.18):
        x = np.linspace(0, 1, 20001)
        assert 2 * naca4_thickness(x, t).max() == pytest.approx(t, rel=2e-3)


def test_symmetric_section_stays_symmetric():
    coords = naca4(0.0, 0.0, 0.12, n_panels=120)
    mirrored = coords[::-1] * np.array([1.0, -1.0])
    assert np.array_equal(coords[:, 0], mirrored[:, 0])
    assert np.allclose(coords[:, 1], mirrored[:, 1], atol=1e-15)


def test_symmetric_ignores_camber_position():
    assert np.array_equal(naca4(0.0, 0.0, 0.1), naca4(0.0, 0.5, 0.1))


def test_camber_line_peaks_at_p():
    m, p = 0.04, 0.4
    yc, dyc = naca4_camber(np.array([0.0, p, 1.0]), m, p)
    assert yc == pytest.approx([0.0, m, 0.0], abs=1e-15)
    assert dyc[1] == pytest.approx(0.0, abs=1e-15)
    # continuous slope at x = p
    left, _ = naca4_camber(p - 1e-9, m, p)
    assert float(left) == pytest.approx(m, abs=1e-8)


def test_cambered_section_is_above_symmetric_one():
    cambered = naca4_from_code("4412", 100)
    # mid-chord upper-surface points sit higher than for the symmetric 0012
    sym = naca4_from_code("0012", 100)
    assert cambered[25, 1] > sym[25, 1]
    assert cambered[75, 1] > sym[75, 1]


def test_contour_area_matches_integrated_thickness():
    t = 0.12
    area = signed_area(naca4(0.0, 0.0, t, n_panels=400))
    exact, _ = quad(lambda x: 2 * naca4_thickness(x, t), 0, 1)
    assert area == pytest.approx(exact, rel=1e-3)
    assert area > 0  # counter-clockwise


@pytest.mark.parametrize(
    "args",
    [
        dict(m=0.02, p=0.0, t=0.12),
        dict(m=0.02, p=1.0, t=0.12),
        dict(m=0.0, p=0.0, t=0.0),
        dict(m=0.0, p=0.0, t=1.5),
        dict(m=0.0, p=0.0, t=0.1, n_panels=7),
        dict(m=0.0, p=0.0, t=0.1, n_panels=2),
    ],
)
def test_invalid_shapes_rejected(args):
    with pytest.raises(ValueError):
        naca4(**args)


def test_panels_on_circle():
    n = 64
    ang = np.linspace(0, 2 * np.pi, n + 1)
    circle = np.column_stack([np.cos(ang), np.sin(ang)])
    panels = build_panels(circle)
    assert panels.n == n
    radial = panels.midpoint / np.linalg.norm(panels.midpoint, axis=1)[:, None]
    assert np.allclose(np.sum(panels.normal * radial, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.sum(panels.normal * panels.tangent, axis=1), 0.0, atol=1e-12)
    assert panels.length.sum() == pytest.approx(2 * np.pi, rel=2e-3)


def test_panel_normals_point_out_of_aerofoil():
    panels = build_panels(naca4_from_code("2412", 100))
    upper = slice(10, 40)  # mid-chord on the upper surface
    lower = slice(60, 90)
    assert np.all(panels.normal[upper, 1] > 0)
    assert np.all(panels.normal[lower, 1] < 0)
    assert np.allclose(np.linalg.norm(panels.normal, axis=1), 1.0)
    assert panels.n == 100
    assert np.all(panels.length > 0)


def test_build_panels_rejects_open_contour():
    coords = naca4_from_code("0012", 20)[:-1]
    with pytest.raises(ValueError, match="not closed"):
        build_panels(coords)


def test_build_panels_rejects_clockwise_contour():
    coords = naca4_from_code("0012", 20)[::-1]
    with pytest.raises(ValueError, match="counter-clockwise"):
        build_panels(coords)
