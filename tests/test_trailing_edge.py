"""The trailing-edge treatment: the spurious spike is gone and nothing else got worse."""

import numpy as np
import pytest

from foilnet.data import ShapeFamily
from foilnet.geometry import naca4, naca4_from_code
from foilnet.solver import LEGACY_SOLVER_ID, SOLVER_ID, PanelSolver, solve
from foilnet.solver.analytic import KarmanTrefftz
from foilnet.solver.panel import _influence_matrix, _trailing_edge_map

# A cambered, thin section at a high angle: the legacy system puts Cp = -23 on the last panel.
SPIKE_SHAPE = (0.05, 0.225, 0.071)
SPIKE_ALPHA = 11.374


def spike(cp: np.ndarray) -> bool:
    """A trailing-edge panel more than 1 below its five neighbours (on either surface)."""
    return bool(cp[0] < cp[1:6].min() - 1.0 or cp[-1] < cp[-6:-1].min() - 1.0)


def test_solver_ids_differ():
    assert SOLVER_ID != LEGACY_SOLVER_ID


def test_rejects_unknown_treatment():
    with pytest.raises(ValueError, match="te_treatment"):
        PanelSolver(naca4(0.02, 0.4, 0.12, 40), "wrong")


def test_legacy_treatment_still_shows_the_spike():
    """Kept as a record of the problem: if this stops failing, the comparison is meaningless."""
    legacy = solve(naca4(*SPIKE_SHAPE, 160), SPIKE_ALPHA, "legacy")
    assert legacy.cp[0] < -10.0 and legacy.cp[-1] < -10.0


def test_smooth_treatment_removes_the_spike_at_every_resolution():
    for n_panels in (80, 160, 320, 640):
        cp = solve(naca4(*SPIKE_SHAPE, n_panels), SPIKE_ALPHA).cp
        assert not spike(cp)
        assert 0.0 < cp[0] < 1.0 and 0.0 < cp[-1] < 1.0


def test_the_weak_mode_is_the_cause():
    """The legacy system has a near-null mode at the trailing edge; the smooth one does not."""
    panels = PanelSolver(naca4(*SPIKE_SHAPE, 160)).panels
    matrix = _influence_matrix(panels)
    for smooth, bound in ((False, 1e-4), (True, 1e-3)):
        system = matrix @ _trailing_edge_map(panels.n, smooth)
        sv = np.linalg.svd(system, compute_uv=False)
        if smooth:
            assert sv[-1] / sv[0] > bound
        else:
            assert sv[-1] / sv[0] < bound


def test_kutta_condition_still_holds_exactly():
    sol = solve(naca4_from_code("2412", 160), 4.0)
    assert sol.gamma[0] + sol.gamma[-1] == pytest.approx(0.0, abs=1e-14)
    assert sol.gamma[0] == pytest.approx(0.5 * (sol.gamma[1] - sol.gamma[-2]), abs=1e-14)


@pytest.mark.parametrize("code", ["0012", "2412", "4412", "6409"])
@pytest.mark.parametrize("alpha", [-4.0, 4.0, 10.0])
def test_equations_are_consistent(code, alpha):
    """n equations, n - 1 unknowns: they must agree to discretisation error."""
    solver = PanelSolver(naca4_from_code(code, 160))
    assert solver.max_residual < 1e-3


@pytest.mark.parametrize("code", ["0012", "2412", "4412", "6409", "2430"])
@pytest.mark.parametrize("alpha", [-4.0, 4.0, 10.0])
def test_lift_and_moment_barely_change(code, alpha):
    coords = naca4_from_code(code, 160)
    new, old = solve(coords, alpha), solve(coords, alpha, "legacy")
    assert new.cl == pytest.approx(old.cl, abs=5e-4)
    assert new.cm == pytest.approx(old.cm, abs=5e-4)


def test_no_spikes_over_random_shapes():
    rng = np.random.default_rng(0)
    for _ in range(60):
        m, p, t = rng.uniform(0.0, 0.06), rng.uniform(0.2, 0.6), rng.uniform(0.06, 0.18)
        cp = solve(naca4(m, p, t, 160), rng.uniform(-5, 12)).cp
        assert not spike(cp)


def test_no_spikes_over_random_cst_shapes():
    family = ShapeFamily("cst", 4)
    rng = np.random.default_rng(1)
    lo = np.array([0.08, 0.06, 0.05, 0.05, -0.26, -0.24, -0.22, -0.22])
    hi = np.array([0.48, 0.40, 0.42, 0.55, 0.04, 0.18, 0.12, 0.25])
    done = 0
    while done < 25:
        shape = rng.uniform(lo, hi)
        if not family.is_valid(shape):
            continue
        done += 1
        cp = solve(family.build(shape, 160), rng.uniform(-5, 12)).cp
        assert not spike(cp)


def test_pressure_at_the_trailing_edge_is_closer_to_exact():
    kt = KarmanTrefftz(n=1.95, xc=-0.08, yc=0.0)
    coords, phi = kt.contour(160)
    exact = kt.cp(0.5 * (phi[:-1] + phi[1:]), 10.0)
    new = np.abs(solve(coords, 10.0).cp - exact)[[0, -1]].max()
    old = np.abs(solve(coords, 10.0, "legacy").cp - exact)[[0, -1]].max()
    assert new < 0.15 and new < 0.5 * old


@pytest.mark.parametrize("n", [1.8, 1.9, 1.95])
def test_lift_still_matches_the_exact_solution(n):
    kt = KarmanTrefftz(n=n, xc=-0.1, yc=0.05)
    coords, _ = kt.contour(160)
    sol = solve(coords, 6.0)
    assert abs(sol.cl - kt.cl(6.0, sol.chord)) < 1e-3
