"""Panel-count convergence: results must stop changing as panels increase."""

import numpy as np
import pytest

from foilnet.geometry import naca4_from_code
from foilnet.solver import solve

REFERENCE_PANELS = 1280
PANEL_COUNTS = [80, 160, 320]


@pytest.fixture(scope="module")
def study():
    out = {}
    for n in [*PANEL_COUNTS, REFERENCE_PANELS]:
        sol = solve(naca4_from_code("2412", n), 4.0)
        out[n] = (sol.cl, sol.cm)
    return out


@pytest.mark.parametrize("which,name", [(0, "cl"), (1, "cm")])
def test_error_shrinks_with_every_doubling(study, which, name):
    ref = study[REFERENCE_PANELS][which]
    errors = [abs(study[n][which] - ref) for n in PANEL_COUNTS]
    assert errors[0] > errors[1] > errors[2], f"{name} errors not decreasing: {errors}"
    # a second-order method should cut the error about 4x per doubling; require better than 2x
    assert errors[0] / errors[1] > 2.0
    assert errors[1] / errors[2] > 2.0


def test_converged_values_are_stable(study):
    assert abs(study[320][0] - study[REFERENCE_PANELS][0]) < 1e-4
    assert abs(study[160][0] - study[REFERENCE_PANELS][0]) < 5e-4
    assert abs(study[320][1] - study[REFERENCE_PANELS][1]) < 1e-4


def test_default_panel_count_is_converged_enough_for_dataset_use():
    # 160 panels is the planned dataset setting; lift must be within 0.1% of the fine result.
    ref = solve(naca4_from_code("0012", REFERENCE_PANELS), 8.0).cl
    cl = solve(naca4_from_code("0012", 160), 8.0).cl
    assert cl == pytest.approx(ref, rel=1e-3)
    assert np.isfinite(cl)
