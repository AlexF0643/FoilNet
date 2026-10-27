import numpy as np
import pytest

from foilnet.analysis.calibration import (
    calibration_curve,
    calibration_report,
    coverage,
    error_spread_correlation,
    expected_coverage,
    rms_z,
)


def gaussian_case(n=20000, seed=0, scale=1.0):
    """Errors drawn as N(0, (scale * sigma)^2) with a different sigma per sample.

    scale = 1 is a perfectly calibrated ensemble, > 1 over-confident, < 1 under-confident."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.5, 2.0, n)
    return rng.normal(0.0, 1.0, n) * sigma * scale, sigma


def test_expected_coverage_matches_the_gaussian_rule():
    assert expected_coverage(1.0) == pytest.approx(0.6827, abs=1e-4)
    assert expected_coverage(2.0) == pytest.approx(0.9545, abs=1e-4)
    assert expected_coverage(3.0) == pytest.approx(0.9973, abs=1e-4)


def test_calibrated_ensemble_hits_the_nominal_coverage():
    err, sigma = gaussian_case()
    for z in (1.0, 2.0, 3.0):
        assert coverage(err, sigma, z) == pytest.approx(expected_coverage(z), abs=0.01)
    nominal, observed = calibration_curve(err, sigma)
    assert np.abs(nominal - observed).max() < 0.02
    assert rms_z(err, sigma) == pytest.approx(1.0, abs=0.03)


def test_overconfident_ensemble_is_detected():
    err, sigma = gaussian_case(scale=3.0)
    assert coverage(err, sigma, 1.0) < 0.3
    assert rms_z(err, sigma) == pytest.approx(3.0, rel=0.05)
    nominal, observed = calibration_curve(err, sigma)
    assert np.all(observed < nominal)  # below the diagonal everywhere


def test_underconfident_ensemble_is_detected():
    err, sigma = gaussian_case(scale=0.3)
    assert coverage(err, sigma, 1.0) > 0.99
    assert rms_z(err, sigma) == pytest.approx(0.3, rel=0.05)


def test_spearman_separates_informative_from_useless_spread():
    err, sigma = gaussian_case()
    assert error_spread_correlation(err, sigma) > 0.2  # spread really scales the errors
    shuffled = np.random.default_rng(1).permutation(sigma)
    assert abs(error_spread_correlation(err, shuffled)) < 0.05


def test_report_collects_everything():
    err, sigma = gaussian_case(n=500)
    rep = calibration_report(err, sigma)
    assert rep.n == 500
    assert set(rep.coverage) == {1.0, 2.0, 3.0}
    expected, observed = rep.coverage[2.0]
    assert expected == pytest.approx(0.9545, abs=1e-4) and 0.9 < observed <= 1.0
    assert len(rep.nominal) == len(rep.observed)


@pytest.mark.parametrize(
    "err,sigma",
    [
        (np.zeros(3), np.ones(4)),
        (np.zeros(3), np.array([1.0, 0.0, 1.0])),
        (np.zeros(3), -np.ones(3)),
        (np.array([]), np.array([])),
        (np.zeros((2, 2)), np.ones((2, 2))),
    ],
)
def test_bad_inputs_rejected(err, sigma):
    with pytest.raises(ValueError):
        coverage(err, sigma, 1.0)
