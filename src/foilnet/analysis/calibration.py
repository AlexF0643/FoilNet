"""Is the ensemble's uncertainty honest?

A prediction with spread ``sigma`` is *calibrated* if the real error falls inside ``+/- z * sigma``
about as often as a Gaussian says it should (68% of the time for z = 1, 95% for z = 2). These
functions measure that, for one target at a time.

``err`` is ``prediction - truth`` and ``sigma`` is the ensemble spread, both 1-D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, spearmanr


def _check(err: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    err, sigma = np.asarray(err, dtype=float), np.asarray(sigma, dtype=float)
    if err.shape != sigma.shape or err.ndim != 1 or len(err) == 0:
        raise ValueError(
            f"err and sigma must be equal-length 1-D arrays, got {err.shape} and {sigma.shape}"
        )
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    return err, sigma


def coverage(err: np.ndarray, sigma: np.ndarray, z: float) -> float:
    """Fraction of samples with ``|err| <= z * sigma``."""
    err, sigma = _check(err, sigma)
    return float(np.mean(np.abs(err) <= z * sigma))


def expected_coverage(z: float) -> float:
    """What a perfectly calibrated Gaussian gives for ``|err| <= z * sigma``."""
    return float(2.0 * norm.cdf(z) - 1.0)


def calibration_curve(
    err: np.ndarray, sigma: np.ndarray, levels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(nominal, observed)``: for each central-interval level, how often the error was inside.

    A calibrated ensemble lies on the diagonal; below it means over-confident.
    """
    err, sigma = _check(err, sigma)
    levels = np.linspace(0.05, 0.99, 20) if levels is None else np.asarray(levels, dtype=float)
    z = norm.ppf(0.5 + 0.5 * levels)
    observed = np.array([np.mean(np.abs(err) <= zi * sigma) for zi in z])
    return levels, observed


def rms_z(err: np.ndarray, sigma: np.ndarray) -> float:
    """Root-mean-square of ``err / sigma``. 1 is calibrated; above 1 is over-confident
    (the spread is smaller than the real error), below 1 under-confident."""
    err, sigma = _check(err, sigma)
    return float(np.sqrt(np.mean((err / sigma) ** 2)))


def error_spread_correlation(err: np.ndarray, sigma: np.ndarray) -> float:
    """Spearman rank correlation between ``|err|`` and ``sigma``: does a larger spread
    point at the samples that are actually worse? 1 is perfect, 0 is no relationship."""
    err, sigma = _check(err, sigma)
    rho = spearmanr(np.abs(err), sigma).statistic
    return float(rho)


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    rms_z: float
    spearman: float
    coverage: dict[float, tuple[float, float]]  # z -> (expected, observed)
    nominal: np.ndarray
    observed: np.ndarray


def calibration_report(
    err: np.ndarray, sigma: np.ndarray, zs: tuple[float, ...] = (1.0, 2.0, 3.0)
) -> CalibrationReport:
    err, sigma = _check(err, sigma)
    nominal, observed = calibration_curve(err, sigma)
    return CalibrationReport(
        n=len(err),
        rms_z=rms_z(err, sigma),
        spearman=error_spread_correlation(err, sigma),
        coverage={z: (expected_coverage(z), coverage(err, sigma, z)) for z in zs},
        nominal=nominal,
        observed=observed,
    )
