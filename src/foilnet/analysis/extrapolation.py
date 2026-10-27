"""Interpolation versus extrapolation: train on thin aerofoils, test on thick ones.

The training and validation data come only from samples below a threshold in one feature
(thickness below 12% by default). The network is then asked about the rest of the range. The
questions are how fast the error grows as shapes get further from the training data, and
whether the ensemble's spread grows with it, that is, whether the network can tell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foilnet.analysis.calibration import (
    coverage,
    error_spread_correlation,
    rms_z,
)
from foilnet.data import Dataset, random_split
from foilnet.nn.ensemble import Ensemble, train_ensemble
from foilnet.nn.train import TrainConfig


@dataclass(frozen=True)
class ExtrapolationSplit:
    train: Dataset
    val: Dataset
    test_inside: Dataset  # held-out samples from the training range
    outside: Dataset  # everything at or above the threshold


def split_by_feature(
    dataset: Dataset, feature: str, threshold: float, seed: int = 0
) -> ExtrapolationSplit:
    """Samples with ``feature < threshold`` are split 70/15/15 into train, validation and an
    in-range test set; every sample at or above the threshold is the out-of-range test set."""
    inside = dataset.column(feature) < threshold
    if not inside.any() or inside.all():
        raise ValueError(f"threshold {threshold} on {feature!r} leaves one side empty")
    train, val, test_inside = random_split(dataset.subset(inside), seed=seed)
    return ExtrapolationSplit(train, val, test_inside, dataset.subset(~inside))


@dataclass(frozen=True)
class BandResult:
    label: str
    low: float
    high: float
    n: int
    rmse: dict[str, float]
    max_abs_error: dict[str, float]
    mean_sigma: dict[str, float]
    spearman: dict[str, float]  # |error| against spread
    coverage_1sigma: dict[str, float]  # share of errors inside +/- 1 sigma (68% if calibrated)
    rms_z: dict[str, float]  # RMS of error / spread (1 if calibrated)


@dataclass
class ExtrapolationResult:
    feature: str
    threshold: float
    n_train: int
    bands: list[BandResult]
    ensemble: Ensemble
    # per-sample arrays over the in-range test set and the out-of-range set, for plotting
    values: np.ndarray  # feature value
    abs_error: np.ndarray  # (n, n_targets)
    sigma: np.ndarray  # (n, n_targets)
    target_names: tuple[str, ...]


def _band(
    label: str, low: float, high: float, data: Dataset, mean: np.ndarray, sigma: np.ndarray
) -> BandResult:
    err = mean - data.y
    names = data.target_names

    def per_target(fn) -> dict[str, float]:
        return {name: float(fn(err[:, j], sigma[:, j])) for j, name in enumerate(names)}

    return BandResult(
        label=label,
        low=low,
        high=high,
        n=len(data),
        rmse=per_target(lambda e, s: np.sqrt(np.mean(e**2))),
        max_abs_error=per_target(lambda e, s: np.max(np.abs(e))),
        mean_sigma=per_target(lambda e, s: np.mean(s)),
        spearman=per_target(error_spread_correlation),
        coverage_1sigma=per_target(lambda e, s: coverage(e, s, 1.0)),
        rms_z=per_target(rms_z),
    )


def run_extrapolation(
    dataset: Dataset,
    config: TrainConfig,
    feature: str = "thickness",
    threshold: float = 0.12,
    n_bands: int = 3,
    n_members: int = 5,
    workers: int | None = None,
    seed: int = 0,
) -> ExtrapolationResult:
    """Train an ensemble on the in-range samples and score it in bands of increasing distance."""
    split = split_by_feature(dataset, feature, threshold, seed)
    ensemble, _ = train_ensemble(split.train, split.val, config, n_members, workers)

    bands = []
    mean, sigma = ensemble.predict(split.test_inside.x)
    low = float(dataset.column(feature).min())
    bands.append(_band("in range (held out)", low, threshold, split.test_inside, mean, sigma))
    values = [split.test_inside.column(feature)]
    abs_err = [np.abs(mean - split.test_inside.y)]
    sigmas = [sigma]

    out_values = split.outside.column(feature)
    edges = np.linspace(threshold, float(out_values.max()), n_bands + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the maximum in the last band
    mean_out, sigma_out = ensemble.predict(split.outside.x)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (out_values >= lo) & (out_values < hi)
        if not mask.any():
            continue
        label = f"{feature} {lo:.3f} to {min(hi, float(out_values.max())):.3f}"
        bands.append(_band(label, float(lo), float(hi), split.outside.subset(mask),
                           mean_out[mask], sigma_out[mask]))
    values.append(out_values)
    abs_err.append(np.abs(mean_out - split.outside.y))
    sigmas.append(sigma_out)

    return ExtrapolationResult(
        feature=feature,
        threshold=threshold,
        n_train=len(split.train),
        bands=bands,
        ensemble=ensemble,
        values=np.concatenate(values),
        abs_error=np.concatenate(abs_err),
        sigma=np.concatenate(sigmas),
        target_names=dataset.target_names,
    )
