"""Run the panel solver over Latin hypercube samples and save a dataset with provenance."""

from __future__ import annotations

import dataclasses
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from foilnet import __version__
from foilnet.config import ConfigError, load_config, require
from foilnet.data.dataset import Dataset, save_dataset
from foilnet.data.families import NACA4_FEATURES, ShapeFamily
from foilnet.data.sampling import latin_hypercube
from foilnet.geometry.cst import ShapeLimits
from foilnet.parallel import process_pool
from foilnet.solver import SOLVER_ID, PanelSolver

FEATURE_NAMES = NACA4_FEATURES  # the default (NACA) family; others come from ShapeFamily
TARGET_NAMES = ("cl", "cm")
_MAX_DRAWS = 10  # batches of candidate shapes before giving up on filling a CST dataset


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    n_samples: int
    seed: int
    n_panels: int
    ranges: dict[str, tuple[float, float]]
    family: ShapeFamily = field(default_factory=lambda: ShapeFamily("naca4"))
    # Safety net after solving: drop samples whose pressure drag (which should be zero) is
    # larger than this, which flags a failed solve. None keeps everything.
    max_abs_cd: float | None = None


def parse_dataset_config(raw: dict) -> DatasetConfig:
    """Validate a loaded TOML dict (see ``configs/dataset-naca4.toml``, ``dataset-cst.toml``)."""
    family_name = str(raw.get("dataset", {}).get("family", "naca4"))
    try:
        if family_name == "cst":
            shape = raw.get("shape", {})
            allowed = {f.name for f in dataclasses.fields(ShapeLimits)}
            unknown = set(shape) - allowed
            if unknown:
                raise ConfigError(f"unknown key(s) in [shape]: {sorted(unknown)}")
            family = ShapeFamily(
                "cst",
                int(raw["dataset"].get("n_coefficients", 4)),
                ShapeLimits(**{k: float(v) for k, v in shape.items()}),
            )
        else:
            family = ShapeFamily(family_name)
    except ValueError as exc:
        raise ConfigError(f"dataset.family: {exc}") from exc

    cfg = DatasetConfig(
        name=str(require(raw, "dataset", "name")),
        n_samples=int(require(raw, "dataset", "n_samples")),
        seed=int(require(raw, "dataset", "seed")),
        n_panels=int(require(raw, "solver", "n_panels")),
        ranges={
            key: (float(require(raw, "ranges", key)[0]), float(require(raw, "ranges", key)[1]))
            for key in family.feature_names
        },
        family=family,
        max_abs_cd=(
            float(raw["solver"]["max_abs_cd"]) if "max_abs_cd" in raw.get("solver", {}) else None
        ),
    )
    if cfg.max_abs_cd is not None and cfg.max_abs_cd <= 0:
        raise ConfigError(f"solver.max_abs_cd must be positive, got {cfg.max_abs_cd}")
    if cfg.n_samples < 1:
        raise ConfigError(f"dataset.n_samples must be at least 1, got {cfg.n_samples}")
    if cfg.n_panels < 4 or cfg.n_panels % 2:
        raise ConfigError(f"solver.n_panels must be even and at least 4, got {cfg.n_panels}")
    for key, (low, high) in cfg.ranges.items():
        if not low < high:
            raise ConfigError(f"ranges.{key} must have low < high, got [{low}, {high}]")
    if family.name == "naca4":
        checks = {
            "camber": (0.0, 0.2),
            "camber_position": (0.05, 0.95),
            "thickness": (0.02, 0.4),
        }
        for key, (lo, hi) in checks.items():
            low, high = cfg.ranges[key]
            if low < lo or high > hi:
                raise ConfigError(f"ranges.{key} must lie within [{lo}, {hi}], got [{low}, {high}]")
    return cfg


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return parse_dataset_config(load_config(path))


def solve_sample(
    family: ShapeFamily, params: np.ndarray, n_panels: int
) -> tuple[float, float, float]:
    """``(cl, cm, cd_pressure)`` for one row of features (shape parameters, then alpha)."""
    try:
        contour = family.build(params[:-1], n_panels)
        sol = PanelSolver(contour).solve(float(params[-1]))
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, np.nan
    return sol.cl, sol.cm, sol.cd_pressure


def _solve_row(task: tuple[ShapeFamily, np.ndarray, int]) -> tuple[float, float, float]:
    """Module level so worker processes can import it."""
    return solve_sample(*task)


def _sample_shapes(config: DatasetConfig) -> tuple[np.ndarray, int]:
    """Latin hypercube samples of the features, and how many candidates were rejected.

    NACA parameters are always valid, so ``n_samples`` rows are drawn directly. For CST a
    batch of candidates is drawn and the unusable shapes dropped (surfaces that cross,
    near-cusped trailing edges, extreme thickness or camber), repeating with a new seed
    until enough survive. Dropping rows slightly weakens the one-per-bin property of the
    Latin hypercube, but coverage stays even.
    """
    if not config.family.needs_filter:
        return latin_hypercube(config.ranges, config.n_samples, config.seed), 0
    kept: list[np.ndarray] = []
    rejected = 0
    for draw in range(_MAX_DRAWS):
        batch = latin_hypercube(config.ranges, 2 * config.n_samples, config.seed + 7919 * draw)
        ok = np.array([config.family.is_valid(row[:-1]) for row in batch])
        kept.append(batch[ok])
        rejected += int((~ok).sum())
        if sum(len(k) for k in kept) >= config.n_samples:
            return np.concatenate(kept)[: config.n_samples], rejected
    raise ConfigError(
        f"too few usable shapes: {sum(len(k) for k in kept)} of {config.n_samples} wanted after "
        f"{_MAX_DRAWS} batches; narrow the ranges or relax the [shape] limits"
    )


def generate_dataset(config: DatasetConfig, workers: int | None = None) -> Dataset:
    """Sample, solve (in parallel if ``workers`` > 1) and return the dataset.

    ``workers=None`` uses the CPU count. The sampled shapes never depend on the worker count;
    the labels agree to rounding error (about 1e-14), not bit for bit, because the in-process
    path uses a multi-threaded maths library and the workers use one thread each. A script
    that calls this with more than one worker must guard its entry point with
    ``if __name__ == "__main__":``.
    """
    x, n_rejected = _sample_shapes(config)
    family = config.family
    tasks = [(family, row, config.n_panels) for row in x]

    workers = workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        results = [_solve_row(t) for t in tasks]
    else:
        chunk = max(1, len(tasks) // (workers * 8))
        with process_pool(workers) as pool:
            results = list(pool.map(_solve_row, tasks, chunksize=chunk))

    out = np.array(results)
    ok = np.all(np.isfinite(out), axis=1)
    if config.max_abs_cd is not None:
        with np.errstate(invalid="ignore"):
            ok &= np.abs(out[:, 2]) <= config.max_abs_cd
    x, out = x[ok], out[ok]

    descriptors = [family.descriptors(row[:-1]) for row in x]
    extras = {k: np.array([d[k] for d in descriptors]) for k in (descriptors[0] if x.size else {})}

    metadata = {
        "name": config.name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "foilnet_version": __version__,
        "solver_id": SOLVER_ID,
        "solver": "linear-vortex panel method, Kutta condition (inviscid, incompressible)",
        "n_panels": config.n_panels,
        "sampling": "latin hypercube (scipy.stats.qmc.LatinHypercube, scrambled)",
        "seed": config.seed,
        "ranges": {k: list(v) for k, v in config.ranges.items()},
        **family.describe(),
        "n_samples_requested": config.n_samples,
        "n_samples": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "max_abs_cd_filter": config.max_abs_cd,
        "n_rejected_invalid_shape": n_rejected,
        "max_abs_cd_pressure": float(np.abs(out[:, 2]).max()) if len(out) else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": sys.platform,
    }
    return Dataset(
        x=x,
        y=out[:, :2],
        feature_names=family.feature_names,
        target_names=TARGET_NAMES,
        metadata=metadata,
        cd_pressure=out[:, 2],
        extras=extras,
    )


def generate_to_file(
    config: DatasetConfig, path: str | Path, workers: int | None = None
) -> Dataset:
    dataset = generate_dataset(config, workers)
    save_dataset(path, dataset)
    return dataset
