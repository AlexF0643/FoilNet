"""Run the panel solver over Latin hypercube samples and save a dataset with provenance."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from foilnet import __version__
from foilnet.config import ConfigError, load_config, require
from foilnet.data.dataset import Dataset, save_dataset
from foilnet.data.sampling import latin_hypercube
from foilnet.geometry import naca4
from foilnet.solver import SOLVER_ID, PanelSolver

FEATURE_NAMES = ("camber", "camber_position", "thickness", "alpha_deg")
TARGET_NAMES = ("cl", "cm")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    n_samples: int
    seed: int
    n_panels: int
    ranges: dict[str, tuple[float, float]]


def parse_dataset_config(raw: dict) -> DatasetConfig:
    """Validate a loaded TOML dict (see ``configs/dataset-naca4.toml``)."""
    cfg = DatasetConfig(
        name=str(require(raw, "dataset", "name")),
        n_samples=int(require(raw, "dataset", "n_samples")),
        seed=int(require(raw, "dataset", "seed")),
        n_panels=int(require(raw, "solver", "n_panels")),
        ranges={
            key: (float(require(raw, "ranges", key)[0]), float(require(raw, "ranges", key)[1]))
            for key in FEATURE_NAMES
        },
    )
    if cfg.n_samples < 1:
        raise ConfigError(f"dataset.n_samples must be at least 1, got {cfg.n_samples}")
    if cfg.n_panels < 4 or cfg.n_panels % 2:
        raise ConfigError(f"solver.n_panels must be even and at least 4, got {cfg.n_panels}")
    for key, (low, high) in cfg.ranges.items():
        if not low < high:
            raise ConfigError(f"ranges.{key} must have low < high, got [{low}, {high}]")
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


def _solve_row(row: tuple[float, float, float, float, int]) -> tuple[float, float, float]:
    """Solve one sample. Module level so worker processes can import it."""
    m, p, t, alpha, n_panels = row
    try:
        sol = PanelSolver(naca4(m, p, t, n_panels)).solve(alpha)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, np.nan
    return sol.cl, sol.cm, sol.cd_pressure


def generate_dataset(config: DatasetConfig, workers: int | None = None) -> Dataset:
    """Sample, solve (in parallel if ``workers`` > 1) and return the dataset.

    ``workers=None`` uses the CPU count. Results do not depend on the worker count. On
    Windows (and macOS), a script that calls this with more than one worker must guard its
    entry point with ``if __name__ == "__main__":``.
    """
    x = latin_hypercube(config.ranges, config.n_samples, config.seed)
    rows: Sequence[tuple[float, float, float, float, int]] = [
        (*map(float, sample), config.n_panels) for sample in x
    ]

    workers = workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(rows)))
    if workers == 1:
        results = [_solve_row(r) for r in rows]
    else:
        chunk = max(1, len(rows) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_solve_row, rows, chunksize=chunk))

    out = np.array(results)
    ok = np.all(np.isfinite(out), axis=1)
    x, out = x[ok], out[ok]
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
        "n_samples_requested": config.n_samples,
        "n_samples": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "max_abs_cd_pressure": float(np.abs(out[:, 2]).max()) if len(out) else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": sys.platform,
    }
    return Dataset(
        x=x,
        y=out[:, :2],
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        metadata=metadata,
        cd_pressure=out[:, 2],
    )


def generate_to_file(
    config: DatasetConfig, path: str | Path, workers: int | None = None
) -> Dataset:
    dataset = generate_dataset(config, workers)
    save_dataset(path, dataset)
    return dataset
