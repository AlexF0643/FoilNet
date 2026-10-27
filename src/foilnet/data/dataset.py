"""Dataset container, file format, splitting and normalisation.

A dataset file is a NumPy ``.npz`` (no pickles) holding ``x`` (inputs), ``y`` (labels),
``cd_pressure`` (a per-sample numerical-error diagnostic) and ``metadata``, a JSON string
recording where every number came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray  # (n, n_features)
    y: np.ndarray  # (n, n_targets)
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    metadata: dict[str, Any]
    cd_pressure: np.ndarray | None = None  # (n,)
    # Derived descriptors that are not model inputs (e.g. the maximum thickness of a CST
    # shape), each an array of length n. They let an experiment split on a physical quantity.
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"x has {len(self.x)} rows but y has {len(self.y)}")
        if self.x.shape[1] != len(self.feature_names):
            raise ValueError("feature_names does not match the columns of x")
        if self.y.shape[1] != len(self.target_names):
            raise ValueError("target_names does not match the columns of y")
        for name, values in self.extras.items():
            if len(values) != len(self.x):
                raise ValueError(
                    f"extra column {name!r} has {len(values)} rows, expected {len(self.x)}"
                )

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, index: np.ndarray) -> Dataset:
        """Rows picked by an index array or boolean mask."""
        return Dataset(
            self.x[index],
            self.y[index],
            self.feature_names,
            self.target_names,
            self.metadata,
            None if self.cd_pressure is None else self.cd_pressure[index],
            {name: values[index] for name, values in self.extras.items()},
        )

    def column(self, name: str) -> np.ndarray:
        """A feature column, or a derived extra column, by name, e.g. for building a mask:
        ``ds.column("thickness") < 0.12``."""
        if name in self.feature_names:
            return self.x[:, self.feature_names.index(name)]
        if name in self.extras:
            return self.extras[name]
        raise KeyError(f"no feature or extra column named {name!r}")


def save_dataset(path: str | Path, dataset: Dataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset.metadata)
    meta["format_version"] = FORMAT_VERSION
    meta["feature_names"] = list(dataset.feature_names)
    meta["target_names"] = list(dataset.target_names)
    arrays: dict[str, np.ndarray] = {
        "x": dataset.x,
        "y": dataset.y,
        "metadata": np.array(json.dumps(meta, indent=2)),
    }
    if dataset.cd_pressure is not None:
        arrays["cd_pressure"] = dataset.cd_pressure
    for name, values in dataset.extras.items():
        arrays[f"extra__{name}"] = values
    np.savez_compressed(path, **arrays)


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        version = meta.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"{path} has dataset format {version}, expected {FORMAT_VERSION}")
        return Dataset(
            x=f["x"],
            y=f["y"],
            feature_names=tuple(meta["feature_names"]),
            target_names=tuple(meta["target_names"]),
            metadata=meta,
            cd_pressure=f["cd_pressure"] if "cd_pressure" in f.files else None,
            extras={k[len("extra__") :]: f[k] for k in f.files if k.startswith("extra__")},
        )


def random_split(
    dataset: Dataset, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0
) -> tuple[Dataset, Dataset, Dataset]:
    """Shuffle and split into (train, validation, test); the same seed gives the same split."""
    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1: {fractions}")
    n = len(dataset)
    order = np.random.default_rng(seed).permutation(n)
    n_train = round(fractions[0] * n)
    n_val = round(fractions[1] * n)
    return (
        dataset.subset(order[:n_train]),
        dataset.subset(order[n_train : n_train + n_val]),
        dataset.subset(order[n_train + n_val :]),
    )


@dataclass(frozen=True)
class Standardizer:
    """Zero-mean, unit-variance scaling. Fit on training data only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        std = values.std(axis=0)
        # a constant column would divide by zero; leave it unscaled instead
        return cls(values.mean(axis=0), np.where(std > 0.0, std, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def inverse_std(self, spread: np.ndarray) -> np.ndarray:
        """Convert a standard deviation in scaled units back to original units."""
        return spread * self.std
