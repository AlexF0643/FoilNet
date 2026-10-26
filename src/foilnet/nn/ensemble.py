"""Deep ensemble: several networks trained from different random starts.

After Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty Estimation
using Deep Ensembles", with one simplification: the members here are trained on squared
error alone (they do not each predict a variance). The prediction is the mean of the
members and the uncertainty is the spread between them, so it measures how much the
members *disagree*. Whether that disagreement tracks the real error is an empirical
question, answered by ``foilnet.analysis.calibration`` and the extrapolation experiment.

Members share the training data and architecture and differ only in weight initialisation
and batch order (seeds ``config.seed + i``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from foilnet.data import Dataset
from foilnet.nn.train import Surrogate, TrainConfig, TrainResult, train_surrogate
from foilnet.parallel import process_pool

_FORMAT = 1


@dataclass
class Ensemble:
    members: tuple[Surrogate, ...]

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("an ensemble needs at least two members to measure spread")
        first = self.members[0]
        for m in self.members[1:]:
            if m.feature_names != first.feature_names or m.target_names != first.target_names:
                raise ValueError("all members must share feature and target names")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.members[0].feature_names

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.members[0].target_names

    def predict_members(self, x: np.ndarray) -> np.ndarray:
        """``(n_members, n, n_targets)``: every member's prediction."""
        return np.stack([m.predict(x) for m in self.members])

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)``, each ``(n, n_targets)``; std is the sample standard deviation
        (ddof = 1) across members."""
        stack = self.predict_members(x)
        return stack.mean(axis=0), stack.std(axis=0, ddof=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format": _FORMAT, "members": [m.to_blob() for m in self.members]}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> Ensemble:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ensemble file not found: {path}")
        blob = torch.load(path, weights_only=True)
        if blob.get("format") != _FORMAT or "members" not in blob:
            raise ValueError(f"{path} is not an ensemble file (format {blob.get('format')})")
        return cls(tuple(Surrogate.from_blob(b, str(path)) for b in blob["members"]))


def _train_member(task: tuple[Dataset, Dataset, TrainConfig]) -> TrainResult:
    train, val, config = task
    return train_surrogate(train, val, config)


def train_ensemble(
    train: Dataset,
    val: Dataset,
    config: TrainConfig,
    n_members: int = 5,
    workers: int | None = None,
) -> tuple[Ensemble, list[TrainResult]]:
    """Train ``n_members`` networks (seeds ``config.seed + i``), in parallel if allowed.

    On Windows, a script calling this with more than one worker must guard its entry point
    with ``if __name__ == "__main__":``.
    """
    if n_members < 2:
        raise ValueError(f"n_members must be at least 2, got {n_members}")
    tasks = [(train, val, replace(config, seed=config.seed + i)) for i in range(n_members)]
    workers = max(1, min(workers or os.cpu_count() or 1, n_members))
    if workers == 1:
        results = [_train_member(t) for t in tasks]
    else:
        with process_pool(workers) as pool:
            results = list(pool.map(_train_member, tasks))
    return Ensemble(tuple(r.surrogate for r in results)), results
