"""Learning curves: held-out error against the number of training samples."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from foilnet.data import Dataset
from foilnet.nn.ensemble import train_ensemble
from foilnet.nn.train import TrainConfig


@dataclass(frozen=True)
class CurveRow:
    n_train: int
    epochs: int
    ensemble_rmse: dict[str, float]  # RMSE of the ensemble mean on the test set
    member_rmse: dict[str, float]  # mean over members of each member's own RMSE
    member_rmse_std: dict[str, float]  # spread of that across members
    mean_sigma: dict[str, float]  # average ensemble spread on the test set
    seconds: float


def epochs_for(n_train: int, config: TrainConfig, min_steps: int = 4000) -> int:
    """With a fixed epoch count, small datasets take far fewer optimiser steps and would be
    under-trained, which would confuse "needs more data" with "needs more training". So the
    epoch count is raised until there are at least ``min_steps`` steps."""
    steps_per_epoch = math.ceil(n_train / config.batch_size)
    return max(config.epochs, math.ceil(min_steps / steps_per_epoch))


def learning_curve(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    sizes: Sequence[int],
    config: TrainConfig,
    n_members: int = 5,
    workers: int | None = None,
    val_size: int = 100,
    min_steps: int = 4000,
) -> list[CurveRow]:
    """Train an ensemble on the first ``n`` training samples for each ``n`` in ``sizes``.

    ``train`` should already be in random order (``random_split`` does that), so the first
    ``n`` rows are a random subset and each size contains the smaller ones. The same ``test``
    set is used throughout. A small fixed slice of ``val`` (``val_size`` samples, regardless
    of ``n``) picks the best epoch, so small training sets are not given a large validation
    set to lean on. ``min_steps`` is passed to :func:`epochs_for`.
    """
    if max(sizes) > len(train):
        raise ValueError(f"largest size {max(sizes)} exceeds the {len(train)} training samples")
    val_small = val.subset(np.arange(min(val_size, len(val))))
    rows = []
    for n in sizes:
        cfg = replace(config, epochs=epochs_for(n, config, min_steps))
        start = time.perf_counter()
        ensemble, _ = train_ensemble(train.subset(np.arange(n)), val_small, cfg, n_members, workers)
        seconds = time.perf_counter() - start

        members = ensemble.predict_members(test.x)  # (m, n_test, k)
        mean, sigma = members.mean(axis=0), members.std(axis=0, ddof=1)
        ens_rmse = np.sqrt(np.mean((mean - test.y) ** 2, axis=0))
        mem_rmse = np.sqrt(np.mean((members - test.y) ** 2, axis=1))  # (m, k)
        names = test.target_names
        member_std = mem_rmse.std(axis=0, ddof=1)
        rows.append(
            CurveRow(
                n_train=n,
                epochs=cfg.epochs,
                ensemble_rmse=dict(zip(names, map(float, ens_rmse), strict=True)),
                member_rmse=dict(zip(names, map(float, mem_rmse.mean(axis=0)), strict=True)),
                member_rmse_std=dict(zip(names, map(float, member_std), strict=True)),
                mean_sigma=dict(zip(names, map(float, sigma.mean(axis=0)), strict=True)),
                seconds=seconds,
            )
        )
    return rows
