"""Staged hyperparameter search, scored on the validation set only.

The test set is never touched here. Each stage varies one group of settings around the best
configuration found so far, which is far cheaper than a full grid and easy to read:

1. architecture (hidden layers) x learning rate
2. learning-rate schedule x weight decay
3. number of epochs

Run it with ``python -m foilnet.analysis.hparam_search --data data/naca4-lhs.npz``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from foilnet.data import Dataset, load_dataset, random_split
from foilnet.nn.train import TrainConfig, evaluate, train_surrogate


@dataclass(frozen=True)
class SearchRow:
    config: TrainConfig
    val_loss: float  # best epoch, standardised units (mean over targets)
    val_rmse: dict[str, float]  # physical units
    best_epoch: int
    seconds: float


def _run_one(task: tuple[Dataset, Dataset, TrainConfig]) -> SearchRow:
    train, val, config = task
    start = time.perf_counter()
    result = train_surrogate(train, val, config)
    rmse = {name: m["rmse"] for name, m in evaluate(result.surrogate, val).items()}
    return SearchRow(
        config,
        result.history["val_loss"][result.best_epoch - 1],
        rmse,
        result.best_epoch,
        time.perf_counter() - start,
    )


def run_grid(
    train: Dataset, val: Dataset, configs: Sequence[TrainConfig], workers: int = 1
) -> list[SearchRow]:
    """Train every configuration; rows come back sorted best (lowest validation loss) first."""
    tasks = [(train, val, c) for c in configs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, tasks))
    else:
        rows = [_run_one(t) for t in tasks]
    return sorted(rows, key=lambda r: r.val_loss)


def staged_search(
    train: Dataset,
    val: Dataset,
    base: TrainConfig | None = None,
    *,
    hidden: Sequence[tuple[int, ...]] = ((32, 32), (64, 64), (64, 64, 64), (128, 128)),
    learning_rates: Sequence[float] = (1e-3, 3e-3, 1e-2),
    schedules: Sequence[str] = ("cosine", "constant"),
    weight_decays: Sequence[float] = (0.0, 1e-5, 1e-4),
    epochs: Sequence[int] = (200, 400, 800, 1600),
    stage_epochs: int = 200,
    workers: int = 1,
) -> list[tuple[str, list[SearchRow]]]:
    """Run the three stages; returns ``[(stage name, rows)]``."""
    base = replace(base or TrainConfig(), epochs=stage_epochs)
    stages: list[tuple[str, list[SearchRow]]] = []

    grid = [replace(base, hidden=h, learning_rate=lr) for h in hidden for lr in learning_rates]
    rows = run_grid(train, val, grid, workers)
    stages.append((f"Stage 1: architecture x learning rate ({stage_epochs} epochs)", rows))
    best = rows[0].config

    grid = [replace(best, schedule=s, weight_decay=wd) for s in schedules for wd in weight_decays]
    rows = run_grid(train, val, grid, workers)
    stages.append((f"Stage 2: schedule x weight decay ({stage_epochs} epochs)", rows))
    best = rows[0].config

    grid = [replace(best, epochs=e) for e in epochs]
    rows = run_grid(train, val, grid, workers)
    stages.append(("Stage 3: epochs", rows))
    return stages


def format_markdown(stages: list[tuple[str, list[SearchRow]]], targets: Sequence[str]) -> str:
    lines: list[str] = []
    for title, rows in stages:
        lines += [f"### {title}", ""]
        header = ["hidden", "lr", "schedule", "weight decay", "epochs", "best epoch", "val loss"]
        header += [f"val RMSE {t}" for t in targets] + ["seconds"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for r in rows:
            c = r.config
            cells = [
                "x".join(map(str, c.hidden)),
                f"{c.learning_rate:g}",
                c.schedule,
                f"{c.weight_decay:g}",
                str(c.epochs),
                str(r.best_epoch),
                f"{r.val_loss:.3g}",
            ]
            cells += [f"{r.val_rmse[t]:.3g}" for t in targets] + [f"{r.seconds:.0f}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/naca4-lhs.npz")
    parser.add_argument("--out", help="write the tables here as Markdown")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="seed for the data split")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data)
    train, val, _test = random_split(dataset, TrainConfig().split, seed=args.seed)
    stages = staged_search(train, val, workers=args.workers)
    text = format_markdown(stages, dataset.target_names)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    best = stages[-1][1][0]
    print(f"best configuration: {best.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
