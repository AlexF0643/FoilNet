"""Run every Phase 6 experiment and write ``docs/results.md`` (and the figures).

``python -m foilnet.analysis.experiments --data data/naca4-lhs.npz`` trains about fifty small
networks and takes ten to fifteen minutes on a laptop. Everything is seeded, so a rerun with
the same data and settings reproduces the tables (timings aside).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from foilnet.analysis import timing
from foilnet.analysis.calibration import CalibrationReport, calibration_report
from foilnet.analysis.extrapolation import ExtrapolationResult, run_extrapolation
from foilnet.analysis.learning_curve import CurveRow, learning_curve
from foilnet.data import Dataset, load_dataset, random_split
from foilnet.data.families import ShapeFamily
from foilnet.data.generate import _solve_row
from foilnet.nn.ensemble import Ensemble, train_ensemble
from foilnet.nn.train import TrainConfig
from foilnet.parallel import process_pool

DEFAULT_SIZES = (50, 100, 200, 400, 800, 1600, 3500)


@dataclass
class EnsembleEval:
    ensemble_rmse: float
    member_rmse: list[float]
    mean_sigma: float
    report: CalibrationReport


@dataclass
class Results:
    config: TrainConfig
    n_members: int
    n_train: int
    n_val: int
    n_test: int
    spread: dict[str, float]  # standard deviation of the test labels
    evaluation: dict[str, EnsembleEval]
    label_panels: int  # panel count the dataset's labels were solved with
    n_fine: int
    label_error: dict[str, float]  # RMS of (n_panels solve - n_fine solve) on the test set
    curve: list[CurveRow]
    extrapolation: ExtrapolationResult
    timing: timing.TimingResult | None  # NACA only: the timing sampler draws NACA shapes
    ensemble: Ensemble
    family: str = "naca4"
    label_check_n: int = 0  # how many test samples the discretisation check used


def solver_discretisation_error(
    dataset: Dataset, n_fine: int, workers: int | None = None, max_samples: int | None = None
) -> dict[str, float]:
    """RMS difference between the dataset's labels and a finer-panel solve of the same samples.

    ``max_samples`` uses only the first rows (the test set is already in random order), which
    keeps the check affordable when the fine solves are slow.

    This is how far the labels are from a converged solver. A surrogate trained on them
    reproduces the coarse solver, so against a converged solver it inherits this much
    discrepancy on top of its own error. It is a smooth, learnable offset, not noise.
    """
    family = ShapeFamily.from_metadata(dataset.metadata)
    if max_samples is not None:
        dataset = dataset.subset(np.arange(min(max_samples, len(dataset))))
    rows = [(family, x, n_fine) for x in dataset.x]
    workers = max(1, min(workers or os.cpu_count() or 1, len(rows)))
    if workers == 1:
        fine = np.array([_solve_row(r)[:2] for r in rows])
    else:
        with process_pool(workers) as pool:
            fine = np.array([r[:2] for r in pool.map(_solve_row, rows, chunksize=16)])
    rms = np.sqrt(np.mean((dataset.y - fine) ** 2, axis=0))
    return dict(zip(dataset.target_names, map(float, rms), strict=True))


def run_all(
    dataset: Dataset,
    config: TrainConfig,
    *,
    n_members: int = 5,
    sizes: Sequence[int] = DEFAULT_SIZES,
    n_fine: int = 640,
    workers: int | None = None,
    threshold: float | None = None,
    min_steps: int = 4000,
    label_check_samples: int | None = None,
) -> Results:
    """``threshold`` splits the extrapolation test on thickness; ``None`` means 0.12 for NACA
    data (as in the README) and the median thickness for other families."""
    family = ShapeFamily.from_metadata(dataset.metadata)
    if threshold is None:
        thicknesses = dataset.column("thickness")
        threshold = 0.12 if family.name == "naca4" else float(np.median(thicknesses))
    train, val, test = random_split(dataset, config.split, seed=config.seed)
    ensemble, _ = train_ensemble(train, val, config, n_members, workers)

    members = ensemble.predict_members(test.x)
    mean, sigma = members.mean(axis=0), members.std(axis=0, ddof=1)
    evaluation, spread = {}, {}
    for j, name in enumerate(test.target_names):
        err = mean[:, j] - test.y[:, j]
        evaluation[name] = EnsembleEval(
            ensemble_rmse=float(np.sqrt(np.mean(err**2))),
            member_rmse=[float(np.sqrt(np.mean((m[:, j] - test.y[:, j]) ** 2))) for m in members],
            mean_sigma=float(sigma[:, j].mean()),
            report=calibration_report(err, sigma[:, j]),
        )
        spread[name] = float(test.y[:, j].std())

    return Results(
        config=config,
        n_members=n_members,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        spread=spread,
        evaluation=evaluation,
        label_panels=int(dataset.metadata.get("n_panels", 0)),
        n_fine=n_fine,
        label_error=solver_discretisation_error(test, n_fine, workers, label_check_samples),
        curve=learning_curve(
            train,
            val,
            test,
            [s for s in sizes if s <= len(train)],
            config,
            n_members,
            workers,
            min_steps=min_steps,
        ),
        extrapolation=run_extrapolation(
            dataset, config, threshold=threshold, n_members=n_members, workers=workers
        ),
        timing=timing.measure(ensemble.predict) if family.name == "naca4" else None,
        family=family.name,
        label_check_n=min(label_check_samples or len(test), len(test)),
        ensemble=ensemble,
    )


# --- Markdown ---------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines + [""]


def _g(value: float) -> str:
    return f"{value:.3g}"


def format_markdown(r: Results, figures: bool = False, figure_dir: str = "assets") -> str:
    names = list(r.spread)
    out = [
        f"# Results: {r.family} shapes",
        "",
        "Generated by `python -m foilnet.analysis.experiments`. Every number comes from one seeded "
        "run; the surrounding discussion is in the README.",
        "",
        "## Setup",
        "",
        f"- {r.n_members}-network deep ensembles. Members differ only in seed (`seed + i`) and use "
        f"the settings in `configs/train-mlp.toml`: hidden {list(r.config.hidden)}, "
        f"{r.config.activation}, lr {r.config.learning_rate:g}, {r.config.epochs} epochs.",
        f"- Data split: {r.n_train} train / {r.n_val} validation / {r.n_test} test, seed "
        f"{r.config.seed}. All test numbers below use samples the networks never saw.",
        "- Ensemble prediction is the mean of the members; its uncertainty is the sample standard "
        "deviation across members (called spread, or sigma).",
        "",
        "## Ensemble on the held-out test set",
        "",
    ]
    header = ["target", "label spread", "single network RMSE (mean)", "ensemble RMSE",
              "mean ensemble spread", "RMS(error / spread)", "Spearman(|error|, spread)"]
    rows = []
    for n in names:
        e = r.evaluation[n]
        rows.append([n, _g(r.spread[n]), _g(float(np.mean(e.member_rmse))), _g(e.ensemble_rmse),
                     _g(e.mean_sigma), f"{e.report.rms_z:.2f}", f"{e.report.spearman:.2f}"])
    out += _table(header, rows)
    out += ["RMS(error / spread) is 1 for a calibrated ensemble and above 1 when the spread "
            "understates the error. Spearman measures whether a larger spread marks the samples "
            "that really are worse (1 = perfectly, 0 = not at all).", ""]

    out += ["### How often the error fell inside the claimed band", ""]
    header = ["target", "band", "claimed", "observed"]
    rows = []
    for n in names:
        for z, (exp, obs) in r.evaluation[n].report.coverage.items():
            rows.append([n, f"±{z:g} spread", f"{exp:.1%}", f"{obs:.1%}"])
    out += _table(header, rows)

    check = (
        "the same test samples"
        if r.label_check_n in (0, r.n_test)
        else f"the first {r.label_check_n} of the {r.n_test} test samples"
    )
    out += ["## Solver discretisation error of the labels", "",
            f"RMS difference between the labels (solved with {r.label_panels} panels) and a "
            f"{r.n_fine}-panel solve of {check}:", ""]
    out += _table(["target", "RMS difference", "as share of label spread"],
                  [[n, _g(r.label_error[n]), f"{r.label_error[n] / r.spread[n]:.2%}"]
                   for n in names])
    out += ["This is the distance from a converged solver, not noise: it is a smooth function of "
            "the inputs that a network can learn, and a surrogate trained on these labels "
            "inherits it relative to a converged solver.", ""]

    out += ["## Learning curve", ""]
    if figures:
        out += [f"![learning curve]({figure_dir}/learning-curve.png)", ""]
    header = ["training samples", "epochs"]
    for n in names:
        header += [f"{n} ensemble RMSE", f"{n} single-net RMSE (±std)", f"{n} mean spread"]
    rows = []
    for row in r.curve:
        cells = [str(row.n_train), str(row.epochs)]
        for n in names:
            cells += [_g(row.ensemble_rmse[n]),
                      f"{_g(row.member_rmse[n])} ± {_g(row.member_rmse_std[n])}",
                      _g(row.mean_sigma[n])]
        rows.append(cells)
    out += _table(header, rows)
    out += ["Epochs are raised for small training sets so that every run gets at least 4000 "
            "optimiser steps; otherwise small sets would be under-trained rather than short of "
            "data. Epoch selection uses a fixed slice of 100 validation samples whatever the "
            "training size.", ""]

    ex = r.extrapolation
    out += [f"## Extrapolation: train on {ex.feature} < {ex.threshold:g}, test above it", ""]
    if figures:
        out += [f"![extrapolation]({figure_dir}/extrapolation.png)", ""]
    out += [f"Trained on {ex.n_train} samples. Bands further from the training range are "
            "harder.", ""]
    for n in names:
        out += [f"### {n}", ""]
        header = ["band", "samples", "RMSE", "max |error|", "mean spread", "RMS(error / spread)",
                  "Spearman", "inside ±1 spread (68% if calibrated)"]
        rows = [[b.label, str(b.n), _g(b.rmse[n]), _g(b.max_abs_error[n]), _g(b.mean_sigma[n]),
                 f"{b.rms_z[n]:.2f}", f"{b.spearman[n]:.2f}", f"{b.coverage_1sigma[n]:.0%}"]
                for b in ex.bands]
        out += _table(header, rows)
    if figures:
        out += [f"![calibration]({figure_dir}/calibration.png)", ""]

    t = r.timing
    if t is None:
        return "\n".join(out)
    out += ["## Speed", "",
            f"Solver at {t.n_panels} panels against the {r.n_members}-network ensemble, medians on "
            "this machine, CPU only.", ""]
    out += _table(
        ["what is timed", "time"],
        [["solver, new shape each call (build + solve)", f"{t.solver_new_shape * 1e3:.2f} ms"],
         ["solver, same shape, new angle", f"{t.solver_same_shape * 1e6:.0f} µs"],
         ["ensemble, large batch, per sample", f"{t.surrogate_batch * 1e6:.1f} µs"],
         ["ensemble, one sample per call", f"{t.surrogate_single * 1e6:.0f} µs"]],
    )
    out += [f"Speed-up over a new-shape solve: about {t.speedup_batch:,.0f}x in a large batch, "
            f"about {t.speedup_single:.0f}x one call at a time. For an angle sweep of a single "
            "shape the solver is already fast (it reuses its factorised matrix), so the "
            "surrogate's advantage there is much smaller. Timings on a laptop vary by about 2x "
            "between runs, so read them as orders of magnitude.", ""]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/naca4-lhs.npz")
    parser.add_argument("--config", default="configs/train-mlp.toml")
    parser.add_argument("--out", default="docs/results.md")
    parser.add_argument("--figures", default="docs/assets", help="'' to skip the PNGs")
    parser.add_argument("--ensemble-out", default="runs/ensemble.pt")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--epochs", type=int, help="override the config's epochs (smoke tests)")
    parser.add_argument("--fine-panels", type=int, default=640)
    parser.add_argument("--threshold", type=float,
                        help="thickness that splits the extrapolation test (default: 0.12 for "
                        "NACA data, the median thickness otherwise)")
    parser.add_argument("--label-check-samples", type=int,
                        help="solve only this many test samples at --fine-panels (default: all)")
    parser.add_argument("--min-steps", type=int, default=4000,
                        help="minimum optimiser steps per learning-curve run")
    args = parser.parse_args(argv)

    from foilnet.nn.train import load_train_config

    config = load_train_config(args.config)
    if args.epochs:
        config = replace(config, epochs=args.epochs)
    results = run_all(
        load_dataset(args.data),
        config,
        n_members=args.members,
        sizes=[int(s) for s in args.sizes.split(",")],
        n_fine=args.fine_panels,
        workers=args.workers,
        min_steps=args.min_steps,
        threshold=args.threshold,
        label_check_samples=args.label_check_samples,
    )
    figures = bool(args.figures)
    if figures:
        from foilnet.analysis.plots import save_figures

        for path in save_figures(
            args.figures, results.curve, results.extrapolation, results.label_error
        ):
            print(f"wrote {path}")
    figure_dir = os.path.relpath(args.figures, Path(args.out).parent).replace(os.sep, "/")
    text = format_markdown(results, figures=figures, figure_dir=figure_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    results.ensemble.save(args.ensemble_out)
    print(f"wrote {args.out} and {args.ensemble_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
