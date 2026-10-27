"""Figures for the uncertainty experiments (needs matplotlib: ``pip install -e ".[plots]"``).

Two series per chart at most: blue and orange from the project palette, also told apart by
line style and marker so colour is never the only cue. One y-scale per panel; panels are
side-by-side small multiples (one per target).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from foilnet.analysis.calibration import calibration_curve
from foilnet.analysis.extrapolation import ExtrapolationResult
from foilnet.analysis.learning_curve import CurveRow

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
_TITLES = {"cl": "Lift coefficient Cl", "cm": "Moment coefficient Cm"}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _panels(targets: Sequence[str]):
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 3.9), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, targets, strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_title(_TITLES.get(name, name), color=INK, fontsize=11, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", colors=INK_2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    return fig, axes


def _legend(ax) -> None:
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(INK_2)


def learning_curve_figure(
    rows: Sequence[CurveRow],
    targets: Sequence[str],
    label_error: dict[str, float] | None = None,
):
    fig, axes = _panels(targets)
    n = [r.n_train for r in rows]
    for ax, name in zip(axes, targets, strict=True):
        ens = [r.ensemble_rmse[name] for r in rows]
        mem = np.array([r.member_rmse[name] for r in rows])
        err = np.array([r.member_rmse_std[name] for r in rows])
        ax.errorbar(n, mem, yerr=err, color=ORANGE, linestyle="--", marker="s", markersize=5,
                    linewidth=1.5, capsize=2.5, label="single network (mean, ±1 std over seeds)")
        ax.plot(n, ens, color=BLUE, marker="o", markersize=6, linewidth=1.8,
                label="ensemble of 5")
        if label_error and name in label_error:
            ax.axhline(label_error[name], color=INK_2, linestyle=":", linewidth=1.2)
            ax.annotate("solver discretisation error", (n[0], label_error[name]),
                        textcoords="offset points", xytext=(2, 4), fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training samples", color=INK_2, fontsize=9)
        ax.set_ylabel("test RMSE", color=INK_2, fontsize=9)
    _legend(axes[0])
    fig.tight_layout()
    return fig


def extrapolation_figure(result: ExtrapolationResult, n_bins: int = 12):
    fig, axes = _panels(result.target_names)
    lo, hi = float(result.values.min()), float(result.values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(result.values, edges) - 1, 0, n_bins - 1)
    for j, ax in enumerate(axes):
        rmse, sigma = [], []
        for b in range(n_bins):
            m = which == b
            rmse.append(np.sqrt(np.mean(result.abs_error[m, j] ** 2)) if m.any() else np.nan)
            sigma.append(np.mean(result.sigma[m, j]) if m.any() else np.nan)
        ax.axvspan(lo, result.threshold, color=GRID, alpha=0.55, linewidth=0)
        ax.plot(centres, rmse, color=BLUE, marker="o", markersize=6, linewidth=1.8,
                label="actual error (RMSE)")
        ax.plot(centres, sigma, color=ORANGE, linestyle="--", marker="s", markersize=5,
                linewidth=1.5, label="ensemble spread (mean)")
        ax.set_yscale("log")
        ax.set_xlabel(result.feature, color=INK_2, fontsize=9)
        ax.set_ylabel("error or spread", color=INK_2, fontsize=9)
        ax.annotate("trained here", (lo, 1), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(4, -14), fontsize=8, color=INK_2)
    _legend(axes[0])
    fig.tight_layout()
    return fig


def calibration_figure(result: ExtrapolationResult):
    """Reliability diagram: how often the error was inside the band the ensemble claimed."""
    fig, axes = _panels(result.target_names)
    inside = result.values < result.threshold
    far = result.values >= result.threshold + 2.0 / 3.0 * (result.values.max() - result.threshold)
    for j, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1.2, label="calibrated")
        nominal, observed = calibration_curve(result.abs_error[inside, j], result.sigma[inside, j])
        ax.plot(nominal, observed, color=BLUE, marker="o", markersize=5, linewidth=1.8,
                label=f"in training range (n={int(inside.sum())})")
        nominal, observed = calibration_curve(result.abs_error[far, j], result.sigma[far, j])
        ax.plot(nominal, observed, color=ORANGE, linestyle="--", marker="s", markersize=5,
                linewidth=1.5, label=f"furthest third outside (n={int(far.sum())})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("claimed coverage of the ±σ band", color=INK_2, fontsize=9)
        ax.set_ylabel("observed coverage", color=INK_2, fontsize=9)
    _legend(axes[0])
    fig.tight_layout()
    return fig


def save_figures(
    out_dir: str | Path,
    curve_rows: Sequence[CurveRow],
    extrapolation: ExtrapolationResult,
    label_error: dict[str, float] | None = None,
) -> list[Path]:
    """Write the three PNGs; returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    figures = {
        "learning-curve.png": learning_curve_figure(
            curve_rows, extrapolation.target_names, label_error
        ),
        "extrapolation.png": extrapolation_figure(extrapolation),
        "calibration.png": calibration_figure(extrapolation),
    }
    paths = []
    for name, fig in figures.items():
        path = out / name
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths
