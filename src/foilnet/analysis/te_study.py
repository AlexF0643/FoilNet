"""The trailing-edge treatment: what changed, measured (writes ``docs/trailing-edge.md``).

Compares the solver's two trailing-edge treatments, ``"legacy"`` (the Kutta condition alone,
solver id ``linear-vortex-kutta-v1``) and ``"smooth"`` (the default), on

* the exact Karman-Trefftz solution, over trailing-edge angles from a true cusp to 27 degrees;
* the shapes of the generated datasets: how often the trailing-edge panel has a spurious
  spike, how much the labels move, and how fast each treatment converges with panel count.

    python -m foilnet.analysis.te_study --out docs/trailing-edge.md

On Windows the entry point needs ``if __name__ == "__main__":`` for the worker pool, which this
module has.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from foilnet.data import ShapeFamily, load_dataset
from foilnet.geometry import build_panels
from foilnet.parallel import process_pool
from foilnet.solver import PanelSolver
from foilnet.solver.analytic import KarmanTrefftz
from foilnet.solver.panel import _influence_matrix, _trailing_edge_map

TREATMENTS = ("legacy", "smooth")
# KT shapes for the angle table: the same circle-centre offsets as in docs/cst-validity.md
KT_CENTRE = (-0.1, 0.1)
KT_ANGLES_DEG = (0.0, 1.8, 3.6, 5.4, 9.0, 18.0, 27.0)


def has_spike(cp: np.ndarray, drop: float = 1.0) -> bool:
    """A trailing-edge panel (either surface) more than ``drop`` below its five neighbours."""
    return bool(cp[0] < cp[1:6].min() - drop or cp[-1] < cp[-6:-1].min() - drop)


def _solve(task):
    family, row, n_panels, treatment = task
    s = PanelSolver(family.build(row[:-1], n_panels), treatment).solve(row[-1])
    return s.cl, s.cm, has_spike(s.cp), min(s.cp[0], s.cp[-1])


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def kt_table() -> list[str]:
    """Relative Cl error and trailing-edge-panel Cp error against the exact solution."""
    rows = ["| trailing-edge angle | panels | Cl error, legacy | Cl error, smooth "
            "| Cp error at the 2 end panels, legacy | smooth |", "|---|---|---|---|---|---|"]
    for angle in KT_ANGLES_DEG:
        kt = KarmanTrefftz(n=2.0 - angle / 180.0, xc=KT_CENTRE[0], yc=KT_CENTRE[1])
        for n_panels in (160, 320):
            coords, phi = kt.contour(n_panels)
            mid = 0.5 * (phi[:-1] + phi[1:])
            cells = []
            cp_errors = []
            for treatment in TREATMENTS:
                solver = PanelSolver(coords, treatment)
                sol = solver.solve(3.0)
                exact_cl = kt.cl(3.0, solver.chord)
                cells.append(f"{(sol.cl - exact_cl) / exact_cl:+.1e}")
                err = np.abs(sol.cp - kt.cp(mid, 3.0))
                cp_errors.append(f"{err[[0, -1]].max():.2f}")
            rows.append(f"| {angle:g} degrees | {n_panels} | {cells[0]} | {cells[1]} | "
                        f"{cp_errors[0]} | {cp_errors[1]} |")
    return rows


def weak_mode_line(coords: np.ndarray) -> str:
    panels = build_panels(coords)
    matrix = _influence_matrix(panels)
    out = []
    for smooth in (False, True):
        sv = np.linalg.svd(matrix @ _trailing_edge_map(panels.n, smooth), compute_uv=False)
        out.append(f"{sv[-1] / sv[0]:.1e}")
    return (f"Smallest over largest singular value of the system, {panels.n} panels: "
            f"legacy {out[0]}, smooth {out[1]}.")


def dataset_study(path: str, label: str, samples: int, conv_samples: int, workers: int,
                  fine_panels: int) -> list[str]:
    ds = load_dataset(path)
    family = ShapeFamily.from_metadata(ds.metadata)
    panels = int(ds.metadata["n_panels"])
    rng = np.random.default_rng(0)
    x = ds.x[rng.choice(len(ds), min(samples, len(ds)), replace=False)]
    out = [f"### {label}", "", f"{len(x)} random shapes from `{path}`, solved at the dataset's "
           f"{panels} panels with both treatments.", ""]
    with process_pool(workers) as pool:
        def run(rows, n_panels, treatment):
            return list(pool.map(_solve, [(family, r, n_panels, treatment) for r in rows],
                                 chunksize=4))

        results = {t: np.array(run(x, panels, t), dtype=float) for t in TREATMENTS}
        d_cl = results["smooth"][:, 0] - results["legacy"][:, 0]
        d_cm = results["smooth"][:, 1] - results["legacy"][:, 1]
        out += ["| | legacy | smooth |", "|---|---|---|",
                f"| shapes with a spike on a trailing-edge panel | "
                f"{int(results['legacy'][:, 2].sum())} of {len(x)} | "
                f"{int(results['smooth'][:, 2].sum())} of {len(x)} |",
                f"| lowest Cp on an end panel | {results['legacy'][:, 3].min():.1f} | "
                f"{results['smooth'][:, 3].min():.1f} |", "",
                f"Labels moved by an RMS of {_rms(d_cl):.1e} in Cl (largest {abs(d_cl).max():.1e}) "
                f"and {_rms(d_cm):.1e} in Cm (largest {abs(d_cm).max():.1e}).", ""]
        sub = x[:conv_samples]
        ref = {t: np.array(run(sub, fine_panels, t))[:, :2] for t in TREATMENTS}
        out += [f"RMS difference from a {fine_panels}-panel solve of the same treatment "
                f"({len(sub)} shapes):", "", "| panels | Cl, legacy | Cl, smooth | Cm, legacy "
                "| Cm, smooth |", "|---|---|---|---|---|"]
        for n_panels in (80, 160, 320):
            cells = {}
            for t in TREATMENTS:
                diff = np.array(run(sub, n_panels, t))[:, :2] - ref[t]
                cells[t] = (_rms(diff[:, 0]), _rms(diff[:, 1]))
            out.append(f"| {n_panels} | {cells['legacy'][0]:.1e} | {cells['smooth'][0]:.1e} | "
                       f"{cells['legacy'][1]:.1e} | {cells['smooth'][1]:.1e} |")
        out.append("")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--naca", default="data/naca4-lhs.npz")
    parser.add_argument("--cst", default="data/cst-lhs.npz")
    parser.add_argument("--out", default="docs/trailing-edge.md")
    parser.add_argument("--samples", type=int, default=200, help="shapes per dataset")
    parser.add_argument("--convergence-samples", type=int, default=100)
    parser.add_argument("--fine-panels", type=int, default=640)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    workers = args.workers or os.cpu_count() or 1

    from foilnet.geometry import naca4

    lines = ["# The trailing-edge treatment", "",
             "Generated by `python -m foilnet.analysis.te_study`. `legacy` is the solver as it "
             "was through Phase 8 (Kutta condition alone, id `linear-vortex-kutta-v1`); "
             "`smooth` is the default since Phase 9.1 (`linear-vortex-kutta-v2`).", "",
             "## Exact solution: Karman-Trefftz, alpha = 3 degrees", "",
             "Relative error in Cl, and the largest Cp error on the two trailing-edge panels "
             "(`KarmanTrefftz` with circle centre "
             f"({KT_CENTRE[0]}, {KT_CENTRE[1]}); a 0-degree angle is the Joukowski cusp):", ""]
    lines += kt_table()
    lines += ["", "## Why the spike appeared", "",
              weak_mode_line(naca4(0.05, 0.225, 0.071, 160)) + " (NACA 5-22.5-7.1 shape.)", ""]
    lines += ["## On the datasets", ""]
    for path, label in ((args.naca, "NACA 4-digit"), (args.cst, "CST")):
        if Path(path).is_file():
            lines += dataset_study(path, label, args.samples, args.convergence_samples, workers,
                                   args.fine_panels)
        else:
            print(f"skipping {path}: not found")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
