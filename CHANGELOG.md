# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed (Phase 9.1)
- Trailing-edge treatment of the panel solver. The Kutta condition alone left the strengths of
  the two coincident end nodes nearly undetermined (smallest over largest singular value 2e-5),
  which put a spike in Cp on the last panel (down to -127 on CST shapes, 40 of 120 sampled), hurt
  a true cusp (2% Cl error) and slowed convergence on drooping edges. The solver now also imposes
  a smoothness condition there and solves by least squares (`PanelSolver(te_treatment="smooth")`,
  the default; solver id `linear-vortex-kutta-v2`). `te_treatment="legacy"` keeps the old
  behaviour (`linear-vortex-kutta-v1`). NACA labels move by about 1e-6 in Cl; CST labels by
  9e-4 RMS and are about five times closer to a 640-panel solve.
- Datasets and all results were regenerated with the new solver; the viewer no longer needs to
  explain a trailing-edge artefact.

### Added (Phase 9.1)
- `python -m foilnet.analysis.te_study`, writing `docs/trailing-edge.md`: legacy and smooth
  against the exact Karman-Trefftz solution and on both datasets.

### Added (Phase 8)
- `foilnet serve`: a local interactive viewer (standard-library server, single HTML page).
  Shape and angle sliders, aerofoil outline, solver and network results side by side with the
  ensemble spread, a sweep mode (Cl or Cm against angle of attack, solver curve against the
  network with its uncertainty band), the solver's pressure distribution, and warnings when
  an input is outside the trained range or a CST shape breaks the shape limits.
- `foilnet.viewer.ViewerModel`: the same calculation as a Python API.

### Added (Phase 7)
- CST (Kulfan) shapes: `foilnet.geometry.cst` (coefficients, contours, least-squares fits,
  validity limits), a shape-family abstraction in `foilnet.data`, optional derived columns in
  datasets (thickness, camber, trailing-edge angle), `configs/dataset-cst.toml` and
  `configs/train-mlp-cst.toml`.
- The experiment runner handles CST data (`--label-check-samples`, `--threshold`), writing
  `docs/results-cst.md`. `docs/cst-validity.md` records why each shape limit exists.

### Fixed
- Dataset generation and other worker pools were slowed several times over by every worker
  starting its own multi-threaded maths library (800 CST shapes: 48 s with 8 workers, 182 s
  with 4, 30 s serial). Workers now run single-threaded (`foilnet.parallel`): the same job
  takes 19 s, and the 10,000-shape CST dataset takes under 3 minutes instead of 31.
- Corrected the README and hyperparameter notes, which called the 160-panel label error a
  noise floor no surrogate can beat. It is a deterministic, learnable offset from a
  converged solver.

### Added
- Deep ensemble (`foilnet.nn.ensemble`), `foilnet train --members N`, and the Phase 6
  experiments (`foilnet.analysis`): calibration metrics, learning curves, a thin-to-thick
  extrapolation test, solver speed comparison, figures and `docs/results.md`. Findings are
  in the README.
- CI installs the plotting extra.
- PyTorch surrogate (`foilnet.nn.model`, `foilnet.nn.train`): MLP, training with a
  cosine schedule and best-epoch restore, `Surrogate` (raw in, raw out, save/load),
  evaluation metrics, and `foilnet train`. Tested against the NumPy network's forward
  pass and gradients.
- Staged hyperparameter search scored on the validation set
  (`foilnet.analysis.hparam_search`), with results and caveats in `docs/hyperparameters.md`.
- CI installs the CPU build of PyTorch.
- From-scratch NumPy MLP (`foilnet.nn`): forward pass, hand-derived backprop, Adam,
  mini-batch training, and a finite-difference `gradient_check`.
- Dataset pipeline: Latin hypercube sampling, parallel batch generation with provenance
  metadata, `.npz` dataset format, reproducible train/validation/test split and a
  standardiser. `foilnet generate` command.
- `SOLVER_ID`, recorded in every dataset so results can be traced to a solver version.
- Panel solver: linear-vortex method with a Kutta condition (`PanelSolver`, `solve`),
  returning Cl, Cm about the quarter chord, Cp and vortex strengths. Validated against the
  exact Karman-Trefftz solution, thin-aerofoil theory and a panel-count convergence study.
- `foilnet solve <naca> --alpha <deg>` command.
- Geometry: NACA 4-digit contours (closed trailing edge, cosine spacing, counter-clockwise
  from the trailing edge) and panel construction (midpoints, lengths, tangents, outward
  normals), with tests for closure, thickness, symmetry and normal direction.
- Project spine: packaging, `foilnet` CLI stub, TOML config loader, CI on
  Ubuntu and Windows with Python 3.11 and 3.12.
