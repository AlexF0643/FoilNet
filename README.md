# FoilNet

[![CI](https://github.com/AlexF0643/FoilNet/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexF0643/FoilNet/actions/workflows/ci.yml)

A neural-network surrogate for aerofoil aerodynamics, trained on data from a panel-method solver I wrote myself.

Aerodynamic solvers are accurate but slow, and design work means running them thousands of times. A surrogate is a network trained to copy the solver: give it an aerofoil shape and an angle of attack and it returns lift and moment coefficients in microseconds.

Getting a network that fits isn't the interesting part. What I want to find out is:

- How much training data does the network actually need?
- Where does it stay trustworthy, and where does it fail quietly?
- Can it tell me when it's guessing?

I write and validate the solver myself, so I understand the physics behind every label instead of treating the dataset as a black box.

## Status

**Phases 0 to 5 done (spine, geometry, panel solver, dataset generation, from-scratch NumPy network, PyTorch surrogate).** The surrogate reproduces the solver's Cl and Cm to about 0.1% and 0.3% of their spread on held-out shapes. Uncertainty estimates, the honest failure-mode experiments and the viewer are still to come; `serve` is a stub that says which phase will implement it.

## Install

Needs Python 3.11 or newer.

```bash
git clone https://github.com/AlexF0643/FoilNet.git
cd FoilNet
python -m pip install -e ".[dev]"
pytest
```

The PyTorch and plotting dependencies are optional extras (`.[ml]`, `.[plots]`) and aren't needed until the later phases.

## Usage

```bash
foilnet solve 2412 --alpha 4
```

```
NACA 2412  alpha = 4 deg  panels = 160
Cl      = 0.7415
Cm(c/4) = -0.0612
```

From Python:

```python
from foilnet.geometry import naca4_from_code
from foilnet.solver import PanelSolver

solver = PanelSolver(naca4_from_code("2412", n_panels=160))
print(solver.solve(4.0).cl, solver.solve(8.0).cl)  # new angles of attack are almost free
```

Planned subcommand that is still a stub: `serve`.

## Dataset

```bash
foilnet generate --config configs/dataset-naca4.toml
```

This draws 5000 Latin hypercube samples of (camber, camber position, thickness, angle of attack), runs the panel solver on each in parallel, and writes `data/naca4-lhs.npz` (about 25 s on 8 cores). Every file carries its provenance: solver id and foilnet version, panel count, seed, parameter ranges, library versions and creation time. The same seed gives the same dataset whatever the number of workers.

```python
from foilnet.data import load_dataset, random_split, Standardizer

ds = load_dataset("data/naca4-lhs.npz")
train, val, test = random_split(ds, (0.7, 0.15, 0.15), seed=0)
scaler = Standardizer.fit(train.x)  # fit on training data only
```

Labels are Cl and Cm from 160-panel solves. Against 640 panels they differ by at most 4e-4 in Cl and 8e-4 in Cm (spreads are 0.63 and 0.054), worst for thin, highly cambered sections at high angle of attack. That is a noise floor no surrogate can beat. Each sample also stores its pressure drag, which should be zero and is a per-sample check on numerical error.

## Surrogate

```bash
python -m pip install -e ".[ml]"   # PyTorch (CPU is fine)
foilnet generate
foilnet train
```

`foilnet train` splits the dataset 70/15/15, trains the PyTorch network from `configs/train-mlp.toml`, restores the epoch with the lowest validation loss, saves `runs/mlp.pt`, and prints errors on the validation and test sets (about a minute). The test set is used only for that final report.

```python
from foilnet.nn.train import Surrogate

model = Surrogate.load("runs/mlp.pt")
model.predict([[0.02, 0.4, 0.12, 4.0]])  # camber, camber position, thickness, alpha (deg) -> Cl, Cm
```

Result on 750 held-out shapes (one run, seed 0):

| | RMSE | max error | share of label spread |
|---|---|---|---|
| Cl | 7.0e-4 | 4.6e-3 | 0.11% |
| Cm | 1.65e-4 | 1.25e-3 | 0.31% |

The chosen settings, and how far to trust each, are in [docs/hyperparameters.md](docs/hyperparameters.md). The short version: learning rate, schedule and epochs matter; the differences between reasonable network sizes are inside seed-to-seed noise (about 30%). Cm error is already at the solver's own numerical error.

This is accuracy on shapes drawn from the same ranges as the training data. How it behaves outside them, and whether it can tell, is what Phase 6 tests.

## From-scratch network

`foilnet.nn.numpy_mlp` is a multi-layer perceptron in plain NumPy with backprop derived by hand (the derivation is in the module docstring) and an Adam optimiser. It is a learning exercise and a cross-check, not the final surrogate: PyTorch takes over in Phase 5.

Backprop is checked against finite differences (`gradient_check`), which agree to about 1e-10 for tanh and ReLU, with and without weight decay.

A first fit on the Phase 3 dataset, as a sanity check only (one seed, one split, a 4-64-64-2 tanh network, 300 epochs, hyperparameters not tuned), validation RMSE on 750 held-out samples:

| training samples | Cl RMSE | Cm RMSE |
|---|---|---|
| 100 | 0.021 | 0.0021 |
| 400 | 0.0067 | 0.00075 |
| 1600 | 0.0035 | 0.00032 |
| 3500 | 0.0032 | 0.00029 |

For scale, the labels have a spread of 0.63 in Cl and 0.054 in Cm, so 3500 samples gets within about 0.5% of the spread. This is not yet a proper learning curve; that comes with the PyTorch surrogate in Phase 6.

## Solver validation

The solver is a linear-vortex panel method with a Kutta condition. Nothing downstream is trusted until it passes these, and each one is a test in `tests/`:

- **Exact solution.** Against the Karman-Trefftz aerofoil (conformal map of flow past a circle; Joukowski is the `n = 2` case), at 160 panels, over several shapes and angles from -4 to 10 degrees: Cl agrees to within 0.1%, and Cp to within 0.02 + 2% of the local value away from the trailing edge. The exact solution is itself checked first: integrating its Cp reproduces its Kutta-Joukowski lift to 1e-5 and gives zero drag.
- **Thin-aerofoil theory.** Lift slope follows 2 pi (1 + 0.77 t/c) to 1.5% for 6, 12 and 18% thick sections. For thin cambered sections the zero-lift angle is within 0.1 degrees and Cm within 0.005 of theory.
- **Convergence.** Cl and Cm errors fall by more than 2x per doubling of panels (about 4x in practice, as expected for a second-order method). At 160 panels, Cl is within 0.1% of the 1280-panel result.
- **Consistency.** Pressure drag, which should be zero, is below 1e-3; lift from integrating Cp agrees with Kutta-Joukowski; the flow inside the body is near zero when checked by independent numerical quadrature.

Known weak spots, found while validating:

- **Cusped trailing edges are not handled well.** The pure Joukowski aerofoil (zero trailing-edge angle) gives about 2% Cl error at 160 panels, so the tests use finite-angle Karman-Trefftz shapes. NACA 4-digit sections are fine. This matters again for the CST shapes in Phase 7.
- Cp is worst at the leading-edge suction peak at high angle of attack. It converges, but a coarse panelling under-resolves the peak.

## What this can't do

This section will grow as the project does, but the limits that are already fixed by design:

- The solver is a linear-vortex panel method: inviscid and incompressible. It gives good lift and moment in the linear range and **cannot predict stall or drag**.
- The surrogate learns the solver, not reality, so it inherits those limits and adds its own error on top.
- Training data will cover angles of attack from −5° to +12°. Outside that range, results aren't meaningful.

## Roadmap

Spine, geometry, panel solver (validated against exact solutions before any ML), dataset generation, a from-scratch NumPy MLP, a PyTorch surrogate, deep-ensemble uncertainty with honest extrapolation tests, CST shapes, an interactive viewer, and a v0.1.0 release. See [PROJECT_PLAN.md](PROJECT_PLAN.md).

## License

MIT. See [LICENSE](LICENSE).
