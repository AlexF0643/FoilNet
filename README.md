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

**Phases 0 to 6 done (spine, geometry, panel solver, dataset, NumPy network, PyTorch surrogate, uncertainty and honest testing).** A single network reproduces the solver's Cl and Cm to about 0.1% and 0.3% of their spread on held-out shapes. A five-network ensemble gives an uncertainty estimate, and I have measured how far it can be trusted: it flags extrapolation but understates the error by about 2x or more (see "Uncertainty and honest testing"). CST shapes, the viewer and the release polish are still to come; `serve` is a stub that says which phase will implement it.

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

Labels are Cl and Cm from 160-panel solves. They are not noisy: each label is a deterministic function of the inputs, so a network can learn them exactly. They do differ slightly from a converged solver. On the test set, a 640-panel solve differs from the labels by an RMS of 1.6e-4 in Cl and 1.9e-4 in Cm (spreads are 0.62 and 0.054), and by up to 4e-4 and 8e-4 in a sample of rows, worst for thin, highly cambered sections at high angle of attack. A surrogate trained on these labels reproduces the 160-panel solver, so measured against a converged solver it carries that difference on top of its own error. Each sample also stores its pressure drag, which should be zero and is a per-sample check on numerical error.

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

The chosen settings, and how far to trust each, are in [docs/hyperparameters.md](docs/hyperparameters.md). The short version: learning rate, schedule and epochs matter; the differences between reasonable network sizes are inside seed-to-seed noise (about 30%). Cm error (1.65e-4) is about the size of the labels' distance from a converged solver (1.9e-4); the Cl error is about 4x that.

This is accuracy on shapes drawn from the same ranges as the training data. How it behaves outside them, and whether it can tell, is the next section.

## Uncertainty and honest testing

`foilnet train --members 5` trains a deep ensemble (after Lakshminarayanan et al., 2017): five networks that differ only in their random start. The prediction is their mean and the uncertainty is the spread between them. Full tables and the code that produced them are in [docs/results.md](docs/results.md); `python -m foilnet.analysis.experiments` regenerates them in about ten minutes. Everything below is one seeded run (one data split, five members), so treat differences of tens of percent with caution.

```python
from foilnet.nn.ensemble import Ensemble

ensemble = Ensemble.load("runs/ensemble.pt")
mean, spread = ensemble.predict([[0.02, 0.4, 0.12, 4.0]])  # Cl and Cm, and their uncertainty
```

**The ensemble is more accurate than one network.** On the 750 test shapes its RMSE is 5.3e-4 in Cl (a single network averages 7.8e-4) and 1.68e-4 in Cm (1.81e-4).

### How much data does it need?

![learning curve](docs/assets/learning-curve.png)

| training samples | Cl RMSE | share of Cl spread | Cm RMSE | share of Cm spread |
|---|---|---|---|---|
| 50 | 9.1e-3 | 1.5% | 1.3e-3 | 2.4% |
| 100 | 3.0e-3 | 0.49% | 5.1e-4 | 0.96% |
| 200 | 1.7e-3 | 0.28% | 4.1e-4 | 0.77% |
| 400 | 1.5e-3 | 0.24% | 3.5e-4 | 0.65% |
| 800 | 1.3e-3 | 0.21% | 2.5e-4 | 0.47% |
| 1600 | 8.5e-4 | 0.14% | 2.0e-4 | 0.38% |
| 3500 | 5.6e-4 | 0.09% | 1.7e-4 | 0.32% |

It depends on how accurate you need it. About 100 samples gets the error under 0.5% of the spread in Cl, 400 gets it under 0.25%, and 3500 gets it under 0.1%. Cl error falls roughly as the square root of the number of samples (about n^-0.5 from 100 to 3500) and is still falling at 3500, so more data would help. Cm flattens out near 1.7e-4, close to the dotted line, which is the labels' distance from a converged solver. Whether that is a limit of the data, the network, or the labels, I have not determined. Small training sets also vary a lot from seed to seed (a single network at 50 samples varies by about 25%). To keep small sets from being under-trained, every run gets at least 4000 optimiser steps, which means more epochs.

### Where does it fail?

I trained on thickness below 12% only (1750 samples) and tested on the rest of the range, in three bands out to 18%, which is 1.5 times the thickest training shape.

![extrapolation](docs/assets/extrapolation.png)

| band | Cl RMSE | Cl ensemble spread | error / spread | Cm RMSE | Cm ensemble spread | error / spread |
|---|---|---|---|---|---|---|
| in range (held out) | 9.1e-4 | 6.4e-4 | 1.5 | 1.6e-4 | 6.5e-5 | 3.2 |
| thickness 0.12 to 0.14 | 2.3e-3 | 1.2e-3 | 1.8 | 3.3e-4 | 1.4e-4 | 3.0 |
| thickness 0.14 to 0.16 | 6.6e-3 | 3.1e-3 | 2.0 | 7.1e-4 | 3.7e-4 | 2.4 |
| thickness 0.16 to 0.18 | 1.4e-2 | 6.5e-3 | 2.1 | 1.6e-3 | 7.2e-4 | 3.7 |

The failure is graceful but steep. Error grows about 2 to 3 times with each band, so at the far end Cl is 15 times worse than in range (and Cm 10 times), with a worst single error of 0.052 in Cl. That is still only about 2% of the Cl spread, because Cl depends smoothly on thickness. A network can extrapolate a smooth trend for a while, which is not the same as being right, and the error would not stay this small for a less smooth quantity.

### Can it tell when it is guessing?

Partly. The spread is a useful warning light but not a calibrated error bar.

- **It notices.** The mean Cl spread grows about 10 times across the bands (6.4e-4 to 6.5e-3), and within the bands outside the training range it ranks the worse samples well (Spearman correlation 0.5 to 0.6 for Cl). Inside the training range it is a poor guide to which individual samples are worst (0.1 to 0.2).
- **It understates the size.** The real error is about 1.2 to 1.5 times the spread for Cl inside the training range and about 2 times outside it, and 2.4 to 3.7 times for Cm everywhere. A band of plus or minus one spread should hold 68% of errors. For Cl it holds 73% on the full-range ensemble (so Cl is roughly calibrated in range), but 28% and 26% in the two furthest bands. For Cm it holds only 48% even in range.

![calibration](docs/assets/calibration.png)

**Why.** Each member's error has a part that is private to that member (a bad start) and a part shared by all members (what this network and data cannot capture). The spread can only see the first part. A rough split, assuming the shared part is the same for every member and the private part is independent: for Cl about 30% of a single network's squared error is shared; for Cm about 80% is. That matches what we see. Cm's measured spread (7.0e-5) understates its error badly, while the spread implied by the split (7.7e-5) agrees with it, and the same check for Cl gives 6.5e-4 implied against 5.8e-4 measured, so the split hangs together. Averaging more members cannot reduce the shared part, which is why Cm barely improves with an ensemble. For Cl, multiplying the spread by about 2 would roughly fix the calibration outside the training range (about 1.5 inside). I have not applied or tested that on separate data, and the factor would have to be fitted on held-out samples.

### Speed

On this machine (CPU only; timings varied by about 2 times between runs, so read these as orders of magnitude):

| what is timed | time |
|---|---|
| solver, new shape each call | about 10 to 15 ms |
| solver, same shape, new angle of attack | about 0.05 to 0.2 ms |
| ensemble, large batch, per sample | about 6 to 7 microseconds |
| ensemble, one sample per call | about 2 ms |

In a large batch the ensemble is about 1,500 to 2,200 times faster than solving a new shape. Called one sample at a time it is only about 4 to 6 times faster, because five networks each cost a PyTorch call (a single network is about 30 times faster). And for a sweep of angles on one fixed shape the solver is already fast, since it reuses its factorised matrix. The surrogate's real advantage is bulk evaluation over many different shapes.

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
- Outside the training range of the shape parameters, error grows by roughly 2 to 3 times per 2% of thickness (measured), and the ensemble spread understates it by about 2 times or more. Treat a large spread as a reason to rerun the solver, and a small one as no guarantee.

## Roadmap

Spine, geometry, panel solver (validated against exact solutions before any ML), dataset generation, a from-scratch NumPy MLP, a PyTorch surrogate, deep-ensemble uncertainty with honest extrapolation tests, CST shapes, an interactive viewer, and a v0.1.0 release. See [PROJECT_PLAN.md](PROJECT_PLAN.md).

## License

MIT. See [LICENSE](LICENSE).
