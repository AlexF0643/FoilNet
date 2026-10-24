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

**Phases 0 to 2 done (spine, geometry, panel solver).** The solver is validated (see below). There is no dataset and no network yet. The remaining CLI subcommands are stubs that say which phase will implement them. Progress is tracked in [CHANGELOG.md](CHANGELOG.md) and the full roadmap is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

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

Planned subcommands that are still stubs: `generate`, `train`, `serve`.

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
