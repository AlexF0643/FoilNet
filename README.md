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

**Phases 0 and 1 done (spine and geometry).** NACA 4-digit shapes and panel construction work and are tested. There is no solver, no dataset and no network yet. The CLI subcommands are stubs that say which phase will implement them. Progress is tracked in [CHANGELOG.md](CHANGELOG.md) and the full roadmap is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

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
foilnet --help
```

Planned subcommands: `solve`, `generate`, `train`, `serve`.

## What this can't do

This section will grow as the project does, but the limits that are already fixed by design:

- The solver is a linear-vortex panel method: inviscid and incompressible. It gives good lift and moment in the linear range and **cannot predict stall or drag**.
- The surrogate learns the solver, not reality, so it inherits those limits and adds its own error on top.
- Training data will cover angles of attack from −5° to +12°. Outside that range, results aren't meaningful.

## Roadmap

Spine, geometry, panel solver (validated against exact solutions before any ML), dataset generation, a from-scratch NumPy MLP, a PyTorch surrogate, deep-ensemble uncertainty with honest extrapolation tests, CST shapes, an interactive viewer, and a v0.1.0 release. See [PROJECT_PLAN.md](PROJECT_PLAN.md).

## License

MIT. See [LICENSE](LICENSE).
