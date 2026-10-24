# FoilNet — Project Plan

**A neural-network surrogate for aerofoil aerodynamics, trained on data from a panel-method solver I wrote myself.**

Author: Alex Forbes
Status: planning

---

## 1. The idea

Aerodynamic solvers are accurate but slow, and design work means running them thousands of times. A surrogate model is a neural network trained to copy what the solver does. It takes an aerofoil shape and an angle of attack and returns the lift and moment coefficients (and later drag) in microseconds rather than seconds.

The point of the project isn't only to get a network that fits. The more interesting questions are these:

- How much training data does the network actually need?
- Where does it stay trustworthy, and where does it fail quietly?
- Can it tell me when it's guessing?

The data comes from a vortex panel method I write and validate myself, so I understand the physics behind every label rather than treating the dataset as a black box.

---

## 2. Objectives

1. **Write and validate a panel-method solver.** It should give Cp, Cl and Cm for arbitrary aerofoils and be checked against thin-aerofoil theory and published NACA data.
2. **Parameterise aerofoil shapes.** Start with NACA 4-digit (camber, camber position, thickness), then move to CST (Kulfan) parameters for a richer design space.
3. **Generate a dataset.** Latin hypercube sampling over shape parameters and angle of attack, with every sample's provenance recorded (solver version, panel count, parameters).
4. **Train a surrogate.** An MLP mapping (shape parameters, α) → (Cl, Cm). Write a small NumPy implementation first to understand backprop, then switch to PyTorch for the real experiments.
5. **Quantify uncertainty.** Use a deep ensemble (several networks trained from different initialisations), with spread across the ensemble as the uncertainty estimate. Check whether that spread actually grows on shapes outside the training range.
6. **Run the honest experiments.** Learning curves (error vs dataset size), an interpolation vs extrapolation test (train on thin aerofoils, test on thick), and speed-up vs the solver.
7. **Stretch goal: predict the whole Cp distribution**, not just integrated coefficients. **Stretch goal: viscous drag** from a coupled integral boundary-layer method (Thwaites + Head + Squire–Young).
8. **Build an interactive viewer.** Sliders for shape and α, with the aerofoil, the solver's Cp curve and the network's prediction (with its uncertainty band) overlaid live.

---

## 3. What carries over from missile-intercept-tracker

| From the old project | Use here |
|---|---|
| TOML config loading (`utf-8-sig`, clear errors) | Dataset and training configs |
| Monte Carlo + Spearman sensitivity | Which shape parameters drive Cl most, and does the network's sensitivity match the solver's? |
| Testing discipline (pytest, analytic checks) | Solver validated against closed-form results before any ML |
| GitHub Actions CI (Ubuntu + Windows, Py 3.11/3.12) | Same matrix |
| stdlib HTTP server + hand-rolled canvas viewer | Interactive aerofoil / Cp viewer |
| README voice, CHANGELOG, social preview workflow | Same |

New for this project: the panel method, shape parameterisation, dataset pipeline, neural networks and uncertainty estimation.

---

## 4. Physics notes (what the solver must get right)

- **Method:** linear-strength vortex panel method with a Kutta condition at the trailing edge. Inviscid and incompressible.
- **What it can and can't do:** gives good Cl and Cm in the linear range. It cannot predict stall or drag. The surrogate inherits these limits, and the README must say so plainly.
- **Validation targets:**
  - Symmetric aerofoil: Cl ≈ 2πα for thin sections (thin-aerofoil theory)
  - Cl at α = 0 for cambered NACA sections vs thin-aerofoil theory
  - Cp on a circle/cylinder (via a Kármán–Trefftz or Joukowski aerofoil) vs the exact conformal-mapping solution, which is the strongest check
  - Panel-count convergence study: results must stop changing as panels increase
- **Sampling range:** α from −5° to +12° to stay in the regime where inviscid theory is meaningful.

---

## 5. Repository structure (spine)

```
foilnet/
├── README.md
├── CHANGELOG.md
├── PROJECT_PLAN.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .github/workflows/ci.yml
├── configs/
│   ├── dataset-naca4.toml
│   └── train-mlp.toml
├── src/foilnet/
│   ├── __init__.py
│   ├── config.py              # TOML loading (ported)
│   ├── geometry/
│   │   ├── naca.py            # NACA 4-digit coordinates
│   │   ├── cst.py             # CST / Kulfan parameterisation
│   │   └── paneling.py        # cosine spacing, panel construction
│   ├── solver/
│   │   ├── panel.py           # linear-vortex panel method
│   │   ├── analytic.py        # thin-aerofoil + Joukowski exact solutions
│   │   └── boundary_layer.py  # (stretch) Thwaites/Head/Squire–Young
│   ├── data/
│   │   ├── sampling.py        # Latin hypercube
│   │   ├── generate.py        # run solver over samples → dataset file
│   │   └── dataset.py         # load, split, normalise
│   ├── nn/
│   │   ├── numpy_mlp.py       # from-scratch MLP + backprop (learning exercise)
│   │   ├── model.py           # PyTorch MLP
│   │   ├── train.py
│   │   └── ensemble.py        # deep ensemble + uncertainty
│   ├── analysis/
│   │   ├── learning_curve.py
│   │   ├── extrapolation.py
│   │   └── sensitivity.py     # Spearman, solver vs network
│   ├── web/                   # viewer (ported server, new canvas)
│   └── cli.py                 # `foilnet solve | generate | train | serve`
├── tests/
│   ├── test_geometry.py
│   ├── test_panel_analytic.py # solver vs exact solutions
│   ├── test_convergence.py
│   ├── test_sampling.py
│   ├── test_numpy_mlp.py      # gradient check vs finite differences
│   └── test_ensemble.py
└── docs/
    └── assets/
```

---

## 6. Phases

**Phase 0: Spine.** Repo skeleton, packaging, CI running on an empty test suite, CLI stub.

**Phase 1: Geometry.** NACA 4-digit and cosine panelling. Tests: closed trailing edge, correct thickness at x/c = 0.3, symmetric sections stay symmetric.

**Phase 2: Panel solver.** Linear-vortex method plus Kutta condition. Validate against Joukowski exact Cp and thin-aerofoil Cl, and run the convergence study. *This phase is done when the solver is trusted. No ML before that.*

**Phase 3: Dataset.** Latin hypercube sampling and batch generation (parallel, reusing the ProcessPoolExecutor pattern), saved with metadata.

**Phase 4: NumPy MLP.** Forward pass, backprop, a gradient check against finite differences, and training on a small subset. This is the learning-the-fundamentals phase.

**Phase 5: PyTorch surrogate.** Proper training loop, train/val/test split, and normalisation, then hyperparameter choices documented and justified.

**Phase 6: Uncertainty and honest testing.** Deep ensemble, learning curves, the extrapolation experiment, a calibration check (is the error inside the uncertainty band as often as it claims to be?) and timing vs the solver.

**Phase 7: CST shapes.** Richer design space. Does the network still cope?

**Phase 8: Viewer.** Interactive sliders and the live solver-vs-network comparison.

**Phase 9 (stretch): Viscous drag, full-Cp prediction.**

**Phase 10: Polish.** README, figures, CHANGELOG, v0.1.0 release.

---

## 7. Definition of done for v0.1.0

- Solver matches the Joukowski exact solution to within a stated tolerance, with a test that enforces it
- Surrogate Cl error under a stated threshold on the in-distribution test set
- A documented, quantified failure mode (extrapolation) rather than a hidden one
- Ensemble uncertainty shown to be calibrated, or shown not to be with an explanation of why
- CI green on Ubuntu + Windows, Python 3.11 + 3.12
- README written in my own voice, with an explicit "what this can't do" section

---

## 8. Dependencies

- Core: `numpy`, `scipy` (Latin hypercube via `scipy.stats.qmc`)
- ML: `torch` (CPU is fine; the networks are small)
- Plots: `matplotlib`
- Dev: `pytest`, `ruff`

## 9. Key references

- Katz & Plotkin, *Low-Speed Aerodynamics*: panel methods
- Anderson, *Fundamentals of Aerodynamics*: thin-aerofoil theory
- Kulfan (2008), "Universal Parametric Geometry Representation Method": CST
- Lakshminarayanan et al. (2017), "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
- Drela, XFOIL documentation: for cross-checking results
