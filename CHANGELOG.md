# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
