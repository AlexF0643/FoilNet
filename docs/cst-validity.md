# CST shapes: which shapes are usable, and why

Random CST coefficients make many shapes that are not usable aerofoils, and the panel solver
is not equally accurate on the ones that are. This page records what I measured and the limits
that follow from it (`ShapeLimits` in `src/foilnet/geometry/cst.py`, set in
`configs/dataset-cst.toml`). Everything here was measured on this repository's solver and code.

## 1. Trailing-edge angle: the solver is more tolerant than I first thought

The only exact reference with a finite trailing-edge angle is the Karman-Trefftz aerofoil, whose
included angle is `(2 - n) * 180` degrees. Relative error in Cl against the exact lift, 160 panels,
alpha = 3 degrees, for the solver as it was through Phase 8 (`legacy`) and the Phase 9.1 trailing-edge
treatment (`smooth`; see [trailing-edge.md](trailing-edge.md)):

| trailing-edge angle | relative Cl error, legacy | smooth |
|---|---|---|
| 0 degrees (pure Joukowski cusp) | about 2% | 5.3e-4 |
| 1.8 | 9.5e-4 | 4.5e-4 |
| 3.6 | 6.8e-4 | 4.0e-4 |
| 5.4 | 5.2e-4 | 3.7e-4 |
| 9.0 | 3.8e-4 | 3.4e-4 |
| 18 | 2.6e-4 | 3.1e-4 |
| 27 | 2.2e-4 | 3.0e-4 |

With the legacy treatment only a true cusp was a problem; the smooth one handles it too. Error
grows slowly as the angle closes, and 320 panels cuts it by about 3 (legacy) to 4 (smooth). An earlier note in the README said CST shapes might be risky because of cusped trailing
edges; the measurements say the risk is a perfect cusp, which CST shapes do not produce unless a
shape is degenerate. The limit is **5 degrees**, where the error is about 5e-4 at 160 panels. That
limit was chosen for the legacy treatment; with the smooth one it could be relaxed, and I have not
done so.

## 2. Leading edge: a knife-edge nose has no valid answer

Near the leading edge each CST surface is `a * sqrt(x)`, a parabola with nose radius `a^2 / 2`
(`a = au0` above, `a = al0` below, which is negative). If `al0` is near zero, the lower surface
leaves the nose flat while the upper one rises steeply: a knife edge. Potential flow has infinite
suction at a convex corner, and the panel solver correctly tries to reproduce it.

In the first 10,000 samples drawn without any nose limit, 17% had a pressure-drag diagnostic above
0.05 (it should be zero), and Cm reached 3.4 against a typical spread of 0.12. The failure rate by
`al0` (rows) and `au0` (columns), at 160 panels:

| `al0` \ `au0` | 0.08-0.20 | 0.20-0.30 | 0.30-0.40 | 0.40-0.48 |
|---|---|---|---|---|
| below -0.15 | 0% | 0% | 0% | 0% |
| -0.15 to -0.08 | 0% | 0% | 0% | 2% |
| -0.08 to -0.03 | 2% | 7% | 11% | 23% |
| -0.03 to 0 | 37% | 51% | 66% | 76% |
| 0 to 0.04 | 60% | 76% | 83% | 83% |

These are **not** a resolution problem. For the worst sample, the smallest surface pressure
coefficient was -1.4e5 at 80 panels, -1.4e6 at 160, -1.2e7 at 320, -8e7 at 640 and -3.9e8 at 1280,
and Cl and Cm drifted at every refinement. More panels make it worse because the shape really is
singular, so these samples have no converged label to learn.

The rule: both `au0` and `-al0` must be at least **0.10** (a NACA 0006, a thin nose, is about 0.09),
and the smaller must be at least **0.25** times the larger. Tested on the original 10,000 samples at
160 panels, the share with a pressure-drag diagnostic above 0.05 falls from 17% to 0.09% with the
lower-nose half of this rule. In the final 320-panel dataset the largest diagnostic is 0.024. A
safety net still drops any sample whose diagnostic is above 0.05 (`max_abs_cd` in the config).

The pressure-drag diagnostic turned out to be a poor predictor of label accuracy (section 3), so it
is only a net for catastrophic failures, not a quality filter.

## 3. A drooping trailing edge slowed the old solver's convergence (fixed in Phase 9.1)

This section records what was measured with the legacy trailing-edge treatment; the cause turned
out to be the trailing edge itself (a nearly undetermined pair of end nodes) and is fixed by the
smoothness condition described in [trailing-edge.md](trailing-edge.md). With it, CST labels at 160
panels differ from a 640-panel solve by an RMS of 2.3e-4 in Cl (legacy: 1.2e-3), and at 320
panels by 4.4e-5 (legacy: 4.2e-4).

After the nose rule, the solver's labels were still less accurate than for NACA shapes. Against a
640-panel solve of the same shapes (900 random samples that passed the rules):

| | RMS error in Cl | RMS error in Cm |
|---|---|---|
| NACA dataset, 160 panels | 1.6e-4 | 1.9e-4 |
| CST, 160 panels | 1.5e-3 | 5.7e-4 |
| CST, 320 panels vs 640 | 5.2e-4 | 1.5e-4 |

What drives it is the slope of the upper surface where it runs into the trailing edge, which is
`au3` (Spearman correlation with the Cl error: +0.71; with the included trailing-edge angle: +0.44;
with nothing at the leading edge: about 0). Holding the included angle between 10 and 20 degrees
fixed, 160-panel Cl error rises with the upper-surface slope at the trailing edge:

| upper surface slope into the trailing edge | RMS Cl error (160 panels) |
|---|---|
| under 8 degrees | 1.6e-4 (the NACA level) |
| 8 to 14 | 2.8e-4 |
| 14 to 20 | 7.3e-4 |
| 20 to 30 | 1.2e-3 |

NACA sections have a near-symmetric wedge at the trailing edge, so they never meet this; random
CST shapes often have a drooped one. Doubling the panels reduces the error by about 3 (a clean
second-order method would give 4), so the convergence is somewhat slower than ideal for these
shapes. **This was a weakness of the solver, not of CST.** At the time I worked around it: the CST
labels use **320 panels** instead of 160, and the remaining difference from a converged solve is
reported with the results (`docs/results-cst.md`). The dataset keeps 320 panels after the fix, since
it brings the label difference to about 5e-5 in Cl, a tenth of the network's own error.

## 4. CST with four coefficients is not NACA

A four-coefficient fit cannot reproduce every NACA leading edge: the Cl of a fitted NACA 2412 is
within 0.4% of the original (six coefficients) and up to about 1.2% for cambered thin sections
(four). So the CST family is its own family, not a re-parameterisation of the NACA one. The
coefficient box in the config was built from fits to a grid of NACA shapes and then widened.

## 5. What is kept

All limits together (`ShapeLimits`): trailing-edge included angle at least 5 degrees; maximum
thickness between 4% and 22%; maximum camber at most 8%; both nose coefficients at least 0.10,
the smaller at least 0.25 of the larger; the surfaces never cross. About 41% of candidate shapes
from the box pass, so the generator draws candidates in batches and keeps the usable ones
(`n_rejected_invalid_shape` in the dataset metadata records how many were dropped).

## 6. What this does not cover

- Shapes with a drooped trailing edge have larger label error than the rest at any given panel
  count (section 3).
- The limits were chosen to keep the *solver* trustworthy. Whether a given shape is a good
  aerofoil, or how it would behave in a real flow, is not judged.
- The solver is still inviscid: no stall and no drag.
