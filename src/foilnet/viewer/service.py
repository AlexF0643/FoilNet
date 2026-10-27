"""What the viewer computes for one set of slider values.

Kept apart from the HTTP layer so it can be tested without a server. Given a trained model
(a single surrogate or a deep ensemble) it reads the shape family, training ranges and panel
count from the model's own provenance, so a viewer needs nothing but the model file.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from foilnet.data.families import ShapeFamily
from foilnet.geometry import check_shape, fit_cst, naca4_from_code, parse_naca4
from foilnet.nn.ensemble import Ensemble
from foilnet.nn.train import Surrogate
from foilnet.solver import PanelSolver

# Sliders reach this far beyond the training range (as a fraction of its width), so the
# extrapolation can be explored. Values past the training range are flagged in the result.
SLIDER_MARGIN = 0.25
# Hard limits on what a client may ask for.
MAX_ABS_ALPHA_DEG = 30.0
SWEEP_POINTS = 53
SWEEP_PAD_DEG = 6.0  # the sweep runs this far beyond the training range of alpha
NACA_PRESETS = ("0012", "2412", "4412")


class ViewerError(ValueError):
    """A request that cannot be answered; the message is safe to show to the user."""


def _round(values: np.ndarray, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values).ravel()]


class ViewerModel:
    """A trained model plus the solver, ready to answer slider requests."""

    def __init__(self, model: Ensemble | Surrogate):
        self.model = model
        provenance = (model.members[0] if isinstance(model, Ensemble) else model).provenance
        try:
            self.family = ShapeFamily.from_metadata(provenance)
            self.ranges: dict[str, tuple[float, float]] = {
                k: (float(v[0]), float(v[1])) for k, v in provenance["ranges"].items()
            }
            self.n_panels = int(provenance["n_panels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ViewerError(f"the model file lacks training information ({exc!r})") from exc
        self.feature_names = tuple(model.feature_names)
        if self.feature_names != self.family.feature_names:
            raise ViewerError("the model's inputs do not match its shape family")
        if set(self.ranges) != set(self.feature_names):
            raise ViewerError("the model's training ranges do not match its inputs")
        self.shape_names = self.feature_names[:-1]
        self.target_names = tuple(model.target_names)
        self.n_members = len(model.members) if isinstance(model, Ensemble) else 1
        self._preset_list = self._presets()

    # ---- description of the model, sent once when the page loads ----

    def info(self) -> dict[str, Any]:
        features = []
        for name in self.feature_names:
            lo, hi = self.ranges[name]
            pad = SLIDER_MARGIN * (hi - lo)
            features.append({
                "name": name,
                "train_min": lo,
                "train_max": hi,
                "slider_min": lo - pad,
                "slider_max": hi + pad,
            })
        a_lo, a_hi = self.ranges["alpha_deg"]
        return {
            "family": self.family.name,
            "features": features,
            "targets": list(self.target_names),
            "n_members": self.n_members,
            "n_panels": self.n_panels,
            "has_spread": self.n_members > 1,
            "sweep_alpha": [a_lo - SWEEP_PAD_DEG, a_hi + SWEEP_PAD_DEG],
            "presets": self._preset_list,
            "default": self._default(),
        }

    def _default(self) -> dict[str, float]:
        values = {n: 0.5 * sum(self.ranges[n]) for n in self.feature_names}
        presets = self._preset_list
        if presets:
            values.update(presets[min(1, len(presets) - 1)]["values"])
        a_lo, a_hi = self.ranges["alpha_deg"]
        values["alpha_deg"] = min(max(4.0, a_lo), a_hi)
        return values

    def _presets(self) -> list[dict[str, Any]]:
        out = []
        for code in NACA_PRESETS:
            if self.family.name == "naca4":
                m, p, t = parse_naca4(code)
                if m == 0.0:  # a symmetric section has no camber position; any value will do
                    p = 0.5 * sum(self.ranges["camber_position"])
                values = dict(zip(self.shape_names, (m, p, t), strict=True))
            else:
                n = self.family.n_coefficients
                upper, lower = fit_cst(naca4_from_code(code, 160), n)
                values = dict(zip(self.shape_names, (*upper, *lower), strict=True))
            suffix = " (CST fit)" if self.family.name == "cst" else ""
            out.append({"name": f"NACA {code}{suffix}",
                        "values": {k: float(v) for k, v in values.items()}})
        return out

    # ---- one evaluation ----

    def _parse(self, request: dict[str, Any]) -> dict[str, float]:
        values = {}
        for name in self.feature_names:
            raw = request.get(name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise ViewerError(f"{name} must be a finite number")
            values[name] = float(raw)
        if abs(values["alpha_deg"]) > MAX_ABS_ALPHA_DEG:
            raise ViewerError(f"alpha_deg must be within +/-{MAX_ABS_ALPHA_DEG:g}")
        for name in self.shape_names:
            if abs(values[name]) > 2.0:
                raise ViewerError(f"{name} is far outside any sensible shape")
        return values

    def _warnings(self, values: dict[str, float]) -> tuple[list[str], list[str]]:
        outside = [
            n for n in self.feature_names
            if not self.ranges[n][0] <= values[n] <= self.ranges[n][1]
        ]
        warnings: list[str] = []
        if self.family.name == "cst":
            n = self.family.n_coefficients
            shape = np.array([values[k] for k in self.shape_names])
            problems = check_shape(shape[:n], shape[n:], self.family.limits)
            warnings = [f"outside the shape limits: {p}" for p in problems]
        return outside, warnings

    def evaluate(self, request: dict[str, Any], sweep: bool = True) -> dict[str, Any]:
        values = self._parse(request)
        shape = np.array([values[n] for n in self.shape_names])
        outside, warnings = self._warnings(values)

        start = time.perf_counter()
        try:
            coords = self.family.build(shape, self.n_panels)
            solver = PanelSolver(coords)
            solution = solver.solve(values["alpha_deg"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ViewerError(f"the solver could not handle this shape ({exc})") from exc
        if not (math.isfinite(solution.cl) and math.isfinite(solution.cm)):
            raise ViewerError("the solver returned a non-finite result for this shape")
        solver_ms = (time.perf_counter() - start) * 1000.0

        row = np.array([[values[n] for n in self.feature_names]])
        start = time.perf_counter()
        mean, std = self._predict(row)
        surrogate_ms = (time.perf_counter() - start) * 1000.0

        mid = solution.panels.midpoint
        i_le = int(np.argmin(mid[:, 0]))
        result: dict[str, Any] = {
            "values": values,
            "outside": outside,
            "warnings": warnings,
            "contour": {"x": _round(coords[:, 0]), "y": _round(coords[:, 1])},
            "cp": {
                # contour order runs from the upper trailing edge round the nose to the lower one
                "upper": {"x": _round(mid[: i_le + 1, 0]), "cp": _round(solution.cp[: i_le + 1])},
                "lower": {"x": _round(mid[i_le:, 0]), "cp": _round(solution.cp[i_le:])},
            },
            "solver": {"cl": solution.cl, "cm": solution.cm},
            "surrogate": {
                "mean": dict(zip(self.target_names, (float(v) for v in mean[0]), strict=True)),
                "std": None if std is None else dict(
                    zip(self.target_names, (float(v) for v in std[0]), strict=True)
                ),
            },
            "timing_ms": {"solver": solver_ms, "surrogate": surrogate_ms},
        }
        if sweep:
            result["sweep"] = self._sweep(solver, values)
        return result

    def _predict(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(self.model, Ensemble):
            return self.model.predict(rows)
        return self.model.predict(rows), None

    def _sweep(self, solver: PanelSolver, values: dict[str, float]) -> dict[str, Any]:
        a_lo, a_hi = self.ranges["alpha_deg"]
        alphas = np.linspace(a_lo - SWEEP_PAD_DEG, a_hi + SWEEP_PAD_DEG, SWEEP_POINTS)
        solved = [solver.solve(float(a)) for a in alphas]
        rows = np.tile([values[n] for n in self.feature_names], (len(alphas), 1))
        rows[:, -1] = alphas
        mean, std = self._predict(rows)
        out: dict[str, Any] = {
            "alpha_deg": _round(alphas, 4),
            "train_alpha": [a_lo, a_hi],
            "solver": {
                "cl": _round(np.array([s.cl for s in solved])),
                "cm": _round(np.array([s.cm for s in solved])),
            },
            "surrogate": {t: _round(mean[:, j]) for j, t in enumerate(self.target_names)},
            "spread": None if std is None else {
                t: _round(std[:, j]) for j, t in enumerate(self.target_names)
            },
        }
        return out

