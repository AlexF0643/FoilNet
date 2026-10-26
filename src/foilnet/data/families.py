"""Shape families: how a row of numbers becomes an aerofoil contour.

A family fixes the input features (shape parameters, then the angle of attack, which is always
last), how to build the contour, which parameter sets are acceptable, and a few derived
descriptors worth keeping alongside each sample.

Families are small frozen dataclasses so they can be sent to worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from foilnet.geometry import naca4
from foilnet.geometry.cst import (
    ShapeLimits,
    check_shape,
    cst_coordinates,
    te_included_angle_deg,
    thickness_and_camber,
)

NACA4_FEATURES = ("camber", "camber_position", "thickness", "alpha_deg")


def cst_feature_names(n_coefficients: int) -> tuple[str, ...]:
    return (
        *(f"au{i}" for i in range(n_coefficients)),
        *(f"al{i}" for i in range(n_coefficients)),
        "alpha_deg",
    )


@dataclass(frozen=True)
class ShapeFamily:
    name: str  # "naca4" or "cst"
    n_coefficients: int = 4  # per surface, CST only
    limits: ShapeLimits = field(default_factory=ShapeLimits)  # CST only

    def __post_init__(self) -> None:
        if self.name not in ("naca4", "cst"):
            raise ValueError(f"unknown shape family {self.name!r}; choose 'naca4' or 'cst'")
        if self.name == "cst" and not 2 <= self.n_coefficients <= 10:
            raise ValueError(f"n_coefficients must be between 2 and 10, got {self.n_coefficients}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.name == "naca4":
            return NACA4_FEATURES
        return cst_feature_names(self.n_coefficients)

    @property
    def needs_filter(self) -> bool:
        """Whether some random parameter sets are not usable shapes."""
        return self.name == "cst"

    def build(self, shape_params: np.ndarray, n_panels: int) -> np.ndarray:
        """Contour for the shape parameters (every feature except the angle of attack)."""
        if self.name == "naca4":
            m, p, t = shape_params
            return naca4(float(m), float(p), float(t), n_panels)
        n = self.n_coefficients
        return cst_coordinates(shape_params[:n], shape_params[n:], n_panels)

    def is_valid(self, shape_params: np.ndarray) -> bool:
        if self.name == "naca4":
            return True
        n = self.n_coefficients
        return not check_shape(shape_params[:n], shape_params[n:], self.limits)

    def descriptors(self, shape_params: np.ndarray) -> dict[str, float]:
        """Derived quantities stored with each sample (empty for NACA, whose parameters
        already are thickness and camber)."""
        if self.name == "naca4":
            return {}
        n = self.n_coefficients
        upper, lower = shape_params[:n], shape_params[n:]
        thickness, camber = thickness_and_camber(upper, lower)
        return {
            "thickness": thickness,
            "camber": camber,
            "te_angle_deg": te_included_angle_deg(upper, lower),
        }

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary for the dataset metadata."""
        out: dict[str, Any] = {"family": self.name}
        if self.name == "cst":
            out["n_coefficients"] = self.n_coefficients
            out["limits"] = {
                "min_te_angle_deg": self.limits.min_te_angle_deg,
                "min_thickness": self.limits.min_thickness,
                "max_thickness": self.limits.max_thickness,
                "max_camber": self.limits.max_camber,
                "min_le_coefficient": self.limits.min_le_coefficient,
                "min_le_ratio": self.limits.min_le_ratio,
            }
        return out

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> ShapeFamily:
        """Rebuild the family a dataset was made with (files from before CST are NACA)."""
        name = metadata.get("family", "naca4")
        if name == "naca4":
            return cls("naca4")
        return cls("cst", int(metadata["n_coefficients"]), ShapeLimits(**metadata["limits"]))
