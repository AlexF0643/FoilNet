"""Aerofoil shapes and panelling."""

from foilnet.geometry.cst import (
    ShapeLimits,
    bernstein_basis,
    check_shape,
    class_function,
    cst_coordinates,
    cst_surface,
    fit_cst,
    fit_cst_surface,
    te_included_angle_deg,
    thickness_and_camber,
)
from foilnet.geometry.naca import naca4, naca4_camber, naca4_from_code, naca4_thickness, parse_naca4
from foilnet.geometry.paneling import Panels, build_panels, cosine_spacing, signed_area

__all__ = [
    "Panels",
    "ShapeLimits",
    "bernstein_basis",
    "check_shape",
    "class_function",
    "cst_coordinates",
    "cst_surface",
    "fit_cst",
    "fit_cst_surface",
    "te_included_angle_deg",
    "thickness_and_camber",
    "build_panels",
    "cosine_spacing",
    "naca4",
    "naca4_camber",
    "naca4_from_code",
    "naca4_thickness",
    "parse_naca4",
    "signed_area",
]
