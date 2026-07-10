"""Shared unit conversion helpers for mocap point data."""

from __future__ import annotations

_MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
_CM_UNITS = {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}
_M_UNITS = {"m", "meter", "meters", "metre", "metres"}


def point_unit_scale_to_mm(unit: str) -> float:
    """Return the multiplier from a C3D point unit to millimetres.

    Unknown or empty units default to millimetres because that is the most common
    convention in the Captury/Motive datasets in this repo.
    """

    normalized_unit = str(unit).strip().lower()
    if normalized_unit in _MM_UNITS:
        return 1.0
    if normalized_unit in _CM_UNITS:
        return 10.0
    if normalized_unit in _M_UNITS:
        return 1000.0
    return 1.0


def point_unit_scale_to_m(unit: str) -> float:
    """Return the multiplier from a C3D point unit to metres."""

    return point_unit_scale_to_mm(unit) / 1000.0
