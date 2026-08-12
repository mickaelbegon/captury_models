"""Conservative decoder and diagnostics for Captury C3D angle channels.

Captury P6 exports thirteen joint-angle triplets in the C3D ``POINT`` group.
``POINT:LABELS`` contains abbreviated names (for example ``RHip``), while
``POINT:ANGLES`` contains the corresponding long names (``RHipAngles``).
Those triplets are not marker trajectories, but the files do not document the
Euler sequence, component signs, or anatomical meaning of X/Y/Z. This module
therefore decodes channel identity separately from component semantics.

The decoder never treats ``POINT:UNITS`` as an angle unit. That parameter
describes physical point coordinates and is commonly ``mm`` in Captury files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from c3d_point_channels import c3d_parameter_values, c3d_string_list

CAPTURY_ANGLE_REGISTRY_PATH = Path(__file__).with_name("captury_c3d_angles.json")
CAPTURY_ANGLE_REGISTRY = json.loads(
    CAPTURY_ANGLE_REGISTRY_PATH.read_text(encoding="utf-8")
)
CAPTURY_ANGLE_CHANNELS: tuple[dict[str, str], ...] = tuple(
    CAPTURY_ANGLE_REGISTRY["channels"]
)

_AXIS_NAMES = ("X", "Y", "Z")
_UNIT_PARAMETER_NAMES = ("ANGLE_UNITS", "ANGLES_UNITS", "ANGLE_UNIT")


def _normalized_angle_unit(value: Any) -> str | None:
    values = c3d_string_list(value)
    if not values:
        return None
    normalized = values[0].strip().lower()
    if normalized in {"deg", "degree", "degrees", "degre", "degres"}:
        return "deg"
    if normalized in {"rad", "radian", "radians"}:
        return "rad"
    return None


def resolve_captury_angle_unit(
    c3d: dict, requested_unit: str = "auto"
) -> dict[str, Any]:
    """Resolve the angle unit without consulting physical ``POINT:UNITS``."""

    requested = str(requested_unit).strip().lower()
    if requested in {"deg", "rad"}:
        return {
            "status": "documented_assumption",
            "unit": requested,
            "source": "cli_override",
            "parameter": None,
        }
    if requested != "auto":
        raise ValueError("requested_unit must be 'auto', 'deg', or 'rad'.")
    for parameter_name in _UNIT_PARAMETER_NAMES:
        raw = c3d_parameter_values(c3d, "POINT", parameter_name, None)
        unit = _normalized_angle_unit(raw)
        if unit is not None:
            return {
                "status": "metadata",
                "unit": unit,
                "source": f"POINT:{parameter_name}",
                "parameter": c3d_string_list(raw),
            }
    return {
        "status": "unknown",
        "unit": None,
        "source": None,
        "parameter": None,
        "reason": "No dedicated POINT angle-unit parameter is present.",
    }


def _channel_locations(c3d: dict) -> list[dict[str, Any]]:
    labels = c3d_string_list(c3d_parameter_values(c3d, "POINT", "LABELS", []))
    parameter_names = c3d_string_list(c3d_parameter_values(c3d, "POINT", "ANGLES", []))
    parameter_index = {name: index for index, name in enumerate(parameter_names)}
    first_tail_index = (
        len(labels) - len(parameter_names)
        if parameter_names and len(parameter_names) <= len(labels)
        else None
    )
    locations: list[dict[str, Any]] = []
    for expected in CAPTURY_ANGLE_CHANNELS:
        parameter_position = parameter_index.get(expected["parameter_name"])
        point_index: int | None = None
        mapping_source: str | None = None
        if parameter_position is not None and first_tail_index is not None:
            candidate = first_tail_index + parameter_position
            if 0 <= candidate < len(labels):
                point_index = candidate
                mapping_source = "POINT:ANGLES_tail_order"
        if point_index is None and expected["point_label"] in labels:
            point_index = labels.index(expected["point_label"])
            mapping_source = "POINT:LABELS"
        actual_label = labels[point_index] if point_index is not None else None
        identity_decoded = bool(
            point_index is not None
            and actual_label == expected["point_label"]
            and (
                parameter_position is None
                or parameter_names[parameter_position] == expected["parameter_name"]
            )
        )
        locations.append(
            {
                **expected,
                "expected_articulation": expected["articulation"],
                "articulation": expected["articulation"] if identity_decoded else None,
                "point_index": point_index,
                "actual_point_label": actual_label,
                "mapping_source": mapping_source,
                "identity_decoded": identity_decoded,
            }
        )
    return locations


def _component_diagnostics(
    values_native: np.ndarray, unit: str | None
) -> dict[str, Any]:
    finite = np.asarray(values_native, dtype=float)
    finite = finite[np.isfinite(finite)]
    scale_to_deg = 180.0 / np.pi if unit == "rad" else 1.0
    if finite.size == 0:
        return {
            "finite_samples": 0,
            "constant": False,
            "discontinuous": False,
            "outside_euler_range": False,
            "range_native": None,
            "range_deg": None,
            "max_abs_step_native": None,
            "max_abs_step_deg": None,
        }
    span_native = float(np.max(finite) - np.min(finite))
    finite_pairs = np.isfinite(values_native[:-1]) & np.isfinite(values_native[1:])
    steps_native = np.abs(np.diff(values_native))[finite_pairs]
    max_step_native = float(np.max(steps_native)) if steps_native.size else 0.0
    span_deg = span_native * scale_to_deg if unit is not None else None
    max_step_deg = max_step_native * scale_to_deg if unit is not None else None
    max_abs_deg = (
        float(np.max(np.abs(finite))) * scale_to_deg if unit is not None else None
    )
    return {
        "finite_samples": int(finite.size),
        "constant": bool(span_native <= (1e-10 if unit != "rad" else 1e-12)),
        "discontinuous": bool(max_step_deg is not None and max_step_deg > 170.0),
        "outside_euler_range": bool(
            max_abs_deg is not None and max_abs_deg > 180.0 + 1e-6
        ),
        "range_native": span_native,
        "range_deg": span_deg,
        "max_abs_step_native": max_step_native,
        "max_abs_step_deg": max_step_deg,
        "min_native": float(np.min(finite)),
        "max_native": float(np.max(finite)),
    }


def _exact_duplicate_groups(
    channel_values: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    names = list(channel_values)
    assigned: set[str] = set()
    groups: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        if name in assigned:
            continue
        matches = [name]
        reference = channel_values[name]
        for candidate in names[index + 1 :]:
            if candidate in assigned:
                continue
            if candidate.split("/", 1)[0] == name.split("/", 1)[0]:
                continue
            if np.array_equal(reference, channel_values[candidate], equal_nan=True):
                matches.append(candidate)
        if len(matches) > 1:
            assigned.update(matches)
            groups.append({"members": matches, "samples": int(reference.shape[0])})
    return groups


def analyze_captury_angle_channels(
    c3d: dict,
    *,
    requested_unit: str = "auto",
) -> dict[str, Any]:
    """Decode Captury channel identity and report numerical quality flags."""

    points = np.asarray(c3d["data"]["points"], dtype=float)[:3, :, :]
    unit = resolve_captury_angle_unit(c3d, requested_unit=requested_unit)
    channels: list[dict[str, Any]] = []
    scalar_values: dict[str, np.ndarray] = {}
    for location in _channel_locations(c3d):
        point_index = location["point_index"]
        if point_index is None or point_index >= points.shape[1]:
            values = np.full((3, points.shape[2]), np.nan, dtype=float)
        else:
            values = points[:, point_index, :]
        components = []
        ranges = []
        for axis_index, axis_name in enumerate(_AXIS_NAMES):
            component = {
                "source_component": axis_name,
                **_component_diagnostics(values[axis_index], unit["unit"]),
            }
            components.append(component)
            ranges.append(component["range_native"] or 0.0)
            if location["identity_decoded"] and np.isfinite(values[axis_index]).any():
                scalar_values[f"{location['articulation']}/{axis_name}"] = values[
                    axis_index
                ]
        dominant_index = int(np.argmax(ranges))
        sorted_ranges = sorted(ranges, reverse=True)
        dominant_range = sorted_ranges[0]
        second_range = sorted_ranges[1] if len(sorted_ranges) > 1 else 0.0
        threshold = 5.0 if unit["unit"] == "deg" else 0.0
        uniplanar = bool(
            dominant_range > 0.0
            and second_range <= max(0.1 * dominant_range, threshold)
        )
        channels.append(
            {
                **location,
                "component_semantics_decoded": False,
                "rotation_sequence": None,
                "eligible_for_anatomical_agreement": False,
                "exclusion_reason": (
                    "Captury Euler sequence, signs, and anatomical component semantics "
                    "are not documented by the C3D metadata."
                ),
                "observed_dominant_component": _AXIS_NAMES[dominant_index],
                "observed_uniplanar": uniplanar,
                "components": components,
            }
        )
    return {
        "schema_version": 1,
        "source": "captury_c3d_angles",
        "expected_channel_count": len(channels),
        "observed_channel_count": sum(
            channel["point_index"] is not None for channel in channels
        ),
        "decoded_channel_count": sum(
            channel["identity_decoded"] for channel in channels
        ),
        "channel_count": len(channels),
        "unit": unit,
        "value_scale_to_deg": (180.0 / np.pi if unit["unit"] == "rad" else 1.0),
        "component_semantics_status": "unknown",
        "eligible_for_anatomical_agreement": False,
        "channels": channels,
        "exact_duplicate_component_groups": _exact_duplicate_groups(scalar_values),
    }
