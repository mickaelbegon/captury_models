"""Build a traceable ISB D1-D3 audit of segment coordinate systems.

The audit deliberately separates three kinds of information:

* the declarative D1-D3 status stored in ``kinematic_conventions.json``;
* the segment-frame construction actually exposed by the imported BioBuddy
  Motive 57 template (origin, axes, landmarks, functional methods and
  anatomical fallbacks);
* numeric source-to-ISB deviations, which are emitted only when an explicit
  proper rotation is present in the convention registry.

This module does not infer ISB compliance from marker names or from an axis
label. In particular, an anatomically plausible BioBuddy frame remains
``documente_non_evalue`` until it has been compared with a cited ISB target
frame. Captury and Motive exporter definitions remain ``inconnu`` when their
local coordinate-system construction is unavailable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from kinematic_conventions import validate_kinematic_conventions
from kinematic_rotations import rotation_vector

AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}
BIOBUDDY_SEGMENT_IDS = {
    "Pelvis": "pelvis",
    "Thorax": "thorax",
    "Head": "head",
    "RThigh": "right_thigh",
    "RShank": "right_shank",
    "RFoot": "right_foot",
    "LThigh": "left_thigh",
    "LShank": "left_shank",
    "LFoot": "left_foot",
    "RUpperArm": "right_upper_arm",
    "RForearm": "right_forearm",
    "RHand": "right_hand",
    "LUpperArm": "left_upper_arm",
    "LForearm": "left_forearm",
    "LHand": "left_hand",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _axis_name(value: Any) -> str:
    value = _enum_value(value)
    if isinstance(value, str):
        normalized = value.upper().split(".")[-1]
        return normalized if normalized in {"X", "Y", "Z"} else str(value)
    try:
        return AXIS_NAMES[int(value)]
    except (TypeError, ValueError, KeyError):
        return str(value)


def _marker_names(value: Any) -> list[str] | None:
    names = getattr(value, "marker_names", None)
    if names is None:
        return None
    return [str(name) for name in names]


def _endpoint_definition(endpoint: Any) -> dict[str, Any]:
    """Serialize a marker or functional endpoint without evaluating data."""

    definition: dict[str, Any] = {"kind": type(endpoint).__name__}
    names = _marker_names(endpoint)
    if names is not None:
        definition["marker_names"] = names
    method = getattr(endpoint, "method", None)
    if method is not None:
        definition["functional_method"] = str(_enum_value(method))
        definition["functional_trial"] = str(getattr(endpoint, "trial_name", ""))
        for attribute in (
            "parent_marker_names",
            "child_marker_names",
            "origin_marker_names",
            "point_marker_names",
        ):
            values = getattr(endpoint, attribute, None)
            if values is not None:
                definition[attribute] = [str(value) for value in values]
        threshold = getattr(endpoint, "max_static_axis_deviation_degrees", None)
        if threshold is not None:
            definition["max_static_axis_deviation_degrees"] = float(threshold)
    fallback = getattr(endpoint, "fallback", None)
    if fallback is not None:
        definition["fallback"] = (
            _axis_definition(fallback)
            if hasattr(fallback, "start") and hasattr(fallback, "end")
            else _endpoint_definition(fallback)
        )
    expected_axis = getattr(endpoint, "expected_axis", None)
    if expected_axis is not None:
        definition["expected_axis"] = _axis_definition(expected_axis)
    return definition


def _axis_definition(axis: Any) -> dict[str, Any]:
    definition = {
        "kind": type(axis).__name__,
        "axis": _axis_name(getattr(axis, "name", "unknown")),
    }
    method = getattr(axis, "method", None)
    if method is not None:
        definition["functional_method"] = str(_enum_value(method))
        definition["functional_trial"] = str(getattr(axis, "trial_name", ""))
        for attribute in (
            "parent_marker_names",
            "child_marker_names",
            "origin_marker_names",
        ):
            values = getattr(axis, attribute, None)
            if values is not None:
                definition[attribute] = [str(value) for value in values]
        expected_axis = getattr(axis, "expected_axis", None)
        if expected_axis is not None:
            definition["expected_axis"] = _axis_definition(expected_axis)
        threshold = getattr(axis, "max_static_axis_deviation_degrees", None)
        if threshold is not None:
            definition["max_static_axis_deviation_degrees"] = float(threshold)
        fallback = getattr(axis, "fallback", None)
        if fallback is not None:
            definition["fallback"] = _axis_definition(fallback)
        return definition
    definition["start"] = _endpoint_definition(getattr(axis, "start", None))
    definition["end"] = _endpoint_definition(getattr(axis, "end", None))
    return definition


def extract_biobuddy_template_frames(template: Any) -> dict[str, dict[str, Any]]:
    """Serialize local-frame definitions from one loaded BioBuddy template."""

    frames: dict[str, dict[str, Any]] = {}
    for segment in template.segments:
        frame = getattr(segment, "frame", None)
        if frame is None:
            continue
        frames[str(segment.name)] = {
            "parent_name": str(segment.parent_name),
            "origin": _endpoint_definition(frame.origin),
            "first_axis": _axis_definition(frame.first_axis),
            "second_axis": _axis_definition(frame.second_axis),
            "axis_to_keep": _axis_name(frame.axis_to_keep),
        }
    return frames


def _load_installed_biobuddy_template() -> tuple[Any, Path]:
    template_module = importlib.import_module("biobuddy.gui.motive_57_template")

    return template_module.motive_57_template(use_functional=True), Path(
        template_module.__file__
    )


def _rotation_deviation(rotation: Any) -> tuple[float, list[float]]:
    matrix = np.asarray(rotation, dtype=float)
    vector = rotation_vector(np.eye(3), matrix)
    return float(np.degrees(np.linalg.norm(vector))), np.degrees(vector).tolist()


def _status_evidence(value: Mapping[str, Any]) -> tuple[str, list[str]]:
    return str(value["status"]), [str(item) for item in value.get("evidence", [])]


def build_isb_d1_d3_audit(
    registry: Mapping[str, Any],
    *,
    biobuddy_template_loader: Callable[[], tuple[Any, Path]] | None = None,
) -> dict[str, Any]:
    """Build an audit report from the registry and the active BioBuddy runtime."""

    validate_kinematic_conventions(registry)
    loader = biobuddy_template_loader or _load_installed_biobuddy_template
    template_frames: dict[str, dict[str, Any]] = {}
    try:
        template, template_path = loader()
        resolved_template_path = Path(template_path).expanduser().resolve()
        template_frames = extract_biobuddy_template_frames(template)
        template_provenance: dict[str, Any] = {
            "status": "available",
            "template_name": str(template.name),
            "path": str(resolved_template_path),
            "sha256": _sha256(resolved_template_path),
            "frames": template_frames,
        }
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        template_provenance = {
            "status": "unavailable",
            "reason": f"{type(error).__name__}: {error}",
            "frames": {},
        }

    template_by_segment_id = {
        BIOBUDDY_SEGMENT_IDS[name]: definition
        for name, definition in template_frames.items()
        if name in BIOBUDDY_SEGMENT_IDS
    }
    template_sha256 = template_provenance.get("sha256")
    rows: list[dict[str, Any]] = []
    for source_id, source in registry["sources"].items():
        for segment_id, segment in source["segments"].items():
            frame_status, frame_evidence = _status_evidence(segment["frame_definition"])
            row: dict[str, Any] = {
                "source_id": source_id,
                "segment_id": segment_id,
                "source_name": segment["source_name"],
                "parent": segment["parent"],
                "runtime_template_status": (
                    str(template_provenance["status"])
                    if source_id == "biobuddy_motive57"
                    else "not_applicable"
                ),
                "frame_definition_status": frame_status,
                "frame_definition_evidence": frame_evidence,
                "template_frame": (
                    template_by_segment_id.get(segment_id)
                    if source_id == "biobuddy_motive57"
                    else None
                ),
                "origin_deviation_mm": None,
            }
            for deviation in ("D1", "D2", "D3"):
                status, evidence = _status_evidence(segment["isb_audit"][deviation])
                if (
                    source_id == "biobuddy_motive57"
                    and status == "documente_non_evalue"
                    and not evidence
                    and segment_id in template_by_segment_id
                    and template_sha256
                ):
                    evidence = [
                        "Runtime BioBuddy template construction serialized from "
                        f"SHA-256 {template_sha256}; ISB target comparison pending."
                    ]
                row[f"{deviation}_status"] = status
                row[f"{deviation}_evidence"] = evidence
            correction = segment["source_to_isb_rotation"]
            if correction is None:
                row.update(
                    {
                        "numeric_status": "source_to_isb_rotation_unknown",
                        "angular_deviation_deg": None,
                        "rotation_vector_deg": None,
                    }
                )
            else:
                angle, vector = _rotation_deviation(correction)
                row.update(
                    {
                        "numeric_status": "available",
                        "angular_deviation_deg": angle,
                        "rotation_vector_deg": vector,
                    }
                )
            rows.append(row)

    return {
        "audit_version": 1,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "scope": "ISB segment-frame deviations D1-D3",
        "interpretation": (
            "A null numeric deviation means that no justified source-to-ISB "
            "rotation is available; it is not a zero deviation. Origin deviation "
            "requires paired source and ISB origins and is therefore unavailable "
            "until those landmarks are defined and measured."
        ),
        "biobuddy_template": template_provenance,
        "segments": rows,
    }


def _table_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in report["segments"]:
        row = dict(source)
        for key, value in tuple(row.items()):
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, sort_keys=True)
        rows.append(row)
    return rows


def _write_table_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    dataframe = pd.DataFrame(rows)
    columns = np.asarray(list(dataframe.columns), dtype=str)
    payload: dict[str, np.ndarray] = {"columns": columns}
    for index, column in enumerate(columns):
        series = dataframe[str(column)]
        if pd.api.types.is_numeric_dtype(series):
            payload[f"col_{index}"] = series.to_numpy()
        else:
            payload[f"col_{index}"] = series.fillna("").astype(str).to_numpy(dtype=str)
    np.savez_compressed(path, **payload)


def write_isb_d1_d3_audit(
    output_dir: Path | str, report: Mapping[str, Any]
) -> dict[str, Path]:
    """Write the human-readable JSON and compact tabular NPZ audit outputs."""

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "isb_d1_d3_audit.json"
    table_path = root / "isb_d1_d3_audit.npz"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_table_npz(table_path, _table_rows(report))
    return {"json": json_path, "table": table_path}
