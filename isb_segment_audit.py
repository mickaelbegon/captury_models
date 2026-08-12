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
DEFAULT_ISB_TARGETS_PATH = Path(__file__).with_name("isb_segment_frames.json")
_SIDED_SEGMENT_BASE = {
    "right_thigh": "thigh",
    "left_thigh": "thigh",
    "right_shank": "shank",
    "left_shank": "shank",
    "right_foot": "foot",
    "left_foot": "foot",
    "right_upper_arm": "upper_arm",
    "left_upper_arm": "upper_arm",
    "right_forearm": "forearm",
    "left_forearm": "forearm",
    "right_hand": "hand",
    "left_hand": "hand",
}


def validate_isb_segment_targets(targets: Mapping[str, Any]) -> None:
    """Validate the compact, versioned Wu 2002/2005 target registry."""

    if targets.get("schema_version") != 1:
        raise ValueError("Unsupported ISB segment-target schema version")
    segments = targets.get("segments")
    if not isinstance(segments, Mapping):
        raise ValueError("ISB segment targets must define a segments mapping")
    required = {
        "pelvis",
        "thorax",
        "head",
        "thigh",
        "shank",
        "foot",
        "upper_arm",
        "forearm",
        "hand",
    }
    missing = required - set(segments)
    if missing:
        raise ValueError(f"Missing ISB segment targets: {sorted(missing)}")
    for segment_id, target in segments.items():
        applicability = target.get("applicability")
        if applicability not in {"applicable", "surrogate", "non_applicable"}:
            raise ValueError(f"Invalid applicability for {segment_id}: {applicability}")
        citation = target.get("citation", {})
        if not str(citation.get("doi", "")).startswith("10.1016/"):
            raise ValueError(f"Primary-source DOI missing for {segment_id}")
        if not str(target.get("origin", {}).get("definition", "")):
            raise ValueError(f"Origin definition missing for {segment_id}")
        axes = target.get("axes", {})
        if applicability != "non_applicable" and set(axes) != {"X", "Y", "Z"}:
            raise ValueError(f"X/Y/Z target axes missing for {segment_id}")


def load_isb_segment_targets(
    path: Path | str = DEFAULT_ISB_TARGETS_PATH,
) -> dict[str, Any]:
    """Load and validate the selected ISB target definitions."""

    targets = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    validate_isb_segment_targets(targets)
    return targets


def load_biobuddy_audit_sidecars(biomod_path: Path | str | None) -> dict[str, Any]:
    """Load optional roundtrip/static evidence adjacent to one selected bioMod."""

    if biomod_path is None:
        return {
            "biomod_verification": {"status": "not_run"},
            "static_evaluation": {"status": "not_run"},
        }
    biomod = Path(biomod_path).expanduser()
    paths = {
        "biomod_verification": biomod.with_suffix(".roundtrip.json"),
        "static_evaluation": biomod.with_suffix(".isb_static.json"),
    }
    loaded: dict[str, Any] = {}
    biomod_sha256 = _sha256(biomod) if biomod.exists() else None
    for key, path in paths.items():
        if path.exists():
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
            loaded[key]["evidence_path"] = str(path.resolve())
            loaded[key]["evidence_sha256"] = _sha256(path)
            if (
                biomod_sha256 is not None
                and loaded[key].get("biomod_sha256") != biomod_sha256
            ):
                loaded[key] = {
                    "status": "stale",
                    "reason": "Sidecar bioMod SHA-256 does not match the selected model.",
                    "evidence_path": str(path.resolve()),
                    "selected_biomod_sha256": biomod_sha256,
                    "sidecar_biomod_sha256": loaded[key].get("biomod_sha256"),
                }
        else:
            loaded[key] = {
                "status": "not_run",
                "reason": f"Expected sidecar not found: {path}",
            }
    return loaded


def _biobuddy_symbolic_profile(segment_id: str) -> dict[str, str]:
    """Return conservative D1-D3 conclusions supported by the active template formula."""

    base = _SIDED_SEGMENT_BASE.get(segment_id, segment_id)
    profiles = {
        "pelvis": {"D1": "documente_non_evalue", "D2": "deviation", "D3": "deviation"},
        "thorax": {"D1": "documente_non_evalue", "D2": "deviation", "D3": "deviation"},
        "head": {
            "D1": "non_applicable",
            "D2": "non_applicable",
            "D3": "non_applicable",
        },
        "thigh": {
            "D1": "documente_non_evalue",
            "D2": "deviation",
            "D3": "documente_non_evalue",
        },
        "shank": {"D1": "documente_non_evalue", "D2": "deviation", "D3": "deviation"},
        "foot": {"D1": "documente_non_evalue", "D2": "deviation", "D3": "deviation"},
        "upper_arm": {
            "D1": "documente_non_evalue",
            "D2": "deviation",
            "D3": "documente_non_evalue",
        },
        "forearm": {"D1": "documente_non_evalue", "D2": "deviation", "D3": "deviation"},
        "hand": {"D1": "documente_non_evalue", "D2": "deviation", "D3": "deviation"},
    }
    return profiles[base]


def assess_biobuddy_template_against_isb(
    template_frames: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare documented template formulas with the selected ISB targets.

    This is a symbolic assessment. It can demonstrate a construction or origin
    mismatch (D2/D3), but it deliberately cannot declare D1 conformity without
    a paired numerical frame evaluation.
    """

    validate_isb_segment_targets(targets)
    result: dict[str, dict[str, Any]] = {}
    for template_name, frame in template_frames.items():
        segment_id = BIOBUDDY_SEGMENT_IDS.get(template_name)
        if segment_id is None:
            continue
        base = _SIDED_SEGMENT_BASE.get(segment_id, segment_id)
        target = targets["segments"][base]
        profile = _biobuddy_symbolic_profile(segment_id)
        citation = target["citation"]
        reference = f"ISB Wu, DOI {citation['doi']}, section {citation['section']}"
        assessed: dict[str, Any] = {
            "target_segment": base,
            "target": target,
            "template_frame": frame,
        }
        for deviation, status in profile.items():
            if status == "non_applicable":
                evidence = [f"{reference}: no selected target applies to this segment."]
            elif status == "deviation":
                subject = (
                    "landmark/axis construction"
                    if deviation == "D2"
                    else "origin construction"
                )
                evidence = [
                    f"{reference}: BioBuddy {subject} differs from the selected target; see serialized template_frame."
                ]
            else:
                evidence = [
                    f"{reference}: direction is documented but requires paired static numerical evaluation."
                ]
            assessed[f"{deviation}_status"] = status
            assessed[f"{deviation}_evidence"] = evidence
        result[segment_id] = assessed
    return result


def parse_biomod_segment_transforms(
    text_or_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Parse parent-local RT matrices from a bioMod without loading biorbd."""

    candidate = Path(text_or_path) if isinstance(text_or_path, Path) else None
    text = (
        candidate.read_text(encoding="utf-8")
        if candidate is not None
        else str(text_or_path)
    )
    parsed: dict[str, dict[str, Any]] = {}
    current_name: str | None = None
    current_parent = "root"
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        parts = stripped.split()
        if parts and parts[0].lower() == "segment" and len(parts) >= 2:
            current_name = parts[1]
            current_parent = "root"
        elif (
            current_name and parts and parts[0].lower() == "parent" and len(parts) >= 2
        ):
            current_parent = parts[1]
        elif current_name and stripped == "RT":
            rows = []
            for offset in range(1, 5):
                values = [float(value) for value in lines[index + offset].split()]
                if len(values) != 4:
                    raise ValueError(f"Invalid RT row for segment {current_name}")
                rows.append(values)
            parsed[current_name] = {
                "parent": current_parent,
                "transform": np.asarray(rows, dtype=float),
            }
            index += 4
        elif stripped.lower() == "endsegment":
            current_name = None
        index += 1
    return parsed


def compare_template_frames_to_biomod(
    template_global_frames: Mapping[str, np.ndarray],
    biomod_text_or_path: str | Path,
    *,
    template_translation_scale_to_biomod: float = 1.0,
    expected_parents: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify bioMod segment names, parents and parent-local RT matrices."""

    biomod = parse_biomod_segment_transforms(biomod_text_or_path)
    segments: dict[str, Any] = {}
    overall = "match"
    expected_names = set(template_global_frames)
    serialized_names = set(biomod)
    missing_segments = sorted(expected_names - serialized_names)
    unexpected_segments = sorted(serialized_names - expected_names)
    parent_mismatches: dict[str, dict[str, str]] = {}
    if expected_parents is not None:
        for name in sorted(expected_names & serialized_names):
            expected_parent = str(expected_parents[name])
            serialized_parent = str(biomod[name]["parent"])
            root_aliases = {"root", "base"}
            parents_match = (
                expected_parent == serialized_parent
                or {
                    expected_parent.lower(),
                    serialized_parent.lower(),
                }
                <= root_aliases
            )
            if not parents_match:
                parent_mismatches[name] = {
                    "expected": expected_parent,
                    "serialized": serialized_parent,
                }
    if missing_segments or unexpected_segments or parent_mismatches:
        overall = "mismatch"
    for name, item in biomod.items():
        if name not in template_global_frames:
            continue
        parent = item["parent"]
        expected_global = np.asarray(template_global_frames[name], dtype=float).copy()
        expected_global[:3, 3] *= template_translation_scale_to_biomod
        if parent.lower() == "root" or parent not in template_global_frames:
            expected_local = expected_global
        else:
            parent_global = np.asarray(
                template_global_frames[parent], dtype=float
            ).copy()
            parent_global[:3, 3] *= template_translation_scale_to_biomod
            expected_local = np.linalg.inv(parent_global) @ expected_global
        actual = np.asarray(item["transform"], dtype=float)
        actual_rotation = actual[:3, :3]
        orthogonality_error = float(
            np.linalg.norm(actual_rotation.T @ actual_rotation - np.eye(3), ord="fro")
        )
        determinant = float(np.linalg.det(actual_rotation))
        rotation_error_deg, _ = _rotation_deviation(
            expected_local[:3, :3].T @ _closest_proper_rotation(actual_rotation)
        )
        origin_error = float(np.linalg.norm(expected_local[:3, 3] - actual[:3, 3]))
        status = (
            "match"
            if rotation_error_deg <= 1e-4 and origin_error <= 1e-6
            else "mismatch"
        )
        if status != "match":
            overall = "mismatch"
        segments[name] = {
            "parent": parent,
            "status": status,
            "rotation_error_deg": rotation_error_deg,
            "origin_error_mm": origin_error * 1000.0,
            "serialized_rotation_orthogonality_error": orthogonality_error,
            "serialized_rotation_determinant": determinant,
        }
    if not segments:
        overall = "unavailable"
    return {
        "status": overall,
        "segments": segments,
        "expected_segment_count": len(expected_names),
        "serialized_segment_count": len(serialized_names),
        "missing_segments": missing_segments,
        "unexpected_segments": unexpected_segments,
        "parent_mismatches": parent_mismatches,
    }


def compare_real_model_to_biomod(model: Any, biomod_path: Path | str) -> dict[str, Any]:
    """Compare a BioBuddy real model with the RT matrices serialized to bioMod."""

    global_frames = {
        str(name): np.asarray(
            model.segment_coordinate_system_in_global(str(name)).rt_matrix,
            dtype=float,
        )
        for name in model.segment_names
    }
    expected_parents = {
        str(name): str(model.segments[str(name)].parent_name)
        for name in model.segment_names
    }
    comparison = compare_template_frames_to_biomod(
        global_frames,
        Path(biomod_path),
        expected_parents=expected_parents,
    )
    comparison["biomod_path"] = str(Path(biomod_path).expanduser().resolve())
    comparison["biomod_sha256"] = _sha256(Path(biomod_path).expanduser())
    comparison["method"] = (
        "BioBuddy real global frames -> parent-local RT -> serialized bioMod RT"
    )
    return comparison


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


def _closest_proper_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a rounded 3x3 matrix onto SO(3) for angular comparison."""

    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def evaluate_static_frame_pair(
    source_frame: np.ndarray, target_frame: np.ndarray
) -> dict[str, Any]:
    """Evaluate one source/ISB static-frame pair in metres and degrees."""

    source = np.asarray(source_frame, dtype=float)
    target = np.asarray(target_frame, dtype=float)
    if source.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("Static source and target frames must both be 4x4")
    source_rotation = source[:3, :3]
    target_rotation = target[:3, :3]
    angular_deviation_deg, rotation_vector_deg = _rotation_deviation(
        _closest_proper_rotation(target_rotation).T
        @ _closest_proper_rotation(source_rotation)
    )
    return {
        "status": "available",
        "angular_deviation_deg": angular_deviation_deg,
        "rotation_vector_deg": rotation_vector_deg,
        "origin_deviation_mm": float(
            np.linalg.norm(source[:3, 3] - target[:3, 3]) * 1000.0
        ),
        "source_determinant": float(np.linalg.det(source_rotation)),
        "target_determinant": float(np.linalg.det(target_rotation)),
        "source_orthogonality_error": float(
            np.linalg.norm(source_rotation.T @ source_rotation - np.eye(3), ord="fro")
        ),
        "target_orthogonality_error": float(
            np.linalg.norm(target_rotation.T @ target_rotation - np.eye(3), ord="fro")
        ),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a null or non-finite anatomical axis")
    return np.asarray(vector, dtype=float) / norm


def _oriented(vector: np.ndarray, sense: np.ndarray) -> np.ndarray:
    value = _unit(vector)
    return value if float(value @ _unit(sense)) >= 0.0 else -value


def _frame(
    origin: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((_unit(x), _unit(y), _unit(z)))
    frame[:3, 3] = np.asarray(origin, dtype=float)
    return frame


def evaluate_biobuddy_static_against_isb(
    model: Any, marker_data: Any
) -> dict[str, Any]:
    """Evaluate constructable Motive 57 static frames against selected ISB targets.

    No anatomical landmark is silently substituted. In particular, TV7 is not
    accepted as T8 and HM2 is not accepted as the third metacarpal.
    """

    del marker_data  # The collapsed model already expresses markers in root coordinates.
    marker_names = [str(name) for name in model.marker_names]
    model_markers = np.asarray(model.markers_in_global(), dtype=float)

    def point(name: str) -> np.ndarray:
        if name not in marker_names:
            raise KeyError(name)
        return model_markers[:3, marker_names.index(name), 0]

    def midpoint(*names: str) -> np.ndarray:
        return np.mean([point(name) for name in names], axis=0)

    source_frames = {
        str(name): np.asarray(
            model.segment_coordinate_system_in_global(str(name)).rt_matrix,
            dtype=float,
        )
        for name in model.segment_names
    }
    anterior = midpoint("LIAS", "RIAS") - midpoint("LIPS", "RIPS")
    right = point("RIAS") - point("LIAS")
    z_pelvis = _oriented(right, right)
    x_pelvis = _oriented(anterior - z_pelvis * float(anterior @ z_pelvis), anterior)
    y_pelvis = _unit(np.cross(z_pelvis, x_pelvis))

    targets: dict[str, np.ndarray] = {}
    unavailable: dict[str, str] = {
        "thorax": "T8 is required by Wu 2005; Motive 57 provides TV7 and no substitution is permitted.",
        "head": "No head target is selected in the Wu 2002/2005 registry.",
        "right_hand": "The selected global-wrist surrogate requires third-metacarpal landmarks; Motive 57 provides HM2.",
        "left_hand": "The selected global-wrist surrogate requires third-metacarpal landmarks; Motive 57 provides HM2.",
        "right_foot": "Wu 2002 defines the calcaneal frame in a documented neutral ankle configuration; P6 Static does not encode that posture as validated metadata.",
        "left_foot": "Wu 2002 defines the calcaneal frame in a documented neutral ankle configuration; P6 Static does not encode that posture as validated metadata.",
        "right_upper_arm": "Wu 2005 option 2 requires elbow flexed 90 degrees in the sagittal plane and fully pronated forearm; this posture is not validated for P6 Static.",
        "left_upper_arm": "Wu 2005 option 2 requires elbow flexed 90 degrees in the sagittal plane and fully pronated forearm; this posture is not validated for P6 Static.",
    }

    # The single model pelvis has a centroid origin. The selected ISB target uses
    # an ipsilateral HJC; use the right HJC and record this choice explicitly.
    targets["pelvis"] = _frame(
        source_frames["RThigh"][:3, 3], x_pelvis, y_pelvis, z_pelvis
    )
    origin_comparison_kinds: dict[str, str] = {
        "pelvis": "different constructions: BioBuddy pelvic centroid vs selected right functional HJC"
    }
    for side, prefix in (("right", "R"), ("left", "L")):
        thigh_name = f"{prefix}Thigh"
        shank_name = f"{prefix}Shank"
        foot_name = f"{prefix}Foot"
        upper_name = f"{prefix}UpperArm"
        forearm_name = f"{prefix}Forearm"

        hjc = source_frames[thigh_name][:3, 3]
        knee = midpoint(f"{prefix}FME", f"{prefix}FLE")
        y = _unit(hjc - knee)
        z = _oriented(point(f"{prefix}FLE") - point(f"{prefix}FME"), right)
        z = _unit(z - y * float(z @ y))
        x = _oriented(np.cross(y, z), anterior)
        z = _unit(np.cross(x, y))
        targets[f"{side}_thigh"] = _frame(hjc, x, y, z)
        origin_comparison_kinds[f"{side}_thigh"] = (
            "identity_by_definition: BioBuddy and selected ISB target both use "
            "the same functional HJC; zero is not independent anatomical validation"
        )

        medial_ankle = point(f"{prefix}TAM")
        lateral_ankle = point(f"{prefix}FAL")
        ankle = (medial_ankle + lateral_ankle) / 2.0
        z = _oriented(lateral_ankle - medial_ankle, right)
        intercondylar = knee
        long_axis = intercondylar - ankle
        x = _oriented(np.cross(long_axis, z), anterior)
        y = _unit(np.cross(z, x))
        z = _unit(np.cross(x, y))
        shank_target = _frame(ankle, x, y, z)
        targets[f"{side}_shank"] = shank_target
        origin_comparison_kinds[f"{side}_shank"] = (
            "different constructions: BioBuddy knee centre vs ISB inter-malleolar midpoint"
        )
        shoulder = source_frames[upper_name][:3, 3]
        elbow = midpoint(f"{prefix}HME", f"{prefix}HLE")
        y = _unit(shoulder - elbow)
        us = point(f"{prefix}USP")
        rs = point(f"{prefix}RSP")
        forearm_y = _unit(elbow - us)
        z = _oriented(np.cross(y, forearm_y), right)
        x = _oriented(np.cross(y, z), anterior)
        z = _unit(np.cross(x, y))
        # Keep the option-2 construction explicit, but do not publish a numeric
        # comparison unless its prescribed calibration posture is validated.
        _ = _frame(shoulder, x, y, z)

        y = _unit(elbow - us)
        x = _oriented(np.cross(y, rs - us), anterior)
        z = _unit(np.cross(x, y))
        targets[f"{side}_forearm"] = _frame(us, x, y, z)
        origin_comparison_kinds[f"{side}_forearm"] = (
            "different constructions: BioBuddy elbow midpoint vs ISB ulnar styloid"
        )

    source_name_by_id = {value: key for key, value in BIOBUDDY_SEGMENT_IDS.items()}
    rows: dict[str, Any] = {}
    for segment_id in BIOBUDDY_SEGMENT_IDS.values():
        source_name = source_name_by_id[segment_id]
        if segment_id in unavailable:
            rows[segment_id] = {
                "status": "unavailable",
                "reason": unavailable[segment_id],
            }
        elif segment_id not in targets or source_name not in source_frames:
            rows[segment_id] = {
                "status": "unavailable",
                "reason": "No justified paired source/ISB frame was constructed.",
            }
        else:
            rows[segment_id] = evaluate_static_frame_pair(
                source_frames[source_name], targets[segment_id]
            )
            rows[segment_id]["source_segment_name"] = source_name
            rows[segment_id]["origin_comparison_kind"] = origin_comparison_kinds[
                segment_id
            ]
    return {
        "status": "available",
        "unit": "m",
        "coordinate_frame": "BioBuddy root (Pelvis) coordinates",
        "pelvis_target_origin_choice": "right functional HJC",
        "segments": rows,
    }


def _status_evidence(value: Mapping[str, Any]) -> tuple[str, list[str]]:
    return str(value["status"]), [str(item) for item in value.get("evidence", [])]


def build_isb_d1_d3_audit(
    registry: Mapping[str, Any],
    *,
    biobuddy_template_loader: Callable[[], tuple[Any, Path]] | None = None,
    isb_targets_loader: Callable[[], dict[str, Any]] | None = None,
    biomod_verification: Mapping[str, Any] | None = None,
    static_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an audit report from the registry and the active BioBuddy runtime."""

    validate_kinematic_conventions(registry)
    loader = biobuddy_template_loader or _load_installed_biobuddy_template
    targets_loader = isb_targets_loader or load_isb_segment_targets
    targets = targets_loader()
    validate_isb_segment_targets(targets)
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
    symbolic_assessment = assess_biobuddy_template_against_isb(template_frames, targets)
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
                symbolic = symbolic_assessment.get(segment_id)
                if source_id == "biobuddy_motive57" and symbolic is not None:
                    status = str(symbolic[f"{deviation}_status"])
                    evidence = [str(item) for item in symbolic[f"{deviation}_evidence"]]
                    if template_sha256:
                        evidence.append(
                            "Runtime BioBuddy template serialized from SHA-256 "
                            f"{template_sha256}."
                        )
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
            static_result = (
                static_evaluation.get("segments", {}).get(segment_id)
                if source_id == "biobuddy_motive57" and static_evaluation
                else None
            )
            if static_result and static_result.get("status") == "available":
                row.update(
                    {
                        "numeric_status": "available_from_static_landmarks",
                        "angular_deviation_deg": static_result["angular_deviation_deg"],
                        "rotation_vector_deg": static_result["rotation_vector_deg"],
                        "origin_deviation_mm": static_result["origin_deviation_mm"],
                        "static_frame_quality": {
                            key: static_result[key]
                            for key in (
                                "source_determinant",
                                "target_determinant",
                                "source_orthogonality_error",
                                "target_orthogonality_error",
                            )
                        },
                    }
                )
            elif static_result:
                row.update(
                    {
                        "numeric_status": "unavailable_from_static_landmarks",
                        "numeric_reason": static_result.get("reason"),
                        "angular_deviation_deg": None,
                        "rotation_vector_deg": None,
                    }
                )
            elif correction is None:
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
        "isb_targets": {
            "status": "available",
            "registry_id": targets["registry_id"],
            "schema_version": targets["schema_version"],
            "path": str(DEFAULT_ISB_TARGETS_PATH.resolve()),
            "sha256": _sha256(DEFAULT_ISB_TARGETS_PATH),
            "segments": targets["segments"],
        },
        "biomod_verification": dict(biomod_verification or {"status": "not_run"}),
        "static_evaluation": dict(static_evaluation or {"status": "not_run"}),
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
