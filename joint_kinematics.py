"""Matrix-first joint kinematics and explicit ISB D4-D6 contracts.

The module uses active rotations, column vectors and intrinsic Euler/Cardan
sequences.  Segment orientations are ``R_lab_segment`` matrices whose columns
are local segment axes expressed in the laboratory frame.  A source frame is
converted to its anatomical frame by right multiplication with a constant
``C_source_to_anatomical`` matrix.  Joint rotation is then computed as::

    R_proximal_distal = (R_lab_proximal @ C_proximal).T
                        @ (R_lab_distal @ C_distal)

Euler components are a presentation derived from this matrix.  The matrix,
quaternion/geodesic interpretation and singularity flags remain authoritative.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_rotations import (
    canonicalize_rotation_series,
    relative_rotation_series,
    rotation_geodesic_degrees,
    validate_rotation_matrix,
    validate_rotation_series,
)

DEFAULT_ISB_JOINT_KINEMATICS_PATH = Path(__file__).with_name(
    "isb_joint_kinematics.json"
)


class JointKinematicsError(ValueError):
    """Raised when a joint convention or kinematic input is ambiguous."""


def _validate_joint_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != 1:
        raise JointKinematicsError("schema_version must be 1")
    if registry.get("matrix_convention") != "column_vectors":
        raise JointKinematicsError("matrix_convention must be column_vectors")
    if not isinstance(registry.get("zero_definition"), str):
        raise JointKinematicsError("zero_definition is required")
    articulations = registry.get("articulations")
    if not isinstance(articulations, Mapping) or not articulations:
        raise JointKinematicsError("articulations must be a non-empty mapping")
    for name, raw in articulations.items():
        context = f"articulations.{name}"
        if not isinstance(raw, Mapping):
            raise JointKinematicsError(f"{context} must be a mapping")
        sequence = str(raw.get("sequence", "")).upper()
        if len(sequence) != 3 or any(axis not in "XYZ" for axis in sequence):
            raise JointKinematicsError(f"{context}.sequence is invalid")
        if raw.get("type") not in {"Cardan", "Euler", "JCS"}:
            raise JointKinematicsError(f"{context}.type is invalid")
        for key in ("proximal", "distal"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise JointKinematicsError(f"{context}.{key} is required")
        for key in ("components", "positive_signs"):
            if not isinstance(raw.get(key), list) or len(raw[key]) != 3:
                raise JointKinematicsError(f"{context}.{key} must contain 3 values")
        laterality = raw.get("laterality")
        if not isinstance(laterality, Mapping):
            raise JointKinematicsError(f"{context}.laterality must be a mapping")
        for side in ("right", "left"):
            side_data = laterality.get(side)
            if not isinstance(side_data, Mapping):
                raise JointKinematicsError(f"{context}.laterality.{side} is required")
            signs = side_data.get("component_signs")
            if not isinstance(signs, list) or len(signs) != 3:
                raise JointKinematicsError(
                    f"{context}.laterality.{side}.component_signs must contain 3 values"
                )
            if any(sign not in (-1, 1) for sign in signs):
                raise JointKinematicsError(
                    f"{context}.laterality.{side}.component_signs must be +/-1"
                )
            if not isinstance(side_data.get("preprocessing"), str):
                raise JointKinematicsError(
                    f"{context}.laterality.{side}.preprocessing is required"
                )
        citation = raw.get("citation")
        if not isinstance(citation, Mapping) or not citation.get("doi"):
            raise JointKinematicsError(f"{context}.citation.doi is required")


def load_isb_joint_kinematics(
    path: Path | str = DEFAULT_ISB_JOINT_KINEMATICS_PATH,
) -> dict[str, Any]:
    """Load and validate the versioned D4-D6 target registry."""

    registry = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    _validate_joint_registry(registry)
    return registry


def apply_anatomical_frame_correction(
    source_rotations: np.ndarray, anatomical_from_source: np.ndarray
) -> np.ndarray:
    """Convert source segment axes to anatomical axes by right multiplication."""

    source, _ = canonicalize_rotation_series(
        source_rotations, context="source segment rotations"
    )
    correction = validate_rotation_matrix(
        anatomical_from_source, context="anatomical_from_source"
    )
    corrected = np.einsum("ijf,jk->ikf", source, correction)
    return validate_rotation_series(corrected, context="anatomical segment rotations")


def joint_rotation_series(
    proximal_rotations: np.ndarray,
    distal_rotations: np.ndarray,
    *,
    proximal_anatomical_from_source: np.ndarray | None = None,
    distal_anatomical_from_source: np.ndarray | None = None,
) -> np.ndarray:
    """Return corrected parent-to-child rotations for all frames."""

    identity = np.eye(3)
    proximal = apply_anatomical_frame_correction(
        proximal_rotations,
        (
            identity
            if proximal_anatomical_from_source is None
            else proximal_anatomical_from_source
        ),
    )
    distal = apply_anatomical_frame_correction(
        distal_rotations,
        (
            identity
            if distal_anatomical_from_source is None
            else distal_anatomical_from_source
        ),
    )
    return relative_rotation_series(proximal, distal)


def euler_matrix(angles_rad: np.ndarray, sequence: str) -> np.ndarray:
    """Compose one active intrinsic Euler/Cardan rotation matrix."""

    sequence = str(sequence).upper()
    angles = np.asarray(angles_rad, dtype=float).reshape(-1)
    if len(sequence) != 3 or any(axis not in "XYZ" for axis in sequence):
        raise JointKinematicsError(f"Unsupported Euler sequence: {sequence!r}")
    if angles.size != 3 or not np.all(np.isfinite(angles)):
        raise JointKinematicsError("angles_rad must contain three finite values")
    return Rotation.from_euler(sequence, angles).as_matrix()


def _singularity_flags(
    middle_angles: np.ndarray,
    sequence: str,
    near_singularity_margin_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    margin = np.deg2rad(float(near_singularity_margin_deg))
    if not np.isfinite(margin) or margin < 0.0:
        raise JointKinematicsError("near_singularity_margin_deg must be non-negative")
    proper_euler = sequence[0] == sequence[2]
    distance = (
        np.abs(np.sin(middle_angles)) if proper_euler else np.abs(np.cos(middle_angles))
    )
    singular = distance <= 1e-8
    near = distance <= np.sin(margin)
    return singular, near


def extract_euler_series(
    rotations: np.ndarray,
    sequence: str,
    *,
    unwrap: bool = True,
    near_singularity_margin_deg: float = 1.0,
) -> dict[str, Any]:
    """Extract Euler components with singularity and matrix-roundtrip evidence."""

    sequence = str(sequence).upper()
    if len(sequence) != 3 or any(axis not in "XYZ" for axis in sequence):
        raise JointKinematicsError(f"Unsupported Euler sequence: {sequence!r}")
    values, quality = canonicalize_rotation_series(
        rotations, context=f"{sequence} Euler input"
    )
    frame_first = np.moveaxis(values, 2, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        angles = Rotation.from_matrix(frame_first).as_euler(sequence)
    singular, near_singular = _singularity_flags(
        angles[:, 1], sequence, near_singularity_margin_deg
    )
    if unwrap:
        angles = np.unwrap(angles, axis=0)
    reconstructed = Rotation.from_euler(sequence, angles).as_matrix()
    roundtrip = np.asarray(
        [
            rotation_geodesic_degrees(reference, test)
            for reference, test in zip(frame_first, reconstructed, strict=True)
        ],
        dtype=float,
    )
    return {
        "sequence": sequence,
        "convention": "active_intrinsic_column_vectors",
        "angles_rad": angles.T,
        "singular": singular,
        "near_singular": near_singular,
        "euler_components_reliable": np.repeat((~near_singular)[None, :], 3, axis=0),
        "branch_policy": "scipy_canonical_branch_then_independent_component_unwrap",
        "roundtrip_geodesic_deg": roundtrip,
        "max_roundtrip_geodesic_deg": float(np.max(roundtrip)),
        "rotation_quality": quality,
    }


def extract_jcs_series(
    rotations: np.ndarray,
    sequence: str,
    *,
    unwrap: bool = True,
    near_singularity_margin_deg: float = 1.0,
) -> dict[str, Any]:
    """Extract a Grood-Suntay linkage and record its three physical axes.

    For a sequence ``ABC``, axis A is fixed in the proximal frame, axis C is
    fixed in the distal frame and expressed proximally, and the floating axis
    is perpendicular to both. This is the JCS linkage represented by the same
    intrinsic three-axis decomposition.
    """

    sequence = str(sequence).upper()
    if len(sequence) != 3 or len(set(sequence)) != 3:
        raise JointKinematicsError("JCS requires a three-axis Cardan sequence")
    values = validate_rotation_series(rotations, context=f"{sequence} JCS input")
    euler = extract_euler_series(
        values,
        sequence,
        unwrap=unwrap,
        near_singularity_margin_deg=near_singularity_margin_deg,
    )
    proximal_axis = np.eye(3)[:, "XYZ".index(sequence[0])]
    distal_local_axis = np.eye(3)[:, "XYZ".index(sequence[2])]
    fixed_proximal = np.repeat(proximal_axis[:, None], values.shape[2], axis=1)
    fixed_distal = np.einsum("ijf,j->if", values, distal_local_axis)
    floating = np.cross(fixed_distal.T, fixed_proximal.T).T
    norms = np.linalg.norm(floating, axis=0)
    valid = norms > 1e-12
    floating[:, valid] /= norms[valid]
    floating[:, ~valid] = np.nan
    euler.update(
        {
            "fixed_proximal_axis": fixed_proximal,
            "floating_axis": floating,
            "fixed_distal_axis": fixed_distal,
            "jcs_axis_order": {
                "first": f"proximal_{sequence[0]}",
                "second": "floating_axis",
                "third": f"distal_{sequence[2]}",
            },
        }
    )
    return euler


def express_joint_translation_in_proximal(
    proximal_rotations: np.ndarray,
    proximal_common_point_lab: np.ndarray,
    distal_common_point_lab: np.ndarray,
) -> np.ndarray:
    """Express distal-minus-proximal common-point translation in proximal axes."""

    proximal, _ = canonicalize_rotation_series(
        proximal_rotations, context="D5 proximal rotations"
    )
    proximal_point = np.asarray(proximal_common_point_lab, dtype=float)
    distal_point = np.asarray(distal_common_point_lab, dtype=float)
    expected_shape = (3, proximal.shape[2])
    if proximal_point.shape != expected_shape or distal_point.shape != expected_shape:
        raise JointKinematicsError(
            "D5 common points must have shape (3, n_frames) matching rotations"
        )
    if not np.all(np.isfinite(proximal_point)) or not np.all(np.isfinite(distal_point)):
        raise JointKinematicsError("D5 common points must be finite")
    displacement_lab = distal_point - proximal_point
    return np.einsum("jif,jf->if", proximal, displacement_lab)


def classify_shoulder_motion(
    proximal_segment: str, distal_segment: str, available_segments: set[str]
) -> dict[str, str]:
    """Distinguish glenohumeral from thoracohumeral shoulder motion."""

    proximal = str(proximal_segment).lower()
    distal = str(distal_segment).lower()
    available = {str(name).lower() for name in available_segments}
    if distal not in {"upper_arm", "humerus"}:
        raise JointKinematicsError("shoulder distal segment must be upper_arm")
    if proximal == "scapula":
        if "scapula" not in available:
            raise JointKinematicsError(
                "glenohumeral motion is unavailable without a scapula segment"
            )
        return {"motion": "glenohumeral", "D6_status": "conforme_target"}
    if proximal == "thorax":
        return {"motion": "thoracohumeral", "D6_status": "deviation"}
    raise JointKinematicsError(f"unsupported shoulder proximal segment: {proximal}")


def frame_corrections_from_static_audit(
    static_audit: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Load only numerically available source-to-target frame corrections."""

    corrections: dict[str, np.ndarray] = {}
    for row in static_audit.get("segments", {}).values():
        if row.get("status") != "available":
            continue
        name = row.get("source_segment_name")
        value = row.get("source_to_target_rotation")
        if not name or value is None:
            continue
        corrections[str(name)] = validate_rotation_matrix(
            np.asarray(value, dtype=float), context=f"static correction {name}"
        )
    return corrections


def _target_joint_name(articulation_id: str) -> str:
    name = str(articulation_id).lower()
    if "shoulder" in name:
        return "thoracohumeral"
    for candidate in ("ankle", "hip", "knee", "elbow", "wrist", "spine"):
        if candidate in name:
            return candidate
    if name == "neck":
        return "spine"
    raise JointKinematicsError(
        f"no D4-D6 target is registered for articulation {articulation_id!r}"
    )


def audit_joint_kinematics_source(
    source_id: str,
    rotations: Mapping[str, np.ndarray],
    time: np.ndarray,
    articulations: Mapping[str, Mapping[str, str]],
    anatomical_from_source: Mapping[str, np.ndarray],
    *,
    target_registry: Mapping[str, Any] | None = None,
    frame_correction_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit matrix-first joint kinematics for one model source.

    Relative native SO(3) series remain available when segment-to-anatomical
    corrections are missing. Euler components are withheld in that case.
    """

    registry = (
        dict(target_registry)
        if target_registry is not None
        else load_isb_joint_kinematics()
    )
    _validate_joint_registry(registry)
    time_values = np.asarray(time, dtype=float).reshape(-1)
    if not np.all(np.isfinite(time_values)):
        raise JointKinematicsError("time must contain finite values")
    metadata: dict[str, Any] = {}
    timeseries: dict[str, dict[str, np.ndarray]] = {}
    for articulation_id, endpoints in articulations.items():
        proximal = str(endpoints["proximal"])
        distal = str(endpoints["distal"])
        proximal_segment_id = endpoints.get("proximal_segment_id")
        distal_segment_id = endpoints.get("distal_segment_id")
        if proximal not in rotations or distal not in rotations:
            metadata[str(articulation_id)] = {
                "proximal": proximal,
                "distal": distal,
                "proximal_segment_id": proximal_segment_id,
                "distal_segment_id": distal_segment_id,
                "status": "unavailable_missing_segment",
                "missing_segments": sorted(
                    name for name in (proximal, distal) if name not in rotations
                ),
                "euler_status": "unavailable",
            }
            continue
        native_relative = joint_rotation_series(rotations[proximal], rotations[distal])
        if native_relative.shape[2] != time_values.size:
            raise JointKinematicsError(
                f"time length does not match articulation {articulation_id}"
            )
        target_id = _target_joint_name(str(articulation_id))
        target = registry["articulations"][target_id]
        laterality = (
            "left"
            if str(articulation_id).lower().startswith("left_")
            else (
                "right"
                if str(articulation_id).lower().startswith("right_")
                else "right"
            )
        )
        laterality_contract = target["laterality"][laterality]
        row: dict[str, Any] = {
            "proximal": proximal,
            "distal": distal,
            "proximal_segment_id": proximal_segment_id,
            "distal_segment_id": distal_segment_id,
            "target": target_id,
            "type": target["type"],
            "sequence": target["sequence"],
            "components": target["components"],
            "positive_signs": target["positive_signs"],
            "laterality": laterality,
            "component_signs": laterality_contract["component_signs"],
            "laterality_preprocessing": laterality_contract["preprocessing"],
            "unit": "rad",
            "zero_definition": registry["zero_definition"],
            "matrix_status": "available_native_source_frames",
            "euler_status": "unavailable",
            "D4_status": "blocked_missing_frame_corrections",
            "D5_status": "non_applicable_no_joint_translation",
            "D6_status": target["D6_status"],
            "citation": target["citation"],
        }
        if target_id == "thoracohumeral":
            row.update(
                classify_shoulder_motion("thorax", "upper_arm", {proximal, distal})
            )
        timeseries[str(articulation_id)] = {
            "time": time_values.copy(),
            "relative_rotation_matrix": native_relative,
            "quaternion_xyzw": Rotation.from_matrix(np.moveaxis(native_relative, 2, 0))
            .as_quat()
            .T,
            "rotation_magnitude_deg": np.rad2deg(
                Rotation.from_matrix(np.moveaxis(native_relative, 2, 0)).magnitude()
            ),
        }
        if proximal in anatomical_from_source and distal in anatomical_from_source:
            corrected = joint_rotation_series(
                rotations[proximal],
                rotations[distal],
                proximal_anatomical_from_source=anatomical_from_source[proximal],
                distal_anatomical_from_source=anatomical_from_source[distal],
            )
            extractor = (
                extract_jcs_series if target["type"] == "JCS" else extract_euler_series
            )
            euler = extractor(corrected, target["sequence"])
            component_signs = np.asarray(
                laterality_contract["component_signs"], dtype=float
            )[:, None]
            published_angles = euler["angles_rad"] * component_signs
            row.update(
                {
                    "matrix_status": "available_anatomical_target_frames",
                    "euler_status": "available",
                    "D4_status": (
                        "jcs_target_evaluated"
                        if target["type"] == "JCS"
                        else "euler_target_evaluated"
                    ),
                    "proximal_frame_correction": np.asarray(
                        anatomical_from_source[proximal], dtype=float
                    ).tolist(),
                    "distal_frame_correction": np.asarray(
                        anatomical_from_source[distal], dtype=float
                    ).tolist(),
                    "frame_correction_provenance": dict(
                        frame_correction_provenance or {}
                    ),
                    "laterality_preprocessing_status": (
                        "encoded_in_source_to_target_frame_correction"
                    ),
                    "euler_branch_policy": euler["branch_policy"],
                    "singular_frame_count": int(np.sum(euler["singular"])),
                    "near_singular_frame_count": int(np.sum(euler["near_singular"])),
                    "max_roundtrip_geodesic_deg": euler["max_roundtrip_geodesic_deg"],
                }
            )
            timeseries[str(articulation_id)].update(
                {
                    "relative_rotation_matrix": corrected,
                    "quaternion_xyzw": Rotation.from_matrix(
                        np.moveaxis(corrected, 2, 0)
                    )
                    .as_quat()
                    .T,
                    "rotation_magnitude_deg": np.rad2deg(
                        Rotation.from_matrix(np.moveaxis(corrected, 2, 0)).magnitude()
                    ),
                    "angles_rad": published_angles,
                    "singular": euler["singular"],
                    "near_singular": euler["near_singular"],
                    "euler_components_reliable": euler["euler_components_reliable"],
                    "roundtrip_geodesic_deg": euler["roundtrip_geodesic_deg"],
                }
            )
            if target["type"] == "JCS":
                timeseries[str(articulation_id)].update(
                    {
                        "fixed_proximal_axis": euler["fixed_proximal_axis"],
                        "floating_axis": euler["floating_axis"],
                        "fixed_distal_axis": euler["fixed_distal_axis"],
                    }
                )
        metadata[str(articulation_id)] = row
    return {
        "source_id": str(source_id),
        "matrix_convention": registry["matrix_convention"],
        "articulations": metadata,
        "timeseries": timeseries,
    }


def _json_without_timeseries(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _json_without_timeseries(item)
            for key, item in value.items()
            if key != "timeseries"
        }
    if isinstance(value, (list, tuple)):
        return [_json_without_timeseries(item) for item in value]
    return value


def write_joint_kinematics_audit(
    output_dir: Path | str, audits_by_source: Mapping[str, Mapping[str, Any]]
) -> dict[str, Path]:
    """Write D4-D6 metadata as JSON and numeric time series as compressed NPZ."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "joint_kinematics_d4_d6.json"
    npz_path = destination / "joint_kinematics_d4_d6.npz"
    payload = {
        "schema_version": 1,
        "sources": {
            str(source): _json_without_timeseries(audit)
            for source, audit in audits_by_source.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    for source, audit in audits_by_source.items():
        for articulation, values in audit.get("timeseries", {}).items():
            for field, array in values.items():
                arrays[f"{source}/{articulation}/{field}"] = np.asarray(array)
    np.savez_compressed(npz_path, **arrays)
    return {"json": json_path, "timeseries": npz_path}
