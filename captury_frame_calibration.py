"""Captury-only aliases and static segment-frame calibration helpers.

The functions in this module deliberately accept only Captury ``Q_*`` C3D
points, Captury model joint centres and Captury segment rotations.  Motive
markers, Motive models and inter-system rigid transforms are intentionally not
part of this API: those belong to the later *comparison* stage.

The generated frames are anatomical surrogates, not an automatic declaration
of ISB compliance.  The pelvis uses the accepted ``Q_Wa`` ASIS/PSIS aliases.
Each thigh uses its Captury model hip and knee centres for its long axis and
the accepted knee-marker pair for its transverse direction.  Other segments
remain explicitly uncalibrated until a validated Captury landmark definition
is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mocap_labels import marker_display_labels

DEFAULT_CAPTURY_Q_LANDMARKS_PATH = Path(__file__).with_name("captury_q_landmarks.json")

# A global relabelling only. It maps (X, Y, Z) local Captury axes to
# (Z, Y, -X), i.e. R_y(+90 deg). It is intentionally not a per-segment
# anatomical calibration.
CAPTURY_SIMPLE_AXIS_RENAME_MATRIX = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float
)


def is_static_frame_calibration_trial(trial_name: str, static_trial: str) -> bool:
    """Return whether a Captury frame fit is authorized for this trial.

    Segment matrices are participant/static calibration quantities. A dynamic
    trial may only reuse a persisted static fit; it must never silently become
    the source of a new one when ``compare_trial`` is used outside ``main``.
    """

    return str(trial_name).strip() == str(static_trial).strip()


def load_captury_q_landmark_config(
    path: Path | str = DEFAULT_CAPTURY_Q_LANDMARKS_PATH,
) -> dict[str, Any]:
    """Load the versioned Captury Q-landmark alias configuration."""

    config_path = Path(path).expanduser()
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if values.get("schema_version") != 1:
        raise ValueError("Unsupported Captury Q-landmark configuration schema")
    markers = values.get("markers")
    if not isinstance(markers, Mapping):
        raise ValueError("Captury Q-landmark configuration must define markers")
    return values


def _marker_config(config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return (config or load_captury_q_landmark_config()).get("markers", {})


def captury_display_labels(
    labels: list[str], config: Mapping[str, Any] | None = None
) -> list[str]:
    """Return display aliases while preserving raw label order and indices.

    The raw C3D labels are never changed.  Each configured alias retains its
    unique occurrence key, e.g. ``CAP_RASIS (Q_Wa#1)``, so an operator can
    validate the accepted correspondence visually in the GUI.
    """

    numbered = marker_display_labels(labels)
    configured = _marker_config(config)
    result: list[str] = []
    for label in numbered:
        entry = configured.get(label, {})
        display = str(entry.get("display", "")).strip()
        result.append(f"{display} ({label})" if display else label)
    return result


def captury_display_to_numbered_labels(
    labels: list[str], config: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Map GUI display aliases back to stable raw occurrence labels."""

    return dict(
        zip(
            captury_display_labels(labels, config),
            marker_display_labels(labels),
            strict=True,
        )
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Degenerate Captury landmark direction")
    return np.asarray(vector, dtype=float) / norm


def _frame_from_y_z(y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """Build a right-handed frame from its long ``Y`` and transverse ``Z`` axes."""

    y_axis = _unit(y_axis)
    z_axis = _unit(z_axis - y_axis * np.dot(y_axis, z_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    left, _singular, right_t = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _rotation_error_deg(reference: np.ndarray, value: np.ndarray) -> float:
    relative = np.asarray(reference, dtype=float).T @ np.asarray(value, dtype=float)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _role_indices(labels: list[str], config: Mapping[str, Any]) -> dict[str, int]:
    numbered = marker_display_labels(labels)
    positions = {label: index for index, label in enumerate(numbered)}
    roles: dict[str, int] = {}
    for numbered_label, entry in _marker_config(config).items():
        role = str(entry.get("role", "")).strip()
        if role and numbered_label in positions:
            roles[role] = positions[numbered_label]
    return roles


def _finite_frame_indices(*values: np.ndarray) -> np.ndarray:
    """Return finite indices shared by arrays that may have different lengths."""

    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays or any(array.ndim != 2 or array.shape[0] != 3 for array in arrays):
        raise ValueError("Each landmark trajectory must have shape (3, frames)")
    n_frames = min(array.shape[1] for array in arrays)
    valid = np.ones(n_frames, dtype=bool)
    for array in arrays:
        valid &= np.all(np.isfinite(array[:, :n_frames]), axis=0)
    return np.flatnonzero(valid)


def captury_static_target_frames(
    labels: list[str],
    points_mm: np.ndarray,
    joint_centres_mm: Mapping[str, np.ndarray],
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive pelvis and thigh target frames from one Captury static C3D.

    The femoral local ``Y`` is explicitly oriented hip -> knee, as requested
    for this project.  It uses the model centres ``LeftUpLeg``/``LeftLeg`` and
    ``RightUpLeg``/``RightLeg``; the Q knee pair only supplies the transverse
    axis.  This function returns no frame for segments without validated Q
    roles, rather than extrapolating a frame from Motive data or from a global
    axis name.
    """

    values = np.asarray(points_mm, dtype=float)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("points_mm must have shape (3, markers, frames)")
    loaded_config = config or load_captury_q_landmark_config()
    roles = _role_indices(labels, loaded_config)
    required_pelvis = {"right_asis", "left_asis", "psis_mid"}
    missing_pelvis = sorted(required_pelvis - set(roles))
    report: dict[str, Any] = {
        "status": "ok",
        "scope": "Captury-only Q_* markers plus Captury model joint centres",
        "target_frame": {
            "pelvis": "X anterior surrogate, Y cranial, Z right from Q_Wa",
            "thigh": "Y hip_to_knee model-centre direction; Z knee-marker transverse direction",
        },
        "segments": {},
    }
    if missing_pelvis:
        report.update(
            {
                "status": "missing_pelvis_roles",
                "missing_roles": missing_pelvis,
            }
        )
        return {}, report
    right_asis = values[:, roles["right_asis"], :]
    left_asis = values[:, roles["left_asis"], :]
    psis_mid = values[:, roles["psis_mid"], :]
    pelvis_frames = np.full((3, 3, values.shape[2]), np.nan, dtype=float)
    pelvis_indices = _finite_frame_indices(right_asis, left_asis, psis_mid)
    for frame in pelvis_indices:
        right_axis = _unit(right_asis[:, frame] - left_asis[:, frame])
        anterior = (right_asis[:, frame] + left_asis[:, frame]) / 2.0 - psis_mid[
            :, frame
        ]
        anterior = _unit(anterior - right_axis * np.dot(anterior, right_axis))
        cranial = _unit(np.cross(right_axis, anterior))
        pelvis_frames[:, :, frame] = np.column_stack((anterior, cranial, right_axis))
    if not np.any(np.isfinite(pelvis_frames)):
        report.update({"status": "no_finite_pelvis_frames"})
        return {}, report
    result: dict[str, np.ndarray] = {"Hips": pelvis_frames}
    report["segments"]["Hips"] = {
        "frame_indices": pelvis_indices.tolist(),
        "roles": ["right_asis", "left_asis", "psis_mid"],
        "status": "ok",
    }
    for side, upper_leg, lower_leg, medial_role, lateral_role in (
        ("Left", "LeftUpLeg", "LeftLeg", "left_knee_medial", "left_knee_lateral"),
        ("Right", "RightUpLeg", "RightLeg", "right_knee_medial", "right_knee_lateral"),
    ):
        missing = [role for role in (medial_role, lateral_role) if role not in roles]
        if upper_leg not in joint_centres_mm or lower_leg not in joint_centres_mm:
            missing.extend(
                centre
                for centre in (upper_leg, lower_leg)
                if centre not in joint_centres_mm
            )
        if missing:
            report["segments"][upper_leg] = {
                "status": "missing_inputs",
                "missing": missing,
            }
            continue
        hip = np.asarray(joint_centres_mm[upper_leg], dtype=float)
        knee = np.asarray(joint_centres_mm[lower_leg], dtype=float)
        medial = values[:, roles[medial_role], :]
        lateral = values[:, roles[lateral_role], :]
        indices = _finite_frame_indices(hip, knee, medial, lateral)
        n_frames = min(values.shape[2], hip.shape[1], knee.shape[1])
        frames = np.full((3, 3, n_frames), np.nan, dtype=float)
        kept_indices: list[int] = []
        for frame in indices:
            try:
                frames[:, :, frame] = _frame_from_y_z(
                    knee[:, frame] - hip[:, frame],
                    lateral[:, frame] - medial[:, frame],
                )
                kept_indices.append(int(frame))
            except ValueError:
                continue
        if not np.any(np.isfinite(frames)):
            report["segments"][upper_leg] = {"status": "degenerate_landmarks"}
            continue
        result[upper_leg] = frames
        report["segments"][upper_leg] = {
            "frame_indices": kept_indices,
            "roles": [medial_role, lateral_role],
            "centres": [upper_leg, lower_leg],
            "status": "ok",
        }
    return result, report


def fit_captury_segment_frame_corrections(
    rotations: Mapping[str, np.ndarray],
    joint_centres_mm: Mapping[str, np.ndarray],
    labels: list[str],
    points_mm: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit static source-to-target local matrices for Captury segments.

    Each per-frame candidate is ``R_source.T @ R_target``.  A robust SO(3)
    mean excludes large static outliers; the resulting constant matrix is
    suitable for every dynamic trial of the same participant and source model.
    """

    targets, target_report = captury_static_target_frames(
        labels, points_mm, joint_centres_mm, config
    )
    report: dict[str, Any] = {
        "status": target_report["status"],
        "method": "captury_static_q_landmarks_and_own_model_centres",
        "matrix_convention": "R_target = R_source @ C_source_to_target",
        "target_frames": target_report,
        "segments": {},
    }
    corrections: dict[str, np.ndarray] = {}
    for segment, target in targets.items():
        source = rotations.get(segment)
        if source is None:
            report["segments"][segment] = {"status": "missing_source_rotation"}
            continue
        source_array = np.asarray(source, dtype=float)
        n_frames = min(source_array.shape[2], target.shape[2])
        if source_array.shape[:2] != (3, 3) or n_frames == 0:
            report["segments"][segment] = {"status": "invalid_source_rotation"}
            continue
        candidates = [
            source_array[:, :, frame].T @ target[:, :, frame]
            for frame in range(n_frames)
            if np.all(np.isfinite(source_array[:, :, frame]))
            and np.all(np.isfinite(target[:, :, frame]))
        ]
        if not candidates:
            report["segments"][segment] = {"status": "no_finite_frame_pair"}
            continue
        initial = _project_to_so3(np.sum(candidates, axis=0))
        errors = np.asarray(
            [_rotation_error_deg(initial, value) for value in candidates]
        )
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        threshold = median + max(1.0, 3.0 * 1.4826 * mad)
        inliers = errors <= threshold
        correction = _project_to_so3(
            np.sum(
                [
                    value
                    for value, accepted in zip(candidates, inliers, strict=True)
                    if accepted
                ],
                axis=0,
            )
        )
        final_errors = np.asarray(
            [_rotation_error_deg(correction, value) for value in candidates]
        )
        corrections[segment] = correction
        report["segments"][segment] = {
            "status": "ok",
            "matrix": correction.tolist(),
            "n_frames": int(len(candidates)),
            "n_inlier_frames": int(np.sum(inliers)),
            "outlier_threshold_deg": float(threshold),
            "median_residual_deg": float(np.median(final_errors)),
            "p95_residual_deg": float(np.percentile(final_errors, 95.0)),
            "max_residual_deg": float(np.max(final_errors)),
        }
    if not corrections and report["status"] == "ok":
        report["status"] = "no_calibrated_segments"
    return corrections, report


def apply_segment_frame_corrections(
    rotations: Mapping[str, np.ndarray], corrections: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Right-multiply each available local segment correction matrix."""

    return {
        name: (
            np.einsum("ijf,jk->ikf", values, np.asarray(corrections[name], dtype=float))
            if name in corrections
            else np.asarray(values, dtype=float).copy()
        )
        for name, values in rotations.items()
    }


def apply_simple_captury_axis_rename(
    rotations: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply only the global Captury axis relabelling to every segment.

    This intentionally does not use any Q marker or joint centre.  It is kept
    as an optional diagnostic baseline to demonstrate why per-segment static
    calibration is needed for anatomical comparisons.
    """

    return apply_segment_frame_corrections(
        rotations,
        {name: CAPTURY_SIMPLE_AXIS_RENAME_MATRIX for name in rotations},
    )
