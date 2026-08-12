"""Canonical rotation operations for biomechanical segment comparisons.

The public contract uses column vectors.  A segment orientation
``R_lab_segment`` is a 3x3 matrix whose columns are the segment-local axes
expressed in the laboratory frame.  Rotation series are stored as arrays with
shape ``(3, 3, n_frames)``.

This module deliberately operates on rotation matrices rather than generalized
coordinates or Euler angles.  It provides the representation layer needed to
compare model exports without assuming that similarly named q channels use the
same local frames or rotation sequence.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class RotationSeriesValidationError(ValueError):
    """Raised when an array does not represent a valid SO(3) time series."""


def canonicalize_segment_rotation_mapping(
    source_rotations: Mapping[str, np.ndarray],
    canonical_to_source_name: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Map exporter-specific segment names to stable anatomical identifiers.

    Missing expected source names and extra exporter segments are reported
    rather than silently conflated. One raw segment cannot represent two
    canonical segments because such a mapping would duplicate evidence.
    """

    reverse: dict[str, str] = {}
    for canonical_id, source_name in canonical_to_source_name.items():
        if source_name in reverse:
            raise ValueError(
                f"source segment {source_name!r} is mapped more than once: "
                f"{reverse[source_name]!r} and {canonical_id!r}"
            )
        reverse[source_name] = canonical_id
    mapped = {
        canonical_id: source_rotations[source_name]
        for canonical_id, source_name in canonical_to_source_name.items()
        if source_name in source_rotations
    }
    expected_source_names = set(canonical_to_source_name.values())
    missing_canonical = sorted(
        canonical_id
        for canonical_id, source_name in canonical_to_source_name.items()
        if source_name not in source_rotations
    )
    return mapped, {
        "missing_canonical_segments": missing_canonical,
        "missing_source_names": sorted(expected_source_names - set(source_rotations)),
        "unmapped_source_names": sorted(set(source_rotations) - expected_source_names),
    }


def assess_rotation_source_equivalence(
    comparison: Mapping[str, Any], *, max_p95_geodesic_deg: float
) -> dict[str, Any]:
    """Decide whether two model exports may be selected interchangeably.

    Equivalence requires at least one common canonical segment, complete
    canonical coverage in both sources, and a per-segment p95 SO(3) geodesic
    deviation below the declared tolerance. This is an engineering gate, not a
    claim that either source follows ISB conventions.
    """

    if not np.isfinite(max_p95_geodesic_deg):
        raise ValueError("max_p95_geodesic_deg must be finite")
    if max_p95_geodesic_deg < 0.0:
        raise ValueError("max_p95_geodesic_deg must be non-negative")
    summary = comparison.get("summary", {})
    over_tolerance = sorted(
        segment
        for segment, metrics in summary.items()
        if float(metrics["p95_geodesic_deg"]) > max_p95_geodesic_deg
    )
    missing = sorted(
        set(comparison.get("missing_in_reference", ())).union(
            comparison.get("missing_in_test", ())
        )
    )
    common = list(comparison.get("common_segments", ()))
    without_metrics = sorted(set(common) - set(summary))
    blocked = (
        not common or bool(missing) or bool(over_tolerance) or bool(without_metrics)
    )
    return {
        "status": "blocked" if blocked else "equivalent_within_tolerance",
        "blocks_automatic_source_selection": blocked,
        "max_p95_geodesic_deg": float(max_p95_geodesic_deg),
        "common_segment_count": len(common),
        "missing_segments": missing,
        "segments_over_tolerance": over_tolerance,
        "segments_without_metrics": without_metrics,
        "meaning": (
            "BVH and FBX must remain explicitly selected and cannot be treated "
            "as interchangeable."
            if blocked
            else "BVH and FBX agree within the declared numerical gate for all "
            "mapped segments."
        ),
    }


def canonicalize_rotation_series(
    values: np.ndarray,
    *,
    context: str = "rotation series",
    max_raw_orthonormal_error: float = 1e-4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project a near-rotation series onto SO(3) and report the correction.

    FBX transforms read through biorbd can contain scale roundoff of a few parts
    per million.  Projection by polar decomposition (SVD) removes that numerical
    residue without hiding malformed transforms: reflections and raw errors over
    ``max_raw_orthonormal_error`` are rejected before projection.
    """

    rotations = _as_rotation_series(values, context)
    frame_first = np.moveaxis(rotations, 2, 0)
    gram = np.einsum("fji,fjk->fik", frame_first, frame_first)
    orthonormal_errors = np.max(np.abs(gram - np.eye(3)[None, :, :]), axis=(1, 2))
    determinants = np.linalg.det(frame_first)
    if np.any(determinants <= 0.0):
        raise RotationSeriesValidationError(
            f"{context} contains a reflection or singular transform"
        )
    worst_raw = float(np.max(orthonormal_errors))
    if worst_raw > max_raw_orthonormal_error:
        raise RotationSeriesValidationError(
            f"{context} is too far from SO(3): raw orthonormal error={worst_raw:.3g}"
        )
    left, _, right_transpose = np.linalg.svd(frame_first)
    projected = left @ right_transpose
    projected_determinants = np.linalg.det(projected)
    negative = projected_determinants < 0.0
    if np.any(negative):
        left[negative, :, -1] *= -1.0
        projected = left @ right_transpose
    projection_errors = np.linalg.norm(projected - frame_first, axis=(1, 2))
    result = np.moveaxis(projected, 0, 2)
    validate_rotation_series(result, context=f"canonical {context}", atol=1e-10)
    return result, {
        "max_raw_orthonormal_error": worst_raw,
        "max_raw_determinant_error": float(np.max(np.abs(determinants - 1.0))),
        "max_projection_frobenius": float(np.max(projection_errors)),
    }


def _as_rotation_series(values: np.ndarray, context: str) -> np.ndarray:
    rotations = np.asarray(values, dtype=float)
    if rotations.ndim != 3 or rotations.shape[:2] != (3, 3):
        raise RotationSeriesValidationError(
            f"{context} must have shape (3, 3, n_frames), got {rotations.shape}"
        )
    if rotations.shape[2] == 0:
        raise RotationSeriesValidationError(
            f"{context} must contain at least one frame"
        )
    if not np.all(np.isfinite(rotations)):
        raise RotationSeriesValidationError(f"{context} contains non-finite values")
    return rotations


def validate_rotation_series(
    values: np.ndarray,
    *,
    context: str = "rotation series",
    atol: float = 1e-6,
) -> np.ndarray:
    """Validate and return a ``(3, 3, n_frames)`` SO(3) series.

    Every frame must be orthonormal and have determinant +1.  Reflections are
    rejected explicitly because they cannot be interpreted as segment-frame
    rotations.
    """

    rotations = _as_rotation_series(values, context)
    frame_first = np.moveaxis(rotations, 2, 0)
    gram = np.einsum("fji,fjk->fik", frame_first, frame_first)
    if not np.allclose(gram, np.eye(3)[None, :, :], atol=atol, rtol=0.0):
        worst = float(np.max(np.abs(gram - np.eye(3)[None, :, :])))
        raise RotationSeriesValidationError(
            f"{context} must contain orthonormal rotations; worst error={worst:.3g}"
        )
    determinants = np.linalg.det(frame_first)
    if not np.allclose(determinants, 1.0, atol=atol, rtol=0.0):
        raise RotationSeriesValidationError(
            f"{context} must have determinant +1; range="
            f"[{determinants.min():.6g}, {determinants.max():.6g}]"
        )
    return rotations


def validate_rotation_matrix(
    value: np.ndarray, *, context: str = "rotation", atol: float = 1e-6
) -> np.ndarray:
    """Validate and return one 3x3 SO(3) matrix."""

    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise RotationSeriesValidationError(
            f"{context} must have shape (3, 3), got {matrix.shape}"
        )
    validate_rotation_series(matrix[:, :, None], context=context, atol=atol)
    return matrix


def change_lab_basis(
    rotations: np.ndarray, target_from_source: np.ndarray
) -> np.ndarray:
    """Express segment frames in another laboratory basis.

    ``target_from_source`` maps source-laboratory column vectors into the target
    laboratory.  Therefore ``R_target_segment = target_from_source @
    R_source_segment``.
    """

    source, _ = canonicalize_rotation_series(rotations, context="source rotations")
    basis = validate_rotation_matrix(target_from_source, context="target_from_source")
    converted = np.einsum("ij,jkf->ikf", basis, source)
    return validate_rotation_series(converted, context="converted rotations")


def relative_rotation_series(proximal: np.ndarray, distal: np.ndarray) -> np.ndarray:
    """Return ``R_proximal_distal = R_lab_proximal.T @ R_lab_distal``."""

    proximal_values, _ = canonicalize_rotation_series(
        proximal, context="proximal rotations"
    )
    distal_values, _ = canonicalize_rotation_series(distal, context="distal rotations")
    if proximal_values.shape[2] != distal_values.shape[2]:
        raise RotationSeriesValidationError(
            "proximal and distal rotations must contain the same number of frames"
        )
    relative = np.einsum("jif,jkf->ikf", proximal_values, distal_values)
    return validate_rotation_series(relative, context="relative rotations")


def rotation_vector(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Return the SO(3) log vector of ``reference.T @ test`` in radians."""

    reference_matrix = validate_rotation_matrix(reference, context="reference")
    test_matrix = validate_rotation_matrix(test, context="test")
    relative = reference_matrix.T @ test_matrix
    return Rotation.from_matrix(relative).as_rotvec()


def rotation_geodesic_degrees(reference: np.ndarray, test: np.ndarray) -> float:
    """Return the shortest angular distance between two rotations in degrees."""

    return float(np.rad2deg(np.linalg.norm(rotation_vector(reference, test))))


def slerp_rotation_series(
    rotations: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Interpolate a rotation series on ``target_time`` using quaternion SLERP.

    Target samples must lie inside the source interval; extrapolation is
    rejected because it has no defensible interpretation for measured motion.
    """

    source = validate_rotation_series(rotations, context="SLERP source rotations")
    source_time_values = np.asarray(source_time, dtype=float).reshape(-1)
    target_time_values = np.asarray(target_time, dtype=float).reshape(-1)
    if source_time_values.size != source.shape[2]:
        raise ValueError("source_time length must match the rotation frame count")
    if target_time_values.size == 0:
        return np.empty((3, 3, 0), dtype=float)
    if not np.all(np.isfinite(source_time_values)) or not np.all(
        np.isfinite(target_time_values)
    ):
        raise ValueError("SLERP times must be finite")
    if np.any(np.diff(source_time_values) <= 0.0):
        raise ValueError("source_time must be strictly increasing")
    tolerance = 1e-12
    if (
        target_time_values.min() < source_time_values[0] - tolerance
        or target_time_values.max() > source_time_values[-1] + tolerance
    ):
        raise ValueError("target_time must lie inside the source time interval")
    target_time_values = np.clip(
        target_time_values, source_time_values[0], source_time_values[-1]
    )
    frame_first = np.moveaxis(source, 2, 0)
    if source_time_values.size == 1:
        if not np.allclose(target_time_values, source_time_values[0], atol=tolerance):
            raise ValueError(
                "a single-frame series can only be sampled at its timestamp"
            )
        interpolated = np.repeat(frame_first, target_time_values.size, axis=0)
    else:
        interpolated = Slerp(source_time_values, Rotation.from_matrix(frame_first))(
            target_time_values
        ).as_matrix()
    result = np.moveaxis(interpolated, 0, 2)
    return validate_rotation_series(result, context="SLERP result")


def _overlap_reference_time(
    reference_time: np.ndarray, test_time: np.ndarray
) -> np.ndarray:
    reference = np.asarray(reference_time, dtype=float).reshape(-1)
    test = np.asarray(test_time, dtype=float).reshape(-1)
    if reference.size == 0 or test.size == 0:
        return np.asarray([], dtype=float)
    start = max(float(reference[0]), float(test[0]))
    end = min(float(reference[-1]), float(test[-1]))
    return reference[(reference >= start) & (reference <= end)]


def _elapsed_time(values: np.ndarray, context: str) -> np.ndarray:
    time = np.asarray(values, dtype=float).reshape(-1)
    if time.size == 0:
        return time
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{context} must contain finite values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{context} must be strictly increasing")
    return time - time[0]


def compare_segment_rotation_mappings(
    reference_rotations: Mapping[str, np.ndarray],
    reference_time: np.ndarray,
    test_rotations: Mapping[str, np.ndarray],
    test_time: np.ndarray,
) -> dict[str, Any]:
    """Compare common segment-frame series with SO(3) geodesic metrics.

    Test rotations are interpolated by SLERP onto reference timestamps in the
    temporal overlap.  The output keeps both rotation-vector components and the
    invariant geodesic magnitude for downstream reports.
    """

    reference_names = set(reference_rotations)
    test_names = set(test_rotations)
    common = sorted(reference_names.intersection(test_names))
    missing_in_reference = sorted(test_names - reference_names)
    missing_in_test = sorted(reference_names - test_names)
    reference_time_values = _elapsed_time(reference_time, "reference_time")
    test_time_values = _elapsed_time(test_time, "test_time")
    overlap_time = _overlap_reference_time(reference_time_values, test_time_values)
    if common and overlap_time.size == 0:
        raise ValueError("reference and test rotation series have no temporal overlap")

    reference_indices = np.searchsorted(reference_time_values, overlap_time)
    summary: dict[str, dict[str, float | int]] = {}
    timeseries: dict[str, dict[str, np.ndarray]] = {}
    for segment in common:
        reference_series, reference_quality = canonicalize_rotation_series(
            reference_rotations[segment], context=f"reference {segment}"
        )
        test_series, test_quality = canonicalize_rotation_series(
            test_rotations[segment], context=f"test {segment}"
        )
        if reference_series.shape[2] != reference_time_values.size:
            raise ValueError(f"reference time length does not match {segment}")
        if test_series.shape[2] != test_time_values.size:
            raise ValueError(f"test time length does not match {segment}")
        reference_overlap = reference_series[:, :, reference_indices]
        test_overlap = slerp_rotation_series(
            test_series, test_time_values, overlap_time
        )
        vectors = np.zeros((3, overlap_time.size), dtype=float)
        for frame in range(overlap_time.size):
            vectors[:, frame] = rotation_vector(
                reference_overlap[:, :, frame], test_overlap[:, :, frame]
            )
        geodesic_deg = np.rad2deg(np.linalg.norm(vectors, axis=0))
        summary[segment] = {
            "n_frames": int(overlap_time.size),
            "median_geodesic_deg": float(np.median(geodesic_deg)),
            "p95_geodesic_deg": float(np.percentile(geodesic_deg, 95.0)),
            "max_geodesic_deg": float(np.max(geodesic_deg)),
            "rms_geodesic_deg": float(np.sqrt(np.mean(geodesic_deg**2))),
            "reference_max_raw_orthonormal_error": reference_quality[
                "max_raw_orthonormal_error"
            ],
            "test_max_raw_orthonormal_error": test_quality["max_raw_orthonormal_error"],
            "reference_max_projection_frobenius": reference_quality[
                "max_projection_frobenius"
            ],
            "test_max_projection_frobenius": test_quality["max_projection_frobenius"],
        }
        timeseries[segment] = {
            "time": overlap_time.copy(),
            "rotation_vector_rad": vectors,
            "geodesic_deg": geodesic_deg,
        }
    return {
        "time_alignment": {
            "method": "elapsed_from_first_sample",
            "lag_estimation": False,
            "meaning": (
                "Each export starts at elapsed time zero; no temporal lag or "
                "missing-leading-frame correction is estimated."
            ),
        },
        "common_segments": common,
        "missing_in_reference": missing_in_reference,
        "missing_in_test": missing_in_test,
        "summary": summary,
        "timeseries": timeseries,
    }
