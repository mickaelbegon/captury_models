"""Non-circular spatial calibration for model and C3D comparisons.

The module uses row-vector rigid transforms throughout:

``p_target = p_source @ rotation + translation``

Calibration and evaluation centres are disjoint by contract. A transform may
be estimated on a static trial and serialized once, then reused unchanged for
all dynamic trials. Unit conversion, model-axis conversion and root-offset
interpretation remain upstream operations and are recorded by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mocap_alignment import apply_row_alignment, compose_row_alignment, kabsch_rows


@dataclass(frozen=True)
class RowRigidTransform:
    """One validated row-vector rigid transform."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(
                f"translation must have shape (3,), got {translation.shape}"
            )
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("rigid transform must contain finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("rotation must have determinant +1")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation", translation.copy())

    @classmethod
    def identity(cls) -> "RowRigidTransform":
        return cls(np.eye(3), np.zeros(3))

    def apply_rows(self, points: np.ndarray) -> np.ndarray:
        """Apply the transform to an array whose trailing dimension is XYZ."""

        return apply_row_alignment(points, self.rotation, self.translation)

    def compose(self, second: "RowRigidTransform") -> "RowRigidTransform":
        """Return the transform obtained by applying self then ``second``."""

        rotation, translation = compose_row_alignment(
            self.rotation,
            self.translation,
            second.rotation,
            second.translation,
        )
        return RowRigidTransform(rotation, translation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotation": self.rotation.tolist(),
            "translation_mm": self.translation.tolist(),
            "convention": "row_vectors: p_target = p_source @ R + t",
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RowRigidTransform":
        return cls(
            np.asarray(values["rotation"], dtype=float),
            np.asarray(values["translation_mm"], dtype=float),
        )


@dataclass(frozen=True)
class SpatialCalibration:
    """Two static transforms reused as one immutable calibration protocol."""

    static_trial: str
    calibration_centres: tuple[str, ...]
    evaluation_centres: tuple[str, ...]
    captury_to_motive: RowRigidTransform
    motive_to_c3d: RowRigidTransform
    status: str
    captury_root_offset_mode: str = "unknown"
    motive_root_offset_mode: str = "unknown"

    def __post_init__(self) -> None:
        overlap = set(self.calibration_centres).intersection(self.evaluation_centres)
        if overlap:
            raise ValueError(
                f"calibration and evaluation centres must be disjoint: {sorted(overlap)}"
            )

    def captury_to_c3d(self) -> RowRigidTransform:
        return self.captury_to_motive.compose(self.motive_to_c3d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "static_trial": self.static_trial,
            "calibration_centres": list(self.calibration_centres),
            "evaluation_centres": list(self.evaluation_centres),
            "root_translation_policy": {
                "captury": self.captury_root_offset_mode,
                "motive": self.motive_root_offset_mode,
            },
            "ordered_stages": [
                "root_translation_policy_in_native_q",
                "forward_kinematics_in_native_model_frame",
                "source_native_to_c3d_mm",
                "captury_to_motive_static",
                "motive_to_c3d_static",
            ],
            "captury_to_motive": self.captury_to_motive.to_dict(),
            "motive_to_c3d": self.motive_to_c3d.to_dict(),
            "captury_to_c3d_composed": self.captury_to_c3d().to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SpatialCalibration":
        return cls(
            static_trial=str(values["static_trial"]),
            calibration_centres=tuple(values["calibration_centres"]),
            evaluation_centres=tuple(values["evaluation_centres"]),
            captury_to_motive=RowRigidTransform.from_dict(values["captury_to_motive"]),
            motive_to_c3d=RowRigidTransform.from_dict(values["motive_to_c3d"]),
            status=str(values["status"]),
            captury_root_offset_mode=str(
                values.get("root_translation_policy", {}).get("captury", "unknown")
            ),
            motive_root_offset_mode=str(
                values.get("root_translation_policy", {}).get("motive", "unknown")
            ),
        )


def _mean_common_rows(
    moving_centres: Mapping[str, np.ndarray],
    reference_centres: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    moving_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    used: list[str] = []
    for name in names:
        if name not in moving_centres or name not in reference_centres:
            continue
        moving = np.nanmean(np.asarray(moving_centres[name], dtype=float), axis=1)
        reference = np.nanmean(np.asarray(reference_centres[name], dtype=float), axis=1)
        if np.all(np.isfinite(moving)) and np.all(np.isfinite(reference)):
            moving_rows.append(moving)
            reference_rows.append(reference)
            used.append(name)
    if not used:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.vstack(moving_rows), np.vstack(reference_rows), used


def _distance_report(
    moving_rows: np.ndarray,
    reference_rows: np.ndarray,
    transform: RowRigidTransform,
) -> dict[str, float | int]:
    if moving_rows.size == 0:
        return {"n_points": 0, "median_mm": np.nan, "p95_mm": np.nan, "max_mm": np.nan}
    errors = np.linalg.norm(transform.apply_rows(moving_rows) - reference_rows, axis=1)
    return {
        "n_points": int(errors.size),
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95.0)),
        "max_mm": float(np.max(errors)),
    }


def fit_held_out_centre_alignment(
    moving_centres: Mapping[str, np.ndarray],
    reference_centres: Mapping[str, np.ndarray],
    calibration_centres: Sequence[str],
    *,
    min_points: int = 4,
) -> tuple[RowRigidTransform, dict[str, Any]]:
    """Fit on reserved static centres and report errors on held-out centres.

    A full 3-D rigid transform requires calibration points spanning at least a
    plane. Collinear or duplicated mean centres are rejected rather than
    allowing an arbitrary Kabsch rotation about the degenerate direction.
    """

    common = sorted(set(moving_centres).intersection(reference_centres))
    requested = list(dict.fromkeys(str(name) for name in calibration_centres))
    moving_cal, reference_cal, used_cal = _mean_common_rows(
        moving_centres, reference_centres, requested
    )
    evaluation = sorted(set(common) - set(used_cal))
    moving_eval, reference_eval, used_eval = _mean_common_rows(
        moving_centres, reference_centres, evaluation
    )
    identity = RowRigidTransform.identity()
    base_report: dict[str, Any] = {
        "method": "static_reserved_centres_kabsch_rows",
        "calibration_centres_requested": requested,
        "calibration_centres": used_cal,
        "evaluation_centres": used_eval,
        "calibration_before": _distance_report(moving_cal, reference_cal, identity),
        "evaluation_before": _distance_report(moving_eval, reference_eval, identity),
    }
    if len(used_cal) < min_points:
        base_report["status"] = "not_enough_calibration_centres"
        base_report["transform"] = identity.to_dict()
        return identity, base_report
    rank = int(np.linalg.matrix_rank(moving_cal - np.mean(moving_cal, axis=0)))
    reference_rank = int(
        np.linalg.matrix_rank(reference_cal - np.mean(reference_cal, axis=0))
    )
    base_report["moving_geometry_rank"] = rank
    base_report["reference_geometry_rank"] = reference_rank
    if min(rank, reference_rank) < 2:
        base_report["status"] = "degenerate_calibration_geometry"
        base_report["transform"] = identity.to_dict()
        return identity, base_report
    rotation, translation = kabsch_rows(reference_cal, moving_cal)
    transform = RowRigidTransform(rotation, translation)
    base_report.update(
        {
            "status": "ok",
            "transform": transform.to_dict(),
            "calibration_after": _distance_report(moving_cal, reference_cal, transform),
            "evaluation_after": _distance_report(
                moving_eval, reference_eval, transform
            ),
        }
    )
    return transform, base_report
