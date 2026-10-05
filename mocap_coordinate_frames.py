"""Resolve source-specific model and C3D coordinate-frame conversions.

The comparison uses a canonical ``+Z``-up laboratory frame.  Captury and
Motive do not currently export their C3D files in the same laboratory basis:
the validated Captury static exports are ``+Y`` up whereas Motive C3D exports
are ``+Z`` up.  A model must first be expressed in *its own* C3D frame before
the two C3D frames are compared in the common frame.

All matrices use column vectors.  For points ``p_target = M @ p_source`` and
for segment frames ``R_target = M @ R_source``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Y_UP_TO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float
)

_DEFAULT_C3D_VERTICAL_AXIS = {"captury": "y", "motive": "z", "biobuddy": "z"}


@dataclass(frozen=True)
class SystemCoordinateFrames:
    """Resolved coordinate changes for one source system.

    ``model_to_own_c3d`` is used for root-offset scoring and for every
    calibration built from this system's own C3D markers.  ``own_c3d_to_common``
    is only used when comparing Captury and Motive together.  Keeping the two
    stages distinct prevents an accidental Captury ``Y``-to-``Z`` conversion
    while deriving Captury-only segment frames.
    """

    system: str
    own_c3d_vertical_axis: str
    model_to_own_c3d: np.ndarray
    own_c3d_to_common: np.ndarray

    @property
    def model_to_common_c3d(self) -> np.ndarray:
        """Return the composed model-to-common-laboratory basis matrix."""

        return self.own_c3d_to_common @ self.model_to_own_c3d

    def to_report(self) -> dict[str, object]:
        """Serialize the resolved matrices for the per-trial provenance report."""

        return {
            "system": self.system,
            "own_c3d_vertical_axis": f"+{self.own_c3d_vertical_axis.upper()}",
            "common_comparison_vertical_axis": "+Z",
            "model_to_own_c3d": self.model_to_own_c3d.tolist(),
            "own_c3d_to_common": self.own_c3d_to_common.tolist(),
            "model_to_common_c3d": self.model_to_common_c3d.tolist(),
            "matrix_convention": "column_vectors: p_target = M @ p_source",
        }


def _normalize_system(system: str) -> str:
    normalized = str(system).strip().lower()
    if normalized not in _DEFAULT_C3D_VERTICAL_AXIS:
        raise ValueError(f"Unsupported capture system: {system!r}")
    return normalized


def _normalize_axis_mode(axis_mode: str) -> str:
    normalized = str(axis_mode).strip().lower()
    if normalized not in {"auto", "identity", "y_up_to_z_up"}:
        raise ValueError(f"Unsupported model-to-C3D axis mode: {axis_mode!r}")
    return normalized


def resolve_system_coordinate_frames(
    system: str,
    model_to_c3d_axis: str = "auto",
) -> SystemCoordinateFrames:
    """Resolve model and C3D bases for Captury, Motive or BioBuddy.

    ``auto`` follows the validated export conventions: Captury model and C3D
    are both ``+Y`` up, so root-offset selection and Captury-only frame fitting
    use identity.  Motive models are ``+Y`` up and its C3D is ``+Z`` up, so its
    own-frame conversion is :data:`Y_UP_TO_Z_UP`.  Captury is then converted to
    the common Motive-like ``+Z`` comparison frame in a distinct second stage.

    An explicit ``identity`` or ``y_up_to_z_up`` overrides only the
    model-to-own-C3D stage; it does not silently change the source C3D's known
    laboratory convention.
    """

    normalized_system = _normalize_system(system)
    normalized_mode = _normalize_axis_mode(model_to_c3d_axis)
    own_vertical = _DEFAULT_C3D_VERTICAL_AXIS[normalized_system]
    if normalized_mode == "auto":
        model_to_own = np.eye(3) if own_vertical == "y" else Y_UP_TO_Z_UP
    elif normalized_mode == "identity":
        model_to_own = np.eye(3)
    else:
        model_to_own = Y_UP_TO_Z_UP
    own_to_common = np.eye(3) if own_vertical == "z" else Y_UP_TO_Z_UP
    return SystemCoordinateFrames(
        system=normalized_system,
        own_c3d_vertical_axis=own_vertical,
        model_to_own_c3d=np.asarray(model_to_own, dtype=float).copy(),
        own_c3d_to_common=np.asarray(own_to_common, dtype=float).copy(),
    )


def transform_point_trajectories(
    points: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Apply a column-vector basis change to ``(XYZ, marker, frame)`` points.

    C3D point arrays deliberately retain the marker and frame axes.  Using
    ``matrix @ points`` would instead try to multiply the matrix with the
    marker axis; the explicit Einstein expression documents and protects the
    expected storage convention.
    """

    values = np.asarray(points, dtype=float)
    matrix = np.asarray(basis, dtype=float)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("points must have shape (3, markers, frames)")
    if matrix.shape != (3, 3):
        raise ValueError("basis must have shape (3, 3)")
    return np.einsum("ij,jkf->ikf", matrix, values)
