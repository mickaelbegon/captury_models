"""Shared row-vector rigid-alignment helpers for mocap data."""

from __future__ import annotations

import numpy as np


def kabsch_rows(
    reference: np.ndarray, moving: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-vector rigid transform aligning ``moving`` to ``reference``.

    Points are represented as rows and transformed as ``points @ rotation +
    translation``.
    """

    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    reference_mean = np.nanmean(reference, axis=0)
    moving_mean = np.nanmean(moving, axis=0)
    covariance = (moving - moving_mean).T @ (reference - reference_mean)
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = reference_mean - moving_mean @ rotation
    return rotation, translation


def apply_row_alignment(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply a row-vector rigid transform to points with trailing XYZ axis."""

    return np.asarray(points, dtype=float) @ np.asarray(
        rotation, dtype=float
    ) + np.asarray(translation, dtype=float)


def compose_row_alignment(
    first_rotation: np.ndarray,
    first_translation: np.ndarray,
    second_rotation: np.ndarray,
    second_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two row-vector rigid transforms.

    Applying ``first`` then ``second`` gives ``p @ (R1 @ R2) + (t1 @ R2 + t2)``.
    """

    first_rotation = np.asarray(first_rotation, dtype=float)
    first_translation = np.asarray(first_translation, dtype=float)
    second_rotation = np.asarray(second_rotation, dtype=float)
    second_translation = np.asarray(second_translation, dtype=float)
    return (
        first_rotation @ second_rotation,
        first_translation @ second_rotation + second_translation,
    )
