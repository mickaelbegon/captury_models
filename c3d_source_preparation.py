"""Source-specific C3D point preparation helpers.

The Captury/Motive workflows often need to test a source-specific interpretation
before comparing marker clouds or model-derived centres. This module keeps that
preparation outside GUI callbacks and plotting scripts: optional root-translation
subtraction is applied in the source coordinate system first, then an optional
diagnostic transform expresses the source in the candidate comparison frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

POINT_TRANSFORM_CHOICES = ("none", "rx_plus_90")


@dataclass(frozen=True)
class SourcePreparationConfig:
    """Preparation options applied independently to one C3D source.

    ``root_offset_mm`` is expressed in the source C3D coordinate system. When
    ``subtract_root_offset`` is true, it is subtracted before applying
    ``transform``. ``transform`` is an active diagnostic rotation used to express
    one source in a candidate comparison frame.
    """

    root_offset_mm: np.ndarray
    subtract_root_offset: bool = False
    transform: str = "none"


def rotation_x_degrees(angle_degrees: float) -> np.ndarray:
    """Return an active 3D rotation matrix around the X axis."""

    angle = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=float,
    )


def apply_point_transform(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a 3D rotation to C3D points stored as ``(3, n_markers, n_frames)``."""

    return np.einsum("ij,jkf->ikf", rotation, points)


def subtract_root_offset(points: np.ndarray, root_offset_mm: np.ndarray) -> np.ndarray:
    """Subtract a root offset from C3D points before axis conversion."""

    offset = np.asarray(root_offset_mm, dtype=float).reshape(3, 1, 1)
    return np.asarray(points, dtype=float) - offset


def transform_points(points: np.ndarray, mode: str) -> np.ndarray:
    """Apply an optional diagnostic point transform."""

    if mode == "none":
        return np.asarray(points, dtype=float).copy()
    if mode == "rx_plus_90":
        return apply_point_transform(points, rotation_x_degrees(90.0))
    raise ValueError(
        f"Unsupported point transform {mode!r}. Expected one of {POINT_TRANSFORM_CHOICES}."
    )


def prepare_source_points(
    points: np.ndarray,
    config: SourcePreparationConfig,
) -> np.ndarray:
    """Apply one source's diagnostic preparation in pipeline order."""

    prepared = np.asarray(points, dtype=float)
    if config.subtract_root_offset:
        prepared = subtract_root_offset(prepared, config.root_offset_mm)
    return transform_points(prepared, config.transform)


def effective_root_offset_subtractions(args: argparse.Namespace) -> tuple[bool, bool]:
    """Return effective Motive/Captury root-offset flags from parsed CLI args."""

    legacy_global_flag = bool(getattr(args, "legacy_subtract_root_offsets", False))
    return (
        bool(getattr(args, "motive_subtract_root_offset", False) or legacy_global_flag),
        bool(
            getattr(args, "captury_subtract_root_offset", False) or legacy_global_flag
        ),
    )


def source_preparation_configs_from_args(
    args: argparse.Namespace,
) -> tuple[SourcePreparationConfig, SourcePreparationConfig]:
    """Build Motive and Captury source preparation configs from CLI args."""

    motive_subtract_root_offset, captury_subtract_root_offset = (
        effective_root_offset_subtractions(args)
    )
    motive_root_offset = np.asarray(
        getattr(args, "motive_root_offset_mm", np.zeros(3)), dtype=float
    )
    captury_root_offset = np.asarray(
        getattr(args, "captury_root_offset_mm", np.zeros(3)), dtype=float
    )
    return (
        SourcePreparationConfig(
            root_offset_mm=motive_root_offset,
            subtract_root_offset=motive_subtract_root_offset,
            transform=getattr(args, "motive_transform", "none"),
        ),
        SourcePreparationConfig(
            root_offset_mm=captury_root_offset,
            subtract_root_offset=captury_subtract_root_offset,
            transform=getattr(args, "captury_transform", "none"),
        ),
    )
