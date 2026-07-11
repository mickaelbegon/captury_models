"""Classify C3D ``POINT`` channels as physical markers or joint angles.

Some Captury exports store joint-angle waveforms in the C3D ``POINT`` group.
They look like 3D trajectories to generic readers but must never be supplied to
marker-based IK, marker viewers, or skin-marker comparison metrics. This module
keeps that distinction consistent for all C3D consumers without reading files
or imposing a unit convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_C3D_ANGLE_LABELS = {
    "RHip",
    "LHip",
    "RKne",
    "LKne",
    "RAnk",
    "LAnk",
    "RSho",
    "LSho",
    "RElb",
    "LElb",
    "RWri",
    "LWri",
    "Neck",
}


@dataclass(frozen=True)
class C3DPointChannelClassification:
    """Indices and labels split between marker and angle ``POINT`` channels."""

    labels: list[str]
    marker_indices: list[int]
    angle_indices: list[int]

    @property
    def marker_labels(self) -> list[str]:
        """Return labels for physical marker channels in source order."""

        return [self.labels[index] for index in self.marker_indices]

    @property
    def angle_labels(self) -> list[str]:
        """Return labels for angle channels in source order."""

        return [self.labels[index] for index in self.angle_indices]


def c3d_parameter_values(c3d: dict, group: str, name: str, default: Any = None) -> Any:
    """Return one C3D parameter value, using ``default`` when absent."""

    try:
        return c3d["parameters"][group][name]["value"]
    except KeyError:
        return default


def c3d_string_list(value: Any) -> list[str]:
    """Normalize an ezc3d parameter value to stripped strings."""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [str(value).strip()]
    return [str(item).strip() for item in value]


def angle_labels_from_c3d_parameters(c3d: dict) -> set[str]:
    """Return explicit C3D angle labels, including compact whitespace variants."""

    labels: set[str] = set()
    for parameter_name in ("ANGLES", "ANGLE_LABELS"):
        for label in c3d_string_list(
            c3d_parameter_values(c3d, "POINT", parameter_name, [])
        ):
            if label:
                labels.add(label)
                labels.add(label.replace(" ", ""))
    return labels


def point_angle_tail_indices(c3d: dict, n_points: int) -> list[tuple[str, int]]:
    """Map legacy ``POINT:ANGLES`` names to the trailing point-channel indices.

    Some exporters name angle channels only in ``POINT:ANGLES``. In those files
    the entries correspond to the final N ``POINT`` channels even when their
    ``POINT:LABELS`` values differ. Invalid counts intentionally return no
    mapping rather than guessing an index.
    """

    angle_names = c3d_string_list(c3d_parameter_values(c3d, "POINT", "ANGLES", []))
    if not angle_names or len(angle_names) > n_points:
        return []
    first_index = n_points - len(angle_names)
    return [(name, first_index + index) for index, name in enumerate(angle_names)]


def classify_c3d_point_channels(
    c3d: dict,
    *,
    angle_label_regex: str,
    extra_angle_labels: list[str] | None = None,
    default_angle_labels: set[str] | None = None,
    point_angles_tail_fallback: bool = False,
) -> C3DPointChannelClassification:
    """Split C3D point channels while retaining their original index order.

    A channel is an angle when it is named in ``POINT:ANGLES`` or
    ``POINT:ANGLE_LABELS``, appears in known/extra labels, or matches the
    caller's regex against its label or description. Empty regex disables the
    regex criterion. ``point_angles_tail_fallback`` preserves the convention of
    legacy C3D exports where ``POINT:ANGLES`` names the last N point channels
    but does not repeat their labels. The function intentionally operates on an
    in-memory C3D dictionary so file loading remains owned by the calling CLI
    or GUI.
    """

    labels = c3d_string_list(c3d_parameter_values(c3d, "POINT", "LABELS", []))
    descriptions = c3d_string_list(
        c3d_parameter_values(c3d, "POINT", "DESCRIPTIONS", [])
    )
    descriptions += [""] * max(0, len(labels) - len(descriptions))
    explicit_angle_labels = angle_labels_from_c3d_parameters(c3d)
    tail_angle_indices = point_angle_tail_indices(c3d, len(labels))
    known_angle_labels = set(default_angle_labels or DEFAULT_C3D_ANGLE_LABELS)
    known_angle_labels.update(label.strip() for label in (extra_angle_labels or []))
    compact_known_angle_labels = {
        label.replace(" ", "") for label in known_angle_labels
    }
    regex = re.compile(angle_label_regex) if angle_label_regex else None

    angle_indices: list[int] = []
    for index, label in enumerate(labels):
        compact_label = label.replace(" ", "")
        description = descriptions[index]
        is_angle = (
            label in explicit_angle_labels
            or compact_label in explicit_angle_labels
            or label in known_angle_labels
            or compact_label in compact_known_angle_labels
            or (
                regex is not None
                and bool(regex.search(label) or regex.search(description))
            )
        )
        if is_angle:
            angle_indices.append(index)
    if point_angles_tail_fallback and tail_angle_indices:
        angle_indices.extend(index for _, index in tail_angle_indices)
        angle_indices = sorted(set(angle_indices))
    angle_index_set = set(angle_indices)
    return C3DPointChannelClassification(
        labels=labels,
        marker_indices=[
            index for index in range(len(labels)) if index not in angle_index_set
        ],
        angle_indices=angle_indices,
    )
