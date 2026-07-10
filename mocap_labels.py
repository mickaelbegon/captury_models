"""Shared marker-label helpers for C3D, GUI and IK workflows."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_MARKER_PREFIXES_TO_STRIP = ("Skeleton_001_",)
JOINT_CENTRE_PREFIXES = ("CAPJC_", "MOTJC_", "BVHJC_", "FBXJC_")


def stripped_marker_label(
    label: str,
    prefixes: tuple[str, ...] = DEFAULT_MARKER_PREFIXES_TO_STRIP,
) -> str:
    """Return ``label`` without the first matching acquisition prefix."""

    clean_label = str(label).strip()
    for prefix in prefixes:
        if prefix and clean_label.startswith(prefix):
            return clean_label[len(prefix) :]
    return clean_label


def display_marker_name(label: str) -> str:
    """Return the marker name shown in GUI lists and viewer labels."""

    return stripped_marker_label(label, DEFAULT_MARKER_PREFIXES_TO_STRIP)


def is_joint_centre_marker_label(label: str) -> bool:
    """Return True for synthetic joint-centre point labels, not skin markers."""

    clean_label = display_marker_name(label).upper()
    return clean_label.startswith(JOINT_CENTRE_PREFIXES)


def marker_display_labels(labels: Iterable[str]) -> list[str]:
    """Return unique GUI labels while preserving the C3D marker order.

    Some Captury exports contain several POINT entries with the same label. A
    plain listbox would collapse these names conceptually, making it impossible
    to map a specific duplicate marker to a Motive marker. Duplicated display
    names are therefore numbered as ``Name#1``, ``Name#2``; unique names are
    left unchanged.
    """

    base_labels = [display_marker_name(str(label)) for label in labels]
    totals: dict[str, int] = {}
    for base_label in base_labels:
        totals[base_label] = totals.get(base_label, 0) + 1
    seen: dict[str, int] = {}
    display_labels: list[str] = []
    for base_label in base_labels:
        seen[base_label] = seen.get(base_label, 0) + 1
        if totals[base_label] > 1:
            display_labels.append(f"{base_label}#{seen[base_label]}")
        else:
            display_labels.append(base_label)
    return display_labels


def marker_indices_by_display_label(labels: list[str]) -> dict[str, list[int]]:
    """Return lookup entries for clean labels and numbered duplicate labels."""

    lookup: dict[str, list[int]] = {}
    unique_labels = marker_display_labels(labels)
    for index, (label, unique_label) in enumerate(zip(labels, unique_labels)):
        clean_label = display_marker_name(label)
        lookup.setdefault(clean_label, []).append(index)
        if unique_label != clean_label:
            lookup.setdefault(unique_label, []).append(index)
    return lookup


def marker_name_index(names: list[str]) -> dict[str, int]:
    """Return unique marker names and drop duplicates from automatic matching."""

    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {name: index for index, name in enumerate(names) if counts[name] == 1}
