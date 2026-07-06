"""Pure helpers for GUI-edited Motive/Captury marker correspondences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def split_marker_labels(value: object) -> list[str]:
    """Normalize one tree/listbox value into landmark-map marker labels."""

    if isinstance(value, str):
        labels = value.split(";")
    elif isinstance(value, Iterable):
        labels = [str(item) for item in value]
    else:
        labels = [str(value)]
    return [label.strip() for label in labels if label and label.strip()]


def marker_pair_name(motive_label: str, captury_label: str) -> str:
    """Return a stable display name for a one-to-one marker pair."""

    return f"{str(motive_label).strip()}_to_{str(captury_label).strip()}"


def tree_values_to_payload(values: Sequence[object]) -> dict[str, object]:
    """Convert Treeview values to the JSON landmark-map item format."""

    name, motive_labels, captury_labels = values
    return {
        "name": str(name),
        "reference": split_marker_labels(motive_labels),
        "test": split_marker_labels(captury_labels),
    }


def payload_to_tree_values(payload: Mapping[str, object]) -> tuple[str, str, str]:
    """Convert one JSON landmark-map item to compact Treeview values."""

    return (
        str(payload.get("name", "")),
        ";".join(split_marker_labels(payload.get("reference", []))),
        ";".join(split_marker_labels(payload.get("test", []))),
    )


def marker_pair_to_payload(motive_label: str, captury_label: str) -> dict[str, object]:
    """Build a JSON payload row from selected Motive and Captury markers."""

    return {
        "name": marker_pair_name(motive_label, captury_label),
        "reference": [str(motive_label)],
        "test": [str(captury_label)],
    }


def marker_pair_key(payload: Mapping[str, object]) -> tuple[str, str] | None:
    """Return the first-label pair used by the GUI for duplicate detection."""

    reference = split_marker_labels(payload.get("reference", []))
    test = split_marker_labels(payload.get("test", []))
    if not reference or not test:
        return None
    return reference[0], test[0]


def save_marker_correspondence_payload(
    path: Path, rows: Iterable[Mapping[str, object]]
) -> Path:
    """Write marker correspondences using the existing landmark-map JSON schema."""

    payload = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
