"""Presentation-only helpers for the filterable ISB D1-D6 GUI table."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

SOURCE_LABELS = {
    "captury_model": "Captury BVH/FBX",
    "captury_c3d_angles": "Captury C3D angles",
    "motive_model": "Motive BVH/FBX",
    "biobuddy_motive57": "BioBuddy Motive 57",
}
SOURCE_IDS_BY_LABEL = {label: source_id for source_id, label in SOURCE_LABELS.items()}


def isb_filter_values(rows: Iterable[Mapping[str, Any]], key: str) -> tuple[str, ...]:
    values = sorted({str(row.get(key, "")) for row in rows if row.get(key)})
    return ("Tous", *values)


def filter_isb_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_id: str = "Tous",
    deviation_id: str = "Tous",
    entity_id: str = "Tous",
    status_class: str = "Tous",
    selected_trial: str = "",
) -> list[dict[str, Any]]:
    """Filter rows while keeping trial-independent D1-D3 evidence visible."""

    filtered: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if source_id != "Tous" and str(row.get("source_id")) != source_id:
            continue
        if deviation_id != "Tous" and str(row.get("deviation_id")) != deviation_id:
            continue
        if entity_id != "Tous" and str(row.get("entity_id")) != entity_id:
            continue
        if status_class != "Tous" and str(row.get("status_class")) != status_class:
            continue
        trial_id = str(row.get("trial_id") or "")
        if selected_trial and selected_trial != "Tous les essais" and trial_id:
            if trial_id != selected_trial:
                continue
        filtered.append(row)
    return filtered


def _list_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def isb_table_values(row: Mapping[str, Any]) -> tuple[str, ...]:
    blockers = _list_text(row.get("blocker_codes"))
    blocking = f"Oui: {blockers}" if row.get("blocking") else "Non"
    return (
        SOURCE_LABELS.get(str(row.get("source_id")), str(row.get("source_id", ""))),
        str(row.get("source_format", "")),
        str(row.get("trial_id") or "Global"),
        str(row.get("entity_id", "")),
        str(row.get("deviation_id", "")),
        str(row.get("status_class", "")),
        str(row.get("confidence", "")),
        blocking,
    )
