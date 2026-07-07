"""Small presentation helpers for trial-level run reports in the GUI."""

from __future__ import annotations

from typing import Any, Mapping


def _nested(mapping: Mapping[str, Any], *keys: str, default: object = "") -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _model_line(report: Mapping[str, Any], model_key: str, label: str) -> str | None:
    model = _nested(report, "models", model_key, default={})
    if not isinstance(model, Mapping) or not model:
        return None
    source_kind = str(model.get("source_kind", "?"))
    policy = _nested(model, "root_offset_policy", "selected_policy", default="?")
    score = _nested(model, "root_offset_policy", "score_mm", default=None)
    line = f"{label}: {source_kind}, root offset {policy}"
    if isinstance(score, (float, int)):
        line += f" ({score:.2f} mm)"
    return line


def summarize_run_report(report: Mapping[str, Any]) -> str:
    """Return a compact human-readable summary of automatic analysis choices."""

    if not report:
        return "Aucun rapport sélectionné."
    lines: list[str] = []
    trial = report.get("trial")
    if trial:
        lines.append(f"Essai: {trial}")
    axis = report.get("axis_conversion")
    if axis:
        lines.append(f"Axe modèle -> C3D: {axis}")
    for model_key, label in (("captury", "Captury"), ("motive", "Motive")):
        model_line = _model_line(report, model_key, label)
        if model_line:
            lines.append(model_line)
    alignment_status = _nested(report, "alignment", "status", default="")
    marker_method = _nested(
        report,
        "alignment",
        "motive_model_to_c3d_markers",
        "method",
        default="",
    )
    if alignment_status or marker_method:
        suffix = f", marqueurs Motive: {marker_method}" if marker_method else ""
        lines.append(f"Recalage: {alignment_status}{suffix}")
    segment_status = _nested(report, "segment_rotations", "status", default="")
    requested = _nested(report, "segment_rotations", "requested_reference", default="")
    effective = _nested(report, "segment_rotations", "effective_reference", default="")
    if requested or effective or segment_status:
        if requested and effective:
            line = f"Référence segments: {requested} -> {effective}"
        else:
            line = f"Référence segments: {requested or effective}"
        if segment_status:
            line += f" ({segment_status})"
        lines.append(line)
    corrections = _nested(
        report, "segment_orientation_corrections", "applied", default=[]
    )
    if corrections:
        lines.append(f"Corrections segments: {', '.join(map(str, corrections))}")
    return (
        "\n".join(lines) if lines else "Rapport disponible, aucun choix critique listé."
    )
