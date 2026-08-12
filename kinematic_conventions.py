"""Load and validate scientific conventions used in kinematic comparisons.

The comparison pipelines operate on representations whose segment frames and
angle conventions are not necessarily equivalent.  This module provides a
small, GUI-independent contract for documenting those conventions before a
quantity is accepted for final biomechanical comparison.

Unknown information is represented explicitly.  Validation guarantees schema
consistency and valid correction rotations, but it never promotes an unknown
or merely documented convention to ISB compliance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_KINEMATIC_CONVENTIONS_PATH = Path(__file__).with_name(
    "kinematic_conventions.json"
)

CONVENTION_STATUSES = {
    "conforme",
    "deviation",
    "documente_non_evalue",
    "inconnu",
    "non_applicable",
}
EVIDENCE_REQUIRED_STATUSES = {"conforme", "deviation"}
ISB_DEVIATIONS = tuple(f"D{index}" for index in range(1, 7))


class ConventionValidationError(ValueError):
    """Raised when a kinematic-convention registry violates its contract."""


def load_kinematic_conventions(
    path: Path | str = DEFAULT_KINEMATIC_CONVENTIONS_PATH,
) -> dict[str, Any]:
    """Load and validate a kinematic-convention registry from JSON."""

    registry_path = Path(path).expanduser()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _resolve_source_inheritance(registry)
    validate_kinematic_conventions(registry)
    return registry


def _resolve_source_inheritance(registry: dict[str, Any]) -> None:
    """Expand source definitions that reuse another source's model topology.

    Inheritance is limited to segment and articulation definitions.  Source-level
    provenance, units, and laboratory-frame declarations always remain local to
    the child source, preventing a Captury convention from silently becoming a
    Motive convention.
    """

    sources = _require_mapping(registry.get("sources"), "sources")
    resolving: set[str] = set()
    resolved: set[str] = set()

    def resolve(source_id: str) -> None:
        if source_id in resolved:
            return
        if source_id in resolving:
            raise ConventionValidationError(
                f"source inheritance contains a cycle at {source_id!r}"
            )
        if source_id not in sources:
            raise ConventionValidationError(
                f"source inheritance references unknown source {source_id!r}"
            )
        resolving.add(source_id)
        source = sources[source_id]
        parent_id = source.get("segments_from")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ConventionValidationError(
                    f"sources.{source_id}.segments_from must be a source id"
                )
            resolve(parent_id)
            parent = sources[parent_id]
            if source.get("segments"):
                raise ConventionValidationError(
                    f"sources.{source_id} cannot combine segments_from with segments"
                )
            if source.get("articulations"):
                raise ConventionValidationError(
                    f"sources.{source_id} cannot combine segments_from with articulations"
                )
            source["segments"] = deepcopy(parent["segments"])
            source["articulations"] = deepcopy(parent["articulations"])
            del source["segments_from"]
        resolving.remove(source_id)
        resolved.add(source_id)

    for source_id in sources:
        resolve(source_id)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConventionValidationError(f"{context} must be an object")
    return value


def _validate_status(value: Any, context: str) -> None:
    status = _require_mapping(value, context).get("status")
    if status not in CONVENTION_STATUSES:
        raise ConventionValidationError(
            f"{context}.status must be one of {sorted(CONVENTION_STATUSES)}"
        )
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        raise ConventionValidationError(f"{context}.evidence must be a list")
    if status in EVIDENCE_REQUIRED_STATUSES and not evidence:
        raise ConventionValidationError(
            f"{context} status {status!r} requires evidence"
        )


def _validate_rotation(value: Any, context: str) -> None:
    if value is None:
        return
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ConventionValidationError(f"{context} must be a 3x3 rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ConventionValidationError(f"{context} must be an orthonormal rotation")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-8):
        raise ConventionValidationError(
            f"{context} must have determinant +1, got {determinant}"
        )


def _validate_segment_graph(
    source_id: str, segments: Mapping[str, Mapping[str, Any]]
) -> None:
    for segment_id, segment in segments.items():
        visited: set[str] = set()
        current: str | None = segment_id
        while current is not None:
            if current in visited:
                raise ConventionValidationError(
                    f"sources.{source_id}.segments contains a cycle at {current}"
                )
            visited.add(current)
            current_segment = segments.get(current)
            if current_segment is None:
                raise ConventionValidationError(
                    f"sources.{source_id}.segments.{segment_id} has unknown parent "
                    f"{current!r}"
                )
            parent = current_segment.get("parent")
            current = None if parent in (None, "root") else str(parent)


def _validate_segments(source_id: str, source: Mapping[str, Any]) -> None:
    segments = _require_mapping(source.get("segments"), f"sources.{source_id}.segments")
    if not segments:
        raise ConventionValidationError(
            f"sources.{source_id}.segments must not be empty"
        )
    used_names: dict[str, str] = {}
    for segment_id, raw_segment in segments.items():
        context = f"sources.{source_id}.segments.{segment_id}"
        segment = _require_mapping(raw_segment, context)
        source_name = segment.get("source_name")
        aliases = segment.get("aliases", [])
        if not isinstance(source_name, str) or not source_name:
            raise ConventionValidationError(f"{context}.source_name is required")
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias for alias in aliases
        ):
            raise ConventionValidationError(f"{context}.aliases must contain strings")
        for name in (source_name, *aliases):
            normalized = name.casefold()
            if normalized in used_names:
                raise ConventionValidationError(
                    f"sources.{source_id} has duplicate segment name or alias "
                    f"{name!r} in {segment_id!r} and {used_names[normalized]!r}"
                )
            used_names[normalized] = segment_id
        _validate_status(segment.get("frame_definition"), f"{context}.frame_definition")
        _validate_rotation(
            segment.get("source_to_isb_rotation"),
            f"{context}.source_to_isb_rotation",
        )
        isb_audit = _require_mapping(segment.get("isb_audit"), f"{context}.isb_audit")
        for deviation in ("D1", "D2", "D3"):
            _validate_status(
                isb_audit.get(deviation), f"{context}.isb_audit.{deviation}"
            )
    _validate_segment_graph(source_id, segments)
    names_by_format = source.get("segment_names_by_format", {})
    if not isinstance(names_by_format, Mapping):
        raise ConventionValidationError(
            f"sources.{source_id}.segment_names_by_format must be an object"
        )
    for source_format, raw_names in names_by_format.items():
        context = f"sources.{source_id}.segment_names_by_format.{source_format}"
        if not isinstance(source_format, str) or not source_format:
            raise ConventionValidationError(
                f"sources.{source_id}.segment_names_by_format keys must be strings"
            )
        names = _require_mapping(raw_names, context)
        used_format_names: dict[str, str] = {}
        for segment_id, source_name in names.items():
            if segment_id not in segments:
                raise ConventionValidationError(
                    f"{context} references unknown segment {segment_id!r}"
                )
            if not isinstance(source_name, str) or not source_name:
                raise ConventionValidationError(
                    f"{context}.{segment_id} must be a non-empty string"
                )
            normalized = source_name.casefold()
            if normalized in used_format_names:
                raise ConventionValidationError(
                    f"{context} has duplicate format-specific segment name "
                    f"{source_name!r} for {segment_id!r} and "
                    f"{used_format_names[normalized]!r}"
                )
            used_format_names[normalized] = str(segment_id)


def _validate_articulations(source_id: str, source: Mapping[str, Any]) -> None:
    articulations = _require_mapping(
        source.get("articulations"), f"sources.{source_id}.articulations"
    )
    if not articulations:
        raise ConventionValidationError(
            f"sources.{source_id}.articulations must not be empty"
        )
    segments = source["segments"]
    for articulation_id, raw_articulation in articulations.items():
        context = f"sources.{source_id}.articulations.{articulation_id}"
        articulation = _require_mapping(raw_articulation, context)
        for endpoint in ("proximal", "distal"):
            value = articulation.get(endpoint)
            if value not in segments and value != "unknown":
                raise ConventionValidationError(
                    f"{context}.{endpoint} references unknown segment {value!r}"
                )
        _validate_status(
            articulation.get("rotation_sequence"), f"{context}.rotation_sequence"
        )
        _validate_status(
            articulation.get("anatomical_components"),
            f"{context}.anatomical_components",
        )
        isb_audit = _require_mapping(
            articulation.get("isb_audit"), f"{context}.isb_audit"
        )
        for deviation in ("D4", "D5", "D6"):
            _validate_status(
                isb_audit.get(deviation), f"{context}.isb_audit.{deviation}"
            )


def validate_kinematic_conventions(registry: Mapping[str, Any]) -> None:
    """Validate structure, graph integrity, statuses and correction rotations."""

    root = _require_mapping(registry, "registry")
    if root.get("schema_version") != 1:
        raise ConventionValidationError("schema_version must be 1")
    matrix_convention = _require_mapping(
        root.get("matrix_convention"), "matrix_convention"
    )
    required_matrix_fields = {
        "vector_storage",
        "orientation_definition",
        "point_transform",
        "relative_rotation",
    }
    missing_matrix_fields = required_matrix_fields.difference(matrix_convention)
    if missing_matrix_fields:
        raise ConventionValidationError(
            f"matrix_convention is missing {sorted(missing_matrix_fields)}"
        )
    sources = _require_mapping(root.get("sources"), "sources")
    if not sources:
        raise ConventionValidationError("sources must not be empty")
    for source_id, raw_source in sources.items():
        source = _require_mapping(raw_source, f"sources.{source_id}")
        if not isinstance(source.get("representation"), str):
            raise ConventionValidationError(
                f"sources.{source_id}.representation is required"
            )
        _validate_status(
            source.get("laboratory_frame"), f"sources.{source_id}.laboratory_frame"
        )
        _validate_status(source.get("length_unit"), f"sources.{source_id}.length_unit")
        _validate_segments(source_id, source)
        _validate_articulations(source_id, source)


def segment_source_names(
    registry: Mapping[str, Any], source_id: str, source_format: str
) -> dict[str, str]:
    """Return canonical segment ids mapped to names used by one file format.

    ``source_name`` is the conservative default. A source can override only the
    names that differ in a specific export format through
    ``segment_names_by_format``. The returned keys remain stable anatomical ids
    and are therefore suitable for cross-format comparisons.
    """

    validate_kinematic_conventions(registry)
    source = registry["sources"][source_id]
    result = {
        segment_id: segment["source_name"]
        for segment_id, segment in source["segments"].items()
    }
    overrides = source.get("segment_names_by_format", {}).get(source_format, {})
    result.update(overrides)
    return result


def final_comparison_blockers(
    registry: Mapping[str, Any], source_id: str, articulation_id: str
) -> list[str]:
    """Return unresolved conventions that block a final angular comparison."""

    validate_kinematic_conventions(registry)
    source = registry["sources"][source_id]
    articulation = source["articulations"][articulation_id]
    blockers: list[str] = []
    for key in ("rotation_sequence", "anatomical_components"):
        status = articulation[key]["status"]
        if status in {"inconnu", "documente_non_evalue"}:
            blockers.append(f"{key}.status={status}")
    for deviation in ("D4", "D6"):
        status = articulation["isb_audit"][deviation]["status"]
        if status in {"inconnu", "documente_non_evalue"}:
            blockers.append(f"isb_audit.{deviation}={status}")
    for endpoint in ("proximal", "distal"):
        segment_id = articulation[endpoint]
        if segment_id == "unknown":
            blockers.append(f"{endpoint}=unknown")
            continue
        segment = source["segments"][segment_id]
        frame_status = segment["frame_definition"]["status"]
        if frame_status in {"inconnu", "documente_non_evalue"}:
            blockers.append(f"segments.{segment_id}.frame_definition={frame_status}")
        for deviation in ("D1", "D2", "D3"):
            status = segment["isb_audit"][deviation]["status"]
            if status in {"inconnu", "documente_non_evalue"}:
                blockers.append(f"segments.{segment_id}.isb_audit.{deviation}={status}")
        if segment["source_to_isb_rotation"] is None:
            blockers.append(f"segments.{segment_id}.source_to_isb_rotation=unknown")
    return blockers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": _sha256(resolved),
    }


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _comparison_readiness(registry: Mapping[str, Any]) -> dict[str, Any]:
    blockers: dict[str, list[str]] = {}
    for source_id, source in registry["sources"].items():
        for articulation_id in source["articulations"]:
            articulation_blockers = final_comparison_blockers(
                registry, source_id, articulation_id
            )
            if articulation_blockers:
                blockers[f"{source_id}/{articulation_id}"] = articulation_blockers
    return {
        "status": "diagnostic_only" if blockers else "final_comparison_ready",
        "blockers": blockers,
        "meaning": (
            "At least one required frame or angular convention remains unresolved; "
            "affected comparisons are diagnostic and cannot support a final "
            "biomechanical agreement conclusion."
            if blockers
            else "All convention gates represented by this registry are resolved."
        ),
    }


def build_provenance_manifest(
    *,
    registry_path: Path | str = DEFAULT_KINEMATIC_CONVENTIONS_PATH,
    input_files: Mapping[str, Path | str],
    derived_artifacts: Mapping[str, Path | str] | None = None,
    command: Sequence[str],
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for one comparison execution."""

    resolved_registry_path = Path(registry_path).expanduser().resolve()
    registry = load_kinematic_conventions(resolved_registry_path)
    return {
        "manifest_version": 1,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "registry": _file_provenance(resolved_registry_path),
        "matrix_convention": registry["matrix_convention"],
        "comparison_readiness": _comparison_readiness(registry),
        "inputs": {
            str(name): _file_provenance(Path(path))
            for name, path in sorted(input_files.items())
        },
        "derived_artifacts": {
            str(name): _file_provenance(Path(path))
            for name, path in sorted((derived_artifacts or {}).items())
        },
        "command": [str(value) for value in command],
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "packages": _package_versions(
                ("biobuddy", "biorbd", "ezc3d", "numpy", "pandas", "scipy")
            ),
        },
    }
