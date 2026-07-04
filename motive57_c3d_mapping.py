"""Motive 57 C3D role discovery and mapping helpers.

The BioBuddy Motive 57 template expects one static/anatomical C3D and six
functional calibration trials. Real datasets are often named with participant
prefixes such as ``P6_LHip.c3d`` instead of BioBuddy's glob patterns such as
``*Func_LHip.c3d``. This module keeps that filename knowledge out of the GUI:
it inventories C3D files, proposes role assignments, stores them as JSON, and
can materialize a temporary folder whose filenames match BioBuddy's template.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

MOTIVE_57_MAPPING_VERSION = 1
MOTIVE_57_MAPPING_FILENAME = ".motive_57_c3d_mapping.json"


@dataclass(frozen=True)
class Motive57C3dRole:
    key: str
    label: str
    expected_pattern: str
    prepared_filename: str
    method: str
    purpose: str
    aliases: tuple[str, ...]


MOTIVE_57_C3D_ROLES: tuple[Motive57C3dRole, ...] = (
    Motive57C3dRole(
        key="static",
        label="Statique/anatomique",
        expected_pattern="*Static*.c3d",
        prepared_filename="selected_Static.c3d",
        method="static",
        purpose="Repères anatomiques statiques, frames segmentaires et points virtuels LGJC/RGJC.",
        aliases=("static", "test_anato", "test_main", "main_markers", "anat"),
    ),
    Motive57C3dRole(
        key="left_hip_score",
        label="Hanche gauche SCoRE",
        expected_pattern="*Func_LHip.c3d",
        prepared_filename="selected_Func_LHip.c3d",
        method="SCoRE",
        purpose="CoR hanche gauche: bassin vs cuisse gauche.",
        aliases=("func_lhip", "lhip", "left_hip"),
    ),
    Motive57C3dRole(
        key="left_knee_sara",
        label="Genou gauche SARA",
        expected_pattern="*Func_LKnee.c3d",
        prepared_filename="selected_Func_LKnee.c3d",
        method="SARA",
        purpose="AoR genou gauche et projection du centre du genou.",
        aliases=("func_lknee", "lknee", "left_knee"),
    ),
    Motive57C3dRole(
        key="left_ankle_score",
        label="Cheville gauche SCoRE",
        expected_pattern="*Func_LAnkle.c3d",
        prepared_filename="selected_Func_LAnkle.c3d",
        method="SCoRE",
        purpose="CoR cheville gauche: jambe gauche vs pied gauche.",
        aliases=("func_lankle", "lankle", "left_ankle"),
    ),
    Motive57C3dRole(
        key="right_hip_score",
        label="Hanche droite SCoRE",
        expected_pattern="*Func_RHip.c3d",
        prepared_filename="selected_Func_RHip.c3d",
        method="SCoRE",
        purpose="CoR hanche droite: bassin vs cuisse droite.",
        aliases=("func_rhip", "rhip", "right_hip"),
    ),
    Motive57C3dRole(
        key="right_knee_sara",
        label="Genou droit SARA",
        expected_pattern="*Func_RKnee.c3d",
        prepared_filename="selected_Func_RKnee.c3d",
        method="SARA",
        purpose="AoR genou droit et projection du centre du genou.",
        aliases=("func_rknee", "rknee", "right_knee"),
    ),
    Motive57C3dRole(
        key="right_ankle_score",
        label="Cheville droite SCoRE",
        expected_pattern="*Func_RAnkle.c3d",
        prepared_filename="selected_Func_RAnkle.c3d",
        method="SCoRE",
        purpose="CoR cheville droite: jambe droite vs pied droit.",
        aliases=("func_rankle", "rankle", "right_ankle"),
    ),
)


def motive57_mapping_path(folder: str | Path) -> Path:
    return Path(folder).expanduser() / MOTIVE_57_MAPPING_FILENAME


def discover_c3d_files(folder: str | Path) -> list[str]:
    folder_path = Path(folder).expanduser()
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    return sorted(path.name for path in folder_path.glob("*.c3d"))


def _normalized_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(name).name.lower()).strip("_")


def _matches_role_alias(filename: str, role: Motive57C3dRole) -> bool:
    normalized = _normalized_filename(filename)
    for alias in role.aliases:
        alias_normalized = _normalized_filename(alias)
        if re.search(rf"(^|_){re.escape(alias_normalized)}($|_)", normalized):
            return True
    return False


def infer_motive57_role_assignments(c3d_files: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for role in MOTIVE_57_C3D_ROLES:
        pattern_matches = [
            name
            for name in c3d_files
            if fnmatch.fnmatchcase(name.lower(), role.expected_pattern.lower())
        ]
        alias_matches = [name for name in c3d_files if _matches_role_alias(name, role)]
        candidates = pattern_matches or alias_matches
        if candidates:
            assignments[role.key] = sorted(candidates)[0]
    return assignments


def motive57_mapping_payload(
    folder: str | Path, assignments: dict[str, str] | None = None
) -> dict[str, object]:
    folder_path = Path(folder).expanduser()
    c3d_files = discover_c3d_files(folder_path)
    role_assignments = (
        infer_motive57_role_assignments(c3d_files)
        if assignments is None
        else assignments
    )
    return {
        "version": MOTIVE_57_MAPPING_VERSION,
        "preset": "motive_57",
        "folder": str(folder_path),
        "c3d_files": c3d_files,
        "roles": [
            {
                "key": role.key,
                "label": role.label,
                "method": role.method,
                "purpose": role.purpose,
                "expected_pattern": role.expected_pattern,
                "file": role_assignments.get(role.key, ""),
            }
            for role in MOTIVE_57_C3D_ROLES
        ],
    }


def assignments_from_payload(payload: dict[str, object]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for item in payload.get("roles", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        filename = str(item.get("file", "")).strip()
        if key and filename:
            assignments[key] = filename
    return assignments


def save_motive57_mapping(
    folder: str | Path,
    assignments: dict[str, str],
    mapping_path: str | Path | None = None,
) -> Path:
    folder_path = Path(folder).expanduser()
    output_path = (
        Path(mapping_path).expanduser()
        if mapping_path
        else motive57_mapping_path(folder_path)
    )
    payload = motive57_mapping_payload(folder_path, assignments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_motive57_mapping(mapping_path: str | Path) -> dict[str, object] | None:
    path = Path(mapping_path).expanduser()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_create_motive57_mapping(folder: str | Path) -> dict[str, object]:
    path = motive57_mapping_path(folder)
    payload = load_motive57_mapping(path)
    if payload is not None:
        return payload
    payload = motive57_mapping_payload(folder)
    save_motive57_mapping(folder, assignments_from_payload(payload), path)
    return payload


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.symlink(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@contextmanager
def prepared_motive57_c3d_folder(
    source_folder: str | Path, mapping_path: str | Path
) -> Iterator[Path]:
    """Yield a temporary folder whose filenames satisfy the Motive 57 template."""
    source = Path(source_folder).expanduser()
    payload = load_motive57_mapping(mapping_path)
    if payload is None:
        raise FileNotFoundError(f"Mapping Motive 57 introuvable: {mapping_path}")
    assignments = assignments_from_payload(payload)
    with TemporaryDirectory(prefix="motive57_c3d_") as tmp:
        prepared = Path(tmp)
        for role in MOTIVE_57_C3D_ROLES:
            filename = assignments.get(role.key, "")
            if not filename:
                continue
            source_file = source / filename
            if not source_file.exists():
                raise FileNotFoundError(
                    f"Fichier C3D sélectionné introuvable pour {role.key}: {source_file}"
                )
            _link_or_copy(source_file, prepared / role.prepared_filename)
        yield prepared
