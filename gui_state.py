"""Persist the minimal GUI context needed to reopen existing results.

Scientific options and visualization choices deliberately remain outside this
file. The session state only remembers where the dataset, comparison outputs
and BioBuddy model live, plus the last selected trial. Existing output files
remain the source of truth for whether a viewer layer is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

GUI_SESSION_STATE_VERSION = 1
GUI_SESSION_STATE_KEYS = (
    "p6_data_root",
    "p6_out_dir",
    "biobuddy_c3d_output",
    "selected_trial",
)


def default_gui_session_state_path() -> Path:
    """Return the per-user state path, with an environment override for tests."""

    override = os.environ.get("CAPTURY_MODELS_GUI_STATE_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    config_root = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ).expanduser()
    return config_root / "captury_models" / "gui_session.json"


def load_gui_session_state(path: Path) -> dict[str, str]:
    """Read a valid state file, returning an empty state on any read failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != GUI_SESSION_STATE_VERSION
    ):
        return {}
    values = payload.get("values")
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key in GUI_SESSION_STATE_KEYS
        if isinstance((value := values.get(key)), str)
    }


def save_gui_session_state(path: Path, values: Mapping[str, object]) -> None:
    """Atomically write the allow-listed session values to ``path``."""

    payload = {
        "version": GUI_SESSION_STATE_VERSION,
        "values": {
            key: str(values[key])
            for key in GUI_SESSION_STATE_KEYS
            if key in values and isinstance(values[key], (str, Path))
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
