"""Launch BioBuddy's model editor and optionally pre-open a selected model."""

from __future__ import annotations

import argparse
from pathlib import Path


def _patch_open_dialog(model_path: Path) -> None:
    """Make BioBuddy's first Open dialog return the model selected by the caller."""
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFileDialog, QMainWindow
    except ImportError:
        from PyQt5.QtCore import QTimer  # type: ignore
        from PyQt5.QtWidgets import QFileDialog, QMainWindow  # type: ignore

    original_get_open_file_name = QFileDialog.getOpenFileName
    original_show = QMainWindow.show
    state = {"dialog_used": False, "open_scheduled": False}

    def get_open_file_name_once(*args, **kwargs):
        if not state["dialog_used"]:
            state["dialog_used"] = True
            return str(model_path), "BioBuddy model"
        return original_get_open_file_name(*args, **kwargs)

    def show_and_open(window, *args, **kwargs):
        result = original_show(window, *args, **kwargs)
        if not state["open_scheduled"] and hasattr(window, "_open_model"):
            state["open_scheduled"] = True
            QTimer.singleShot(0, window._open_model)
        return result

    QFileDialog.getOpenFileName = get_open_file_name_once
    QMainWindow.show = show_and_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open BioBuddy's model editor.")
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        help="Optional .bioMod, .osim, .urdf or .bvh model to pre-open in the editor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is not None:
        model_path = args.model.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        _patch_open_dialog(model_path)

    from biobuddy.gui.model_editor import launch_model_editor

    launch_model_editor()


if __name__ == "__main__":
    main()
