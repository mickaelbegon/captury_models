from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from create_biobuddy_c3d_model import (
    create_biobuddy_c3d_model,
    marker_prefixes_to_strip_for_preset,
)


class _FakeModel:
    def to_biomod(self, filepath: str, with_mesh: bool = False) -> None:
        Path(filepath).write_text(f"with_mesh={with_mesh}\n", encoding="utf-8")


class CreateBioBuddyC3dModelTests(unittest.TestCase):
    def test_motive_57_strips_skeleton_marker_prefix_by_default(self) -> None:
        self.assertEqual(
            marker_prefixes_to_strip_for_preset("motive_57"), ("Skeleton_001_",)
        )

    def test_create_model_passes_default_motive_57_marker_prefix(self) -> None:
        calls: dict[str, object] = {}

        def fake_create_model(*args: object, **kwargs: object) -> object:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                model=_FakeModel(),
                preset=SimpleNamespace(value="motive_57"),
            )

        fake_api = {
            "preset_from_cli": lambda value: SimpleNamespace(value=value),
            "create_model": fake_create_model,
            "default_static_virtual_points": lambda _preset: (),
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Motive"
            folder.mkdir()
            output = Path(tmp) / "motive_57.bioMod"
            with patch(
                "create_biobuddy_c3d_model._load_biobuddy_c3d_api",
                return_value=fake_api,
            ):
                create_biobuddy_c3d_model(folder, preset="motive_57", output=output)

        self.assertEqual(
            calls["kwargs"]["marker_name_prefixes_to_strip"], ("Skeleton_001_",)
        )

    def test_explicit_marker_prefixes_override_default(self) -> None:
        self.assertEqual(
            marker_prefixes_to_strip_for_preset(
                "motive_57", explicit_prefixes=["Subject:"]
            ),
            ("Subject:",),
        )


if __name__ == "__main__":
    unittest.main()
