from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    from compare_p6_motive_captury import (
        captury_flat_trial_name,
        discover_flat_trials,
        model_to_c3d_matrix,
        motive_flat_trial_name,
    )
except ImportError as exc:  # pragma: no cover - depends on optional scientific env
    captury_flat_trial_name = None
    discover_flat_trials = None
    model_to_c3d_matrix = None
    motive_flat_trial_name = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    IMPORT_ERROR is not None, f"optional dependencies missing: {IMPORT_ERROR}"
)
class FlatTrialDiscoveryTests(unittest.TestCase):
    def test_flat_trial_names_strip_system_specific_suffixes(self) -> None:
        assert captury_flat_trial_name is not None
        assert motive_flat_trial_name is not None

        self.assertEqual(captury_flat_trial_name(Path("Static_P6.c3d")), "Static")
        self.assertEqual(
            motive_flat_trial_name(Path("P6_Static_Skeleton 001.bvh")), "Static"
        )
        self.assertEqual(
            motive_flat_trial_name(Path("P6_Marche_001.c3d")), "Marche_001"
        )

    def test_discover_flat_trials_matches_only_complete_c3d_pairs(self) -> None:
        assert discover_flat_trials is not None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captury = root / "Captury"
            motive = root / "Motive"
            captury.mkdir()
            motive.mkdir()

            for suffix in ("c3d", "bvh", "fbx"):
                (captury / f"Static_P6.{suffix}").write_text(
                    "captury", encoding="utf-8"
                )
            (motive / "P6_Static.c3d").write_text("motive", encoding="utf-8")
            (motive / "P6_Static.fbx").write_text("motive", encoding="utf-8")
            (motive / "P6_Static_Skeleton 001.bvh").write_text(
                "motive", encoding="utf-8"
            )

            (captury / "MissingMotive_P6.c3d").write_text("captury", encoding="utf-8")
            (motive / "P6_MissingCaptury.c3d").write_text("motive", encoding="utf-8")

            trials = discover_flat_trials(root)

            self.assertEqual([trial.name for trial in trials], ["Static"])
            bundle = trials[0]
            self.assertEqual(bundle.captury_c3d.name, "Static_P6.c3d")
            self.assertEqual(bundle.captury_bvh.name, "Static_P6.bvh")
            self.assertEqual(bundle.captury_fbx.name, "Static_P6.fbx")
            self.assertEqual(bundle.motive_c3d.name, "P6_Static.c3d")
            self.assertEqual(bundle.motive_bvh.name, "P6_Static_Skeleton 001.bvh")
            self.assertEqual(bundle.motive_fbx.name, "P6_Static.fbx")

    def test_auto_axis_uses_current_y_up_to_z_up_conversion(self) -> None:
        assert model_to_c3d_matrix is not None

        np.testing.assert_allclose(
            model_to_c3d_matrix("auto"), model_to_c3d_matrix("y_up_to_z_up")
        )


if __name__ == "__main__":
    unittest.main()
