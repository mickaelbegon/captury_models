from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motive57_c3d_mapping import (
    assignments_from_payload,
    discover_c3d_files,
    infer_motive57_role_assignments,
    motive57_mapping_payload,
    prepared_motive57_c3d_folder,
    save_motive57_mapping,
)


class Motive57C3dMappingTests(unittest.TestCase):
    def test_infers_p6_motive57_role_assignments(self) -> None:
        files = [
            "P6_Static.c3d",
            "P6_LHip.c3d",
            "P6_LKnee.c3d",
            "P6_LAnkle.c3d",
            "P6_RHip.c3d",
            "P6_RKnee.c3d",
            "P6_RAnkle.c3d",
            "P6_Marche_001.c3d",
        ]

        assignments = infer_motive57_role_assignments(files)

        self.assertEqual(assignments["static"], "P6_Static.c3d")
        self.assertEqual(assignments["left_hip_score"], "P6_LHip.c3d")
        self.assertEqual(assignments["left_knee_sara"], "P6_LKnee.c3d")
        self.assertEqual(assignments["left_ankle_score"], "P6_LAnkle.c3d")
        self.assertEqual(assignments["right_hip_score"], "P6_RHip.c3d")
        self.assertEqual(assignments["right_knee_sara"], "P6_RKnee.c3d")
        self.assertEqual(assignments["right_ankle_score"], "P6_RAnkle.c3d")

    def test_mapping_payload_lists_c3d_files_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "P6_Static.c3d").write_text("static", encoding="utf-8")
            (folder / "P6_LHip.c3d").write_text("lhip", encoding="utf-8")

            payload = motive57_mapping_payload(folder)

        self.assertIn("P6_Static.c3d", payload["c3d_files"])
        assignments = assignments_from_payload(payload)
        self.assertEqual(assignments["static"], "P6_Static.c3d")
        self.assertEqual(assignments["left_hip_score"], "P6_LHip.c3d")

    def test_prepared_folder_uses_template_compatible_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("P6_Static.c3d", "P6_LHip.c3d"):
                (folder / name).write_text(name, encoding="utf-8")
            mapping_path = save_motive57_mapping(
                folder,
                {"static": "P6_Static.c3d", "left_hip_score": "P6_LHip.c3d"},
            )

            with prepared_motive57_c3d_folder(folder, mapping_path) as prepared:
                prepared_files = discover_c3d_files(prepared)

        self.assertIn("selected_Static.c3d", prepared_files)
        self.assertIn("selected_Func_LHip.c3d", prepared_files)


if __name__ == "__main__":
    unittest.main()
