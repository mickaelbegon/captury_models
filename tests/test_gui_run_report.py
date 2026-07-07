from __future__ import annotations

import unittest

from gui_run_report import summarize_run_report


class GuiRunReportTests(unittest.TestCase):
    def test_summarize_run_report_exposes_automatic_choices_and_fallbacks(self) -> None:
        report = {
            "trial": "Static",
            "axis_conversion": "y_up_to_z_up",
            "models": {
                "captury": {
                    "source_kind": "bvh",
                    "root_offset_policy": {
                        "selected_policy": "subtract",
                        "score_mm": 12.3,
                    },
                },
                "motive": {
                    "source_kind": "fbx",
                    "root_offset_policy": {
                        "selected_policy": "keep",
                    },
                },
            },
            "alignment": {
                "status": "ok",
                "motive_model_to_c3d_markers": {"method": "horizontal_pca_fallback"},
            },
            "segment_rotations": {
                "requested_reference": "biobuddy",
                "effective_reference": "motive",
                "status": "fallback_missing_reference",
            },
            "segment_orientation_corrections": {
                "applied": ["captury_thigh_y_axis_from_hip_to_knee_cor"]
            },
        }

        summary = summarize_run_report(report)

        self.assertIn("Essai: Static", summary)
        self.assertIn("Axe modèle -> C3D: y_up_to_z_up", summary)
        self.assertIn("Captury: bvh, root offset subtract", summary)
        self.assertIn("Motive: fbx, root offset keep", summary)
        self.assertIn("Référence segments: biobuddy -> motive", summary)
        self.assertIn("fallback_missing_reference", summary)
        self.assertIn("captury_thigh_y_axis_from_hip_to_knee_cor", summary)

    def test_summarize_run_report_handles_empty_report(self) -> None:
        self.assertEqual(summarize_run_report({}), "Aucun rapport sélectionné.")


if __name__ == "__main__":
    unittest.main()
