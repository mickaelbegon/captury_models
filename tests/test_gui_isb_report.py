from __future__ import annotations

import unittest

from gui_isb_report import filter_isb_rows, isb_filter_values, isb_table_values


class GuiIsbReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "source_id": "captury_model",
                "source_format": "fbx",
                "trial_id": None,
                "scope_kind": "segment",
                "entity_id": "pelvis",
                "deviation_id": "D1",
                "status_class": "unknown",
                "confidence": "unknown",
                "blocking": True,
                "blocker_codes": ["unknown_convention"],
            },
            {
                "source_id": "biobuddy_motive57",
                "source_format": "biomod",
                "trial_id": "Static",
                "scope_kind": "articulation",
                "entity_id": "right_hip",
                "deviation_id": "D4",
                "status_class": "conformant",
                "confidence": "computed_anatomical",
                "blocking": False,
                "blocker_codes": [],
            },
        ]

    def test_filters_are_composable_and_keep_global_segment_rows(self) -> None:
        filtered = filter_isb_rows(
            self.rows,
            source_id="captury_model",
            deviation_id="D1",
            entity_id="pelvis",
            selected_trial="Static",
        )

        self.assertEqual(filtered, [self.rows[0]])
        self.assertEqual(
            isb_filter_values(self.rows, "deviation_id"), ("Tous", "D1", "D4")
        )
        self.assertEqual(
            isb_filter_values(self.rows, "entity_id"),
            ("Tous", "pelvis", "right_hip"),
        )

    def test_table_values_make_blockers_explicit(self) -> None:
        values = isb_table_values(self.rows[0])

        self.assertEqual(values[0], "Captury BVH/FBX")
        self.assertEqual(values[3], "pelvis")
        self.assertEqual(values[7], "Oui: unknown_convention")


if __name__ == "__main__":
    unittest.main()
