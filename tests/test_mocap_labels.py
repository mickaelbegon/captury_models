from __future__ import annotations

import unittest

from mocap_labels import (
    DEFAULT_MARKER_PREFIXES_TO_STRIP,
    display_marker_name,
    is_joint_centre_marker_label,
    marker_display_labels,
    marker_indices_by_display_label,
    marker_name_index,
    stripped_marker_label,
)


class MocapLabelsTests(unittest.TestCase):
    def test_stripped_marker_label_removes_first_matching_prefix(self) -> None:
        self.assertEqual(
            stripped_marker_label(
                "Skeleton_001_LIAS", DEFAULT_MARKER_PREFIXES_TO_STRIP
            ),
            "LIAS",
        )
        self.assertEqual(
            stripped_marker_label("LIAS", DEFAULT_MARKER_PREFIXES_TO_STRIP),
            "LIAS",
        )

    def test_display_marker_name_uses_default_motive_prefixes(self) -> None:
        self.assertEqual(display_marker_name("Skeleton_001_LIAS"), "LIAS")
        self.assertEqual(
            display_marker_name("FooSkeleton_001_LIAS"), "FooSkeleton_001_LIAS"
        )

    def test_marker_display_labels_number_duplicate_clean_names(self) -> None:
        self.assertEqual(
            marker_display_labels(["Q_Hip", "Q_Knee", "Q_Hip"]),
            ["Q_Hip#1", "Q_Knee", "Q_Hip#2"],
        )
        self.assertEqual(
            marker_display_labels(["Skeleton_001_LIAS", "Skeleton_001_LIAS"]),
            ["LIAS#1", "LIAS#2"],
        )

    def test_joint_centre_marker_labels_are_excluded_from_skin_markers(self) -> None:
        self.assertTrue(is_joint_centre_marker_label("CAPJC_Hips"))
        self.assertTrue(is_joint_centre_marker_label("MOTJC_LeftKnee"))
        self.assertTrue(is_joint_centre_marker_label("FBXJC_RightAnkle"))
        self.assertTrue(is_joint_centre_marker_label("BVHJC_RightFoot"))
        self.assertFalse(is_joint_centre_marker_label("Skeleton_001_LIAS"))
        self.assertFalse(is_joint_centre_marker_label("Q_LH#1"))

    def test_marker_indices_by_display_label_maps_clean_and_numbered_labels(
        self,
    ) -> None:
        lookup = marker_indices_by_display_label(
            ["Skeleton_001_LIAS", "Q_Hip", "Q_Hip"]
        )

        self.assertEqual(lookup["LIAS"], [0])
        self.assertEqual(lookup["Q_Hip"], [1, 2])
        self.assertEqual(lookup["Q_Hip#1"], [1])
        self.assertEqual(lookup["Q_Hip#2"], [2])

    def test_marker_name_index_keeps_only_unique_names_for_ik_matching(self) -> None:
        self.assertEqual(marker_name_index(["A", "B", "A", "C"]), {"B": 1, "C": 3})


if __name__ == "__main__":
    unittest.main()
