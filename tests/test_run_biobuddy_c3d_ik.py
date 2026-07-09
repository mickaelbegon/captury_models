import unittest

from run_biobuddy_c3d_ik import marker_name_index, stripped_marker_label


class RunBioBuddyC3dIkTests(unittest.TestCase):
    def test_stripped_marker_label_removes_motive_prefix(self) -> None:
        self.assertEqual(
            stripped_marker_label("Skeleton_001_LIAS", ("Skeleton_001_",)), "LIAS"
        )
        self.assertEqual(stripped_marker_label("LIAS", ("Skeleton_001_",)), "LIAS")

    def test_marker_name_index_keeps_only_unique_labels(self) -> None:
        self.assertEqual(marker_name_index(["A", "B", "A", "C"]), {"B": 1, "C": 3})


if __name__ == "__main__":
    unittest.main()
