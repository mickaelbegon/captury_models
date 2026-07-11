from __future__ import annotations

import unittest

from c3d_point_channels import DEFAULT_C3D_ANGLE_LABELS, classify_c3d_point_channels
from compare_capture_systems import detect_angle_indices


def synthetic_c3d(*, point_angles: list[str] | None = None) -> dict:
    point = {
        "LABELS": {"value": ["LASI", "RHip", "Knee Angle", "Custom"]},
        "DESCRIPTIONS": {
            "value": ["pelvis", "joint angle", "technical marker", "derived angle"]
        },
    }
    if point_angles is not None:
        point["ANGLES"] = {"value": point_angles}
    return {"parameters": {"POINT": point}}


class C3DPointChannelTests(unittest.TestCase):
    def test_classification_combines_parameter_label_and_description_rules(
        self,
    ) -> None:
        result = classify_c3d_point_channels(
            synthetic_c3d(point_angles=["Custom"]),
            angle_label_regex=r"(?i)angle",
        )

        self.assertEqual(result.marker_indices, [0])
        self.assertEqual(result.angle_indices, [1, 2, 3])
        self.assertEqual(result.marker_labels, ["LASI"])
        self.assertEqual(result.angle_labels, ["RHip", "Knee Angle", "Custom"])

    def test_explicit_angle_parameter_matches_labels_independently_of_order(
        self,
    ) -> None:
        c3d = synthetic_c3d(point_angles=["LASI"])

        result = classify_c3d_point_channels(c3d, angle_label_regex="")

        self.assertEqual(result.angle_indices, [0, 1])
        self.assertEqual(result.marker_indices, [2, 3])

    def test_extra_angle_labels_accept_compact_whitespace_variant(self) -> None:
        result = classify_c3d_point_channels(
            synthetic_c3d(),
            angle_label_regex="",
            extra_angle_labels=["KneeAngle"],
        )

        self.assertEqual(result.angle_indices, [1, 2])
        self.assertEqual(result.marker_indices, [0, 3])

    def test_known_captury_labels_remain_angle_channels(self) -> None:
        self.assertIn("RHip", DEFAULT_C3D_ANGLE_LABELS)
        self.assertIn("Neck", DEFAULT_C3D_ANGLE_LABELS)

        result = classify_c3d_point_channels(synthetic_c3d(), angle_label_regex="")

        self.assertEqual(result.angle_labels, ["RHip"])
        self.assertEqual(result.marker_labels, ["LASI", "Knee Angle", "Custom"])

    def test_optional_point_angles_tail_fallback_preserves_legacy_exports(self) -> None:
        result = classify_c3d_point_channels(
            synthetic_c3d(point_angles=["Hip flexion"]),
            angle_label_regex="",
            point_angles_tail_fallback=True,
        )

        self.assertEqual(result.angle_indices, [1, 3])
        self.assertEqual(result.marker_indices, [0, 2])

    def test_capture_comparison_preserves_legacy_angle_parameter_name(self) -> None:
        c3d = synthetic_c3d(point_angles=["Hip flexion"])

        indices = detect_angle_indices(
            c3d,
            c3d["parameters"]["POINT"]["LABELS"]["value"],
            angle_label_regex="",
        )

        self.assertEqual(indices["hip_flexion"], 3)
        self.assertEqual(indices["right_hip"], 1)


if __name__ == "__main__":
    unittest.main()
