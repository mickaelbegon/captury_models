from __future__ import annotations

import unittest
import json

import numpy as np

from captury_c3d_angles import (
    CAPTURY_ANGLE_CHANNELS,
    CAPTURY_ANGLE_REGISTRY_PATH,
    analyze_captury_angle_channels,
    resolve_captury_angle_unit,
)


def synthetic_c3d(
    values: np.ndarray,
    *,
    angle_names: list[str] | None = None,
    point_labels: list[str] | None = None,
    angle_unit: str | None = None,
) -> dict:
    point = {
        "LABELS": {
            "value": point_labels
            or [channel["point_label"] for channel in CAPTURY_ANGLE_CHANNELS]
        },
        "ANGLES": {
            "value": angle_names
            or [channel["parameter_name"] for channel in CAPTURY_ANGLE_CHANNELS]
        },
        # POINT:UNITS describes physical point coordinates and must not be
        # mistaken for an angle unit.
        "UNITS": {"value": ["mm"]},
    }
    if angle_unit is not None:
        point["ANGLE_UNITS"] = {"value": [angle_unit]}
    points = np.ones((4, values.shape[1], values.shape[2]), dtype=float)
    points[:3, :, :] = values
    return {"parameters": {"POINT": point}, "data": {"points": points}}


class CapturyC3dAngleTests(unittest.TestCase):
    def test_versioned_registry_contains_the_runtime_channel_mapping(self) -> None:
        registry = json.loads(CAPTURY_ANGLE_REGISTRY_PATH.read_text(encoding="utf-8"))

        self.assertEqual(registry["schema_version"], 1)
        self.assertEqual(tuple(registry["channels"]), CAPTURY_ANGLE_CHANNELS)
        self.assertFalse(registry["eligible_for_anatomical_agreement"])

    def test_mapping_resolves_all_thirteen_channels_without_decoding_axes(self) -> None:
        values = np.zeros((3, len(CAPTURY_ANGLE_CHANNELS), 5))

        report = analyze_captury_angle_channels(
            synthetic_c3d(values, angle_unit="deg"), requested_unit="auto"
        )

        self.assertEqual(report["channel_count"], 13)
        self.assertEqual(
            [channel["articulation"] for channel in report["channels"]],
            [channel["articulation"] for channel in CAPTURY_ANGLE_CHANNELS],
        )
        self.assertTrue(
            all(channel["identity_decoded"] for channel in report["channels"])
        )
        self.assertTrue(
            all(
                not channel["component_semantics_decoded"]
                for channel in report["channels"]
            )
        )
        self.assertTrue(
            all(
                not channel["eligible_for_anatomical_agreement"]
                for channel in report["channels"]
            )
        )

    def test_angle_unit_ignores_point_units_and_requires_metadata_or_override(
        self,
    ) -> None:
        unknown = resolve_captury_angle_unit(
            synthetic_c3d(np.zeros((3, 13, 2))), requested_unit="auto"
        )
        metadata = resolve_captury_angle_unit(
            synthetic_c3d(np.zeros((3, 13, 2)), angle_unit="radians"),
            requested_unit="auto",
        )
        override = resolve_captury_angle_unit(
            synthetic_c3d(np.zeros((3, 13, 2))), requested_unit="deg"
        )

        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["unit"])
        self.assertEqual(metadata["unit"], "rad")
        self.assertEqual(metadata["source"], "POINT:ANGLE_UNITS")
        self.assertEqual(override["unit"], "deg")
        self.assertEqual(override["source"], "cli_override")

    def test_radian_data_are_diagnosed_in_degrees(self) -> None:
        values = np.zeros((3, 13, 4))
        values[0, 0, :] = np.deg2rad([0.0, 30.0, 60.0, 90.0])

        report = analyze_captury_angle_channels(
            synthetic_c3d(values, angle_unit="rad"), requested_unit="auto"
        )

        component = report["channels"][0]["components"][0]
        self.assertAlmostEqual(component["range_deg"], 90.0)
        self.assertAlmostEqual(component["max_abs_step_deg"], 30.0)

    def test_diagnostics_find_constant_duplicate_discontinuous_and_out_of_range(
        self,
    ) -> None:
        values = np.zeros((3, 13, 5))
        values[0, 0, :] = [0.0, 10.0, 20.0, -170.0, -160.0]
        values[0, 1, :] = values[0, 0, :]
        values[1, 2, :] = 7.0
        values[2, 3, :] = [0.0, 50.0, 100.0, 150.0, 210.0]

        report = analyze_captury_angle_channels(
            synthetic_c3d(values, angle_unit="deg"), requested_unit="auto"
        )

        duplicate_members = {
            member
            for group in report["exact_duplicate_component_groups"]
            for member in group["members"]
        }
        self.assertIn("right_hip/X", duplicate_members)
        self.assertIn("left_hip/X", duplicate_members)
        self.assertTrue(report["channels"][2]["components"][1]["constant"])
        self.assertTrue(report["channels"][0]["components"][0]["discontinuous"])
        self.assertTrue(report["channels"][3]["components"][2]["outside_euler_range"])

    def test_uniplanar_motion_is_reported_without_assigning_anatomical_axis(
        self,
    ) -> None:
        values = np.zeros((3, 13, 8))
        values[1, 4, :] = np.linspace(0.0, 80.0, 8)
        values[0, 4, :] = np.linspace(0.0, 2.0, 8)

        report = analyze_captury_angle_channels(
            synthetic_c3d(values, angle_unit="deg"), requested_unit="auto"
        )

        ankle = report["channels"][4]
        self.assertEqual(ankle["observed_dominant_component"], "Y")
        self.assertTrue(ankle["observed_uniplanar"])
        self.assertFalse(ankle["component_semantics_decoded"])

    def test_mismatched_tail_label_does_not_assign_an_articulation(self) -> None:
        values = np.zeros((3, len(CAPTURY_ANGLE_CHANNELS), 5))
        labels = [channel["point_label"] for channel in CAPTURY_ANGLE_CHANNELS]
        labels[0] = "Unexpected"

        report = analyze_captury_angle_channels(
            synthetic_c3d(values, point_labels=labels, angle_unit="deg"),
            requested_unit="auto",
        )

        self.assertFalse(report["channels"][0]["identity_decoded"])
        self.assertIsNone(report["channels"][0]["articulation"])

    def test_missing_channels_do_not_form_false_nan_duplicate_groups(self) -> None:
        values = np.zeros((3, 1, 5))
        c3d = synthetic_c3d(
            values,
            angle_names=[],
            point_labels=["RHip"],
            angle_unit="deg",
        )

        report = analyze_captury_angle_channels(c3d, requested_unit="auto")

        self.assertEqual(report["observed_channel_count"], 1)
        self.assertEqual(report["exact_duplicate_component_groups"], [])


if __name__ == "__main__":
    unittest.main()
