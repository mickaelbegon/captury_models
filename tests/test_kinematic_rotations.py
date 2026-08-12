from __future__ import annotations

import unittest

import numpy as np

from kinematic_rotations import (
    RotationSeriesValidationError,
    assess_rotation_source_equivalence,
    canonicalize_rotation_series,
    canonicalize_segment_rotation_mapping,
    change_lab_basis,
    compare_segment_rotation_mappings,
    relative_rotation_series,
    rotation_geodesic_degrees,
    rotation_vector,
    slerp_rotation_series,
    validate_rotation_series,
)


def rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class KinematicRotationTests(unittest.TestCase):
    def test_validation_rejects_reflections_and_non_orthogonal_frames(self) -> None:
        reflection = np.diag([1.0, 1.0, -1.0])[:, :, None]
        with self.assertRaisesRegex(RotationSeriesValidationError, "determinant"):
            validate_rotation_series(reflection)

        scaled = np.diag([1.0, 2.0, 1.0])[:, :, None]
        with self.assertRaisesRegex(RotationSeriesValidationError, "orthonormal"):
            validate_rotation_series(scaled)

    def test_validation_accepts_biorbd_scale_roundoff_but_rejects_drift(self) -> None:
        roundoff = np.diag([1.0 + 2.0e-7, 1.0, 1.0])[:, :, None]
        validate_rotation_series(roundoff)

        drift = np.diag([1.0 + 1.0e-4, 1.0, 1.0])[:, :, None]
        with self.assertRaisesRegex(RotationSeriesValidationError, "orthonormal"):
            validate_rotation_series(drift)

    def test_canonicalization_projects_small_fbx_roundoff_and_reports_it(self) -> None:
        raw = np.diag([1.0 + 2.5e-6, 1.0 - 1.0e-6, 1.0])[:, :, None]

        canonical, report = canonicalize_rotation_series(raw)

        validate_rotation_series(canonical, atol=1e-12)
        np.testing.assert_allclose(canonical[:, :, 0], np.eye(3), atol=1e-12)
        self.assertGreater(report["max_raw_orthonormal_error"], 1e-6)
        self.assertLess(report["max_projection_frobenius"], 1e-5)

    def test_canonicalization_rejects_reflection_and_large_drift(self) -> None:
        with self.assertRaisesRegex(RotationSeriesValidationError, "reflection"):
            canonicalize_rotation_series(np.diag([1.0, 1.0, -1.0])[:, :, None])

        with self.assertRaisesRegex(RotationSeriesValidationError, "too far"):
            canonicalize_rotation_series(
                np.diag([1.01, 1.0, 1.0])[:, :, None],
                max_raw_orthonormal_error=1e-4,
            )

    def test_change_lab_basis_left_multiplies_column_axis_frames(self) -> None:
        source = rotation_x(np.deg2rad(20.0))[:, :, None]
        basis = rotation_z(np.deg2rad(90.0))

        converted = change_lab_basis(source, basis)

        np.testing.assert_allclose(converted[:, :, 0], basis @ source[:, :, 0])
        validate_rotation_series(converted)

    def test_relative_rotation_is_invariant_to_common_lab_basis_change(self) -> None:
        proximal = rotation_z(np.deg2rad(15.0))[:, :, None]
        distal = (proximal[:, :, 0] @ rotation_x(np.deg2rad(30.0)))[:, :, None]
        basis = rotation_x(np.deg2rad(90.0))

        native = relative_rotation_series(proximal, distal)
        converted = relative_rotation_series(
            change_lab_basis(proximal, basis), change_lab_basis(distal, basis)
        )

        np.testing.assert_allclose(native, converted, atol=1e-12)
        np.testing.assert_allclose(
            native[:, :, 0], rotation_x(np.deg2rad(30.0)), atol=1e-12
        )

    def test_rotation_vector_is_stable_at_180_degrees(self) -> None:
        vector = rotation_vector(np.eye(3), rotation_x(np.pi))

        self.assertAlmostEqual(np.linalg.norm(vector), np.pi, places=12)
        np.testing.assert_allclose(np.abs(vector), [np.pi, 0.0, 0.0], atol=1e-12)

    def test_slerp_interpolates_rotation_without_leaving_so3(self) -> None:
        rotations = np.stack((np.eye(3), rotation_z(np.pi / 2.0)), axis=2)

        interpolated = slerp_rotation_series(
            rotations,
            source_time=np.asarray([0.0, 1.0]),
            target_time=np.asarray([0.0, 0.5, 1.0]),
        )

        validate_rotation_series(interpolated)
        np.testing.assert_allclose(
            interpolated[:, :, 1], rotation_z(np.pi / 4.0), atol=1e-12
        )

    def test_mapping_comparison_reports_geodesic_deviation_per_segment(self) -> None:
        reference = {
            "Pelvis": np.repeat(np.eye(3)[:, :, None], 3, axis=2),
            "Thigh": np.repeat(np.eye(3)[:, :, None], 3, axis=2),
        }
        test = {
            "Pelvis": reference["Pelvis"].copy(),
            "Thigh": np.stack(
                (rotation_x(0.0), rotation_x(0.1), rotation_x(0.2)), axis=2
            ),
            "Foot": np.repeat(np.eye(3)[:, :, None], 3, axis=2),
        }

        result = compare_segment_rotation_mappings(
            reference,
            np.asarray([0.0, 0.5, 1.0]),
            test,
            np.asarray([0.0, 0.5, 1.0]),
        )

        self.assertEqual(result["common_segments"], ["Pelvis", "Thigh"])
        self.assertEqual(result["missing_in_reference"], ["Foot"])
        self.assertEqual(result["missing_in_test"], [])
        self.assertAlmostEqual(
            result["summary"]["Thigh"]["max_geodesic_deg"], np.rad2deg(0.2)
        )
        np.testing.assert_allclose(
            result["timeseries"]["Pelvis"]["geodesic_deg"], 0.0, atol=1e-12
        )

    def test_mapping_comparison_uses_elapsed_time_from_each_export(self) -> None:
        rotations = {
            "pelvis": np.stack(
                (rotation_x(0.0), rotation_x(0.1), rotation_x(0.2)), axis=2
            )
        }

        result = compare_segment_rotation_mappings(
            rotations,
            np.asarray([10.0, 10.5, 11.0]),
            rotations,
            np.asarray([20.0, 20.5, 21.0]),
        )

        np.testing.assert_allclose(result["timeseries"]["pelvis"]["time"], [0, 0.5, 1])
        np.testing.assert_allclose(
            result["timeseries"]["pelvis"]["geodesic_deg"], 0.0, atol=1e-12
        )

    def test_mapping_comparison_rejects_non_monotonic_reference_time(self) -> None:
        rotations = {"pelvis": np.repeat(np.eye(3)[:, :, None], 3, axis=2)}

        with self.assertRaisesRegex(ValueError, "reference_time.*strictly increasing"):
            compare_segment_rotation_mappings(
                rotations,
                np.asarray([0.0, 1.0, 0.5]),
                rotations,
                np.asarray([0.0, 0.5, 1.0]),
            )

    def test_segment_mapping_uses_canonical_ids_and_reports_unmapped_names(
        self,
    ) -> None:
        raw = {
            "pelvis": np.repeat(np.eye(3)[:, :, None], 2, axis=2),
            "thigh_l": np.repeat(rotation_x(0.1)[:, :, None], 2, axis=2),
            "finger_l": np.repeat(np.eye(3)[:, :, None], 2, axis=2),
        }

        mapped, report = canonicalize_segment_rotation_mapping(
            raw,
            {
                "pelvis": "pelvis",
                "left_thigh": "thigh_l",
                "left_shank": "calf_l",
            },
        )

        self.assertEqual(set(mapped), {"pelvis", "left_thigh"})
        self.assertEqual(report["missing_source_names"], ["calf_l"])
        self.assertEqual(report["unmapped_source_names"], ["finger_l"])

    def test_segment_mapping_rejects_one_source_name_for_two_segments(self) -> None:
        raw = {"same": np.eye(3)[:, :, None]}

        with self.assertRaisesRegex(ValueError, "mapped more than once"):
            canonicalize_segment_rotation_mapping(
                raw, {"pelvis": "same", "thorax": "same"}
            )

    def test_equivalence_gate_blocks_large_or_incomplete_bvh_fbx_differences(
        self,
    ) -> None:
        comparison = {
            "common_segments": ["pelvis", "left_thigh"],
            "missing_in_reference": [],
            "missing_in_test": ["left_shank"],
            "summary": {
                "pelvis": {"p95_geodesic_deg": 0.5},
                "left_thigh": {"p95_geodesic_deg": 18.0},
            },
        }

        verdict = assess_rotation_source_equivalence(
            comparison, max_p95_geodesic_deg=5.0
        )

        self.assertEqual(verdict["status"], "blocked")
        self.assertTrue(verdict["blocks_automatic_source_selection"])
        self.assertEqual(verdict["segments_over_tolerance"], ["left_thigh"])
        self.assertIn("left_shank", verdict["missing_segments"])

    def test_equivalence_gate_accepts_complete_low_deviation_comparison(self) -> None:
        comparison = {
            "common_segments": ["pelvis"],
            "missing_in_reference": [],
            "missing_in_test": [],
            "summary": {"pelvis": {"p95_geodesic_deg": 0.5}},
        }

        verdict = assess_rotation_source_equivalence(
            comparison, max_p95_geodesic_deg=5.0
        )

        self.assertEqual(verdict["status"], "equivalent_within_tolerance")
        self.assertFalse(verdict["blocks_automatic_source_selection"])

    def test_equivalence_gate_rejects_non_finite_tolerance(self) -> None:
        comparison = {
            "common_segments": ["pelvis"],
            "missing_in_reference": [],
            "missing_in_test": [],
            "summary": {"pelvis": {"p95_geodesic_deg": 0.5}},
        }

        for tolerance in (np.nan, np.inf):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "finite"):
                    assess_rotation_source_equivalence(
                        comparison, max_p95_geodesic_deg=tolerance
                    )

    def test_equivalence_gate_blocks_incomplete_summary(self) -> None:
        comparison = {
            "common_segments": ["pelvis", "left_thigh"],
            "missing_in_reference": [],
            "missing_in_test": [],
            "summary": {"pelvis": {"p95_geodesic_deg": 0.5}},
        }

        verdict = assess_rotation_source_equivalence(
            comparison, max_p95_geodesic_deg=5.0
        )

        self.assertEqual(verdict["status"], "blocked")
        self.assertEqual(verdict["segments_without_metrics"], ["left_thigh"])

    def test_geodesic_distance_is_invariant_to_lab_basis(self) -> None:
        reference = rotation_x(np.deg2rad(10.0))
        test = rotation_x(np.deg2rad(25.0))
        basis = rotation_z(np.deg2rad(80.0))

        native = rotation_geodesic_degrees(reference, test)
        converted = rotation_geodesic_degrees(basis @ reference, basis @ test)

        self.assertAlmostEqual(native, 15.0, places=12)
        self.assertAlmostEqual(converted, native, places=12)


if __name__ == "__main__":
    unittest.main()
