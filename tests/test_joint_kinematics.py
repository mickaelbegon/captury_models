from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from joint_kinematics import (
    JointKinematicsError,
    apply_anatomical_frame_correction,
    audit_joint_kinematics_source,
    classify_shoulder_motion,
    euler_matrix,
    express_joint_translation_in_proximal,
    extract_jcs_series,
    frame_corrections_from_static_audit,
    extract_euler_series,
    joint_rotation_series,
    load_isb_joint_kinematics,
    write_joint_kinematics_audit,
)


class JointKinematicsTests(unittest.TestCase):
    def test_isb_joint_registry_is_versioned_and_explicit(self) -> None:
        registry = load_isb_joint_kinematics()

        self.assertEqual(registry["schema_version"], 1)
        self.assertEqual(registry["matrix_convention"], "column_vectors")
        self.assertIn("aligned", registry["zero_definition"])
        self.assertEqual(registry["articulations"]["hip"]["sequence"], "ZXY")
        self.assertEqual(registry["articulations"]["glenohumeral"]["sequence"], "YXY")
        self.assertEqual(
            registry["articulations"]["thoracohumeral"]["D6_status"],
            "deviation",
        )
        for articulation in registry["articulations"].values():
            self.assertIn(articulation["type"], {"Cardan", "Euler", "JCS"})
            self.assertTrue(articulation["proximal"])
            self.assertTrue(articulation["distal"])
            self.assertEqual(len(articulation["components"]), 3)
            self.assertEqual(len(articulation["positive_signs"]), 3)
            self.assertRegex(articulation["citation"]["doi"], r"^10\.")

    def test_elementary_and_combined_zxy_roundtrip(self) -> None:
        angles = np.deg2rad(
            [
                [20.0, 0.0, 0.0],
                [0.0, -15.0, 0.0],
                [0.0, 0.0, 35.0],
                [20.0, -15.0, 35.0],
            ]
        )
        rotations = np.moveaxis(
            np.asarray([euler_matrix(frame, "ZXY") for frame in angles]), 0, 2
        )

        extracted = extract_euler_series(rotations, "ZXY")

        np.testing.assert_allclose(extracted["angles_rad"].T, angles, atol=1e-10)
        self.assertLess(extracted["max_roundtrip_geodesic_deg"], 1e-10)
        self.assertFalse(np.any(extracted["near_singular"]))

    def test_zxy_jcs_axes_match_fixed_floating_and_distal_axes(self) -> None:
        angles = np.deg2rad([25.0, -12.0, 18.0])
        relative = euler_matrix(angles, "ZXY")[:, :, None]

        extracted = extract_jcs_series(relative, "ZXY")

        expected_fixed = np.asarray([0.0, 0.0, 1.0])
        expected_distal = relative[:, 1, 0]
        expected_floating = np.cross(expected_distal, expected_fixed)
        expected_floating /= np.linalg.norm(expected_floating)
        np.testing.assert_allclose(
            extracted["fixed_proximal_axis"][:, 0], expected_fixed
        )
        np.testing.assert_allclose(
            extracted["fixed_distal_axis"][:, 0], expected_distal
        )
        np.testing.assert_allclose(extracted["floating_axis"][:, 0], expected_floating)
        np.testing.assert_allclose(extracted["angles_rad"][:, 0], angles)

    def test_jcs_elementary_components_have_explicit_positive_signs(self) -> None:
        for component in range(3):
            angles = np.zeros(3)
            angles[component] = np.deg2rad(10.0)
            extracted = extract_jcs_series(
                euler_matrix(angles, "ZXY")[:, :, None], "ZXY"
            )
            self.assertAlmostEqual(
                extracted["angles_rad"][component, 0], angles[component]
            )
            other = np.delete(extracted["angles_rad"][:, 0], component)
            np.testing.assert_allclose(other, 0.0, atol=1e-12)

    def test_proper_euler_yxy_roundtrip(self) -> None:
        angles = np.deg2rad([30.0, 40.0, -20.0])
        rotation = euler_matrix(angles, "YXY")[:, :, None]

        extracted = extract_euler_series(rotation, "YXY")

        np.testing.assert_allclose(extracted["angles_rad"][:, 0], angles, atol=1e-10)
        self.assertLess(extracted["roundtrip_geodesic_deg"][0], 1e-10)

    def test_singularity_is_flagged_without_hiding_matrix_result(self) -> None:
        rotations = np.stack(
            (
                euler_matrix(np.deg2rad([10.0, 89.8, 20.0]), "ZXY"),
                euler_matrix(np.deg2rad([10.0, 90.0, 20.0]), "ZXY"),
            ),
            axis=2,
        )

        extracted = extract_euler_series(
            rotations, "ZXY", near_singularity_margin_deg=0.5
        )

        np.testing.assert_array_equal(extracted["near_singular"], [True, True])
        np.testing.assert_array_equal(extracted["singular"], [False, True])
        np.testing.assert_array_equal(
            extracted["euler_components_reliable"],
            [[False, False], [False, False], [False, False]],
        )
        self.assertEqual(
            extracted["branch_policy"],
            "scipy_canonical_branch_then_independent_component_unwrap",
        )
        self.assertLess(extracted["max_roundtrip_geodesic_deg"], 1e-8)

    def test_unwrap_preserves_continuous_rotation_across_180_degrees(self) -> None:
        angles_deg = np.asarray([170.0, 175.0, 179.0, 181.0, 185.0, 190.0])
        rotations = np.stack(
            [
                euler_matrix(np.deg2rad([angle, 10.0, 5.0]), "ZXY")
                for angle in angles_deg
            ],
            axis=2,
        )

        extracted = extract_euler_series(rotations, "ZXY", unwrap=True)

        np.testing.assert_allclose(
            np.rad2deg(extracted["angles_rad"][0]), angles_deg, atol=1e-10
        )

    def test_joint_rotation_applies_distinct_proximal_and_distal_corrections(
        self,
    ) -> None:
        proximal = np.eye(3)[:, :, None]
        distal = euler_matrix(np.deg2rad([15.0, 5.0, -8.0]), "ZXY")[:, :, None]
        proximal_correction = euler_matrix(np.deg2rad([5.0, 0.0, 0.0]), "ZXY")
        distal_correction = euler_matrix(np.deg2rad([-3.0, 2.0, 0.0]), "ZXY")

        relative = joint_rotation_series(
            proximal,
            distal,
            proximal_anatomical_from_source=proximal_correction,
            distal_anatomical_from_source=distal_correction,
        )
        expected = (proximal[:, :, 0] @ proximal_correction).T @ (
            distal[:, :, 0] @ distal_correction
        )

        np.testing.assert_allclose(relative[:, :, 0], expected, atol=1e-10)

    def test_frame_correction_is_a_right_multiplication(self) -> None:
        source = euler_matrix(np.deg2rad([12.0, 4.0, 8.0]), "ZXY")[:, :, None]
        correction = euler_matrix(np.deg2rad([0.0, 0.0, 90.0]), "ZXY")

        corrected = apply_anatomical_frame_correction(source, correction)

        np.testing.assert_allclose(corrected[:, :, 0], source[:, :, 0] @ correction)

    def test_left_right_signs_are_explicit_registry_data(self) -> None:
        registry = load_isb_joint_kinematics()
        shoulder = registry["articulations"]["thoracohumeral"]

        self.assertEqual(shoulder["laterality"]["right"]["component_signs"], [1, 1, 1])
        self.assertEqual(
            shoulder["laterality"]["left"]["preprocessing"],
            "mirror_positions_across_sagittal_plane_before_frame_construction",
        )

    def test_glenohumeral_requires_scapula(self) -> None:
        self.assertEqual(
            classify_shoulder_motion("thorax", "upper_arm", {"thorax", "upper_arm"}),
            {"motion": "thoracohumeral", "D6_status": "deviation"},
        )
        with self.assertRaisesRegex(JointKinematicsError, "scapula"):
            classify_shoulder_motion("scapula", "upper_arm", {"thorax", "upper_arm"})

    def test_d5_translation_is_expressed_in_proximal_frame(self) -> None:
        proximal_rotation = euler_matrix(np.deg2rad([90.0, 0.0, 0.0]), "ZXY")[
            :, :, None
        ]
        proximal_point = np.asarray([[1.0], [2.0], [3.0]])
        distal_point = proximal_point + np.asarray([[0.0], [1.0], [0.0]])

        translation = express_joint_translation_in_proximal(
            proximal_rotation, proximal_point, distal_point
        )

        np.testing.assert_allclose(translation[:, 0], [1.0, 0.0, 0.0], atol=1e-10)

    def test_registry_rejects_implicit_left_signs(self) -> None:
        registry = load_isb_joint_kinematics()
        del registry["articulations"]["hip"]["laterality"]["left"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(JointKinematicsError, "laterality.left"):
                load_isb_joint_kinematics(path)

    def test_source_audit_keeps_native_so3_when_anatomical_frames_are_unknown(
        self,
    ) -> None:
        rotations = {
            "pelvis": np.eye(3)[:, :, None],
            "right_thigh": euler_matrix(np.deg2rad([10.0, 5.0, -2.0]), "ZXY")[
                :, :, None
            ],
        }
        articulations = {"right_hip": {"proximal": "pelvis", "distal": "right_thigh"}}

        audit = audit_joint_kinematics_source(
            "captury_model",
            rotations,
            np.asarray([0.0]),
            articulations,
            {},
        )

        hip = audit["articulations"]["right_hip"]
        self.assertEqual(hip["D4_status"], "blocked_missing_frame_corrections")
        self.assertEqual(hip["euler_status"], "unavailable")
        self.assertEqual(hip["proximal"], "pelvis")
        self.assertEqual(hip["distal"], "right_thigh")
        self.assertIsNone(hip["proximal_segment_id"])
        self.assertIsNone(hip["distal_segment_id"])
        self.assertEqual(hip["laterality"], "right")
        self.assertEqual(hip["component_signs"], [1, 1, 1])
        self.assertEqual(hip["laterality_preprocessing"], "none")
        self.assertEqual(
            audit["timeseries"]["right_hip"]["relative_rotation_matrix"].shape,
            (3, 3, 1),
        )
        self.assertEqual(
            audit["timeseries"]["right_hip"]["quaternion_xyzw"].shape,
            (4, 1),
        )
        self.assertEqual(
            audit["timeseries"]["right_hip"]["rotation_magnitude_deg"].shape,
            (1,),
        )

    def test_source_audit_extracts_euler_only_with_both_frame_corrections(self) -> None:
        rotation = euler_matrix(np.deg2rad([10.0, 5.0, -2.0]), "ZXY")
        audit = audit_joint_kinematics_source(
            "biobuddy_motive57",
            {
                "pelvis": np.eye(3)[:, :, None],
                "right_thigh": rotation[:, :, None],
            },
            np.asarray([0.0]),
            {"right_hip": {"proximal": "pelvis", "distal": "right_thigh"}},
            {"pelvis": np.eye(3), "right_thigh": np.eye(3)},
            frame_correction_provenance={
                "path": "/tmp/model.isb_static.json",
                "biomod_sha256": "abc123",
            },
        )

        hip = audit["articulations"]["right_hip"]
        self.assertEqual(hip["euler_status"], "available")
        self.assertEqual(hip["sequence"], "ZXY")
        self.assertEqual(hip["unit"], "rad")
        self.assertEqual(hip["D4_status"], "jcs_target_evaluated")
        self.assertEqual(hip["proximal_frame_correction"], np.eye(3).tolist())
        self.assertEqual(hip["distal_frame_correction"], np.eye(3).tolist())
        self.assertEqual(hip["frame_correction_provenance"]["biomod_sha256"], "abc123")
        self.assertIn("aligned", hip["zero_definition"])
        self.assertEqual(
            hip["laterality_preprocessing_status"],
            "encoded_in_source_to_target_frame_correction",
        )
        np.testing.assert_allclose(
            audit["timeseries"]["right_hip"]["angles_rad"][:, 0],
            np.deg2rad([10.0, 5.0, -2.0]),
            atol=1e-10,
        )

        left_audit = audit_joint_kinematics_source(
            "biobuddy_motive57",
            {
                "pelvis": np.eye(3)[:, :, None],
                "left_thigh": rotation[:, :, None],
            },
            np.asarray([0.0]),
            {"left_hip": {"proximal": "pelvis", "distal": "left_thigh"}},
            {"pelvis": np.eye(3), "left_thigh": np.eye(3)},
        )
        left_hip = left_audit["articulations"]["left_hip"]
        self.assertEqual(left_hip["laterality"], "left")
        self.assertIn("left", left_hip["laterality_preprocessing"])

    def test_component_signs_are_applied_to_published_angles(self) -> None:
        registry = load_isb_joint_kinematics()
        registry["articulations"]["hip"]["laterality"]["left"]["component_signs"] = [
            -1,
            1,
            -1,
        ]
        rotation = euler_matrix(np.deg2rad([10.0, 5.0, -2.0]), "ZXY")

        audit = audit_joint_kinematics_source(
            "synthetic",
            {
                "pelvis": np.eye(3)[:, :, None],
                "left_thigh": rotation[:, :, None],
            },
            np.asarray([0.0]),
            {"left_hip": {"proximal": "pelvis", "distal": "left_thigh"}},
            {"pelvis": np.eye(3), "left_thigh": np.eye(3)},
            target_registry=registry,
        )

        np.testing.assert_allclose(
            np.rad2deg(audit["timeseries"]["left_hip"]["angles_rad"][:, 0]),
            [-10.0, 5.0, 2.0],
            atol=1e-10,
        )

    def test_joint_audit_writes_json_metadata_and_compressed_npz(self) -> None:
        audit = audit_joint_kinematics_source(
            "captury_model",
            {
                "thorax": np.eye(3)[:, :, None],
                "right_upper_arm": np.eye(3)[:, :, None],
            },
            np.asarray([0.0]),
            {
                "right_shoulder": {
                    "proximal": "thorax",
                    "distal": "right_upper_arm",
                }
            },
            {},
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_joint_kinematics_audit(Path(tmp), {"captury": audit})
            metadata = json.loads(paths["json"].read_text(encoding="utf-8"))
            with np.load(paths["timeseries"], allow_pickle=False) as arrays:
                keys = set(arrays.files)

        shoulder = metadata["sources"]["captury"]["articulations"]["right_shoulder"]
        self.assertEqual(shoulder["motion"], "thoracohumeral")
        self.assertEqual(shoulder["D6_status"], "deviation")
        self.assertIn("captury/right_shoulder/relative_rotation_matrix", keys)
        self.assertIn("captury/right_shoulder/quaternion_xyzw", keys)
        self.assertIn("captury/right_shoulder/rotation_magnitude_deg", keys)

    def test_static_audit_exposes_only_available_frame_corrections(self) -> None:
        correction = euler_matrix(np.deg2rad([2.0, 3.0, 4.0]), "ZXY")
        static_audit = {
            "segments": {
                "pelvis": {
                    "status": "available",
                    "source_segment_name": "Pelvis",
                    "source_to_target_rotation": correction.tolist(),
                },
                "thorax": {
                    "status": "unavailable",
                    "source_segment_name": "Thorax",
                },
            }
        }

        corrections = frame_corrections_from_static_audit(static_audit)

        self.assertEqual(set(corrections), {"Pelvis"})
        np.testing.assert_allclose(corrections["Pelvis"], correction)


if __name__ == "__main__":
    unittest.main()
