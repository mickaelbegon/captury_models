from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinematic_conventions import (
    DEFAULT_KINEMATIC_CONVENTIONS_PATH,
    ConventionValidationError,
    build_provenance_manifest,
    final_comparison_blockers,
    load_kinematic_conventions,
    segment_source_names,
    validate_kinematic_conventions,
)


class KinematicConventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_kinematic_conventions()

    def test_default_registry_is_versioned_and_valid(self) -> None:
        self.assertTrue(DEFAULT_KINEMATIC_CONVENTIONS_PATH.is_file())
        self.assertEqual(self.registry["schema_version"], 1)
        validate_kinematic_conventions(self.registry)

    def test_registry_covers_all_four_kinematic_representations(self) -> None:
        self.assertEqual(
            set(self.registry["sources"]),
            {
                "captury_model",
                "captury_c3d_angles",
                "motive_model",
                "biobuddy_motive57",
            },
        )
        captury_channels = {
            articulation["channel_label"]
            for articulation in self.registry["sources"]["captury_c3d_angles"][
                "articulations"
            ].values()
        }
        self.assertEqual(
            captury_channels,
            {
                "RHipAngles",
                "LHipAngles",
                "RKneeAngles",
                "LKneeAngles",
                "RAnkleAngles",
                "LAnkleAngles",
                "RShoulderAngles",
                "LShoulderAngles",
                "RElbowAngles",
                "LElbowAngles",
                "RWristAngles",
                "LWristAngles",
                "Neck",
            },
        )

    def test_source_inheritance_is_resolved_before_validation(self) -> None:
        registry = load_kinematic_conventions()
        motive = registry["sources"]["motive_model"]

        self.assertNotIn("segments_from", motive)
        self.assertEqual(
            set(motive["segments"]),
            set(registry["sources"]["captury_model"]["segments"]),
        )
        self.assertEqual(
            set(motive["articulations"]),
            set(registry["sources"]["captury_model"]["articulations"]),
        )
        motive["segments"]["pelvis"]["source_name"] = "MotivePelvis"
        self.assertEqual(
            registry["sources"]["captury_model"]["segments"]["pelvis"]["source_name"],
            "Hips",
        )
        self.assertEqual(motive["length_unit"]["value"], "cm")
        self.assertEqual(
            registry["sources"]["captury_model"]["length_unit"]["value"],
            "mm",
        )

    def test_segment_source_names_resolve_format_specific_motive_names(self) -> None:
        motive_bvh = segment_source_names(self.registry, "motive_model", "bvh")
        motive_fbx = segment_source_names(self.registry, "motive_model", "fbx")
        captury_fbx = segment_source_names(self.registry, "captury_model", "fbx")

        self.assertEqual(motive_bvh["pelvis"], "Hips")
        self.assertEqual(motive_fbx["pelvis"], "pelvis")
        self.assertEqual(motive_fbx["thorax"], "spine_04")
        self.assertEqual(motive_fbx["left_thigh"], "thigh_l")
        self.assertEqual(captury_fbx["left_thigh"], "LeftUpLeg")

    def test_format_specific_segment_names_must_reference_known_segments(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["sources"]["motive_model"]["segment_names_by_format"] = {
            "fbx": {"unknown_segment": "pelvis"}
        }

        with self.assertRaisesRegex(
            ConventionValidationError, "references unknown segment"
        ):
            validate_kinematic_conventions(registry)

    def test_format_specific_segment_names_must_be_unique(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["sources"]["motive_model"]["segment_names_by_format"] = {
            "fbx": {"pelvis": "same", "thorax": "same"}
        }

        with self.assertRaisesRegex(
            ConventionValidationError, "duplicate format-specific segment name"
        ):
            validate_kinematic_conventions(registry)

    def test_empty_source_is_rejected_after_resolution(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["sources"]["motive_model"]["segments"] = {}

        with self.assertRaisesRegex(
            ConventionValidationError, "segments must not be empty"
        ):
            validate_kinematic_conventions(registry)

    def test_duplicate_segment_alias_is_rejected_within_one_source(self) -> None:
        registry = copy.deepcopy(self.registry)
        segments = registry["sources"]["biobuddy_motive57"]["segments"]
        segments["left_thigh"]["aliases"].append(segments["right_thigh"]["source_name"])

        with self.assertRaisesRegex(
            ConventionValidationError, "duplicate segment name or alias"
        ):
            validate_kinematic_conventions(registry)

    def test_segment_parent_cycle_is_rejected(self) -> None:
        registry = copy.deepcopy(self.registry)
        segments = registry["sources"]["biobuddy_motive57"]["segments"]
        segments["pelvis"]["parent"] = "left_thigh"

        with self.assertRaisesRegex(ConventionValidationError, "cycle"):
            validate_kinematic_conventions(registry)

    def test_invalid_source_to_isb_rotation_is_rejected(self) -> None:
        registry = copy.deepcopy(self.registry)
        segment = registry["sources"]["biobuddy_motive57"]["segments"]["pelvis"]
        segment["source_to_isb_rotation"] = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        with self.assertRaisesRegex(ConventionValidationError, "orthonormal rotation"):
            validate_kinematic_conventions(registry)

    def test_deviation_or_conformity_requires_evidence(self) -> None:
        registry = copy.deepcopy(self.registry)
        hip = registry["sources"]["biobuddy_motive57"]["articulations"]["right_hip"]
        hip["isb_audit"]["D4"] = {"status": "conforme", "evidence": []}

        with self.assertRaisesRegex(ConventionValidationError, "requires evidence"):
            validate_kinematic_conventions(registry)

    def test_unknown_conventions_block_final_angular_comparison(self) -> None:
        blockers = final_comparison_blockers(
            self.registry,
            source_id="captury_c3d_angles",
            articulation_id="right_hip",
        )

        self.assertIn("rotation_sequence.status=inconnu", blockers)
        self.assertIn("isb_audit.D4=inconnu", blockers)

    def test_isb_matrix_alone_does_not_resolve_frame_definition(self) -> None:
        registry = copy.deepcopy(self.registry)
        source = registry["sources"]["captury_model"]
        articulation = source["articulations"]["right_hip"]
        articulation["rotation_sequence"] = {
            "status": "conforme",
            "evidence": ["synthetic test"],
        }
        articulation["anatomical_components"] = {
            "status": "conforme",
            "evidence": ["synthetic test"],
        }
        articulation["isb_audit"]["D4"] = {
            "status": "conforme",
            "evidence": ["synthetic test"],
        }
        for segment_id in ("pelvis", "right_thigh"):
            source["segments"][segment_id]["source_to_isb_rotation"] = np.eye(
                3
            ).tolist()

        blockers = final_comparison_blockers(
            registry, source_id="captury_model", articulation_id="right_hip"
        )

        self.assertIn("segments.pelvis.frame_definition=inconnu", blockers)
        self.assertIn("segments.pelvis.isb_audit.D1=inconnu", blockers)
        self.assertIn("segments.pelvis.isb_audit.D2=inconnu", blockers)
        self.assertIn("segments.pelvis.isb_audit.D3=inconnu", blockers)

    def test_provenance_manifest_hashes_inputs_and_records_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trial.c3d"
            source.write_bytes(b"c3d-test-payload")
            derived = Path(tmp) / "generated.bioMod"
            derived.write_bytes(b"generated-model")

            manifest = build_provenance_manifest(
                registry_path=DEFAULT_KINEMATIC_CONVENTIONS_PATH,
                input_files={"captury_c3d": source},
                derived_artifacts={"Static/captury/biomod": derived},
                command=["python", "compare.py", "--trial", "Static"],
            )

        self.assertEqual(manifest["manifest_version"], 1)
        self.assertEqual(manifest["inputs"]["captury_c3d"]["size_bytes"], 16)
        self.assertEqual(
            manifest["inputs"]["captury_c3d"]["sha256"],
            "a0dfc707a37db6d8ea61013717a2bfde0f67744d825279f2c1d71565b09f8c2b",
        )
        self.assertIn("python", manifest["runtime"])
        self.assertEqual(
            manifest["matrix_convention"], self.registry["matrix_convention"]
        )
        self.assertEqual(manifest["comparison_readiness"]["status"], "diagnostic_only")
        self.assertIn(
            "captury_c3d_angles/right_hip",
            manifest["comparison_readiness"]["blockers"],
        )
        self.assertIn("packages", manifest["runtime"])
        self.assertEqual(
            manifest["derived_artifacts"]["Static/captury/biomod"]["size_bytes"],
            15,
        )
        json.dumps(manifest)

    def test_known_correction_rotation_must_have_positive_determinant(self) -> None:
        registry = copy.deepcopy(self.registry)
        segment = registry["sources"]["biobuddy_motive57"]["segments"]["pelvis"]
        segment["source_to_isb_rotation"] = np.diag([1.0, 1.0, -1.0]).tolist()

        with self.assertRaisesRegex(ConventionValidationError, "determinant"):
            validate_kinematic_conventions(registry)


if __name__ == "__main__":
    unittest.main()
