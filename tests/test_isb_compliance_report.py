from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isb_compliance_report import (
    build_isb_d1_d6_report,
    build_reproducibility_manifest,
    load_curve_from_manifest,
    read_isb_d1_d6_table,
    validate_reproducibility_manifest,
    write_isb_d1_d6_report,
    write_reproducibility_manifest,
)
from isb_segment_audit import build_isb_d1_d3_audit
from kinematic_conventions import load_kinematic_conventions


class IsbComplianceReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_kinematic_conventions()
        self.d1_d3 = build_isb_d1_d3_audit(
            self.registry,
            biobuddy_template_loader=lambda: (_Template(), Path(__file__)),
        )

    def _trial_report(self) -> dict[str, object]:
        return {
            "trial": "Static",
            "models": {
                "captury": {"source_kind": "fbx"},
                "motive": {"source_kind": "bvh"},
                "biobuddy": {"status": "missing"},
            },
            "joint_kinematics_d4_d6": {
                "captury": {
                    "source_id": "captury_model",
                    "articulations": {
                        "right_hip": {
                            "D4_status": "blocked_missing_frame_corrections",
                            "D5_status": "non_applicable_no_joint_translation",
                            "D6_status": "non_applicable",
                            "matrix_status": "available_native_source_frames",
                            "euler_status": "unavailable",
                            "sequence": "XYZ",
                            "proximal_segment_id": "pelvis",
                            "distal_segment_id": "right_thigh",
                        }
                    },
                },
                "motive": {
                    "source_id": "motive_model",
                    "articulations": {},
                },
            },
            "captury_c3d_angle_decode": {
                "schema_version": 1,
                "source": "captury_c3d_angles",
                "decoded_channel_count": 1,
                "component_semantics_status": "unknown",
                "channels": [
                    {
                        "articulation": "right_hip",
                        "identity_decoded": True,
                        "rotation_sequence": None,
                        "eligible_for_anatomical_agreement": False,
                    }
                ],
            },
        }

    def test_report_covers_d1_d6_and_never_falls_back_between_sources(self) -> None:
        report = build_isb_d1_d6_report(
            self.d1_d3, [self._trial_report()], self.registry
        )

        self.assertEqual(
            set(report["sources"]),
            {
                "captury_model",
                "captury_c3d_angles",
                "motive_model",
                "biobuddy_motive57",
            },
        )
        self.assertEqual(
            {row["deviation_id"] for row in report["rows"]},
            {"D1", "D2", "D3", "D4", "D5", "D6"},
        )
        biobuddy_d4 = next(
            row
            for row in report["rows"]
            if row["trial_id"] == "Static"
            and row["source_id"] == "biobuddy_motive57"
            and row["entity_id"] == "right_hip"
            and row["deviation_id"] == "D4"
        )
        self.assertEqual(biobuddy_d4["status_class"], "unavailable")
        self.assertIn("missing_source", biobuddy_d4["blocker_codes"])
        self.assertNotEqual(biobuddy_d4["raw_status"], "captury_fallback")

        captury_c3d_d4 = next(
            row
            for row in report["rows"]
            if row["trial_id"] == "Static"
            and row["source_id"] == "captury_c3d_angles"
            and row["entity_id"] == "right_hip"
            and row["deviation_id"] == "D4"
        )
        self.assertEqual(captury_c3d_d4["status_class"], "unknown")
        self.assertIn("unknown_euler_sequence", captury_c3d_d4["blocker_codes"])

    def test_blocked_d4_and_non_applicable_d5_are_distinct(self) -> None:
        report = build_isb_d1_d6_report(
            self.d1_d3, [self._trial_report()], self.registry
        )
        rows = {
            row["deviation_id"]: row
            for row in report["rows"]
            if row["trial_id"] == "Static"
            and row["source_id"] == "captury_model"
            and row["entity_id"] == "right_hip"
        }

        self.assertEqual(rows["D4"]["status_class"], "blocked")
        self.assertTrue(rows["D4"]["blocking"])
        self.assertEqual(rows["D5"]["status_class"], "not_applicable")
        self.assertFalse(rows["D5"]["blocking"])

    def test_d4_target_evaluation_remains_blocked_by_unharmonized_segment_frames(
        self,
    ) -> None:
        trial = self._trial_report()
        articulation = trial["joint_kinematics_d4_d6"]["captury"]["articulations"][
            "right_hip"
        ]
        articulation.update(
            {
                "D4_status": "jcs_target_evaluated",
                "euler_status": "available",
            }
        )

        report = build_isb_d1_d6_report(self.d1_d3, [trial], self.registry)
        row = next(
            item
            for item in report["rows"]
            if item["trial_id"] == "Static"
            and item["source_id"] == "captury_model"
            and item["entity_id"] == "right_hip"
            and item["deviation_id"] == "D4"
        )

        self.assertEqual(row["status_class"], "conformant")
        self.assertTrue(row["blocking"])
        self.assertIn("upstream_segment_frames_not_harmonized", row["blocker_codes"])

    def test_d4_blocker_uses_canonical_segment_ids_not_exporter_names(self) -> None:
        trial = self._trial_report()
        articulation = trial["joint_kinematics_d4_d6"]["captury"]["articulations"][
            "right_hip"
        ]
        articulation.update(
            {
                "D4_status": "jcs_target_evaluated",
                "euler_status": "available",
                "proximal": "Hips",
                "distal": "RightUpLeg",
                "proximal_segment_id": "pelvis",
                "distal_segment_id": "right_thigh",
            }
        )
        d1_d3 = {
            **self.d1_d3,
            "segments": [
                {
                    **row,
                    "D1_status": "conforme",
                    "D2_status": "conforme",
                    "D3_status": "conforme",
                }
                for row in self.d1_d3["segments"]
            ],
        }

        report = build_isb_d1_d6_report(d1_d3, [trial], self.registry)
        row = next(
            item
            for item in report["rows"]
            if item["trial_id"] == "Static"
            and item["source_id"] == "captury_model"
            and item["entity_id"] == "right_hip"
            and item["deviation_id"] == "D4"
        )

        self.assertFalse(row["blocking"])
        self.assertNotIn("upstream_segment_frames_not_harmonized", row["blocker_codes"])

    def test_d4_without_canonical_endpoints_is_blocked(self) -> None:
        trial = self._trial_report()
        trial["joint_kinematics_d4_d6"]["captury"]["articulations"]["custom_joint"] = {
            "D4_status": "jcs_target_evaluated",
            "D5_status": "non_applicable_no_joint_translation",
            "D6_status": "non_applicable",
            "euler_status": "available",
            "proximal": "NativeA",
            "distal": "NativeB",
        }

        report = build_isb_d1_d6_report(self.d1_d3, [trial], self.registry)
        row = next(
            item
            for item in report["rows"]
            if item["trial_id"] == "Static"
            and item["source_id"] == "captury_model"
            and item["entity_id"] == "custom_joint"
            and item["deviation_id"] == "D4"
        )

        self.assertTrue(row["blocking"])
        self.assertIn("missing_canonical_segment_endpoints", row["blocker_codes"])

    def test_json_and_npz_roundtrip_preserve_normalized_rows(self) -> None:
        report = build_isb_d1_d6_report(
            self.d1_d3, [self._trial_report()], self.registry
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_isb_d1_d6_report(Path(tmp), report)
            loaded_json = json.loads(paths["json"].read_text(encoding="utf-8"))
            loaded_rows = read_isb_d1_d6_table(paths["table"])
            with np.load(paths["table"], allow_pickle=False) as arrays:
                dtypes = [arrays[key].dtype for key in arrays.files]

        self.assertEqual(loaded_json["schema_version"], 1)
        self.assertEqual(len(loaded_rows), len(report["rows"]))
        self.assertTrue(all(dtype != object for dtype in dtypes))
        for key in ("source_id", "trial_id", "evidence", "blocker_codes", "details"):
            self.assertEqual(loaded_rows[0][key], report["rows"][0][key])


class ReproducibilityManifestTests(unittest.TestCase):
    def _write_table(self, path: Path) -> None:
        np.savez_compressed(
            path,
            columns=np.asarray(["time_s", "joint", "error_mm"], dtype=str),
            col_0=np.asarray([0.0, 0.5, 1.0]),
            col_1=np.asarray(["Hip", "Hip", "Hip"], dtype=str),
            col_2=np.asarray([1.0, 2.0, 3.0]),
        )

    def test_manifest_reconstructs_curve_without_gui_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_dir = root / "Static"
            trial_dir.mkdir()
            series_path = trial_dir / "joint_centre_timeseries.npz"
            self._write_table(series_path)
            reports = [
                {
                    "trial": "Static",
                    "outputs": {
                        "joint_centre_timeseries": str(series_path),
                        "spatial_calibration": str(
                            trial_dir / "spatial_calibration.json"
                        ),
                    },
                    "spatial_calibration": {"status": "available"},
                    "temporal_synchronization": {"status": "not_requested"},
                }
            ]
            (trial_dir / "spatial_calibration.json").write_text(
                json.dumps({"schema_version": 1, "status": "available"}),
                encoding="utf-8",
            )

            manifest = build_reproducibility_manifest(root, reports)
            path = write_reproducibility_manifest(root, manifest)
            loaded = load_curve_from_manifest(
                path,
                "Static/joint_centre_timeseries/error_mm",
                filters={"joint": "Hip"},
            )

        np.testing.assert_allclose(loaded["x"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(loaded["y"], [1.0, 2.0, 3.0])
        self.assertEqual(loaded["filters"]["joint"].tolist(), ["Hip"] * 3)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["validation"]["status"], "valid")
        recipe = next(
            item
            for item in manifest["curve_recipes"]
            if item["curve_id"] == "Static/joint_centre_timeseries/error_mm"
        )
        self.assertEqual(recipe["spatial_alignment_ref"], "Static/spatial_calibration")
        self.assertEqual(
            manifest["transformations"]["Static"]["spatial"],
            "Static/spatial_calibration",
        )

    def test_missing_artifact_is_reported_but_never_referenced_by_a_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "Static" / "missing.npz"
            manifest = build_reproducibility_manifest(
                root,
                [
                    {
                        "trial": "Static",
                        "outputs": {"joint_centre_timeseries": str(missing)},
                    }
                ],
            )

        self.assertEqual(manifest["curve_recipes"], [])
        self.assertEqual(manifest["omitted_artifacts"][0]["reason"], "missing")
        self.assertEqual(validate_reproducibility_manifest(manifest), [])

    def test_hierarchical_joint_kinematics_npz_is_reconstructible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_dir = root / "Static"
            trial_dir.mkdir()
            path = trial_dir / "joint_kinematics_d4_d6.npz"
            np.savez_compressed(
                path,
                **{
                    "captury/right_hip/time": np.asarray([0.0, 0.5]),
                    "captury/right_hip/angles_rad": np.asarray(
                        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    ),
                    "captury/right_hip/relative_rotation_matrix": np.repeat(
                        np.eye(3)[:, :, None], 2, axis=2
                    ),
                },
            )
            manifest = build_reproducibility_manifest(
                root,
                [
                    {
                        "trial": "Static",
                        "outputs": {"joint_kinematics_d4_d6_timeseries": str(path)},
                    }
                ],
            )
            manifest_path = write_reproducibility_manifest(root, manifest)
            curve = load_curve_from_manifest(
                manifest_path,
                "Static/joint_kinematics_d4_d6_timeseries/captury/right_hip/angles_rad[1]",
            )

        np.testing.assert_allclose(curve["x"], [0.0, 0.5])
        np.testing.assert_allclose(curve["y"], [0.3, 0.4])
        self.assertEqual(curve["y_unit"], "rad")

    def test_validation_checks_files_hashes_and_transformation_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_dir = root / "Static"
            trial_dir.mkdir()
            series = trial_dir / "series.npz"
            self._write_table(series)
            manifest = build_reproducibility_manifest(
                root,
                [
                    {
                        "trial": "Static",
                        "outputs": {"joint_centre_timeseries": str(series)},
                    }
                ],
            )
            artifact = manifest["artifacts"]["Static/joint_centre_timeseries"]
            artifact["sha256"] = "0" * 64
            manifest["transformations"]["Static"]["spatial"] = "Static/missing"

            errors = validate_reproducibility_manifest(manifest, base_dir=root)

        self.assertTrue(any(error.startswith("hash_mismatch:") for error in errors))
        self.assertIn("missing_transformation:Static/missing", errors)

    def test_validation_checks_each_curve_transformation_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_dir = root / "Static"
            trial_dir.mkdir()
            series = trial_dir / "series.npz"
            self._write_table(series)
            manifest = build_reproducibility_manifest(
                root,
                [
                    {
                        "trial": "Static",
                        "outputs": {"joint_centre_timeseries": str(series)},
                    }
                ],
            )
            manifest["curve_recipes"][0]["spatial_alignment_ref"] = "Static/missing"

            errors = validate_reproducibility_manifest(manifest, base_dir=root)

        self.assertIn(
            "missing_recipe_transformation:Static/missing",
            errors,
        )

    def test_bundle_can_be_moved_and_curve_still_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            original = parent / "original"
            moved = parent / "moved"
            trial_dir = original / "Static"
            trial_dir.mkdir(parents=True)
            series = trial_dir / "series.npz"
            self._write_table(series)
            manifest = build_reproducibility_manifest(
                original,
                [
                    {
                        "trial": "Static",
                        "outputs": {"joint_centre_timeseries": str(series)},
                    }
                ],
            )
            write_reproducibility_manifest(original, manifest)
            shutil.copytree(original, moved)

            curve = load_curve_from_manifest(
                moved / "comparison_reproducibility_manifest.json",
                "Static/joint_centre_timeseries/error_mm",
                filters={"joint": "Hip"},
            )

        np.testing.assert_allclose(curve["y"], [1.0, 2.0, 3.0])

    def test_csv_metrics_and_explicit_filters_are_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "all_model_dimensions.csv"
            csv_path.write_text(
                "trial,system,dimension,median_length_mm\n"
                "Static,captury,left_thigh,400\n"
                "Static,motive,left_thigh,410\n",
                encoding="utf-8",
            )

            manifest = build_reproducibility_manifest(root, [])

        artifact = manifest["artifacts"]["Batch/all_model_dimensions"]
        self.assertEqual(artifact["media_type"], "text/csv")
        filtered = [
            recipe
            for recipe in manifest["curve_recipes"]
            if recipe["y_key"] == "median_length_mm"
            and {"system": "captury", "trial": "Static", "dimension": "left_thigh"}
            in recipe.get("filter_combinations", [])
        ]
        self.assertTrue(filtered)
        self.assertLess(len(manifest["curve_recipes"]), 10)


class _Template:
    name = "minimal"
    segments: tuple[object, ...] = ()


if __name__ == "__main__":
    unittest.main()
