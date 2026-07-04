from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    from compare_p6_motive_captury import (
        TrialBundle,
        captury_flat_trial_name,
        c3d_angle_scale_to_deg,
        centre_metric_rows,
        discover_flat_trials,
        file_fingerprint,
        marker_proxy_centres_from_c3d,
        model_to_c3d_matrix,
        motive_flat_trial_name,
        occlusion_rows_from_points,
        required_trial_outputs,
        resolve_cut_window,
        rotation_deviation_vector,
        root_alignment_score_mm,
        sanitize_channel_name,
        static_transform_from_report,
        time_window_mask,
        trial_cache_fingerprint,
        yaw_alignment_rows,
    )
except ImportError as exc:  # pragma: no cover - depends on optional scientific env
    TrialBundle = None
    captury_flat_trial_name = None
    c3d_angle_scale_to_deg = None
    centre_metric_rows = None
    discover_flat_trials = None
    file_fingerprint = None
    marker_proxy_centres_from_c3d = None
    model_to_c3d_matrix = None
    motive_flat_trial_name = None
    occlusion_rows_from_points = None
    required_trial_outputs = None
    resolve_cut_window = None
    rotation_deviation_vector = None
    root_alignment_score_mm = None
    sanitize_channel_name = None
    static_transform_from_report = None
    time_window_mask = None
    trial_cache_fingerprint = None
    yaw_alignment_rows = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    IMPORT_ERROR is not None, f"optional dependencies missing: {IMPORT_ERROR}"
)
class FlatTrialDiscoveryTests(unittest.TestCase):
    def test_flat_trial_names_strip_system_specific_suffixes(self) -> None:
        assert captury_flat_trial_name is not None
        assert motive_flat_trial_name is not None

        self.assertEqual(captury_flat_trial_name(Path("Static_P6.c3d")), "Static")
        self.assertEqual(
            motive_flat_trial_name(Path("P6_Static_Skeleton 001.bvh")), "Static"
        )
        self.assertEqual(
            motive_flat_trial_name(Path("P6_Marche_001.c3d")), "Marche_001"
        )

    def test_required_trial_outputs_use_npz_for_timeseries(self) -> None:
        assert required_trial_outputs is not None

        outputs = [path.name for path in required_trial_outputs(Path("out"), "Static")]

        self.assertIn("joint_centre_timeseries.npz", outputs)
        self.assertIn("kinematics_q_timeseries.npz", outputs)
        self.assertIn("captury_c3d_angle_metrics.csv", outputs)
        self.assertIn("captury_c3d_angle_timeseries.npz", outputs)
        self.assertIn("segment_rotation_metrics.csv", outputs)
        self.assertIn("segment_rotation_timeseries.npz", outputs)
        self.assertNotIn("joint_centre_timeseries.csv", outputs)
        self.assertNotIn("kinematics_q_timeseries.csv", outputs)
        self.assertNotIn("segment_rotation_timeseries.csv", outputs)

    def test_rotation_deviation_vector_reports_axis_angle_components(self) -> None:
        assert rotation_deviation_vector is not None

        angle = np.deg2rad(10.0)
        rotation_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )

        vector = rotation_deviation_vector(np.eye(3), rotation_x)

        np.testing.assert_allclose(vector, [angle, 0.0, 0.0], atol=1e-10)

    def test_c3d_angle_scale_to_deg_handles_rad_and_deg(self) -> None:
        assert c3d_angle_scale_to_deg is not None

        self.assertEqual(c3d_angle_scale_to_deg("deg"), 1.0)
        self.assertAlmostEqual(c3d_angle_scale_to_deg("rad"), 180.0 / np.pi)

    def test_sanitize_channel_name_keeps_angle_q_names_valid(self) -> None:
        assert sanitize_channel_name is not None

        self.assertEqual(
            sanitize_channel_name("R Hip.angle", "fallback"), "R_Hip_angle"
        )
        self.assertEqual(sanitize_channel_name("123", "fallback"), "_123")
        self.assertEqual(sanitize_channel_name(" ", "fallback"), "fallback")

    def test_root_alignment_score_uses_nearest_marker_distances_in_mm(self) -> None:
        assert root_alignment_score_mm is not None

        time = np.asarray([0.0, 1.0])
        centres = {"Hips": np.asarray([[0.0, 10.0], [0.0, 0.0], [0.0, 0.0]])}
        markers = np.asarray(
            [
                [[1.0, 11.0], [100.0, 100.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )

        score = root_alignment_score_mm(centres, time, markers, time)

        self.assertAlmostEqual(score, 1.0)

    def test_discover_flat_trials_matches_only_complete_c3d_pairs(self) -> None:
        assert discover_flat_trials is not None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captury = root / "Captury"
            motive = root / "Motive"
            captury.mkdir()
            motive.mkdir()

            for suffix in ("c3d", "bvh", "fbx"):
                (captury / f"Static_P6.{suffix}").write_text(
                    "captury", encoding="utf-8"
                )
            (motive / "P6_Static.c3d").write_text("motive", encoding="utf-8")
            (motive / "P6_Static.fbx").write_text("motive", encoding="utf-8")
            (motive / "P6_Static_Skeleton 001.bvh").write_text(
                "motive", encoding="utf-8"
            )

            (captury / "MissingMotive_P6.c3d").write_text("captury", encoding="utf-8")
            (motive / "P6_MissingCaptury.c3d").write_text("motive", encoding="utf-8")

            trials = discover_flat_trials(root)

            self.assertEqual([trial.name for trial in trials], ["Static"])
            bundle = trials[0]
            self.assertEqual(bundle.captury_c3d.name, "Static_P6.c3d")
            self.assertEqual(bundle.captury_bvh.name, "Static_P6.bvh")
            self.assertEqual(bundle.captury_fbx.name, "Static_P6.fbx")
            self.assertEqual(bundle.motive_c3d.name, "P6_Static.c3d")
            self.assertEqual(bundle.motive_bvh.name, "P6_Static_Skeleton 001.bvh")
            self.assertEqual(bundle.motive_fbx.name, "P6_Static.fbx")

    def test_auto_axis_uses_current_y_up_to_z_up_conversion(self) -> None:
        assert model_to_c3d_matrix is not None

        np.testing.assert_allclose(
            model_to_c3d_matrix("auto"), model_to_c3d_matrix("y_up_to_z_up")
        )

    def test_time_window_mask_keeps_requested_seconds(self) -> None:
        assert time_window_mask is not None

        mask = time_window_mask(np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]), 0.5, 1.5)

        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_resolve_cut_window_can_use_detected_movement(self) -> None:
        assert resolve_cut_window is not None

        start, end, mode = resolve_cut_window(
            "movement",
            None,
            None,
            {"movement_start_time": 0.25, "movement_end_time": 1.5},
        )

        self.assertEqual((start, end, mode), (0.25, 1.5, "movement"))

    def test_trial_cache_fingerprint_changes_when_input_changes(self) -> None:
        assert TrialBundle is not None
        assert trial_cache_fingerprint is not None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in (
                "captury.c3d",
                "captury.bvh",
                "captury.fbx",
                "motive.c3d",
                "motive.bvh",
                "motive.fbx",
            ):
                paths[name] = root / name
                paths[name].write_text(name, encoding="utf-8")
            bundle = TrialBundle(
                name="Static",
                captury_c3d=paths["captury.c3d"],
                captury_bvh=paths["captury.bvh"],
                captury_fbx=paths["captury.fbx"],
                motive_c3d=paths["motive.c3d"],
                motive_bvh=paths["motive.bvh"],
                motive_fbx=paths["motive.fbx"],
            )
            args = argparse.Namespace(
                model_source="bvh",
                model_to_c3d_axis="auto",
                captury_unit_scale_to_m=None,
                motive_unit_scale_to_m=None,
                root_offset_mode="auto",
                angle_label_regex="angle",
                c3d_angle_unit="deg",
                segment_reference="biobuddy",
                disable_static_model_alignment=False,
                disable_motive_marker_alignment=False,
                joint_filter=[],
                no_mesh=True,
                max_mesh_points=0,
                run_ik_batch=False,
                ik_max_frames=0,
                cut_mode="manual",
                time_start=None,
                time_end=None,
                no_figures=True,
            )

            first = trial_cache_fingerprint(bundle, args)
            args.root_offset_mode = "keep"
            third = trial_cache_fingerprint(bundle, args)
            args.root_offset_mode = "auto"
            paths["motive.c3d"].write_text("changed", encoding="utf-8")
            second = trial_cache_fingerprint(bundle, args)

            self.assertNotEqual(first["digest"], second["digest"])
            self.assertNotEqual(first["digest"], third["digest"])

    def test_file_fingerprint_records_missing_files(self) -> None:
        assert file_fingerprint is not None

        with tempfile.TemporaryDirectory() as tmp:
            fingerprint = file_fingerprint(Path(tmp) / "missing.c3d")

        self.assertFalse(fingerprint["exists"])

    def test_static_transform_from_report_returns_numpy_arrays(self) -> None:
        assert static_transform_from_report is not None

        transform = static_transform_from_report(
            {
                "alignment": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_mm": [1.0, 2.0, 3.0],
                }
            }
        )

        self.assertIsNotNone(transform)
        rotation, translation = transform
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(translation, [1.0, 2.0, 3.0])

    def test_occlusion_rows_use_vectorized_missing_points_and_residuals(self) -> None:
        assert occlusion_rows_from_points is not None

        points = np.asarray(
            [
                [[0.0, np.nan, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            ]
        )
        residuals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

        rows = occlusion_rows_from_points(
            "Static", ["Skeleton_001_A", "B"], points, residuals
        )

        self.assertEqual(rows[0]["marker"], "A")
        self.assertEqual(rows[0]["missing_frames"], 2)
        self.assertAlmostEqual(rows[0]["missing_percent"], 100.0 * 2.0 / 3.0)
        self.assertEqual(rows[1]["missing_frames"], 0)

    def test_yaw_alignment_rows_recovers_vertical_axis_rotation(self) -> None:
        assert yaw_alignment_rows is not None

        moving = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 200.0, 0.0],
                [100.0, 200.0, 500.0],
            ]
        )
        angle = np.deg2rad(90.0)
        row_rotation = np.asarray(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [-np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.asarray([10.0, -20.0, 30.0])
        reference = moving @ row_rotation + translation

        rotation, recovered_translation, report = yaw_alignment_rows(reference, moving)

        self.assertEqual(report["status"], "ok")
        np.testing.assert_allclose(rotation, row_rotation, atol=1e-12)
        np.testing.assert_allclose(recovered_translation, translation, atol=1e-12)
        np.testing.assert_allclose(moving @ rotation + recovered_translation, reference)

    def test_marker_proxy_centres_from_c3d_uses_clean_motive_labels(self) -> None:
        assert marker_proxy_centres_from_c3d is not None

        labels = [
            "Skeleton_001_LFLE",
            "Skeleton_001_LFME",
            "Skeleton_001_RFLE",
            "Skeleton_001_RFME",
        ]
        points = np.asarray(
            [
                [[0.0, 2.0], [2.0, 4.0], [10.0, 12.0], [14.0, 16.0]],
                [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [14.0, 14.0]],
                [
                    [100.0, 100.0],
                    [102.0, 102.0],
                    [110.0, 110.0],
                    [114.0, 114.0],
                ],
            ]
        )

        proxies = marker_proxy_centres_from_c3d(labels, points)

        np.testing.assert_allclose(
            proxies["LeftLeg"], [[1.0, 3.0], [1.0, 1.0], [101.0, 101.0]]
        )
        np.testing.assert_allclose(
            proxies["RightLeg"], [[12.0, 14.0], [12.0, 12.0], [112.0, 112.0]]
        )

    def test_joint_filter_keeps_full_timeseries_for_visualization(self) -> None:
        assert centre_metric_rows is not None

        time = np.asarray([0.0, 1.0])
        captury = {
            "Hips": np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            "LeftLeg": np.asarray([[0.0, 1.0], [10.0, 10.0], [0.0, 0.0]]),
            "Spine": np.asarray([[0.0, 1.0], [20.0, 20.0], [0.0, 0.0]]),
        }
        motive = {
            "Hips": np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            "LeftLeg": np.asarray([[0.0, 1.0], [11.0, 11.0], [0.0, 0.0]]),
            "Spine": np.asarray([[0.0, 1.0], [21.0, 21.0], [0.0, 0.0]]),
        }

        summary_rows, timeseries_rows = centre_metric_rows(
            "Walk", captury, motive, time, time, ["Leg"]
        )

        self.assertEqual({row["joint"] for row in summary_rows}, {"LeftLeg"})
        self.assertEqual(
            {row["joint"] for row in timeseries_rows},
            {"Hips", "LeftLeg", "Spine"},
        )


if __name__ == "__main__":
    unittest.main()
