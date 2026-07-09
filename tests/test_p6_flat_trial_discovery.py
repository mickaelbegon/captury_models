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
        compose_row_alignment,
        discover_flat_trials,
        dimension_rows_from_centres,
        euler_matrix_from_sequence,
        euler_zxy_from_matrix,
        file_fingerprint,
        marker_proxy_centres_from_c3d,
        marker_indices_by_clean_label,
        model_to_c3d_matrix,
        motive_flat_trial_name,
        occlusion_rows_from_points,
        orient_segment_y_from_cor,
        propose_marker_correspondences_from_points,
        reexpress_rotational_q_from_segment_rotations,
        required_trial_outputs,
        resolve_cut_window,
        rotation_deviation_vector,
        rotate_segment_frames_180_x,
        root_alignment_score_mm,
        sanitize_channel_name,
        segment_relative_rotation_curves,
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
    compose_row_alignment = None
    discover_flat_trials = None
    dimension_rows_from_centres = None
    euler_matrix_from_sequence = None
    euler_zxy_from_matrix = None
    file_fingerprint = None
    marker_proxy_centres_from_c3d = None
    marker_indices_by_clean_label = None
    model_to_c3d_matrix = None
    motive_flat_trial_name = None
    occlusion_rows_from_points = None
    orient_segment_y_from_cor = None
    propose_marker_correspondences_from_points = None
    reexpress_rotational_q_from_segment_rotations = None
    required_trial_outputs = None
    resolve_cut_window = None
    rotation_deviation_vector = None
    rotate_segment_frames_180_x = None
    root_alignment_score_mm = None
    sanitize_channel_name = None
    segment_relative_rotation_curves = None
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
        self.assertIn("skin_marker_correspondence_timeseries.npz", outputs)
        self.assertNotIn("joint_centre_timeseries.csv", outputs)
        self.assertNotIn("kinematics_q_timeseries.csv", outputs)
        self.assertNotIn("segment_rotation_timeseries.csv", outputs)

    def test_dimension_rows_from_centres_supports_biobuddy_source(self) -> None:
        assert dimension_rows_from_centres is not None

        centres_mm = {
            "LThigh": np.asarray([[0.0], [0.0], [0.0]]),
            "LShank": np.asarray([[0.0], [400.0], [0.0]]),
        }

        rows = dimension_rows_from_centres(
            "Static", "biobuddy", "motive_57", centres_mm
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["system"], "biobuddy")
        self.assertEqual(rows[0]["source_kind"], "motive_57")
        self.assertEqual(rows[0]["dimension"], "left_thigh")
        self.assertAlmostEqual(rows[0]["median_length_mm"], 400.0)

    def test_marker_indices_support_duplicate_number_suffixes(self) -> None:
        assert marker_indices_by_clean_label is not None

        lookup = marker_indices_by_clean_label(["Q_Hip", "Q_Knee", "Q_Hip"])

        self.assertEqual(lookup["Q_Hip"], [0, 2])
        self.assertEqual(lookup["Q_Hip#1"], [0])
        self.assertEqual(lookup["Q_Hip#2"], [2])

    def test_compose_row_alignment_matches_sequential_transforms(self) -> None:
        assert compose_row_alignment is not None

        first_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        first_translation = np.asarray([10.0, 20.0, 30.0])
        second_rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        )
        second_translation = np.asarray([-5.0, 2.0, 8.0])
        points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        rotation, translation = compose_row_alignment(
            first_rotation, first_translation, second_rotation, second_translation
        )

        sequential = (
            points @ first_rotation + first_translation
        ) @ second_rotation + second_translation
        composed = points @ rotation + translation
        np.testing.assert_allclose(composed, sequential)

    def test_marker_correspondence_proposal_pairs_nearest_aligned_skin_markers(
        self,
    ) -> None:
        assert propose_marker_correspondences_from_points is not None

        time = np.asarray([0.0, 1.0])
        motive_points = np.asarray(
            [
                [[0.0, 0.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        captury_points = motive_points[:, [0, 1, 3], :].copy()
        captury_points[:, 0, :] += 500.0
        pairs, report = propose_marker_correspondences_from_points(
            ["Skeleton_001_LIAS", "Skeleton_001_RIAS", "CAPJC_Hips", "LANK"],
            motive_points,
            time,
            ["Q_A", "Q_B", "Q_C"],
            captury_points,
            time,
            np.eye(3),
            np.zeros(3),
            max_median_error_mm=1.0,
        )

        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(
            [(row["reference"][0], row["test"][0]) for row in pairs],
            [("RIAS", "Q_B"), ("LANK", "Q_C")],
        )

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

    def test_rotate_segment_frames_180_x_changes_local_x_and_y_axes(self) -> None:
        assert rotate_segment_frames_180_x is not None

        corrected = rotate_segment_frames_180_x({"Thigh": np.eye(3)[:, :, None]})[
            "Thigh"
        ]

        np.testing.assert_allclose(corrected[:, :, 0], np.diag([-1.0, -1.0, 1.0]))

    def test_euler_zxy_roundtrip_from_rotation_matrix(self) -> None:
        assert euler_matrix_from_sequence is not None
        assert euler_zxy_from_matrix is not None

        expected = np.deg2rad([20.0, -10.0, 35.0])
        rotation = euler_matrix_from_sequence(expected, "ZXY")

        actual = euler_zxy_from_matrix(rotation)

        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(
            euler_matrix_from_sequence(actual, "ZXY"), rotation, atol=1e-10
        )

    def test_reexpress_rotational_q_uses_corrected_segment_matrix(self) -> None:
        assert euler_matrix_from_sequence is not None
        assert reexpress_rotational_q_from_segment_rotations is not None

        q_names = ["Thigh_rotZ", "Thigh_rotX", "Thigh_rotY", "Thigh_transX"]
        original_q = np.asarray([[0.0], [0.0], [0.0], [123.0]])
        original_rotation = euler_matrix_from_sequence(
            np.deg2rad([5.0, 8.0, 11.0]), "ZXY"
        )
        corrected_rotation = original_rotation @ np.diag([-1.0, -1.0, 1.0])

        q, report = reexpress_rotational_q_from_segment_rotations(
            original_q,
            q_names,
            {"Thigh": corrected_rotation[:, :, None]},
            "ZXY",
        )

        np.testing.assert_allclose(
            euler_matrix_from_sequence(q[:3, 0], "ZXY"),
            corrected_rotation,
            atol=1e-10,
        )
        self.assertEqual(float(q[3, 0]), 123.0)
        self.assertEqual(report["changed_segments"], ["Thigh"])

    def test_orient_segment_y_from_cor_uses_proximal_to_distal_direction(
        self,
    ) -> None:
        assert orient_segment_y_from_cor is not None

        rotations = {"LeftUpLeg": np.eye(3)[:, :, None]}
        centres = {
            "LeftUpLeg": np.asarray([[0.0], [0.0], [0.0]]),
            "LeftLeg": np.asarray([[0.0], [0.0], [2.0]]),
        }

        corrected = orient_segment_y_from_cor(
            rotations, centres, "LeftUpLeg", "LeftUpLeg", "LeftLeg"
        )

        np.testing.assert_allclose(
            corrected["LeftUpLeg"][:, 1, 0], [0.0, 0.0, 1.0], atol=1e-10
        )

    def test_segment_relative_rotation_curves_extract_joint_rotation(self) -> None:
        assert segment_relative_rotation_curves is not None

        angle = np.deg2rad(15.0)
        rotation_z = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        curves = segment_relative_rotation_curves(
            {
                "LeftUpLeg": np.eye(3)[:, :, None],
                "LeftLeg": rotation_z[:, :, None],
            }
        )

        np.testing.assert_allclose(
            curves["SegRel_LeftKnee"][:, 0], [0.0, 0.0, angle], atol=1e-10
        )

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
                biobuddy_biomod=None,
                biobuddy_unit_scale_to_m=1.0,
                root_offset_mode="auto",
                angle_label_regex="angle",
                c3d_angle_unit="deg",
                landmark_map=None,
                segment_reference="biobuddy",
                captury_reorient_thigh_y_from_cor=False,
                rotate_body_segments_180_x=False,
                reexpress_rotations_zxy=False,
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
            args.captury_reorient_thigh_y_from_cor = True
            fourth = trial_cache_fingerprint(bundle, args)
            args.captury_reorient_thigh_y_from_cor = False
            args.reexpress_rotations_zxy = True
            fifth = trial_cache_fingerprint(bundle, args)
            args.reexpress_rotations_zxy = False
            args.root_offset_mode = "keep"
            third = trial_cache_fingerprint(bundle, args)
            args.root_offset_mode = "auto"
            paths["motive.c3d"].write_text("changed", encoding="utf-8")
            second = trial_cache_fingerprint(bundle, args)

            self.assertNotEqual(first["digest"], second["digest"])
            self.assertNotEqual(first["digest"], third["digest"])
            self.assertNotEqual(first["digest"], fourth["digest"])
            self.assertNotEqual(first["digest"], fifth["digest"])

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
