from __future__ import annotations

import unittest

import numpy as np

from spatial_calibration import (
    RowRigidTransform,
    SpatialCalibration,
    fit_held_out_centre_alignment,
)


def centre_series(point: np.ndarray, n_frames: int = 3) -> np.ndarray:
    return np.repeat(np.asarray(point, dtype=float)[:, None], n_frames, axis=1)


class SpatialCalibrationTests(unittest.TestCase):
    def test_known_transform_is_recovered_from_reserved_centres(self) -> None:
        angle = np.deg2rad(25.0)
        rotation = np.asarray(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [-np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.asarray([120.0, -35.0, 80.0])
        moving_points = {
            "Hips": [0.0, 0.0, 0.0],
            "Head": [0.0, 0.0, 700.0],
            "LeftShoulder": [-180.0, 0.0, 500.0],
            "RightShoulder": [180.0, 0.0, 500.0],
            "LeftLeg": [-90.0, 25.0, -420.0],
            "RightLeg": [90.0, -20.0, -415.0],
        }
        moving = {name: centre_series(point) for name, point in moving_points.items()}
        reference = {
            name: centre_series(np.asarray(point) @ rotation + translation)
            for name, point in moving_points.items()
        }

        transform, report = fit_held_out_centre_alignment(
            moving,
            reference,
            ("Hips", "Head", "LeftShoulder", "RightShoulder"),
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["evaluation_centres"], ["LeftLeg", "RightLeg"])
        np.testing.assert_allclose(transform.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(transform.translation, translation, atol=1e-12)
        self.assertLess(report["calibration_after"]["max_mm"], 1e-10)
        self.assertLess(report["evaluation_after"]["max_mm"], 1e-10)

    def test_reserved_centres_are_never_evaluation_centres(self) -> None:
        points = {
            "A": centre_series([0.0, 0.0, 0.0]),
            "B": centre_series([1.0, 0.0, 0.0]),
            "C": centre_series([0.0, 1.0, 0.0]),
            "D": centre_series([0.0, 0.0, 1.0]),
            "E": centre_series([1.0, 1.0, 1.0]),
        }

        _transform, report = fit_held_out_centre_alignment(
            points, points, ("A", "B", "C", "D")
        )

        self.assertEqual(report["calibration_centres"], ["A", "B", "C", "D"])
        self.assertEqual(report["evaluation_centres"], ["E"])
        self.assertTrue(
            set(report["calibration_centres"]).isdisjoint(report["evaluation_centres"])
        )

    def test_degenerate_calibration_geometry_is_blocked(self) -> None:
        collinear = {
            name: centre_series([float(index), 0.0, 0.0])
            for index, name in enumerate(("A", "B", "C", "D", "E"))
        }

        transform, report = fit_held_out_centre_alignment(
            collinear, collinear, ("A", "B", "C", "D")
        )

        self.assertEqual(report["status"], "degenerate_calibration_geometry")
        np.testing.assert_allclose(transform.rotation, np.eye(3))
        np.testing.assert_allclose(transform.translation, 0.0)

    def test_shared_rigid_transform_preserves_pairwise_distances(self) -> None:
        transform = RowRigidTransform(
            rotation=np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            translation=np.asarray([30.0, 40.0, 50.0]),
        )
        first = np.asarray([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
        second = np.asarray([[9.0, 3.0, -2.0], [7.0, 6.0, 5.0]])

        before = np.linalg.norm(first - second, axis=1)
        after = np.linalg.norm(
            transform.apply_rows(first) - transform.apply_rows(second), axis=1
        )

        np.testing.assert_allclose(after, before, atol=1e-12)

    def test_spatial_calibration_roundtrip_preserves_ordered_stages(self) -> None:
        calibration = SpatialCalibration(
            static_trial="Static",
            calibration_centres=("Hips", "Head", "LeftShoulder", "RightShoulder"),
            evaluation_centres=("LeftLeg", "RightLeg"),
            captury_to_motive=RowRigidTransform(np.eye(3), np.asarray([1.0, 2.0, 3.0])),
            motive_to_c3d=RowRigidTransform(np.eye(3), np.asarray([4.0, 5.0, 6.0])),
            status="ok",
            captury_root_offset_mode="keep",
            motive_root_offset_mode="subtract",
        )

        restored = SpatialCalibration.from_dict(calibration.to_dict())
        composed = restored.captury_to_c3d()

        self.assertEqual(restored.static_trial, "Static")
        self.assertEqual(restored.calibration_centres, calibration.calibration_centres)
        self.assertEqual(restored.captury_root_offset_mode, "keep")
        self.assertEqual(restored.motive_root_offset_mode, "subtract")
        np.testing.assert_allclose(composed.translation, [5.0, 7.0, 9.0])
        self.assertEqual(
            calibration.to_dict()["ordered_stages"],
            [
                "root_translation_policy_in_native_q",
                "forward_kinematics_in_native_model_frame",
                "source_native_to_c3d_mm",
                "captury_to_motive_static",
                "motive_to_c3d_static",
            ],
        )

    def test_transform_composition_matches_sequential_non_commuting_stages(
        self,
    ) -> None:
        first = RowRigidTransform(
            np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.asarray([10.0, 20.0, 30.0]),
        )
        second = RowRigidTransform(
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
            np.asarray([-4.0, 7.0, 11.0]),
        )
        points = np.asarray([[2.0, 3.0, 5.0], [-1.0, 8.0, 13.0]])

        sequential = second.apply_rows(first.apply_rows(points))
        composed = first.compose(second).apply_rows(points)

        np.testing.assert_allclose(composed, sequential, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
