from __future__ import annotations

import unittest

import numpy as np

from c3d_trial_viewer import (
    camera_matrix_for_plane,
    camera_matrix_for_subject_view,
    default_camera_matrix,
    fit_center_and_scale,
    point_unit_scale_to_mm,
    project_points,
    rotation_matrix_from_drag,
)


class C3DTrialViewerCoreTests(unittest.TestCase):
    def test_point_unit_scale_to_mm_handles_common_c3d_units(self) -> None:
        self.assertEqual(point_unit_scale_to_mm("mm"), 1.0)
        self.assertEqual(point_unit_scale_to_mm("cm"), 10.0)
        self.assertEqual(point_unit_scale_to_mm("m"), 1000.0)

    def test_projection_maps_center_to_widget_center(self) -> None:
        points = np.asarray([[10.0], [20.0], [30.0]])
        screen, _depth = project_points(
            points,
            default_camera_matrix(),
            points[:, 0],
            scale=2.0,
            width=800,
            height=600,
        )

        np.testing.assert_allclose(screen[:, 0], [400.0, 300.0])

    def test_drag_rotation_matrix_is_orthonormal(self) -> None:
        matrix = rotation_matrix_from_drag(15.0, -8.0)

        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)

    def test_plane_and_subject_views_return_camera_matrices(self) -> None:
        points = np.asarray(
            [
                [-100.0, 100.0, -100.0, 100.0],
                [-50.0, -50.0, 50.0, 50.0],
                [0.0, 0.0, 1000.0, 1000.0],
            ]
        )

        self.assertEqual(camera_matrix_for_plane("XY").shape, (3, 3))
        self.assertEqual(camera_matrix_for_plane("XZ").shape, (3, 3))
        self.assertEqual(camera_matrix_for_subject_view("face", points).shape, (3, 3))

    def test_subject_view_pca_has_deterministic_axis_sign(self) -> None:
        points = np.asarray(
            [
                [-200.0, 200.0, -200.0, 200.0],
                [-50.0, -50.0, 50.0, 50.0],
                [0.0, 0.0, 100.0, 100.0],
            ]
        )

        forward = camera_matrix_for_subject_view("face", points)
        reversed_points = points[:, ::-1]
        reversed_forward = camera_matrix_for_subject_view("face", reversed_points)

        np.testing.assert_allclose(forward, reversed_forward)

    def test_fit_scale_uses_finite_points_only(self) -> None:
        points = np.asarray([[0.0, 100.0, np.nan], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        center, scale = fit_center_and_scale(
            points, 400, 300, camera_matrix_for_plane("XY")
        )

        np.testing.assert_allclose(center, [50.0, 0.0, 0.0])
        self.assertGreater(scale, 0)


if __name__ == "__main__":
    unittest.main()
