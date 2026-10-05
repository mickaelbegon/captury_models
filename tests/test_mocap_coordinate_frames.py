from __future__ import annotations

import unittest

import numpy as np

from mocap_coordinate_frames import (
    Y_UP_TO_Z_UP,
    resolve_system_coordinate_frames,
    transform_point_trajectories,
)


class MocapCoordinateFrameTests(unittest.TestCase):
    def test_auto_keeps_captury_model_in_its_own_y_up_c3d_frame(self) -> None:
        resolution = resolve_system_coordinate_frames("captury", "auto")

        self.assertEqual(resolution.own_c3d_vertical_axis, "y")
        np.testing.assert_allclose(resolution.model_to_own_c3d, np.eye(3))
        np.testing.assert_allclose(resolution.own_c3d_to_common, Y_UP_TO_Z_UP)
        np.testing.assert_allclose(
            resolution.model_to_common_c3d,
            Y_UP_TO_Z_UP,
        )

    def test_auto_rotates_motive_model_to_its_z_up_c3d_frame(self) -> None:
        resolution = resolve_system_coordinate_frames("motive", "auto")

        self.assertEqual(resolution.own_c3d_vertical_axis, "z")
        np.testing.assert_allclose(resolution.model_to_own_c3d, Y_UP_TO_Z_UP)
        np.testing.assert_allclose(resolution.own_c3d_to_common, np.eye(3))
        np.testing.assert_allclose(
            resolution.model_to_common_c3d,
            Y_UP_TO_Z_UP,
        )

    def test_explicit_identity_is_scoped_to_the_system_own_c3d_frame(self) -> None:
        resolution = resolve_system_coordinate_frames("motive", "identity")

        np.testing.assert_allclose(resolution.model_to_own_c3d, np.eye(3))
        np.testing.assert_allclose(resolution.own_c3d_to_common, np.eye(3))

    def test_basis_transform_preserves_marker_and_frame_axes(self) -> None:
        points = np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 200.0], [300.0, 400.0]],
            ]
        )

        converted = transform_point_trajectories(points, Y_UP_TO_Z_UP)

        self.assertEqual(converted.shape, points.shape)
        np.testing.assert_allclose(converted[0], points[0])
        np.testing.assert_allclose(converted[1], -points[2])
        np.testing.assert_allclose(converted[2], points[1])


if __name__ == "__main__":
    unittest.main()
