from __future__ import annotations

import unittest

import numpy as np

from c3d_source_preparation import (
    SourcePreparationConfig,
    apply_point_transform,
    effective_root_offset_subtractions,
    prepare_source_points,
    rotation_x_degrees,
    subtract_root_offset,
    transform_points,
)


class C3DSourcePreparationTests(unittest.TestCase):
    def test_prepare_source_points_applies_offset_before_transform(self) -> None:
        points = np.asarray([[[0.0]], [[898.673]], [[43.9328]]])
        config = SourcePreparationConfig(
            root_offset_mm=np.asarray([0.0, 898.673, 43.9328]),
            subtract_root_offset=True,
            transform="rx_plus_90",
        )

        transformed = prepare_source_points(points, config)

        np.testing.assert_allclose(transformed[:, 0, 0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_prepare_source_points_keeps_source_options_independent(self) -> None:
        points = np.asarray([[[0.0]], [[2.0]], [[3.0]]])
        motive_config = SourcePreparationConfig(
            root_offset_mm=np.asarray([0.0, 2.0, 3.0]),
            subtract_root_offset=True,
            transform="none",
        )
        captury_config = SourcePreparationConfig(
            root_offset_mm=np.asarray([0.0, 2.0, 3.0]),
            subtract_root_offset=False,
            transform="rx_plus_90",
        )

        motive_points = prepare_source_points(points, motive_config)
        captury_points = prepare_source_points(points, captury_config)

        np.testing.assert_allclose(motive_points[:, 0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(captury_points[:, 0, 0], [0.0, -3.0, 2.0])

    def test_transform_points_rx_plus_90_rotates_y_to_positive_z(self) -> None:
        points = np.asarray(
            [
                [[0.0], [0.0]],
                [[1.0], [0.0]],
                [[0.0], [1.0]],
            ]
        )

        transformed = transform_points(points, "rx_plus_90")

        expected = np.asarray(
            [
                [[0.0], [0.0]],
                [[0.0], [-1.0]],
                [[1.0], [0.0]],
            ]
        )
        np.testing.assert_allclose(transformed, expected, atol=1e-12)

    def test_subtract_root_offset_keeps_c3d_shape(self) -> None:
        points = np.asarray(
            [
                [[10.0, 20.0]],
                [[30.0, 40.0]],
                [[50.0, 60.0]],
            ]
        )

        shifted = subtract_root_offset(points, np.asarray([1.0, 2.0, 3.0]))

        self.assertEqual(shifted.shape, points.shape)
        np.testing.assert_allclose(shifted[:, 0, 0], [9.0, 28.0, 47.0])

    def test_apply_point_transform_matches_rotation_matrix(self) -> None:
        points = np.asarray([[[1.0]], [[2.0]], [[3.0]]])
        rotation = rotation_x_degrees(90.0)

        transformed = apply_point_transform(points, rotation)

        np.testing.assert_allclose(transformed[:, 0, 0], rotation @ points[:, 0, 0])

    def test_effective_root_offset_flags_are_independent_by_source(self) -> None:
        class Args:
            legacy_subtract_root_offsets = False
            motive_subtract_root_offset = True
            captury_subtract_root_offset = False

        self.assertEqual(effective_root_offset_subtractions(Args()), (True, False))

    def test_legacy_root_offset_flag_still_enables_both_sources(self) -> None:
        class Args:
            legacy_subtract_root_offsets = True
            motive_subtract_root_offset = False
            captury_subtract_root_offset = False

        self.assertEqual(effective_root_offset_subtractions(Args()), (True, True))


if __name__ == "__main__":
    unittest.main()
