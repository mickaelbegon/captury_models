from __future__ import annotations

import unittest

import numpy as np

from c3d_trial_viewer import C3DMarkerData
from plot_c3d_initial_offset import (
    DEFAULT_CAPTURY_ROOT_OFFSET_MM,
    SourcePreparationConfig,
    apply_point_transform,
    centroid,
    effective_root_offset_subtractions,
    offset_summary,
    prepare_c3d_points_for_offset_test,
    prepare_source_points,
    representative_points,
    rotation_x_degrees,
    selected_frame_window,
    subtract_root_offset,
    transform_points,
)


class PlotC3DInitialOffsetTests(unittest.TestCase):
    def test_default_captury_root_offset_is_zero_for_raw_c3d_markers(self) -> None:
        self.assertEqual(DEFAULT_CAPTURY_ROOT_OFFSET_MM, (0.0, 0.0, 0.0))

    def test_selected_frame_window_is_clipped_around_requested_frame(self) -> None:
        self.assertEqual(selected_frame_window(10, 0, 5), (0, 5))
        self.assertEqual(selected_frame_window(10, 9, 5), (5, 10))
        self.assertEqual(selected_frame_window(10, 5, 3), (4, 7))

    def test_representative_points_uses_nanmedian_for_windows(self) -> None:
        points = np.asarray(
            [
                [[1.0, 3.0, np.nan], [10.0, 12.0, 14.0]],
                [[2.0, 4.0, np.nan], [20.0, 22.0, 24.0]],
                [[3.0, 5.0, np.nan], [30.0, 32.0, 34.0]],
            ]
        )

        actual = representative_points(points, 0, 3)

        np.testing.assert_allclose(actual[0], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(actual[1], [12.0, 22.0, 32.0])

    def test_offset_summary_reports_captury_minus_motive_centroid(self) -> None:
        motive = C3DMarkerData(
            labels=["A", "B"],
            points=np.asarray(
                [
                    [[0.0], [10.0]],
                    [[0.0], [0.0]],
                    [[0.0], [0.0]],
                ]
            ),
            rate=100.0,
        )
        captury = C3DMarkerData(
            labels=["A", "B"],
            points=np.asarray(
                [
                    [[100.0], [110.0]],
                    [[20.0], [20.0]],
                    [[30.0], [30.0]],
                ]
            ),
            rate=100.0,
        )

        summary = offset_summary(motive, captury, frame=0, window=1)

        np.testing.assert_allclose(summary.motive_centroid_mm, [5.0, 0.0, 0.0])
        np.testing.assert_allclose(summary.captury_centroid_mm, [105.0, 20.0, 30.0])
        np.testing.assert_allclose(
            summary.delta_captury_minus_motive_mm, [100.0, 20.0, 30.0]
        )
        self.assertAlmostEqual(summary.euclidean_mm, float(np.sqrt(11300.0)))

    def test_offset_summary_ignores_root_offsets_when_flags_are_disabled(self) -> None:
        motive = C3DMarkerData(
            labels=["A"],
            points=np.asarray([[[10.0]], [[0.0]], [[0.0]]]),
            rate=100.0,
        )
        captury = C3DMarkerData(
            labels=["A"],
            points=np.asarray([[[20.0]], [[0.0]], [[0.0]]]),
            rate=100.0,
        )

        summary = offset_summary(
            motive,
            captury,
            frame=0,
            window=1,
            motive_root_offset_mm=np.asarray([1000.0, 0.0, 0.0]),
            captury_root_offset_mm=np.asarray([1000.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(
            summary.delta_captury_minus_motive_mm, [10.0, 0.0, 0.0]
        )

    def test_offset_summary_applies_motive_and_captury_options_independently(
        self,
    ) -> None:
        motive = C3DMarkerData(
            labels=["A"],
            points=np.asarray([[[0.0]], [[2.0]], [[3.0]]]),
            rate=100.0,
        )
        captury = C3DMarkerData(
            labels=["A"],
            points=np.asarray([[[0.0]], [[2.0]], [[3.0]]]),
            rate=100.0,
        )

        summary = offset_summary(
            motive,
            captury,
            frame=0,
            window=1,
            motive_subtract_root_offset=True,
            captury_subtract_root_offset=False,
            motive_root_offset_mm=np.asarray([0.0, 2.0, 3.0]),
            captury_root_offset_mm=np.asarray([0.0, 2.0, 3.0]),
            captury_transform="rx_plus_90",
        )

        np.testing.assert_allclose(summary.motive_centroid_mm, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(summary.captury_centroid_mm, [0.0, -3.0, 2.0])
        np.testing.assert_allclose(
            summary.delta_captury_minus_motive_mm, [0.0, -3.0, 2.0]
        )

    def test_captury_rx_plus_90_transform_rotates_y_to_positive_z(self) -> None:
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

    def test_root_offset_is_subtracted_before_captury_axis_rotation(self) -> None:
        points = np.asarray([[[0.0]], [[898.673]], [[43.9328]]])
        root_offset = np.asarray([0.0, 898.673, 43.9328])

        transformed = prepare_c3d_points_for_offset_test(
            points,
            root_offset_mm=root_offset,
            subtract_offset=True,
            captury_transform="rx_plus_90",
        )

        np.testing.assert_allclose(transformed[:, 0, 0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_source_preparation_config_applies_independent_source_options(self) -> None:
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

    def test_centroid_ignores_non_finite_markers(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]])

        np.testing.assert_allclose(centroid(points), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
