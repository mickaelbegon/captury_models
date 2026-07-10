from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import numpy as np

from c3d_source_preparation import source_preparation_configs_from_args
from c3d_trial_viewer import C3DMarkerData
from plot_c3d_initial_offset import (
    DEFAULT_CAPTURY_ROOT_OFFSET_MM,
    centroid,
    offset_summary,
    parse_args,
    representative_points,
    selected_frame_window,
)


class PlotC3DInitialOffsetTests(unittest.TestCase):
    def test_default_captury_root_offset_is_zero_for_raw_c3d_markers(self) -> None:
        self.assertEqual(DEFAULT_CAPTURY_ROOT_OFFSET_MM, (0.0, 0.0, 0.0))

    def test_parse_args_maps_independent_cli_options_to_source_configs(self) -> None:
        argv = [
            "plot_c3d_initial_offset.py",
            "--motive-transform",
            "rx_plus_90",
            "--captury-transform",
            "none",
            "--motive-subtract-root-offset",
            "--motive-root-offset-mm",
            "1",
            "2",
            "3",
            "--captury-root-offset-mm",
            "4",
            "5",
            "6",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        motive_config, captury_config = source_preparation_configs_from_args(args)

        self.assertEqual(motive_config.transform, "rx_plus_90")
        self.assertEqual(captury_config.transform, "none")
        self.assertTrue(motive_config.subtract_root_offset)
        self.assertFalse(captury_config.subtract_root_offset)
        np.testing.assert_allclose(motive_config.root_offset_mm, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(captury_config.root_offset_mm, [4.0, 5.0, 6.0])

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

    def test_centroid_ignores_non_finite_markers(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]])

        np.testing.assert_allclose(centroid(points), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
