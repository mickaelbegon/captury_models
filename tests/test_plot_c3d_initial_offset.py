from __future__ import annotations

import unittest

import numpy as np

from c3d_trial_viewer import C3DMarkerData
from plot_c3d_initial_offset import (
    centroid,
    offset_summary,
    representative_points,
    selected_frame_window,
)


class PlotC3DInitialOffsetTests(unittest.TestCase):
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

    def test_centroid_ignores_non_finite_markers(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]])

        np.testing.assert_allclose(centroid(points), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
