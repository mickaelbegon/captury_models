from __future__ import annotations

import unittest

import numpy as np

from compare_capture_systems import (
    aggregate_population_metrics,
    angle_values_to_degrees,
    resample_1d,
)


class CaptureSystemPopulationTests(unittest.TestCase):
    def test_population_summary_clusters_trials_within_participant(self) -> None:
        rows = [
            {"participant": "P01", "trial": "Walk1", "angle": "Hip", "rmse_deg": 1.0},
            {"participant": "P01", "trial": "Walk2", "angle": "Hip", "rmse_deg": 3.0},
            {"participant": "P02", "trial": "Walk1", "angle": "Hip", "rmse_deg": 5.0},
        ]

        summary = aggregate_population_metrics(
            rows,
            group_specs=[("population_by_angle", ["angle"])],
            metric_names=["rmse_deg"],
        )

        self.assertEqual(int(summary.loc[0, "n_participants"]), 2)
        self.assertEqual(summary.loc[0, "statistical_unit"], "participant")
        self.assertAlmostEqual(float(summary.loc[0, "rmse_deg_mean"]), 3.5)

    def test_angle_population_exposes_guarded_waveform_metrics(self) -> None:
        from compare_capture_systems import angle_population_summary

        summary = angle_population_summary(
            [
                {
                    "participant": "P01",
                    "trial": "Walk",
                    "angle": "Hip",
                    "component": "x",
                    "rmse_deg": 2.0,
                    "reference_rom_deg": 30.0,
                }
            ]
        )

        self.assertIn("rmse_deg_mean", summary.columns)
        self.assertIn("reference_rom_deg_mean", summary.columns)

    def test_angle_conversion_is_independent_from_spatial_point_units(self) -> None:
        values = np.asarray([[0.0, np.pi / 2.0, np.pi]])

        converted = angle_values_to_degrees(values, "rad")

        np.testing.assert_allclose(converted, [[0.0, 90.0, 180.0]])

    def test_resampling_preserves_internal_missing_intervals(self) -> None:
        signal = np.asarray([0.0, 1.0, np.nan, np.nan, 4.0, 5.0])

        resampled = resample_1d(signal, 11)

        self.assertTrue(np.isnan(resampled[4:7]).all())
        self.assertAlmostEqual(resampled[2], 1.0)
        self.assertAlmostEqual(resampled[8], 4.0)
        self.assertAlmostEqual(resampled[0], 0.0)
        self.assertAlmostEqual(resampled[-1], 5.0)

    def test_population_angle_summary_does_not_merge_movements(self) -> None:
        from compare_capture_systems import angle_population_summary

        summary = angle_population_summary(
            [
                {
                    "participant": "P01",
                    "trial": "Walk",
                    "angle": "Hip",
                    "component": "x",
                    "rmse_deg": 1.0,
                },
                {
                    "participant": "P01",
                    "trial": "Squat",
                    "angle": "Hip",
                    "component": "x",
                    "rmse_deg": 9.0,
                },
                {
                    "participant": "P02",
                    "trial": "Walk",
                    "angle": "Hip",
                    "component": "x",
                    "rmse_deg": 3.0,
                },
                {
                    "participant": "P02",
                    "trial": "Squat",
                    "angle": "Hip",
                    "component": "x",
                    "rmse_deg": 11.0,
                },
            ]
        )

        population = summary[summary["summary_scope"] == "population_by_trial_angle"]
        self.assertEqual(set(population["trial"]), {"Walk", "Squat"})
        self.assertNotIn("population_by_angle", set(summary["summary_scope"]))

    def test_landmark_population_summary_does_not_merge_movements(self) -> None:
        from compare_capture_systems import landmark_population_summary

        summary = landmark_population_summary(
            [
                {
                    "participant": "P01",
                    "trial": "Walk",
                    "variant": "aligned",
                    "landmark": "hip",
                    "rmse_euclidean": 1.0,
                },
                {
                    "participant": "P01",
                    "trial": "Squat",
                    "variant": "aligned",
                    "landmark": "hip",
                    "rmse_euclidean": 9.0,
                },
            ]
        )

        self.assertNotIn("population_by_landmark", set(summary["summary_scope"]))
        self.assertIn("population_by_trial_landmark", set(summary["summary_scope"]))


if __name__ == "__main__":
    unittest.main()
