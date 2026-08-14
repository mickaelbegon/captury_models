from __future__ import annotations

import unittest

import numpy as np

from model_comparison_metrics import (
    aggregate_trial_metrics_by_participant,
    joint_center_error_xyz,
    waveform_metrics,
)


class WaveformMetricContractTests(unittest.TestCase):
    def test_identical_waveforms_have_zero_error_and_full_agreement(self) -> None:
        time = np.linspace(0.0, 1.0, 101)
        curve = 30.0 * np.sin(2.0 * np.pi * time)

        metrics = waveform_metrics(curve, curve, unit="deg", time=time)

        self.assertEqual(metrics["waveform_status"], "ok")
        self.assertTrue(metrics["shape_metrics_eligible"])
        self.assertAlmostEqual(metrics["rmse_deg"], 0.0)
        self.assertAlmostEqual(metrics["nrmse_range"], 0.0)
        self.assertAlmostEqual(metrics["pearson_r_waveform"], 1.0)
        self.assertAlmostEqual(metrics["lin_ccc_waveform"], 1.0)
        self.assertAlmostEqual(metrics["rom_difference_deg"], 0.0)
        self.assertAlmostEqual(metrics["max_time_difference_s"], 0.0)

    def test_constant_bias_is_reported_with_descriptive_limits_of_agreement(
        self,
    ) -> None:
        reference = np.linspace(-20.0, 40.0, 61)
        test = reference + 5.0

        metrics = waveform_metrics(reference, test, unit="deg")

        self.assertAlmostEqual(metrics["bias_deg"], 5.0)
        self.assertAlmostEqual(metrics["loa_lower_deg"], 5.0)
        self.assertAlmostEqual(metrics["loa_upper_deg"], 5.0)
        self.assertEqual(
            metrics["limits_of_agreement_scope"],
            "descriptive_within_trial_frames_not_population_ci",
        )

    def test_gain_and_sign_inversion_are_diagnosed(self) -> None:
        reference = np.linspace(-10.0, 10.0, 101)

        gain = waveform_metrics(reference, 2.0 * reference + 3.0, unit="deg")
        inverted = waveform_metrics(reference, -reference, unit="deg")

        self.assertAlmostEqual(gain["linear_gain"], 2.0)
        self.assertAlmostEqual(gain["linear_offset_deg"], 3.0)
        self.assertAlmostEqual(inverted["linear_gain"], -1.0)
        self.assertAlmostEqual(inverted["pearson_r_waveform"], -1.0)

    def test_extrema_timing_uses_supplied_time(self) -> None:
        time = np.arange(5, dtype=float) * 0.1
        reference = np.asarray([0.0, 1.0, 3.0, 1.0, -2.0])
        test = np.asarray([-2.0, 0.0, 1.0, 3.0, 1.0])

        metrics = waveform_metrics(
            reference,
            test,
            unit="deg",
            time=time,
            minimum_paired_samples=5,
        )

        self.assertAlmostEqual(metrics["max_time_difference_s"], 0.1)
        self.assertAlmostEqual(metrics["min_time_difference_s"], -0.4)

    def test_low_amplitude_blocks_shape_metrics_but_keeps_absolute_errors(
        self,
    ) -> None:
        reference = np.linspace(0.0, 0.2, 101)
        test = reference + 0.1

        metrics = waveform_metrics(reference, test, unit="deg")

        self.assertEqual(metrics["waveform_status"], "low_reference_amplitude")
        self.assertFalse(metrics["shape_metrics_eligible"])
        self.assertTrue(np.isnan(metrics["nrmse_range"]))
        self.assertTrue(np.isnan(metrics["pearson_r_waveform"]))
        self.assertTrue(np.isnan(metrics["lin_ccc_waveform"]))
        self.assertAlmostEqual(metrics["mae_deg"], 0.1)

    def test_missing_samples_report_coverage_and_use_only_finite_pairs(self) -> None:
        reference = np.asarray([0.0, 10.0, np.nan, 30.0, 40.0])
        test = np.asarray([1.0, 11.0, 20.0, np.nan, 41.0])

        metrics = waveform_metrics(reference, test, unit="deg")

        self.assertEqual(metrics["paired_samples"], 3)
        self.assertAlmostEqual(metrics["paired_coverage"], 0.6)
        self.assertAlmostEqual(metrics["bias_deg"], 1.0)
        self.assertEqual(metrics["waveform_status"], "insufficient_pairs")
        self.assertFalse(metrics["shape_metrics_eligible"])

    def test_coverage_gate_applies_when_ten_or_more_pairs_remain(self) -> None:
        reference = np.linspace(-20.0, 20.0, 20)
        test = reference.copy()
        test[15:] = np.nan

        metrics = waveform_metrics(reference, test, unit="deg")

        self.assertEqual(metrics["paired_samples"], 15)
        self.assertEqual(metrics["waveform_status"], "insufficient_coverage")
        self.assertTrue(np.isnan(metrics["linear_gain"]))
        self.assertTrue(np.isnan(metrics["max_index_difference_samples"]))

    def test_low_test_amplitude_blocks_all_shape_metrics(self) -> None:
        reference = np.linspace(-20.0, 20.0, 101)
        test = np.linspace(0.0, 0.2, 101)

        metrics = waveform_metrics(reference, test, unit="deg")

        self.assertEqual(metrics["waveform_status"], "low_test_amplitude")
        self.assertTrue(np.isnan(metrics["linear_gain"]))
        self.assertTrue(np.isnan(metrics["max_index_difference_samples"]))

    def test_native_shape_metrics_require_an_explicit_amplitude_threshold(self) -> None:
        reference = np.linspace(0.0, 1.0, 101)
        test = reference.copy()

        implicit = waveform_metrics(reference, test, unit="native")
        explicit = waveform_metrics(
            reference, test, unit="native", minimum_amplitude=0.1
        )

        self.assertEqual(implicit["waveform_status"], "unknown_native_scale")
        self.assertFalse(implicit["shape_metrics_eligible"])
        self.assertEqual(explicit["waveform_status"], "ok")


class JointCentreMetricContractTests(unittest.TestCase):
    def test_xyz_metrics_include_signed_bias_and_limits_of_agreement(self) -> None:
        reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        test = reference + np.asarray([2.0, -3.0, 4.0])

        metrics = joint_center_error_xyz(reference, test)

        self.assertAlmostEqual(metrics["bias_x"], 2.0)
        self.assertAlmostEqual(metrics["bias_y"], -3.0)
        self.assertAlmostEqual(metrics["bias_z"], 4.0)
        self.assertAlmostEqual(metrics["loa_lower_x"], 2.0)
        self.assertAlmostEqual(metrics["loa_upper_x"], 2.0)
        self.assertEqual(metrics["paired_frames"], 2)


class ParticipantAggregationTests(unittest.TestCase):
    def test_frames_are_not_used_as_population_observations(self) -> None:
        rows = [
            {
                "participant": participant,
                "trial": trial,
                "q_name": "Hip_rotX",
                "rmse_deg": value,
            }
            for participant, trial, value in (
                ("P01", "Walk1", 1.0),
                ("P01", "Walk2", 3.0),
                ("P02", "Walk1", 5.0),
            )
        ]

        summary, report = aggregate_trial_metrics_by_participant(
            rows,
            metric_keys=("rmse_deg",),
            group_keys=("q_name",),
        )

        self.assertEqual(report["statistical_unit"], "participant")
        self.assertEqual(report["trial_rows"], 3)
        self.assertEqual(report["participant_rows"], 2)
        self.assertEqual(summary[0]["participants"], 2)
        self.assertAlmostEqual(summary[0]["mean_rmse_deg"], 3.5)
        self.assertIn("ci95_lower_rmse_deg", summary[0])
        self.assertEqual(
            report["uncertainty_method"], "participant_cluster_bootstrap_mean"
        )

    def test_population_groups_keep_movements_separate(self) -> None:
        rows = [
            {
                "participant": participant,
                "trial": trial,
                "q_name": "Hip_rotX",
                "rmse_deg": value,
            }
            for participant, trial, value in (
                ("P01", "Walk", 1.0),
                ("P01", "Squat", 9.0),
                ("P02", "Walk", 3.0),
                ("P02", "Squat", 11.0),
            )
        ]

        summary, _report = aggregate_trial_metrics_by_participant(
            rows,
            metric_keys=("rmse_deg",),
            group_keys=("trial", "q_name"),
        )

        by_trial = {row["trial"]: row["mean_rmse_deg"] for row in summary}
        self.assertEqual(by_trial, {"Squat": 10.0, "Walk": 2.0})

    def test_missing_participant_identifier_refuses_population_summary(self) -> None:
        summary, report = aggregate_trial_metrics_by_participant(
            [{"trial": "Walk", "q_name": "Hip", "rmse_deg": 2.0}],
            metric_keys=("rmse_deg",),
            group_keys=("q_name",),
        )

        self.assertEqual(summary, [])
        self.assertEqual(report["status"], "missing_participant_identifier")

        summary_nan, report_nan = aggregate_trial_metrics_by_participant(
            [
                {
                    "participant": np.nan,
                    "trial": "Walk",
                    "q_name": "Hip",
                    "rmse_deg": 2.0,
                }
            ],
            metric_keys=("rmse_deg",),
            group_keys=("q_name",),
        )
        self.assertEqual(summary_nan, [])
        self.assertEqual(report_nan["status"], "missing_participant_identifier")

    def test_participant_count_is_not_reduced_by_inapplicable_unit_metrics(
        self,
    ) -> None:
        rows = [
            {
                "participant": participant,
                "trial": "Walk",
                "source": "captury",
                "q_name": "Hip_rotX",
                "rmse_rad": value,
                "rmse_native": np.nan,
            }
            for participant, value in (("P01", 0.1), ("P02", 0.2))
        ]

        summary, _report = aggregate_trial_metrics_by_participant(
            rows,
            metric_keys=("rmse_rad", "rmse_native"),
            group_keys=("source", "q_name"),
        )

        self.assertEqual(summary[0]["participants"], 2)
        self.assertEqual(summary[0]["participants_rmse_rad"], 2)
        self.assertEqual(summary[0]["participants_rmse_native"], 0)

    def test_population_with_one_participant_is_descriptive_only(self) -> None:
        summary, report = aggregate_trial_metrics_by_participant(
            [
                {
                    "participant": "P01",
                    "trial": "Walk",
                    "q_name": "Hip",
                    "rmse_deg": 2.0,
                }
            ],
            metric_keys=("rmse_deg",),
            group_keys=("trial", "q_name"),
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(report["status"], "insufficient_participants")
        self.assertEqual(report["maximum_participants_per_group"], 1)

    def test_groups_without_any_finite_metric_are_excluded(self) -> None:
        summary, report = aggregate_trial_metrics_by_participant(
            [
                {
                    "participant": "P01",
                    "trial": "Walk",
                    "source": "descriptive_only",
                    "rmse_deg": np.nan,
                }
            ],
            metric_keys=("rmse_deg",),
            group_keys=("trial", "source"),
        )

        self.assertEqual(summary, [])
        self.assertEqual(report["status"], "no_finite_metrics")


if __name__ == "__main__":
    unittest.main()
