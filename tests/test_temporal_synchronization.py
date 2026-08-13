from __future__ import annotations

import unittest

import numpy as np

from temporal_synchronization import (
    apply_time_offset,
    estimate_constant_lag,
    interpolate_finite_array,
    normalize_selected_phase,
    normalize_timeseries_groups,
    resolve_temporal_synchronization,
)


def trajectory(time: np.ndarray) -> np.ndarray:
    """Smooth non-periodic 3D trajectory used by the synthetic tests."""

    x = 200.0 * np.exp(-(((time - 1.2) / 0.22) ** 2))
    y = 120.0 * np.exp(-(((time - 2.4) / 0.35) ** 2))
    z = 40.0 * np.sin(2.0 * np.pi * 0.7 * time) * np.exp(-time / 5.0)
    return np.vstack((x, y, z))


class TemporalSynchronizationTests(unittest.TestCase):
    def test_estimates_positive_lag_for_a_moving_clock_that_starts_early(self) -> None:
        reference_time = np.arange(0.0, 4.0, 1.0 / 120.0)
        captury_time = np.arange(0.0, 4.0, 1.0 / 100.0)
        true_lag = 0.18
        reference = {"Hips": trajectory(reference_time)}
        # Captury sample t represents the physical state observed at t + lag.
        captury = {"Hips": trajectory(captury_time + true_lag)}

        result = estimate_constant_lag(
            reference,
            reference_time,
            captury,
            captury_time,
            max_lag_s=0.5,
        )

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["lag_s"], true_lag, delta=0.015)
        self.assertGreater(result["correlation_after"], 0.98)
        self.assertGreater(result["correlation_after"], result["correlation_before"])
        self.assertEqual(
            result["lag_convention"],
            "captury_corrected_time = captury_original_time + lag_s",
        )

    def test_estimation_handles_missing_samples_and_different_rates(self) -> None:
        reference_time = np.arange(0.0, 4.0, 1.0 / 200.0)
        captury_time = np.arange(0.0, 4.0, 1.0 / 75.0)
        true_lag = -0.12
        reference_curve = trajectory(reference_time)
        captury_curve = trajectory(captury_time + true_lag)
        captury_curve[:, 40:55] = np.nan

        result = estimate_constant_lag(
            {"Hips": reference_curve},
            reference_time,
            {"Hips": captury_curve},
            captury_time,
            max_lag_s=0.4,
        )

        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["lag_s"], true_lag, delta=0.02)
        self.assertGreater(result["overlap_samples"], 100)

    def test_static_signal_is_refused_in_auto_mode(self) -> None:
        time = np.linspace(0.0, 2.0, 201)
        static = {"Hips": np.zeros((3, time.size))}

        result = resolve_temporal_synchronization(
            "auto", static, time, static, time, max_lag_s=0.5
        )

        self.assertEqual(result["status"], "insufficient_motion")
        self.assertEqual(result["lag_s"], 0.0)
        self.assertFalse(result["applied"])

    def test_negligible_gain_does_not_shift_an_already_synchronized_signal(
        self,
    ) -> None:
        time = np.arange(0.0, 4.0, 1.0 / 120.0)
        reference = {"Hips": trajectory(time)}
        almost_identical = {
            "Hips": trajectory(time + 0.002) + 1e-6 * np.sin(time)[None, :]
        }

        result = estimate_constant_lag(
            reference,
            time,
            almost_identical,
            time,
            max_lag_s=0.1,
            min_correlation_gain=0.01,
        )

        self.assertEqual(result["lag_s"], 0.0)
        self.assertFalse(result["applied"])
        self.assertIn(result["status"], {"ok", "negligible_gain"})

    def test_manual_mode_uses_the_declared_lag(self) -> None:
        time = np.linspace(0.0, 1.0, 101)
        curves = {"Hips": trajectory(time)}

        result = resolve_temporal_synchronization(
            "manual",
            curves,
            time,
            curves,
            time,
            manual_lag_s=-0.075,
            max_lag_s=0.5,
        )

        self.assertEqual(result["status"], "manual")
        self.assertEqual(result["lag_s"], -0.075)
        self.assertTrue(result["applied"])

    def test_apply_time_offset_does_not_modify_samples(self) -> None:
        time = np.asarray([0.0, 0.1, 0.2])

        corrected = apply_time_offset(time, 0.25)

        np.testing.assert_allclose(corrected, [0.25, 0.35, 0.45])
        np.testing.assert_allclose(time, [0.0, 0.1, 0.2])

    def test_interpolation_returns_nan_outside_temporal_overlap(self) -> None:
        source_time = np.asarray([0.2, 0.3, 0.4])
        target_time = np.asarray([0.1, 0.2, 0.25, 0.4, 0.5])
        values = np.asarray([[2.0, 3.0, 4.0], [20.0, 30.0, 40.0]])

        interpolated = interpolate_finite_array(values, source_time, target_time)

        self.assertTrue(np.all(np.isnan(interpolated[:, [0, 4]])))
        np.testing.assert_allclose(
            interpolated[:, 1:4], [[2.0, 2.5, 4.0], [20.0, 25.0, 40.0]]
        )

    def test_interpolation_accepts_two_samples(self) -> None:
        interpolated = interpolate_finite_array(
            np.asarray([[0.0, 10.0]]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 0.5, 1.0]),
        )

        np.testing.assert_allclose(interpolated, [[0.0, 5.0, 10.0]])

    def test_interpolation_preserves_long_internal_missing_gap(self) -> None:
        source_time = np.arange(6, dtype=float)
        target_time = np.arange(0.0, 5.1, 0.5)
        values = np.asarray([[0.0, 1.0, np.nan, np.nan, 4.0, 5.0]])

        interpolated = interpolate_finite_array(
            values, source_time, target_time, max_gap_s=1.1
        )

        self.assertTrue(np.isnan(interpolated[0, 4:7]).all())

    def test_periodic_ambiguous_peak_is_not_applied(self) -> None:
        reference_time = np.arange(0.0, 8.0, 1.0 / 120.0)
        captury_time = np.arange(0.0, 8.0, 1.0 / 100.0)
        period = 1.0

        def periodic_curve(time: np.ndarray) -> np.ndarray:
            return np.vstack(
                (
                    np.sin(2.0 * np.pi * time / period),
                    0.2 * np.sin(4.0 * np.pi * time / period),
                    np.zeros(time.size),
                )
            )

        reference_curve = periodic_curve(reference_time)
        captury_curve = periodic_curve(captury_time + 0.15)

        result = estimate_constant_lag(
            {"Hips": reference_curve},
            reference_time,
            {"Hips": captury_curve},
            captury_time,
            max_lag_s=1.2,
        )

        self.assertEqual(result["status"], "ambiguous_peak")
        self.assertFalse(result["applied"])
        self.assertEqual(result["lag_s"], 0.0)

    def test_selected_phase_is_normalized_to_requested_points(self) -> None:
        reference_time = np.linspace(0.0, 2.0, 201)
        captury_time = reference_time - 0.1
        reference_signal = np.sin(reference_time)
        captury_signal = np.sin(captury_time + 0.1)

        rows, report = normalize_selected_phase(
            reference_time,
            reference_signal,
            captury_time,
            captury_signal,
            lag_s=0.1,
            start_s=0.5,
            end_s=1.5,
            n_points=51,
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["points"], 51)
        self.assertEqual(rows[0]["phase_percent"], 0.0)
        self.assertEqual(rows[-1]["phase_percent"], 100.0)
        self.assertLess(
            max(abs(row["difference"]) for row in rows),
            1e-10,
        )

    def test_long_timeseries_groups_are_normalized_independently(self) -> None:
        rows = [
            {"time": time, "joint": joint, "distance_mm": multiplier * time}
            for joint, multiplier in (("Hip", 1.0), ("Knee", 2.0))
            for time in (0.0, 0.5, 1.0)
        ]

        normalized = normalize_timeseries_groups(
            rows,
            time_key="time",
            group_keys=("joint",),
            value_keys=("distance_mm",),
            start_s=0.25,
            end_s=0.75,
            n_points=3,
        )

        self.assertEqual(len(normalized), 6)
        hip = [row for row in normalized if row["joint"] == "Hip"]
        np.testing.assert_allclose(
            [row["distance_mm"] for row in hip], [0.25, 0.5, 0.75]
        )
        self.assertTrue(all(row["normalization_status"] == "ok" for row in hip))

    def test_normalization_refuses_a_signal_with_a_large_internal_gap(self) -> None:
        rows = [
            {"time": time, "joint": "Hip", "distance_mm": value}
            for time, value in ((0.0, 0.0), (0.1, 0.1), (0.9, 0.9), (1.0, 1.0))
        ]

        normalized = normalize_timeseries_groups(
            rows,
            time_key="time",
            group_keys=("joint",),
            value_keys=("distance_mm",),
            start_s=0.0,
            end_s=1.0,
            n_points=5,
        )

        self.assertEqual(len(normalized), 5)
        self.assertTrue(
            all(row["normalization_status"] == "unavailable" for row in normalized)
        )
        self.assertTrue(all(np.isnan(row["distance_mm"]) for row in normalized))
        self.assertAlmostEqual(normalized[0]["distance_mm_largest_gap_s"], 0.8)


if __name__ == "__main__":
    unittest.main()
