from __future__ import annotations

import inspect
import unittest

import numpy as np

from captury_frame_calibration import (
    CAPTURY_SIMPLE_AXIS_RENAME_MATRIX,
    captury_display_labels,
    captury_static_target_frames,
    fit_captury_segment_frame_corrections,
    is_static_frame_calibration_trial,
)


def rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


class CapturyFrameCalibrationTests(unittest.TestCase):
    def test_known_q_occurrences_get_anatomical_display_names(self) -> None:
        labels = ["Q_Wa", "Q_Wa", "Q_Wa", "Q_Wa", "Q_Wa", "Q_LK", "Q_LK"]

        display = captury_display_labels(labels)

        self.assertEqual(
            display[:5],
            [
                "CAP_RASIS (Q_Wa#1)",
                "CAP_LASIS (Q_Wa#2)",
                "CAP_LPSIS (Q_Wa#3)",
                "CAP_RPSIS (Q_Wa#4)",
                "CAP_PSIS_mid (Q_Wa#5)",
            ],
        )
        self.assertEqual(display[5], "CAP_LK_medial (Q_LK#1)")
        self.assertEqual(display[6], "CAP_LK_lateral (Q_LK#2)")

    def test_static_q_and_captury_centres_recover_segment_specific_matrices(
        self,
    ) -> None:
        n_frames = 5
        labels = [
            "Q_Wa",
            "Q_Wa",
            "Q_Wa",
            "Q_Wa",
            "Q_Wa",
            "Q_LK",
            "Q_LK",
            "Q_RK",
            "Q_RK",
        ]
        base = np.asarray(
            [
                [100.0, -100.0, -100.0, 100.0, 0.0, -120.0, -80.0, 80.0, 120.0],
                [900.0, 900.0, 900.0, 900.0, 900.0, 500.0, 500.0, 500.0, 500.0],
                [-20.0, -20.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0],
            ]
        )
        points = np.repeat(base[:, :, None], n_frames, axis=2)
        centres = {
            "LeftUpLeg": np.repeat(
                np.asarray([[-100.0], [900.0], [20.0]]), n_frames, axis=1
            ),
            "LeftLeg": np.repeat(
                np.asarray([[-100.0], [500.0], [10.0]]), n_frames, axis=1
            ),
            "RightUpLeg": np.repeat(
                np.asarray([[100.0], [900.0], [20.0]]), n_frames, axis=1
            ),
            "RightLeg": np.repeat(
                np.asarray([[100.0], [500.0], [10.0]]), n_frames, axis=1
            ),
        }
        target_frames, target_report = captury_static_target_frames(
            labels, points, centres
        )
        self.assertEqual(target_report["status"], "ok")
        source_corrections = {
            "Hips": rotation_y(np.pi / 2.0),
            "LeftUpLeg": rotation_y(0.2),
            "RightUpLeg": rotation_y(-0.2),
        }
        rotations = {
            segment: np.einsum("ijf,jk->ikf", frames, correction.T)
            for segment, correction in source_corrections.items()
            for frames in (target_frames[segment],)
        }

        corrections, report = fit_captury_segment_frame_corrections(
            rotations, centres, labels, points
        )

        self.assertEqual(report["status"], "ok")
        for segment, expected in source_corrections.items():
            np.testing.assert_allclose(corrections[segment], expected, atol=1e-10)
        self.assertFalse(
            np.allclose(corrections["LeftUpLeg"], corrections["RightUpLeg"])
        )

    def test_simple_axis_rename_is_a_single_global_matrix(self) -> None:
        self.assertEqual(CAPTURY_SIMPLE_AXIS_RENAME_MATRIX.shape, (3, 3))
        np.testing.assert_allclose(
            CAPTURY_SIMPLE_AXIS_RENAME_MATRIX.T @ CAPTURY_SIMPLE_AXIS_RENAME_MATRIX,
            np.eye(3),
        )
        self.assertAlmostEqual(
            float(np.linalg.det(CAPTURY_SIMPLE_AXIS_RENAME_MATRIX)),
            1.0,
        )

    def test_static_targets_use_the_common_available_frame_range(self) -> None:
        """A longer model time series must not index beyond the static C3D."""

        labels = ["Q_Wa", "Q_Wa", "Q_Wa", "Q_Wa", "Q_Wa"]
        points = np.repeat(
            np.asarray(
                [
                    [100.0, -100.0, -100.0, 100.0, 0.0],
                    [900.0, 900.0, 900.0, 900.0, 900.0],
                    [-20.0, -20.0, 100.0, 100.0, 100.0],
                ]
            )[:, :, None],
            3,
            axis=2,
        )
        centres = {
            "LeftUpLeg": np.zeros((3, 6)),
            "LeftLeg": np.zeros((3, 6)),
            "RightUpLeg": np.zeros((3, 6)),
            "RightLeg": np.zeros((3, 6)),
        }

        frames, report = captury_static_target_frames(labels, points, centres)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(frames["Hips"].shape, (3, 3, 3))

    def test_calibration_signature_has_no_motive_inputs(self) -> None:
        parameters = inspect.signature(fit_captury_segment_frame_corrections).parameters
        self.assertFalse(any("motive" in name.lower() for name in parameters))

    def test_frame_calibration_is_only_fitted_from_the_named_static_trial(self) -> None:
        self.assertTrue(is_static_frame_calibration_trial("Static", "Static"))
        self.assertFalse(is_static_frame_calibration_trial("Marche_001", "Static"))


if __name__ == "__main__":
    unittest.main()
