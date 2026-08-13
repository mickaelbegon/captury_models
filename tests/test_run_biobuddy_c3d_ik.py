import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run_biobuddy_c3d_ik import (
    _sha256,
    build_direct_marker_data,
    cached_ik_is_valid,
    ik_cache_fingerprint,
    marker_name_index,
    marker_residual_summary,
    marker_residuals_by_segment,
    run_direct_biobuddy_ik,
    stripped_marker_label,
)


class _Name:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_string(self) -> str:
        return self.value


class _MarkerModel:
    def markerNames(self) -> list[_Name]:
        return [_Name("A"), _Name("ANATOMICAL"), _Name("B")]

    def technicalMarkerNames(self) -> list[_Name]:
        return [_Name("A"), _Name("B")]

    def nbMarkers(self) -> int:
        return 3

    def nbTechnicalMarkers(self) -> int:
        return 2

    def marker(self, index: int):
        names = ["A", "ANATOMICAL", "B"]
        parents = ["Pelvis", "Virtual", "Foot"]
        return SimpleNamespace(
            name=lambda: _Name(names[index]),
            parent=lambda: _Name(parents[index]),
        )


class _Array:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value, dtype=float)

    def to_array(self) -> np.ndarray:
        return self.value


class _LinearMarkerModel:
    def nbQ(self) -> int:
        return 1

    def technicalMarkers(self, q: np.ndarray) -> list[_Array]:
        value = float(np.asarray(q)[0])
        return [_Array([value, 0.0, 0.0]), _Array([value + 1.0, 0.0, 0.0])]

    def technicalMarkersJacobian(self, _q: np.ndarray) -> list[_Array]:
        jacobian = np.asarray([[1.0], [0.0], [0.0]])
        return [_Array(jacobian), _Array(jacobian)]


class RunBioBuddyC3dIkTests(unittest.TestCase):
    def test_stripped_marker_label_removes_motive_prefix(self) -> None:
        self.assertEqual(
            stripped_marker_label("Skeleton_001_LIAS", ("Skeleton_001_",)), "LIAS"
        )
        self.assertEqual(stripped_marker_label("LIAS", ("Skeleton_001_",)), "LIAS")

    def test_marker_name_index_keeps_only_unique_labels(self) -> None:
        self.assertEqual(marker_name_index(["A", "B", "A", "C"]), {"B": 1, "C": 3})

    def test_marker_data_is_indexed_by_technical_markers_not_all_markers(self) -> None:
        split = SimpleNamespace(
            time=np.asarray([0.0, 0.01]),
            marker_labels=["A", "B"],
            marker_data_native=np.asarray(
                [
                    [[1.0, 2.0], [4.0, 5.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            c3d_unit_scale_to_m=1.0,
            angle_labels=[],
        )
        with patch("run_biobuddy_c3d_ik.split_c3d_points", return_value=split):
            data, _time, names, report = build_direct_marker_data(
                _MarkerModel(),
                Path("trial.c3d"),
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=(),
                angle_label_regex="angle",
                max_frames=0,
            )

        self.assertEqual(data.shape, (3, 2, 2))
        self.assertEqual(names, ["A", "B"])
        np.testing.assert_allclose(data[0], [[1.0, 2.0], [4.0, 5.0]])
        self.assertEqual(report["technical_model_markers"], 2)
        self.assertEqual(report["markers_used"], 2)

    def test_marker_data_converts_c3d_millimetres_to_biomod_metres(self) -> None:
        split = SimpleNamespace(
            time=np.asarray([0.0]),
            marker_labels=["A", "B"],
            marker_data_native=np.asarray(
                [[[1000.0], [250.0]], [[0.0], [0.0]], [[0.0], [0.0]]]
            ),
            c3d_unit_scale_to_m=0.001,
            angle_labels=[],
        )
        with patch("run_biobuddy_c3d_ik.split_c3d_points", return_value=split):
            data, _time, _names, report = build_direct_marker_data(
                _MarkerModel(),
                Path("trial.c3d"),
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=(),
                angle_label_regex="angle",
                max_frames=0,
            )

        np.testing.assert_allclose(data[0, :, 0], [1.0, 0.25])
        self.assertEqual(report["c3d_unit_scale_to_m"], 0.001)

    def test_cache_fingerprint_tracks_files_solver_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            biomod = root / "model.bioMod"
            c3d = root / "trial.c3d"
            biomod.write_bytes(b"model-a")
            c3d.write_bytes(b"c3d-a")
            base = ik_cache_fingerprint(
                biomod,
                c3d,
                method="trf",
                max_frames=0,
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=("Skeleton_001_",),
                angle_label_regex="angle",
                xtol=1e-6,
            )
            changed_parameter = ik_cache_fingerprint(
                biomod,
                c3d,
                method="trf",
                max_frames=10,
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=("Skeleton_001_",),
                angle_label_regex="angle",
                xtol=1e-6,
                ftol=1e-5,
            )
            biomod.write_bytes(b"model-b")
            changed_model = ik_cache_fingerprint(
                biomod,
                c3d,
                method="trf",
                max_frames=0,
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=("Skeleton_001_",),
                angle_label_regex="angle",
                xtol=1e-6,
            )

        self.assertNotEqual(base["digest"], changed_parameter["digest"])
        self.assertNotEqual(base["digest"], changed_model["digest"])
        self.assertEqual(base["payload"]["solver_parameters"]["method"], "trf")
        self.assertIn("scipy_version", base["payload"])
        self.assertEqual(len(base["payload"]["implementation_sha256"]), 64)

    def test_cache_fingerprint_changes_with_ik_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            biomod = root / "model.bioMod"
            c3d = root / "trial.c3d"
            biomod.write_bytes(b"model")
            c3d.write_bytes(b"c3d")
            kwargs = dict(
                method="trf",
                max_frames=0,
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=(),
                angle_label_regex="angle",
                xtol=1e-6,
            )
            with patch(
                "run_biobuddy_c3d_ik.implementation_sha256", return_value="a" * 64
            ):
                first = ik_cache_fingerprint(biomod, c3d, **kwargs)
            with patch(
                "run_biobuddy_c3d_ik.implementation_sha256", return_value="b" * 64
            ):
                second = ik_cache_fingerprint(biomod, c3d, **kwargs)

        self.assertNotEqual(first["digest"], second["digest"])

    def test_marker_residual_summary_reports_per_marker_and_global_error(self) -> None:
        residuals = np.asarray([[1.0, 3.0], [2.0, np.nan]])

        summary = marker_residual_summary(residuals, ["A", "B"])

        self.assertAlmostEqual(summary["global"]["mean_mm"], 2.0)
        self.assertAlmostEqual(summary["global"]["rmse_mm"], np.sqrt(14.0 / 3.0))
        self.assertEqual(summary["per_marker"]["A"]["samples"], 2)
        self.assertAlmostEqual(summary["per_marker"]["B"]["mean_mm"], 2.0)

    def test_marker_residuals_are_grouped_by_model_parent_segment(self) -> None:
        residuals = np.asarray([[1.0, 3.0], [2.0, 4.0]])

        summary = marker_residuals_by_segment(_MarkerModel(), residuals, ["A", "B"])

        self.assertAlmostEqual(summary["Pelvis"]["mean_mm"], 2.0)
        self.assertAlmostEqual(summary["Foot"]["mean_mm"], 3.0)

    def test_trf_solver_reconstructs_synthetic_marker_motion(self) -> None:
        from run_biobuddy_c3d_ik import solve_inverse_kinematics_trf

        expected_q = np.asarray([0.2, 0.5, -0.1])
        marker_data = np.zeros((3, 2, expected_q.size))
        marker_data[0, 0, :] = expected_q
        marker_data[0, 1, :] = expected_q + 1.0
        fake_biorbd = SimpleNamespace(
            get_range_q=lambda _model: (np.asarray([-2.0]), np.asarray([2.0]))
        )
        progress = []
        with patch("run_biobuddy_c3d_ik.require_biorbd", return_value=fake_biorbd):
            q, diagnostics = solve_inverse_kinematics_trf(
                _LinearMarkerModel(),
                marker_data,
                biomod_unit_scale_to_m=1.0,
                progress=progress.append,
            )

        np.testing.assert_allclose(q[0], expected_q, atol=1e-8)
        np.testing.assert_allclose(diagnostics["residuals_mm"], 0.0, atol=1e-6)
        self.assertEqual(len(progress), 2)
        self.assertTrue(np.all(diagnostics["success"]))

    def test_trf_reports_nonzero_residual_in_millimetres(self) -> None:
        from run_biobuddy_c3d_ik import solve_inverse_kinematics_trf

        marker_data = np.zeros((3, 2, 1))
        marker_data[0, 1, 0] = 3.0
        fake_biorbd = SimpleNamespace(
            get_range_q=lambda _model: (np.asarray([-2.0]), np.asarray([2.0]))
        )
        with patch("run_biobuddy_c3d_ik.require_biorbd", return_value=fake_biorbd):
            _q, diagnostics = solve_inverse_kinematics_trf(
                _LinearMarkerModel(),
                marker_data,
                biomod_unit_scale_to_m=0.001,
            )

        # The inconsistent targets leave one model-unit residual per marker.
        np.testing.assert_allclose(
            diagnostics["residuals_mm"][:, 0], [1.0, 1.0], atol=1e-6
        )

    def test_existing_cache_is_reused_without_loading_biorbd_or_solving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            biomod = root / "model.bioMod"
            c3d = root / "trial.c3d"
            cache = root / "cache"
            biomod.write_bytes(b"model")
            c3d.write_bytes(b"c3d")
            fingerprint = ik_cache_fingerprint(
                biomod,
                c3d,
                method="trf",
                max_frames=0,
                biomod_unit_scale_to_m=1.0,
                marker_prefixes_to_strip=("Skeleton_001_",),
                angle_label_regex="angle",
                xtol=1e-6,
            )
            result_dir = cache / fingerprint["digest"]
            result_dir.mkdir(parents=True)
            npz = result_dir / "inverse_kinematics_trf.npz"
            np.savez(
                npz,
                q=np.zeros((1, 1)),
                time=np.zeros(1),
                q_names=np.asarray(["q"], dtype=object),
            )
            summary_path = result_dir / "inverse_kinematics_trf_summary.json"
            summary_path.write_text(
                __import__("json").dumps(
                    {
                        "status": "ok",
                        "cache": fingerprint,
                        "artifacts": {"npz_sha256": _sha256(npz)},
                        "outputs": {"npz": str(npz), "summary": str(summary_path)},
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "run_biobuddy_c3d_ik.require_biorbd",
                side_effect=AssertionError("cache hit must not load biorbd"),
            ):
                report = run_direct_biobuddy_ik(
                    biomod,
                    c3d,
                    root / "requested-output",
                    source_name="static",
                    cache_dir=cache,
                    angle_label_regex="angle",
                )

        self.assertTrue(report["cache"]["hit"])
        self.assertEqual(Path(report["outputs"]["npz"]), npz)

    def test_corrupted_cached_npz_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz = root / "ik.npz"
            np.savez(
                npz,
                q=np.zeros((1, 1)),
                time=np.zeros(1),
                q_names=np.asarray(["q"], dtype=object),
            )
            report = {"artifacts": {"npz_sha256": _sha256(npz)}}
            self.assertTrue(cached_ik_is_valid(report, npz))

            npz.write_bytes(b"corrupted")

            self.assertFalse(cached_ik_is_valid(report, npz))


if __name__ == "__main__":
    unittest.main()
