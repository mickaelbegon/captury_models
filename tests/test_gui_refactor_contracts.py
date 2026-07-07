from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from captury_biobuddy_gui import (
    ALL_TRIALS_LABEL,
    CRITICAL_METHOD_NOTES,
    CapturyBioBuddyGui,
    C3DMarkerData,
    data_source_color,
    data_source_marker_color,
    graph_metric_columns,
    transformed_marker_data,
    vertical_axis_label,
)
from gui_graphs import joint_centre_error_boxplot_series, joint_centre_error_timeseries
from gui_commands import ROOT_OFFSET_MODE_LABELS
from gui_trial_viewer import local_chain_axes


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


def make_gui_stub() -> CapturyBioBuddyGui:
    gui = object.__new__(CapturyBioBuddyGui)
    keys = [
        "command_mode",
        "p6_data_root",
        "p6_out_dir",
        "p6_trials",
        "selected_trial",
        "p6_static_trial",
        "p6_cut_mode",
        "p6_time_start",
        "p6_time_end",
        "p6_joint_filter",
        "p6_auto_analyze",
        "p6_model_source",
        "p6_model_to_c3d_axis",
        "root_offset_mode",
        "c3d_angle_unit",
        "p6_no_figures",
        "p6_no_cache",
        "p6_no_mesh",
        "p6_max_mesh_points",
        "p6_run_ik_batch",
        "p6_ik_max_frames",
        "p6_visualize",
        "p6_visualize_trial",
        "p6_headless",
        "p6_rerun_wait_seconds",
        "biobuddy_c3d_folder",
        "biobuddy_c3d_preset",
        "biobuddy_c3d_output",
        "biobuddy_c3d_mapping_json",
        "biobuddy_c3d_with_mesh",
        "biobuddy_c3d_no_default_virtual_points",
    ]
    gui.vars = {key: FakeVar("") for key in keys}
    gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
    gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
    gui.vars["p6_trials"].set("Marche_001")
    gui.vars["selected_trial"].set("Marche_001")
    gui.vars["p6_static_trial"].set("Static")
    gui.vars["p6_cut_mode"].set("manual")
    gui.vars["p6_time_start"].set("0.5")
    gui.vars["p6_time_end"].set("2.0")
    gui.vars["p6_joint_filter"].set("Hip|Knee\nAnkle")
    gui.vars["p6_auto_analyze"].set(True)
    gui.vars["p6_model_source"].set("fbx")
    gui.vars["p6_model_to_c3d_axis"].set("auto")
    gui.vars["root_offset_mode"].set("auto")
    gui.vars["c3d_angle_unit"].set("deg")
    gui.vars["p6_no_figures"].set(True)
    gui.vars["p6_no_cache"].set(True)
    gui.vars["p6_no_mesh"].set(True)
    gui.vars["p6_max_mesh_points"].set("0")
    gui.vars["p6_run_ik_batch"].set(False)
    gui.vars["p6_visualize"].set(False)
    gui.vars["p6_headless"].set(True)
    gui.vars["p6_rerun_wait_seconds"].set("0")
    gui.vars["biobuddy_c3d_folder"].set("/Users/mickaelbegon/Downloads/data/Motive")
    gui.vars["biobuddy_c3d_preset"].set("motive_57")
    gui.vars["biobuddy_c3d_output"].set("/tmp/motive_57.bioMod")
    gui.vars["biobuddy_c3d_mapping_json"].set(
        "/Users/mickaelbegon/Downloads/data/Motive/.motive_57_c3d_mapping.json"
    )
    gui.vars["biobuddy_c3d_with_mesh"].set(False)
    gui.vars["biobuddy_c3d_no_default_virtual_points"].set(False)
    return gui


class GuiRefactorContracts(unittest.TestCase):
    def test_p6_command_builder_contract(self) -> None:
        gui = make_gui_stub()

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertEqual(args[0], sys.executable)
        self.assertTrue(args[1].endswith("compare_p6_motive_captury.py"))
        self.assertIn("--model-source", args)
        self.assertIn("fbx", args)
        self.assertIn("--root-offset-mode", args)
        self.assertIn("auto", args)
        self.assertIn("--time-start", args)
        self.assertIn("0.5", args)
        self.assertIn("--c3d-angle-unit", args)
        self.assertIn("deg", args)
        self.assertIn("--biobuddy-biomod", args)
        self.assertIn("/tmp/motive_57.bioMod", args)
        self.assertIn("--joint-filter", args)
        self.assertIn("Hip|Knee", args)
        self.assertIn("Ankle", args)
        self.assertIn("--no-cache", args)
        self.assertIn("--headless", args)

    def test_explicit_root_offset_label_maps_to_cli_value(self) -> None:
        gui = make_gui_stub()
        gui.vars["root_offset_mode"].set(ROOT_OFFSET_MODE_LABELS["keep"])

        args = CapturyBioBuddyGui._p6_args(gui)

        option_index = args.index("--root-offset-mode")
        self.assertEqual(args[option_index + 1], "keep")

    def test_empty_csv_reads_as_empty_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")

            dataframe = CapturyBioBuddyGui._read_csv_or_empty(path)

        self.assertTrue(dataframe.empty)

    def test_p6_auto_command_is_lightweight_contract(self) -> None:
        gui = make_gui_stub()

        args = CapturyBioBuddyGui._p6_auto_analysis_args(gui, "Squat_001")

        self.assertIn("--trial", args)
        self.assertIn("Squat_001", args)
        self.assertIn("--no-figures", args)
        self.assertIn("--no-mesh", args)
        self.assertIn("--max-mesh-points", args)
        self.assertIn("--biobuddy-biomod", args)
        self.assertIn("/tmp/motive_57.bioMod", args)
        self.assertNotIn("--visualize", args)
        self.assertNotIn("--run-ik-batch", args)

    def test_biobuddy_c3d_model_command_contract(self) -> None:
        gui = make_gui_stub()

        args = CapturyBioBuddyGui._biobuddy_c3d_model_args(gui)

        self.assertTrue(args[1].endswith("create_biobuddy_c3d_model.py"))
        self.assertIn("/Users/mickaelbegon/Downloads/data/Motive", args)
        self.assertIn("--preset", args)
        self.assertIn("motive_57", args)
        self.assertIn("--output", args)
        self.assertIn("/tmp/motive_57.bioMod", args)
        self.assertIn("--motive-57-mapping-json", args)
        self.assertIn(
            "/Users/mickaelbegon/Downloads/data/Motive/.motive_57_c3d_mapping.json",
            args,
        )
        self.assertNotIn("--with-mesh", args)

    def test_biobuddy_c3d_folder_defaults_to_p6_motive_folder(self) -> None:
        gui = make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "2026-06-30_P6_flat"
            motive = data_root / "Motive"
            motive.mkdir(parents=True)
            gui.vars["p6_data_root"].set(str(data_root))
            gui.vars["biobuddy_c3d_folder"].set("")

            args = CapturyBioBuddyGui._biobuddy_c3d_model_args(gui)

            self.assertEqual(args[2], str(motive))

    def test_biobuddy_c3d_folder_accepts_lowercase_motive_folder(self) -> None:
        gui = make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "2026-06-30_P6_flat"
            motive = data_root / "motive"
            motive.mkdir(parents=True)
            gui.vars["p6_data_root"].set(str(data_root))
            gui.vars["biobuddy_c3d_folder"].set("")

            folder = CapturyBioBuddyGui._biobuddy_c3d_folder_path(gui)

            self.assertEqual(folder, motive)

    def test_critical_method_notes_cover_main_risks(self) -> None:
        titles = " ".join(note["title"] for note in CRITICAL_METHOD_NOTES)

        self.assertIn("Recalage", titles)
        self.assertIn("Cohérence", titles)
        self.assertIn("Orientation", titles)
        self.assertIn("Mise à l'échelle", titles)
        self.assertIn("Angles", titles)
        self.assertGreaterEqual(len(CRITICAL_METHOD_NOTES), 5)

    def test_viewer_helper_contract(self) -> None:
        points = np.asarray([[[1.0]], [[2.0]], [[3.0]]])
        data = C3DMarkerData(labels=["A"], points=points, rate=100.0, unit="mm")

        transformed = transformed_marker_data(
            data, np.eye(3), np.asarray([10.0, 20.0, 30.0])
        )

        np.testing.assert_allclose(
            transformed.points[:, 0, 0], np.asarray([11.0, 22.0, 33.0])
        )
        self.assertEqual(vertical_axis_label("fbx"), "+Y modèle")
        self.assertEqual(vertical_axis_label("c3d"), "+Z labo")
        self.assertEqual(data_source_color("BioBuddy"), "#22c55e")
        self.assertEqual(data_source_color("captury_c3d"), "#f97316")
        self.assertEqual(data_source_marker_color("motive"), "#38bdf8")

    def test_graph_helper_contract(self) -> None:
        dataframe = pd.DataFrame({"trial": ["A"], "metric": [1.0], "text": ["x"]})

        self.assertEqual(
            graph_metric_columns(dataframe, ("metric", "text")), ["metric"]
        )
        self.assertEqual(
            CapturyBioBuddyGui._metric_display_name("bias_rad"), "bias_deg"
        )
        converted = CapturyBioBuddyGui._values_for_display(
            pd.Series([np.pi / 2.0]), "motive", q_name="Knee_rotZ"
        )
        np.testing.assert_allclose(converted.to_numpy(), [90.0])

    def test_joint_centre_error_timeseries_contract(self) -> None:
        dataframe = pd.DataFrame(
            {
                "time": [0.0, 0.1],
                "joint": ["Hips", "Hips"],
                "captury_x_mm": [1.0, 3.0],
                "captury_y_mm": [2.0, 5.0],
                "captury_z_mm": [3.0, 9.0],
                "motive_x_mm": [0.0, 1.0],
                "motive_y_mm": [0.0, 1.0],
                "motive_z_mm": [0.0, 1.0],
            }
        )

        values = joint_centre_error_timeseries(dataframe, "Hips")

        np.testing.assert_allclose(values["abs_error_x_mm"].to_numpy(), [1.0, 2.0])
        np.testing.assert_allclose(values["abs_error_y_mm"].to_numpy(), [2.0, 4.0])
        np.testing.assert_allclose(values["abs_error_z_mm"].to_numpy(), [3.0, 8.0])
        np.testing.assert_allclose(
            values["distance_mm"].to_numpy(),
            [np.sqrt(14.0), np.sqrt(84.0)],
        )

    def test_joint_centre_boxplot_series_are_split_by_joint(self) -> None:
        dataframe = pd.DataFrame(
            {
                "trial": ["A", "A", "A", "A"],
                "time": [0.0, 0.1, 0.0, 0.1],
                "joint": ["Hips", "Hips", "Knee", "Knee"],
                "captury_x_mm": [1.0, 2.0, 10.0, 13.0],
                "captury_y_mm": [0.0, 0.0, 0.0, 0.0],
                "captury_z_mm": [0.0, 0.0, 0.0, 0.0],
                "motive_x_mm": [0.0, 0.0, 0.0, 0.0],
                "motive_y_mm": [0.0, 0.0, 0.0, 0.0],
                "motive_z_mm": [0.0, 0.0, 0.0, 0.0],
            }
        )

        series = joint_centre_error_boxplot_series(
            dataframe, "median_error_mm", trial="A"
        )

        self.assertEqual([item["label"] for item in series], ["Hips", "Knee"])
        np.testing.assert_allclose(series[0]["values"], [1.0, 2.0])
        np.testing.assert_allclose(series[1]["values"], [10.0, 13.0])

    def test_local_chain_axes_are_orthonormal(self) -> None:
        points = {
            "Hips": np.asarray([0.0, 0.0, 0.0]),
            "Spine": np.asarray([0.0, 10.0, 0.0]),
        }

        axes = local_chain_axes("Hips", points, (("Hips", "Spine"),))

        self.assertIsNotNone(axes)
        assert axes is not None
        np.testing.assert_allclose(axes["Y"], [0.0, 1.0, 0.0])
        for axis in ("X", "Y", "Z"):
            self.assertAlmostEqual(float(np.linalg.norm(axes[axis])), 1.0)
        self.assertAlmostEqual(float(np.dot(axes["X"], axes["Y"])), 0.0)
        self.assertAlmostEqual(float(np.dot(axes["Y"], axes["Z"])), 0.0)

    def test_all_trials_label_remains_stable_for_callbacks(self) -> None:
        self.assertEqual(ALL_TRIALS_LABEL, "Tous les essais")


if __name__ == "__main__":
    unittest.main()
