from __future__ import annotations

import unittest

import tempfile
from pathlib import Path

import pandas as pd

from captury_biobuddy_gui import (
    COMMAND_MODES,
    CapturyBioBuddyGui,
    graph_metric_columns,
    inventory_p6_dataset,
)


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class P6DebugGuiTests(unittest.TestCase):
    def make_gui_stub(self) -> CapturyBioBuddyGui:
        gui = object.__new__(CapturyBioBuddyGui)
        keys = [
            "command_mode",
            "p6_data_root",
            "p6_out_dir",
            "p6_trials",
            "selected_trial",
            "p6_static_trial",
            "p6_joint_filter",
            "p6_model_source",
            "p6_model_to_c3d_axis",
            "p6_no_figures",
            "p6_no_mesh",
            "p6_max_mesh_points",
            "p6_run_ik_batch",
            "p6_ik_max_frames",
            "p6_visualize",
            "p6_visualize_trial",
            "p6_headless",
            "p6_rerun_wait_seconds",
        ]
        gui.vars = {key: FakeVar("") for key in keys}
        gui.vars["p6_no_figures"].set(False)
        gui.vars["p6_no_mesh"].set(False)
        gui.vars["p6_run_ik_batch"].set(False)
        gui.vars["p6_visualize"].set(False)
        gui.vars["p6_headless"].set(False)
        gui.status_var = FakeVar()
        return gui

    def test_p6_debug_preset_populates_fast_static_analysis(self) -> None:
        gui = self.make_gui_stub()

        CapturyBioBuddyGui._load_p6_debug_preset(gui)

        self.assertEqual(gui.vars["command_mode"].get(), COMMAND_MODES["kinematic"])
        self.assertEqual(
            gui.vars["p6_data_root"].get(), "local_trials/2026-06-30_P6_flat"
        )
        self.assertEqual(gui.vars["p6_out_dir"].get(), "out_p6_motive_captury_debug")
        self.assertEqual(gui.vars["p6_trials"].get(), "Static")
        self.assertEqual(gui.vars["selected_trial"].get(), "Static")
        self.assertEqual(gui.vars["p6_joint_filter"].get(), "Hip|Knee|Ankle|Leg|Foot")
        self.assertEqual(gui.vars["p6_model_source"].get(), "bvh")
        self.assertEqual(gui.vars["p6_model_to_c3d_axis"].get(), "auto")
        self.assertIs(gui.vars["p6_no_mesh"].get(), True)
        self.assertIs(gui.vars["p6_headless"].get(), True)
        self.assertEqual(gui.vars["p6_rerun_wait_seconds"].get(), "0")
        self.assertEqual(gui.status_var.get(), "Preset P6 debug chargé")

    def test_p6_debug_command_targets_kinematic_cli(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertTrue(args[1].endswith("compare_p6_motive_captury.py"))
        self.assertIn("--trial", args)
        self.assertIn("Static", args)
        self.assertIn("--no-mesh", args)
        self.assertIn("--headless", args)
        self.assertNotIn("--run-ik-batch", args)
        self.assertNotIn("--visualize", args)

    def test_inventory_p6_dataset_collects_flat_files_for_trial_dropdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Captury").mkdir()
            (root / "Motive").mkdir()
            for suffix in ("c3d", "bvh", "fbx"):
                (root / "Captury" / f"Static_P6.{suffix}").write_text(
                    "captury", encoding="utf-8"
                )
            (root / "Motive" / "P6_Static.c3d").write_text("motive", encoding="utf-8")
            (root / "Motive" / "P6_Static_Skeleton 001.bvh").write_text(
                "motive", encoding="utf-8"
            )

            inventory = inventory_p6_dataset(root)

            self.assertEqual(sorted(inventory), ["Static"])
            self.assertEqual(
                inventory["Static"]["Captury"]["c3d"].name, "Static_P6.c3d"
            )
            self.assertEqual(inventory["Static"]["Motive"]["c3d"].name, "P6_Static.c3d")

    def test_selected_trial_c3d_prefers_motive_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motive_c3d = root / "P6_Static.c3d"
            captury_c3d = root / "Static_P6.c3d"
            motive_c3d.write_text("motive", encoding="utf-8")
            captury_c3d.write_text("captury", encoding="utf-8")
            gui = self.make_gui_stub()
            gui.trial_inventory = {
                "Static": {
                    "Captury": {"c3d": captury_c3d},
                    "Motive": {"c3d": motive_c3d},
                }
            }
            gui.vars["selected_trial"].set("Static")

            self.assertEqual(
                CapturyBioBuddyGui._selected_trial_c3d_path(gui), motive_c3d
            )

    def test_graph_metric_columns_keeps_only_numeric_requested_columns(self) -> None:
        dataframe = pd.DataFrame(
            {
                "trial": ["Static"],
                "joint": ["Hip"],
                "median_error_mm": [12.0],
                "comment": ["ignore"],
            }
        )

        self.assertEqual(
            graph_metric_columns(dataframe, ("median_error_mm", "comment", "missing")),
            ["median_error_mm"],
        )


if __name__ == "__main__":
    unittest.main()
