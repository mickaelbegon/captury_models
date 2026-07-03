from __future__ import annotations

import unittest

from captury_biobuddy_gui import COMMAND_MODES, CapturyBioBuddyGui


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
        self.assertEqual(gui.vars["p6_joint_filter"].get(), "Hip|Knee|Ankle|Leg|Foot")
        self.assertEqual(gui.vars["p6_model_source"].get(), "bvh")
        self.assertEqual(gui.vars["p6_model_to_c3d_axis"].get(), "y_up_to_z_up")
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


if __name__ == "__main__":
    unittest.main()
