import json
import tempfile
import unittest
from pathlib import Path

from gui_state import (
    GUI_SESSION_STATE_KEYS,
    load_gui_session_state,
    save_gui_session_state,
)


class GuiSessionStateTests(unittest.TestCase):
    def test_session_state_round_trip_keeps_only_restore_keys(self) -> None:
        values = {
            "p6_data_root": "/data/P6",
            "p6_out_dir": "/results/P6",
            "biobuddy_c3d_output": "/models/motive_57.bioMod",
            "selected_trial": "Static",
            "p6_run_ik_batch": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gui_state.json"

            save_gui_session_state(path, values)
            restored = load_gui_session_state(path)

        self.assertEqual(set(restored), set(GUI_SESSION_STATE_KEYS))
        self.assertEqual(restored["selected_trial"], "Static")
        self.assertNotIn("p6_run_ik_batch", restored)

    def test_missing_or_corrupt_session_state_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gui_state.json"
            self.assertEqual(load_gui_session_state(path), {})

            path.write_text("{not valid json", encoding="utf-8")
            self.assertEqual(load_gui_session_state(path), {})

    def test_session_state_rejects_unknown_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gui_state.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 999,
                        "values": {"selected_trial": "Static"},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_gui_session_state(path), {})


if __name__ == "__main__":
    unittest.main()
