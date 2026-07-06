from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gui_marker_correspondence import (
    marker_pair_key,
    marker_pair_name,
    marker_pair_to_payload,
    payload_to_tree_values,
    save_marker_correspondence_payload,
    split_marker_labels,
    tree_values_to_payload,
)


class GuiMarkerCorrespondenceTests(unittest.TestCase):
    def test_split_marker_labels_accepts_semicolon_groups(self) -> None:
        self.assertEqual(split_marker_labels("LIAS; RIAS;;"), ["LIAS", "RIAS"])
        self.assertEqual(split_marker_labels(["Q_LH", "Q_RH"]), ["Q_LH", "Q_RH"])
        self.assertEqual(split_marker_labels(""), [])

    def test_tree_values_round_trip_payload_groups(self) -> None:
        payload = tree_values_to_payload(("pelvis", "LIAS;RIAS", "Q_LIAS;Q_RIAS"))

        self.assertEqual(
            payload,
            {
                "name": "pelvis",
                "reference": ["LIAS", "RIAS"],
                "test": ["Q_LIAS", "Q_RIAS"],
            },
        )
        self.assertEqual(
            payload_to_tree_values(payload), ("pelvis", "LIAS;RIAS", "Q_LIAS;Q_RIAS")
        )

    def test_marker_pair_helpers_use_first_labels_for_duplicate_detection(self) -> None:
        payload = marker_pair_to_payload("LIAS", "Q_LIAS")

        self.assertEqual(payload["name"], "LIAS_to_Q_LIAS")
        self.assertEqual(marker_pair_name("LIAS", "Q_LIAS"), "LIAS_to_Q_LIAS")
        self.assertEqual(marker_pair_key(payload), ("LIAS", "Q_LIAS"))
        self.assertIsNone(marker_pair_key({"name": "empty"}))

    def test_save_marker_correspondence_payload_writes_json_list(self) -> None:
        rows = [marker_pair_to_payload("LIAS", "Q_LIAS")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "markers.json"

            output = save_marker_correspondence_payload(path, rows)

            self.assertEqual(output, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), rows)


if __name__ == "__main__":
    unittest.main()
