from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gui_graphs import read_table_npz
from isb_segment_audit import (
    build_isb_d1_d3_audit,
    extract_biobuddy_template_frames,
    write_isb_d1_d3_audit,
)
from kinematic_conventions import load_kinematic_conventions


class _Endpoint:
    def __init__(self, *marker_names: str) -> None:
        self.marker_names = marker_names


class _FunctionalCenter:
    def __init__(self) -> None:
        self.method = "score"
        self.trial_name = "right_hip_score"
        self.parent_marker_names = ("LIAS", "RIAS", "LIPS", "RIPS")
        self.child_marker_names = ("RFTC", "RTH", "RFLE", "RFME")
        self.fallback = _Endpoint("RIAS", "RIPS", "RFTC")


class _Axis:
    def __init__(self, name: int, start: object, end: object) -> None:
        self.name = name
        self.start = start
        self.end = end


class _FunctionalAxis:
    def __init__(self) -> None:
        self.method = "sara_direction"
        self.trial_name = "right_knee_sara"
        self.fallback = _Axis(2, _Endpoint("RFME"), _Endpoint("RFLE"))
        self.parent_marker_names = ("RFTC", "RTH", "RFLE", "RFME")
        self.child_marker_names = ("RFAX", "RSK", "RTTC", "RFAL", "RTAM")
        self.expected_axis = self.fallback
        self.origin_marker_names = ("RFLE", "RFME")
        self.max_static_axis_deviation_degrees = 30.0


class _Frame:
    def __init__(self, origin: object, second_axis: object) -> None:
        self.origin = origin
        self.first_axis = _Axis(1, _Endpoint("RFLE", "RFME"), origin)
        self.second_axis = second_axis
        self.axis_to_keep = 1


class _Segment:
    def __init__(self, name: str, frame: object) -> None:
        self.name = name
        self.parent_name = "Pelvis" if name != "Pelvis" else "root"
        self.frame = frame


class _Template:
    def __init__(self) -> None:
        self.name = "synthetic Motive 57"
        self.segments = (
            _Segment(
                "Pelvis",
                _Frame(
                    _Endpoint("LIAS", "RIAS", "LIPS", "RIPS"),
                    _Axis(
                        0,
                        _Endpoint("LIPS", "RIPS"),
                        _Endpoint("LIAS", "RIAS"),
                    ),
                ),
            ),
            _Segment("RThigh", _Frame(_FunctionalCenter(), _FunctionalAxis())),
        )


class IsbSegmentAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_kinematic_conventions()

    def test_audit_covers_every_source_segment_and_preserves_unknowns(self) -> None:
        report = build_isb_d1_d3_audit(
            self.registry,
            biobuddy_template_loader=lambda: (_Template(), Path(__file__)),
        )

        expected = sum(
            len(source["segments"]) for source in self.registry["sources"].values()
        )
        self.assertEqual(len(report["segments"]), expected)
        captury_pelvis = next(
            row
            for row in report["segments"]
            if row["source_id"] == "captury_model" and row["segment_id"] == "pelvis"
        )
        self.assertEqual(captury_pelvis["D1_status"], "inconnu")
        self.assertIsNone(captury_pelvis["angular_deviation_deg"])
        self.assertEqual(
            captury_pelvis["numeric_status"], "source_to_isb_rotation_unknown"
        )

    def test_numeric_deviation_is_only_emitted_for_documented_rotation(self) -> None:
        registry = copy.deepcopy(self.registry)
        segment = registry["sources"]["captury_model"]["segments"]["pelvis"]
        angle = np.deg2rad(30.0)
        segment["source_to_isb_rotation"] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]

        report = build_isb_d1_d3_audit(
            registry, biobuddy_template_loader=lambda: (_Template(), Path(__file__))
        )
        row = next(
            item
            for item in report["segments"]
            if item["source_id"] == "captury_model" and item["segment_id"] == "pelvis"
        )

        self.assertEqual(row["numeric_status"], "available")
        self.assertAlmostEqual(row["angular_deviation_deg"], 30.0)
        np.testing.assert_allclose(row["rotation_vector_deg"], [0.0, 0.0, 30.0])
        self.assertIsNone(row["origin_deviation_mm"])

    def test_half_turn_rotation_has_a_finite_rotation_vector(self) -> None:
        registry = copy.deepcopy(self.registry)
        segment = registry["sources"]["captury_model"]["segments"]["pelvis"]
        segment["source_to_isb_rotation"] = np.diag([1.0, -1.0, -1.0]).tolist()

        report = build_isb_d1_d3_audit(
            registry,
            biobuddy_template_loader=lambda: (_Template(), Path(__file__)),
        )
        row = next(
            item
            for item in report["segments"]
            if item["source_id"] == "captury_model" and item["segment_id"] == "pelvis"
        )

        self.assertAlmostEqual(row["angular_deviation_deg"], 180.0)
        self.assertTrue(np.isfinite(row["rotation_vector_deg"]).all())
        self.assertAlmostEqual(np.linalg.norm(row["rotation_vector_deg"]), 180.0)

    def test_template_extraction_records_axes_functional_methods_and_fallbacks(
        self,
    ) -> None:
        frames = extract_biobuddy_template_frames(_Template())

        pelvis = frames["Pelvis"]
        self.assertEqual(
            pelvis["origin"]["marker_names"], ["LIAS", "RIAS", "LIPS", "RIPS"]
        )
        self.assertEqual(pelvis["first_axis"]["axis"], "Y")
        self.assertEqual(pelvis["axis_to_keep"], "Y")
        thigh = frames["RThigh"]
        self.assertEqual(thigh["origin"]["functional_method"], "score")
        self.assertEqual(
            thigh["origin"]["fallback"]["marker_names"], ["RIAS", "RIPS", "RFTC"]
        )
        self.assertEqual(thigh["second_axis"]["functional_method"], "sara_direction")
        self.assertEqual(thigh["second_axis"]["fallback"]["axis"], "Z")

    def test_template_provenance_uses_the_file_actually_loaded(self) -> None:
        report = build_isb_d1_d3_audit(
            self.registry,
            biobuddy_template_loader=lambda: (_Template(), Path(__file__)),
        )

        provenance = report["biobuddy_template"]
        self.assertEqual(Path(provenance["path"]), Path(__file__).resolve())
        self.assertEqual(
            provenance["sha256"],
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def test_documented_biobuddy_status_has_runtime_construction_evidence(self) -> None:
        report = build_isb_d1_d3_audit(
            self.registry,
            biobuddy_template_loader=lambda: (_Template(), Path(__file__)),
        )
        pelvis = next(
            row
            for row in report["segments"]
            if row["source_id"] == "biobuddy_motive57" and row["segment_id"] == "pelvis"
        )

        self.assertEqual(pelvis["D1_status"], "documente_non_evalue")
        self.assertTrue(pelvis["D1_evidence"])
        self.assertIn(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            pelvis["D1_evidence"][0],
        )

    def test_missing_biobuddy_import_is_explicit_and_does_not_crash(self) -> None:
        def unavailable():
            raise ImportError("synthetic missing BioBuddy")

        report = build_isb_d1_d3_audit(
            self.registry, biobuddy_template_loader=unavailable
        )

        self.assertEqual(report["biobuddy_template"]["status"], "unavailable")
        self.assertIn(
            "synthetic missing BioBuddy", report["biobuddy_template"]["reason"]
        )
        biobuddy_rows = [
            row for row in report["segments"] if row["source_id"] == "biobuddy_motive57"
        ]
        self.assertTrue(biobuddy_rows)
        self.assertTrue(
            all(
                row["runtime_template_status"] == "unavailable"
                and row["template_frame"] is None
                for row in biobuddy_rows
            )
        )
        self.assertTrue(
            all(row["D1_status"] == "documente_non_evalue" for row in biobuddy_rows)
        )

    def test_incompatible_biobuddy_template_api_is_explicit(self) -> None:
        def incompatible():
            raise TypeError("synthetic incompatible template API")

        report = build_isb_d1_d3_audit(
            self.registry, biobuddy_template_loader=incompatible
        )

        self.assertEqual(report["biobuddy_template"]["status"], "unavailable")
        self.assertIn("TypeError", report["biobuddy_template"]["reason"])

    def test_json_and_npz_outputs_roundtrip(self) -> None:
        report = build_isb_d1_d3_audit(
            self.registry,
            biobuddy_template_loader=lambda: (_Template(), Path(__file__)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_isb_d1_d3_audit(Path(tmp), report)
            loaded_json = json.loads(paths["json"].read_text(encoding="utf-8"))
            loaded_table = read_table_npz(paths["table"])

        self.assertEqual(loaded_json["audit_version"], 1)
        self.assertEqual(len(loaded_table), len(report["segments"]))
        self.assertIn("D3_status", loaded_table.columns)
        self.assertFalse(paths["json"].suffix == ".csv")


if __name__ == "__main__":
    unittest.main()
