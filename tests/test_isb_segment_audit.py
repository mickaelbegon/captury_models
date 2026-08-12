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
    assess_biobuddy_template_against_isb,
    build_isb_d1_d3_audit,
    compare_template_frames_to_biomod,
    extract_biobuddy_template_frames,
    load_isb_segment_targets,
    load_biobuddy_audit_sidecars,
    parse_biomod_segment_transforms,
    evaluate_static_frame_pair,
    validate_isb_segment_targets,
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
        self.assertTrue(
            any(
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest() in item
                for item in pelvis["D1_evidence"]
            )
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

    def test_versioned_isb_targets_cover_biobuddy_segments_with_primary_sources(
        self,
    ) -> None:
        targets = load_isb_segment_targets()

        validate_isb_segment_targets(targets)
        self.assertEqual(targets["schema_version"], 1)
        self.assertIn("pelvis", targets["segments"])
        self.assertIn("thorax", targets["segments"])
        self.assertEqual(targets["segments"]["head"]["applicability"], "non_applicable")
        self.assertIn("option 2", targets["segments"]["upper_arm"]["selected_option"])
        self.assertIn("midway", targets["segments"]["hand"]["origin"]["definition"])
        for target in targets["segments"].values():
            if target["applicability"] == "non_applicable":
                continue
            self.assertRegex(target["citation"]["doi"], r"^10\.1016/")
            self.assertTrue(target["origin"]["definition"])
            self.assertEqual(set(target["axes"]), {"X", "Y", "Z"})

    def test_symbolic_assessment_distinguishes_known_deviations_and_non_applicable(
        self,
    ) -> None:
        frames = extract_biobuddy_template_frames(_Template())
        assessment = assess_biobuddy_template_against_isb(
            frames, load_isb_segment_targets()
        )

        self.assertEqual(assessment["pelvis"]["D2_status"], "deviation")
        self.assertEqual(assessment["pelvis"]["D3_status"], "deviation")
        self.assertIn("ISB", assessment["pelvis"]["D2_evidence"][0])

    def test_biomod_parser_and_template_comparison_use_parent_local_transforms(
        self,
    ) -> None:
        biomod = """
segment Pelvis
    parent root
    RTinMatrix 1
    RT
        1 0 0 1
        0 1 0 2
        0 0 1 3
        0 0 0 1
endsegment
segment RThigh
    parent Pelvis
    RTinMatrix 1
    RT
        1 0 0 0
        0 1 0 -1
        0 0 1 0
        0 0 0 1
endsegment
"""
        parsed = parse_biomod_segment_transforms(biomod)
        self.assertEqual(parsed["RThigh"]["parent"], "Pelvis")
        np.testing.assert_allclose(
            parsed["RThigh"]["transform"][:3, 3], [0.0, -1.0, 0.0]
        )

        pelvis = np.eye(4)
        pelvis[:3, 3] = [1.0, 2.0, 3.0]
        thigh = pelvis.copy()
        thigh[1, 3] -= 1.0
        comparison = compare_template_frames_to_biomod(
            {"Pelvis": pelvis, "RThigh": thigh},
            biomod,
            expected_parents={"Pelvis": "base", "RThigh": "Pelvis"},
        )

        self.assertEqual(comparison["status"], "match")
        self.assertLess(comparison["segments"]["RThigh"]["rotation_error_deg"], 1e-10)
        self.assertLess(comparison["segments"]["RThigh"]["origin_error_mm"], 1e-10)
        self.assertEqual(comparison["expected_segment_count"], 2)
        self.assertEqual(comparison["serialized_segment_count"], 2)
        self.assertEqual(comparison["parent_mismatches"], {})

    def test_biomod_roundtrip_rejects_wrong_segment_parent(self) -> None:
        biomod = """
segment Pelvis
    parent root
    RTinMatrix 1
    RT
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
endsegment
segment RThigh
    parent root
    RTinMatrix 1
    RT
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
endsegment
"""

        comparison = compare_template_frames_to_biomod(
            {"Pelvis": np.eye(4), "RThigh": np.eye(4)},
            biomod,
            expected_parents={"Pelvis": "base", "RThigh": "Pelvis"},
        )

        self.assertEqual(comparison["status"], "mismatch")
        self.assertEqual(
            comparison["parent_mismatches"]["RThigh"],
            {"expected": "Pelvis", "serialized": "root"},
        )

    def test_biomod_roundtrip_rejects_missing_expected_segment(self) -> None:
        biomod = """
segment Pelvis
    parent root
    RTinMatrix 1
    RT
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
endsegment
"""

        comparison = compare_template_frames_to_biomod(
            {"Pelvis": np.eye(4), "RThigh": np.eye(4)}, biomod
        )

        self.assertEqual(comparison["status"], "mismatch")
        self.assertEqual(comparison["missing_segments"], ["RThigh"])

    def test_static_frame_pair_reports_rotation_origin_and_frame_quality(self) -> None:
        source = np.eye(4)
        angle = np.deg2rad(10.0)
        target = np.eye(4)
        target[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        target[:3, 3] = [0.001, 0.002, 0.002]

        result = evaluate_static_frame_pair(source, target)

        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["angular_deviation_deg"], 10.0)
        self.assertAlmostEqual(result["origin_deviation_mm"], 3.0)
        self.assertAlmostEqual(result["source_determinant"], 1.0)
        self.assertAlmostEqual(result["target_determinant"], 1.0)

    def test_biobuddy_sidecars_are_loaded_only_from_the_selected_biomod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            biomod = Path(tmp) / "motive_57.bioMod"
            biomod.write_text("version 4\n", encoding="utf-8")
            biomod.with_suffix(".roundtrip.json").write_text(
                json.dumps(
                    {
                        "status": "match",
                        "biomod_sha256": hashlib.sha256(
                            biomod.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            biomod.with_suffix(".isb_static.json").write_text(
                json.dumps(
                    {
                        "status": "available",
                        "segments": {},
                        "biomod_sha256": hashlib.sha256(
                            biomod.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            sidecars = load_biobuddy_audit_sidecars(biomod)

        self.assertEqual(sidecars["biomod_verification"]["status"], "match")
        self.assertEqual(sidecars["static_evaluation"]["status"], "available")

    def test_stale_biobuddy_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            biomod = Path(tmp) / "motive_57.bioMod"
            biomod.write_text("version 4\n", encoding="utf-8")
            biomod.with_suffix(".roundtrip.json").write_text(
                json.dumps({"status": "match", "biomod_sha256": "stale"}),
                encoding="utf-8",
            )

            sidecars = load_biobuddy_audit_sidecars(biomod)

        self.assertEqual(sidecars["biomod_verification"]["status"], "stale")


if __name__ == "__main__":
    unittest.main()
