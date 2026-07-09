#!/usr/bin/env python3
"""Run biorbd QLD inverse kinematics from a BioBuddy C3D-built model.

This script is intentionally narrow: it consumes an existing ``.bioMod`` and a
C3D marker trial, matches C3D marker labels to model marker names, and solves
``biorbd.InverseKinematics`` with the least-squares QLD backend. It is used by
the Tk GUI immediately after creating a Motive 57 BioBuddy model, and by the P6
batch comparison when ``--run-ik-batch`` is enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bvh_c3d_biobuddy_pyorerun_compare import (
    biorbd_strings_to_list,
    finite_difference_by_time,
    require_biorbd,
    solve_inverse_kinematics_least_squares,
    split_c3d_points,
)

ANGLE_LABEL_REGEX = r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)"
DEFAULT_MARKER_PREFIXES_TO_STRIP = ("Skeleton_001_",)


def stripped_marker_label(label: str, prefixes: tuple[str, ...]) -> str:
    """Return ``label`` without the first matching acquisition prefix."""

    for prefix in prefixes:
        if prefix and label.startswith(prefix):
            return label[len(prefix) :]
    return label


def marker_name_index(names: list[str]) -> dict[str, int]:
    """Return unique marker names and drop duplicates from automatic matching."""

    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {name: index for index, name in enumerate(names) if counts[name] == 1}


def build_direct_marker_data(
    model: Any,
    c3d_path: Path,
    *,
    biomod_unit_scale_to_m: float,
    marker_prefixes_to_strip: tuple[str, ...],
    angle_label_regex: str,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Build a ``3 x model_markers x frames`` marker array for biorbd IK."""

    split = split_c3d_points(
        c3d_path,
        bvh_unit_scale_to_m=biomod_unit_scale_to_m,
        angle_label_regex=angle_label_regex,
    )
    n_frames = (
        split.time.shape[0] if max_frames <= 0 else min(max_frames, split.time.shape[0])
    )
    frame_indices = np.arange(n_frames, dtype=int)
    c3d_labels = [
        stripped_marker_label(label, marker_prefixes_to_strip)
        for label in split.marker_labels
    ]
    c3d_index_by_label = marker_name_index(c3d_labels)
    model_marker_names = biorbd_strings_to_list(model.markerNames())
    technical_marker_names = set(biorbd_strings_to_list(model.technicalMarkerNames()))
    marker_data = np.full((3, len(model_marker_names), n_frames), np.nan, dtype=float)
    used: list[str] = []
    missing: list[str] = []
    c3d_markers_model_units = split.marker_data_native * (
        split.c3d_unit_scale_to_m / biomod_unit_scale_to_m
    )
    nb_technical_markers = int(model.nbTechnicalMarkers())
    for model_index, marker_name in enumerate(model_marker_names):
        if (
            marker_name not in technical_marker_names
            or model_index >= nb_technical_markers
        ):
            missing.append(marker_name)
            continue
        c3d_index = c3d_index_by_label.get(marker_name)
        if c3d_index is None:
            missing.append(marker_name)
            continue
        marker_data[:, model_index, :] = c3d_markers_model_units[:, c3d_index, :][
            :, frame_indices
        ]
        used.append(marker_name)
    if not used:
        raise RuntimeError(
            "Aucun marqueur C3D ne correspond aux marqueurs du modèle BioBuddy."
        )
    report = {
        "c3d": str(c3d_path),
        "biomod_unit_scale_to_m": biomod_unit_scale_to_m,
        "marker_prefixes_to_strip": list(marker_prefixes_to_strip),
        "model_markers": int(model.nbMarkers()),
        "technical_model_markers": nb_technical_markers,
        "markers_used": len(used),
        "used_marker_names": used,
        "missing_model_markers": missing,
        "angle_channels_ignored": split.angle_labels,
    }
    return marker_data, split.time[frame_indices], used, report


def run_direct_biobuddy_ik(
    biomod: Path,
    c3d: Path,
    out_dir: Path,
    *,
    source_name: str,
    biomod_unit_scale_to_m: float = 1.0,
    marker_prefixes_to_strip: tuple[str, ...] = DEFAULT_MARKER_PREFIXES_TO_STRIP,
    angle_label_regex: str = ANGLE_LABEL_REGEX,
    max_frames: int = 0,
    method: str = "trf",
) -> dict[str, Any]:
    """Solve QLD IK and store compact NPZ/JSON outputs."""

    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod))
    marker_data, time, used_markers, report = build_direct_marker_data(
        model,
        c3d,
        biomod_unit_scale_to_m=biomod_unit_scale_to_m,
        marker_prefixes_to_strip=marker_prefixes_to_strip,
        angle_label_regex=angle_label_regex,
        max_frames=max_frames,
    )
    q, _qdot, _qddot = solve_inverse_kinematics_least_squares(
        model=model, marker_data=marker_data, time=time, method=method
    )
    qdot = finite_difference_by_time(q, time)
    qddot = finite_difference_by_time(qdot, time)
    q_names = biorbd_strings_to_list(model.nameDof())

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{source_name}_inverse_kinematics_qld.npz"
    summary_path = out_dir / f"{source_name}_inverse_kinematics_qld_summary.json"
    np.savez(
        npz_path,
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        q_names=np.asarray(q_names, dtype=object),
        marker_names=np.asarray(used_markers, dtype=object),
        solver="biorbd.InverseKinematics:least_squares",
        least_squares_method=method,
    )
    report.update(
        {
            "status": "ok",
            "biomod": str(biomod),
            "solver": "biorbd.InverseKinematics:least_squares",
            "least_squares_method": method,
            "frames": int(time.shape[0]),
            "nb_q": int(q.shape[0]),
            "outputs": {"npz": str(npz_path), "summary": str(summary_path)},
        }
    )
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct BioBuddy/biorbd QLD IK from a bioMod and C3D."
    )
    parser.add_argument("--biomod", type=Path, required=True)
    parser.add_argument("--c3d", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-name", default="biobuddy")
    parser.add_argument("--biomod-unit-scale-to-m", type=float, default=1.0)
    parser.add_argument("--strip-marker-prefix", action="append", default=[])
    parser.add_argument("--angle-label-regex", default=ANGLE_LABEL_REGEX)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--method", default="trf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefixes = (
        tuple(args.strip_marker_prefix)
        if args.strip_marker_prefix
        else DEFAULT_MARKER_PREFIXES_TO_STRIP
    )
    report = run_direct_biobuddy_ik(
        args.biomod,
        args.c3d,
        args.out_dir,
        source_name=args.source_name,
        biomod_unit_scale_to_m=args.biomod_unit_scale_to_m,
        marker_prefixes_to_strip=prefixes,
        angle_label_regex=args.angle_label_regex,
        max_frames=args.max_frames,
        method=args.method,
    )
    print(f"BioBuddy QLD IK: {report['outputs']['npz']}")
    print(f"Markers used: {report['markers_used']}/{report['model_markers']}")


if __name__ == "__main__":
    main()
