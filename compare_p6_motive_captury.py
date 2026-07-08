#!/usr/bin/env python3
"""Batch comparison for Captury/Motive kinematic datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bvh_c3d_biobuddy_pyorerun_compare import (
    append_joint_centre_markers_to_biomod,
    as_str_list,
    build_biomod_from_bvh_with_biobuddy,
    build_biomod_from_fbx_with_biobuddy,
    clone_c3d_dict,
    collect_fbx_joint_names_depth_first,
    compute_model_joint_centres_native,
    convert_biobuddy_ply_meshes_to_vtp,
    extract_q_from_biobuddy_bvh_parser,
    extract_q_from_fbx_parser,
    get_c3d_param,
    interpolate_array,
    require_biorbd,
    require_ezc3d,
    require_pyorerun,
    save_model_joint_centres,
    save_q_outputs,
    split_c3d_points,
)
from compare_capture_systems import (
    DEFAULT_LANDMARK_MAP,
    detect_angle_indices,
    load_landmark_map,
    unit_scale_to_mm,
)
from model_comparison_metrics import joint_center_error_xyz, waveform_metrics

DEFAULT_DATA_ROOT = Path("local_trials/2026-06-30_P6_flat")
DEFAULT_OUTPUT_ROOT = Path("out_p6_motive_captury_comparison")
ANGLE_LABEL_REGEX = r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)"
FOOT_MARKER_PATTERN = r"(LFCC|RFCC|LFM|RFM|LDP|RDP|Foot|Toe|Heel)"
CACHE_VERSION = 3

MODEL_JOINT_MARKER_PROXIES = {
    "Hips": ("LIAS", "RIAS", "LIPS", "RIPS"),
    "LeftUpLeg": ("LIAS", "LIPS", "LFTC"),
    "RightUpLeg": ("RIAS", "RIPS", "RFTC"),
    "LeftLeg": ("LFLE", "LFME"),
    "RightLeg": ("RFLE", "RFME"),
    "LeftFoot": ("LFAL", "LTAM", "LFAX"),
    "RightFoot": ("RFAL", "RTAM", "RFAX"),
}


@dataclass
class TrialBundle:
    name: str
    captury_c3d: Path
    captury_bvh: Path | None
    captury_fbx: Path | None
    motive_c3d: Path
    motive_bvh: Path | None
    motive_fbx: Path | None


@dataclass
class ModelRun:
    system: str
    source_kind: str
    biomod_path: Path
    q: np.ndarray
    q_names: list[str]
    q_units: list[str]
    time: np.ndarray
    joint_names: list[str]
    centres_native: dict[str, np.ndarray]
    rotations_native: dict[str, np.ndarray]
    unit_scale_to_m: float
    mesh_report: dict[str, Any]
    root_offset_policy: dict[str, Any]


def file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _static_alignment_cache_payload(
    transform: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any] | None:
    if transform is None:
        return None
    rotation, translation = transform
    return {
        "rotation": np.asarray(rotation, dtype=float).round(12).tolist(),
        "translation_mm": np.asarray(translation, dtype=float).round(12).tolist(),
    }


def trial_cache_fingerprint(
    bundle: TrialBundle,
    args: argparse.Namespace,
    static_alignment_transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    payload = {
        "cache_version": CACHE_VERSION,
        "implementation": file_fingerprint(Path(__file__)),
        "inputs": {
            "captury_c3d": file_fingerprint(bundle.captury_c3d),
            "captury_bvh": file_fingerprint(bundle.captury_bvh),
            "captury_fbx": file_fingerprint(bundle.captury_fbx),
            "motive_c3d": file_fingerprint(bundle.motive_c3d),
            "motive_bvh": file_fingerprint(bundle.motive_bvh),
            "motive_fbx": file_fingerprint(bundle.motive_fbx),
            "biobuddy_biomod": (
                file_fingerprint(args.biobuddy_biomod) if args.biobuddy_biomod else None
            ),
        },
        "options": {
            "model_source": args.model_source,
            "model_to_c3d_axis": args.model_to_c3d_axis,
            "captury_unit_scale_to_m": args.captury_unit_scale_to_m,
            "motive_unit_scale_to_m": args.motive_unit_scale_to_m,
            "biobuddy_unit_scale_to_m": args.biobuddy_unit_scale_to_m,
            "root_offset_mode": args.root_offset_mode,
            "angle_label_regex": args.angle_label_regex,
            "c3d_angle_unit": args.c3d_angle_unit,
            "landmark_map": (
                file_fingerprint(args.landmark_map) if args.landmark_map else None
            ),
            "segment_reference": args.segment_reference,
            "captury_reorient_thigh_y_from_cor": bool(
                args.captury_reorient_thigh_y_from_cor
            ),
            "rotate_body_segments_180_x": bool(args.rotate_body_segments_180_x),
            "disable_static_model_alignment": bool(args.disable_static_model_alignment),
            "disable_motive_marker_alignment": bool(
                args.disable_motive_marker_alignment
            ),
            "joint_filter": list(args.joint_filter),
            "no_mesh": bool(args.no_mesh),
            "max_mesh_points": int(args.max_mesh_points),
            "run_ik_batch": bool(args.run_ik_batch),
            "ik_max_frames": int(args.ik_max_frames),
            "cut_mode": args.cut_mode,
            "time_start": args.time_start,
            "time_end": args.time_end,
            "no_figures": bool(args.no_figures),
        },
        "static_alignment": _static_alignment_cache_payload(static_alignment_transform),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "version": CACHE_VERSION,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def required_trial_outputs(trial_dir: Path, trial_name: str) -> list[Path]:
    safe_trial = safe_name(trial_name)
    return [
        trial_dir / f"{safe_trial}_motive_with_capjc_motjc.c3d",
        trial_dir / "joint_centre_metrics.csv",
        trial_dir / "joint_centre_timeseries.npz",
        trial_dir / "kinematics_q_metrics.csv",
        trial_dir / "kinematics_q_timeseries.npz",
        trial_dir / "captury_c3d_angle_metrics.csv",
        trial_dir / "captury_c3d_angle_timeseries.npz",
        trial_dir / "segment_rotation_metrics.csv",
        trial_dir / "segment_rotation_timeseries.npz",
        trial_dir / "model_dimensions.csv",
        trial_dir / "motive_marker_occlusions.csv",
        trial_dir / "skin_marker_correspondence_metrics.csv",
        trial_dir / "skin_marker_correspondence_timeseries.npz",
        trial_dir / "trial_events_contacts.csv",
        trial_dir / "run_report.json",
    ]


def cached_trial_report(
    trial_dir: Path,
    bundle: TrialBundle,
    args: argparse.Namespace,
    static_alignment_transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    if args.no_cache:
        return None
    report_path = trial_dir / "run_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = trial_cache_fingerprint(bundle, args, static_alignment_transform)
    if report.get("cache", {}).get("fingerprint") != expected:
        return None
    for output_path in required_trial_outputs(trial_dir, bundle.name):
        if not output_path.exists() or output_path.stat().st_size == 0:
            return None
    if args.run_ik_batch and "motive_ik_batch" not in report:
        return None
    return report


def static_transform_from_report(
    report: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    alignment = report.get("alignment", {})
    rotation = alignment.get("rotation")
    translation = alignment.get("translation_mm")
    if rotation is None or translation is None:
        return None
    try:
        return np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    except (TypeError, ValueError):
        return None


def safe_name(value: str) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "trial"


def discover_trials(data_root: Path) -> list[TrialBundle]:
    if (data_root / "Captury").is_dir() and (data_root / "Motive").is_dir():
        return discover_flat_trials(data_root)

    trials: list[TrialBundle] = []
    for trial_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        captury_dir = trial_dir / "captury"
        motive_dir = trial_dir / "squelettes"
        if not captury_dir.is_dir() or not motive_dir.is_dir():
            continue
        captury_c3d = captury_dir / "P6.c3d"
        motive_c3ds = sorted(motive_dir.glob("*.c3d"))
        motive_bvhs = sorted(motive_dir.glob("*Skeleton 001.bvh"))
        motive_fbxs = sorted(motive_dir.glob("*.fbx"))
        if not captury_c3d.exists() or not motive_c3ds:
            continue
        trials.append(
            TrialBundle(
                name=trial_dir.name,
                captury_c3d=captury_c3d,
                captury_bvh=first_existing(captury_dir / "P6.bvh"),
                captury_fbx=first_existing(captury_dir / "P6.fbx"),
                motive_c3d=motive_c3ds[0],
                motive_bvh=motive_bvhs[0] if motive_bvhs else None,
                motive_fbx=motive_fbxs[0] if motive_fbxs else None,
            )
        )
    return trials


def captury_flat_trial_name(path: Path) -> str:
    name = path.stem
    return name[: -len("_P6")] if name.endswith("_P6") else name


def motive_flat_trial_name(path: Path) -> str:
    name = path.stem
    if name.startswith("P6_"):
        name = name[3:]
    if name.endswith("_Skeleton 001"):
        name = name[: -len("_Skeleton 001")]
    return name


def discover_flat_trials(data_root: Path) -> list[TrialBundle]:
    captury_dir = data_root / "Captury"
    motive_dir = data_root / "Motive"
    captury: dict[str, dict[str, Path]] = {}
    motive: dict[str, dict[str, Path]] = {}

    for path in sorted(captury_dir.glob("*")):
        if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
            continue
        captury.setdefault(captury_flat_trial_name(path), {})[
            path.suffix.lower().lstrip(".")
        ] = path
    for path in sorted(motive_dir.glob("*")):
        if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
            continue
        motive.setdefault(motive_flat_trial_name(path), {})[
            path.suffix.lower().lstrip(".")
        ] = path

    trials: list[TrialBundle] = []
    for trial in sorted(set(captury).intersection(motive)):
        if "c3d" not in captury[trial] or "c3d" not in motive[trial]:
            continue
        trials.append(
            TrialBundle(
                name=trial,
                captury_c3d=captury[trial]["c3d"],
                captury_bvh=captury[trial].get("bvh"),
                captury_fbx=captury[trial].get("fbx"),
                motive_c3d=motive[trial]["c3d"],
                motive_bvh=motive[trial].get("bvh"),
                motive_fbx=motive[trial].get("fbx"),
            )
        )
    return trials


def first_existing(path: Path) -> Path | None:
    return path if path.exists() else None


def select_model_file(
    bundle: TrialBundle, system: str, model_source: str
) -> tuple[str, Path]:
    bvh = bundle.captury_bvh if system == "captury" else bundle.motive_bvh
    fbx = bundle.captury_fbx if system == "captury" else bundle.motive_fbx
    if model_source == "bvh":
        if bvh is None:
            raise FileNotFoundError(f"No {system} BVH for {bundle.name}.")
        return "bvh", bvh
    if model_source == "fbx":
        if fbx is None:
            raise FileNotFoundError(f"No {system} FBX for {bundle.name}.")
        return "fbx", fbx
    if bvh is not None:
        return "bvh", bvh
    if fbx is not None:
        return "fbx", fbx
    raise FileNotFoundError(f"No {system} BVH/FBX for {bundle.name}.")


def native_unit_scale_to_m(
    system: str, source_kind: str, user_scale: float | None
) -> float:
    if user_scale is not None:
        return user_scale
    if system == "motive":
        return 0.01
    return 0.001


def biorbd_segment_names(model: Any) -> list[str]:
    return [
        model.segment(index).name().to_string() for index in range(model.nbSegment())
    ]


def compute_model_segment_rotations_native(
    biomod_path: Path,
    q: np.ndarray,
    keep_segment_names: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract segment global rotation matrices from a biorbd model.

    ``biorbd.Model.globalJCS(q, i)`` returns the homogeneous transform of each
    segment. This helper stores the rotational ``3x3`` block for every requested
    non-root segment as an array shaped ``(3, 3, n_frames)``.
    """

    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    if q.shape[0] != model.nbQ():
        raise RuntimeError(f"{biomod_path} expects {model.nbQ()} q, got {q.shape[0]}.")
    names = biorbd_segment_names(model)
    rotations: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        if name == "root":
            continue
        if keep_segment_names is not None and name not in keep_segment_names:
            continue
        rotations[name] = np.zeros((3, 3, q.shape[1]), dtype=float)
    for frame in range(q.shape[1]):
        q_frame = np.ascontiguousarray(q[:, frame], dtype=float)
        for index, name in enumerate(names):
            if name not in rotations:
                continue
            rt = np.asarray(model.globalJCS(q_frame, index).to_array(), dtype=float)
            rotations[name][:, :, frame] = rt[:3, :3]
    return rotations


def build_model_run(
    bundle: TrialBundle,
    system: str,
    model_source: str,
    out_dir: Path,
    include_mesh: bool,
    max_mesh_points: int,
    unit_scale_override: float | None,
    root_offset_mode: str,
    model_to_c3d_axis: str,
    angle_label_regex: str,
) -> ModelRun:
    source_kind, source_path = select_model_file(bundle, system, model_source)
    source_dir = out_dir / system / source_kind
    source_dir.mkdir(parents=True, exist_ok=True)
    biomod_path = source_dir / f"{system}_{source_kind}_biobuddy.bioMod"
    unit_scale_to_m = native_unit_scale_to_m(system, source_kind, unit_scale_override)
    c3d_path = bundle.captury_c3d if system == "captury" else bundle.motive_c3d
    _labels, c3d_points_mm, _residuals, c3d_time = read_c3d_points_mm(
        c3d_path, angle_label_regex
    )
    mesh_report: dict[str, Any] = {
        "mesh_file_count": 0,
        "mesh_vertices": 0,
        "mesh_faces": 0,
    }
    if source_kind == "bvh":
        _, parser = build_biomod_from_bvh_with_biobuddy(
            source_path, biomod_path, add_joint_centre_markers=True
        )
        corrected_runtime = extract_q_from_biobuddy_bvh_parser(
            parser, apply_root_offset_correction=True
        )
        uncorrected_runtime = extract_q_from_biobuddy_bvh_parser(
            parser, apply_root_offset_correction=False
        )
        joint_names = corrected_runtime.joint_names
    else:
        _, parser, mesh_report = build_biomod_from_fbx_with_biobuddy(
            source_path,
            biomod_path,
            add_joint_centre_markers=True,
            include_mesh=include_mesh,
            max_mesh_points=max_mesh_points,
        )
        corrected_runtime = extract_q_from_fbx_parser(
            parser, source_path, apply_root_offset_correction=True
        )
        uncorrected_runtime = extract_q_from_fbx_parser(
            parser, source_path, apply_root_offset_correction=False
        )
        joint_names = collect_fbx_joint_names_depth_first(parser)
        mesh_dir = biomod_path.parent / "meshes"
        if include_mesh and mesh_dir.exists():
            try:
                mesh_report = convert_biobuddy_ply_meshes_to_vtp(mesh_dir)
            except Exception:
                pass
    use_correction, root_offset_policy, centres_native = (
        choose_root_offset_policy_in_c3d(
            source_name=f"{system}_{source_kind}",
            biomod_path=biomod_path,
            corrected_q=corrected_runtime.q,
            uncorrected_q=uncorrected_runtime.q,
            q_names=corrected_runtime.q_names,
            time=corrected_runtime.time,
            joint_names=joint_names,
            unit_scale_to_m=unit_scale_to_m,
            model_to_c3d_axis=model_to_c3d_axis,
            c3d_markers_mm=c3d_points_mm,
            c3d_time=c3d_time,
            requested_mode=root_offset_mode,
            out_dir=source_dir,
        )
    )
    runtime = corrected_runtime if use_correction else uncorrected_runtime
    rotations_native = compute_model_segment_rotations_native(
        biomod_path, runtime.q, set(joint_names)
    )
    root_offset_policy.update(
        {
            "source_file": str(source_path),
            "c3d_file": str(c3d_path),
            "root_offset_native": (
                runtime.root_offset_native.tolist()
                if runtime.root_offset_native is not None
                else None
            ),
            "root_offset_correction_applied": bool(
                runtime.root_offset_correction_applied
            ),
            "source_unit_scale_to_m": unit_scale_to_m,
        }
    )
    (source_dir / f"{system}_{source_kind}_root_translation_policy.json").write_text(
        json.dumps(root_offset_policy, indent=2), encoding="utf-8"
    )
    if source_kind == "fbx":
        append_joint_centre_markers_to_biomod(
            biomod_path, joint_names, marker_prefix=f"{system.upper()}JC_"
        )
    save_q_outputs(
        runtime.q,
        runtime.q_names,
        runtime.time,
        source_dir,
        source_name=system,
        q_units=runtime.q_units,
    )
    save_model_joint_centres(centres_native, runtime.time, source_dir, system)
    return ModelRun(
        system=system,
        source_kind=source_kind,
        biomod_path=biomod_path,
        q=runtime.q,
        q_names=runtime.q_names,
        q_units=runtime.q_units,
        time=runtime.time,
        joint_names=joint_names,
        centres_native=centres_native,
        rotations_native=rotations_native,
        unit_scale_to_m=unit_scale_to_m,
        mesh_report=mesh_report,
        root_offset_policy=root_offset_policy,
    )


def model_to_c3d_matrix(axis_mode: str) -> np.ndarray:
    if axis_mode == "auto":
        axis_mode = "y_up_to_z_up"
    if axis_mode == "identity":
        return np.eye(3)
    if axis_mode == "y_up_to_z_up":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    raise ValueError(f"Unsupported axis conversion: {axis_mode}")


def centres_to_c3d_mm(
    centres_native: dict[str, np.ndarray], unit_scale_to_m: float, axis_mode: str
) -> dict[str, np.ndarray]:
    matrix = model_to_c3d_matrix(axis_mode)
    factor = unit_scale_to_m * 1000.0
    return {name: matrix @ (values * factor) for name, values in centres_native.items()}


def rotations_to_c3d(
    rotations_native: dict[str, np.ndarray],
    axis_mode: str,
    row_global_rotation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    matrix = model_to_c3d_matrix(axis_mode)
    if row_global_rotation is not None:
        matrix = np.asarray(row_global_rotation, dtype=float).T @ matrix
    return {
        name: np.einsum("ij,jkf->ikf", matrix, values)
        for name, values in rotations_native.items()
    }


def trim_rotations(
    rotations: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[:, :, mask] for name, values in rotations.items()}


ROTATION_180_AROUND_LOCAL_X = np.diag([1.0, -1.0, -1.0])


def rotate_segment_frames_180_x(
    rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rotate every segment frame by 180 degrees around its local X axis.

    Segment rotation matrices are stored as columns expressing local axes in the
    C3D/global frame. Right multiplication therefore changes the segment-local
    basis while preserving the global trajectory of the segment origin.
    """

    return {
        name: np.einsum("ijf,jk->ikf", values, ROTATION_180_AROUND_LOCAL_X)
        for name, values in rotations.items()
    }


def _safe_unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-12:
        return fallback / fallback_norm
    return np.asarray([0.0, 1.0, 0.0], dtype=float)


def orient_segment_y_from_cor(
    rotations: dict[str, np.ndarray],
    centres_mm: dict[str, np.ndarray],
    segment: str,
    proximal_joint: str,
    distal_joint: str,
) -> dict[str, np.ndarray]:
    """Return rotations where ``segment`` Y axis follows proximal -> distal CoR.

    The X axis keeps the original segment orientation as much as possible by
    projecting the previous X axis onto the plane orthogonal to the corrected Y.
    This yields a right-handed orthonormal frame per frame.
    """

    if (
        segment not in rotations
        or proximal_joint not in centres_mm
        or distal_joint not in centres_mm
    ):
        return rotations

    corrected = dict(rotations)
    original = np.asarray(rotations[segment], dtype=float)
    proximal = np.asarray(centres_mm[proximal_joint], dtype=float)
    distal = np.asarray(centres_mm[distal_joint], dtype=float)
    n_frames = min(original.shape[2], proximal.shape[1], distal.shape[1])
    if n_frames <= 0:
        return rotations

    frames = original.copy()
    for frame in range(n_frames):
        y_axis = _safe_unit_vector(
            distal[:, frame] - proximal[:, frame], original[:, 1, frame]
        )
        x_axis = original[:, 0, frame]
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        if np.linalg.norm(x_axis) <= 1e-12:
            x_axis = np.cross(original[:, 2, frame], y_axis)
        if np.linalg.norm(x_axis) <= 1e-12:
            helper = (
                np.asarray([1.0, 0.0, 0.0])
                if abs(y_axis[0]) < 0.9
                else np.asarray([0.0, 0.0, 1.0])
            )
            x_axis = np.cross(helper, y_axis)
        x_axis = _safe_unit_vector(x_axis, original[:, 0, frame])
        z_axis = _safe_unit_vector(np.cross(x_axis, y_axis), original[:, 2, frame])
        x_axis = _safe_unit_vector(np.cross(y_axis, z_axis), x_axis)
        frames[:, :, frame] = np.column_stack((x_axis, y_axis, z_axis))

    corrected[segment] = frames
    return corrected


def correct_captury_thigh_y_from_cor(
    rotations: dict[str, np.ndarray], centres_mm: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Orient Captury thigh Y axes from hip CoR toward knee CoR."""

    corrected = rotations
    for segment, proximal, distal in (
        ("LeftUpLeg", "LeftUpLeg", "LeftLeg"),
        ("RightUpLeg", "RightUpLeg", "RightLeg"),
    ):
        corrected = orient_segment_y_from_cor(
            corrected, centres_mm, segment, proximal, distal
        )
    return corrected


SEGMENT_RELATIVE_ROTATION_PAIRS = (
    ("SegRel_HipsSpine", "Hips", "Spine"),
    ("SegRel_LeftHip", "Hips", "LeftUpLeg"),
    ("SegRel_RightHip", "Hips", "RightUpLeg"),
    ("SegRel_LeftKnee", "LeftUpLeg", "LeftLeg"),
    ("SegRel_RightKnee", "RightUpLeg", "RightLeg"),
    ("SegRel_LeftAnkle", "LeftLeg", "LeftFoot"),
    ("SegRel_RightAnkle", "RightLeg", "RightFoot"),
    ("SegRel_LeftShoulder", "Spine3", "LeftArm"),
    ("SegRel_RightShoulder", "Spine3", "RightArm"),
    ("SegRel_LeftElbow", "LeftArm", "LeftForeArm"),
    ("SegRel_RightElbow", "RightArm", "RightForeArm"),
)


def segment_relative_rotation_curves(
    rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract segment-relative rotation-vector curves in radians."""

    curves: dict[str, np.ndarray] = {}
    for joint, proximal, distal in SEGMENT_RELATIVE_ROTATION_PAIRS:
        if proximal not in rotations or distal not in rotations:
            continue
        n_frames = min(rotations[proximal].shape[2], rotations[distal].shape[2])
        if n_frames <= 0:
            continue
        values = np.zeros((3, n_frames), dtype=float)
        for frame in range(n_frames):
            values[:, frame] = rotation_deviation_vector(
                rotations[proximal][:, :, frame],
                rotations[distal][:, :, frame],
            )
        curves[joint] = values
    return curves


def segment_relative_q_metric_rows(
    trial: str,
    captury_rotations: dict[str, np.ndarray],
    motive_rotations: dict[str, np.ndarray],
    captury_time: np.ndarray,
    motive_time: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare derived joint angles from corrected segment coordinate systems."""

    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    captury_curves = segment_relative_rotation_curves(captury_rotations)
    motive_curves = segment_relative_rotation_curves(motive_rotations)
    component_names = ("x", "y", "z")
    for joint in sorted(set(captury_curves).intersection(motive_curves)):
        cap_curve = interpolate_array(captury_curves[joint], captury_time, motive_time)
        mot_curve = motive_curves[joint]
        n_frames = min(cap_curve.shape[1], mot_curve.shape[1], motive_time.shape[0])
        if n_frames <= 0:
            continue
        cap_curve = cap_curve[:, :n_frames]
        mot_curve = mot_curve[:, :n_frames]
        for component_index, component in enumerate(component_names):
            q_name = f"{joint}_{component}"
            reference = mot_curve[component_index]
            test = cap_curve[component_index]
            summary_rows.append(
                {
                    "trial": trial,
                    "q_name": q_name,
                    "unit": "rad",
                    "source": "segment_relative_rotation",
                    **waveform_metrics(reference, test, "rad"),
                }
            )
            for frame, time_value in enumerate(motive_time[:n_frames]):
                timeseries_rows.append(
                    {
                        "trial": trial,
                        "time": float(time_value),
                        "q_name": q_name,
                        "motive": float(reference[frame]),
                        "captury": float(test[frame]),
                        "difference": float(test[frame] - reference[frame]),
                    }
                )
    return summary_rows, timeseries_rows


def nearest_time_indices(
    source_time: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    if source_time.size == 0 or target_time.size == 0:
        return np.zeros(0, dtype=int)
    indices = np.searchsorted(source_time, target_time, side="left")
    indices = np.clip(indices, 0, source_time.size - 1)
    previous = np.clip(indices - 1, 0, source_time.size - 1)
    use_previous = np.abs(target_time - source_time[previous]) < np.abs(
        target_time - source_time[indices]
    )
    indices[use_previous] = previous[use_previous]
    return indices


def rotation_deviation_vector(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Return the rotation-vector deviation that maps ``R1`` to ``R2``.

    The implementation follows the logarithm map supplied in the GUI request:
    ``R = R1.T @ R2`` and the returned vector components are expressed in
    radians around the local X/Y/Z axes of the reference orientation.
    """

    R = np.asarray(R1, dtype=float).T @ np.asarray(R2, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    skew_vector = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    )
    if theta < 1e-8:
        return 0.5 * skew_vector
    return theta / (2.0 * np.sin(theta)) * skew_vector


def segment_rotation_metric_rows(
    trial: str,
    rotations_by_source: dict[str, dict[str, np.ndarray]],
    times_by_source: dict[str, np.ndarray],
    reference_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reference_source = reference_source.lower()
    available = sorted(
        source
        for source, rotations in rotations_by_source.items()
        if rotations and times_by_source.get(source, np.asarray([])).size
    )
    report: dict[str, Any] = {
        "requested_reference": reference_source,
        "available_sources": available,
    }
    if reference_source not in rotations_by_source or not rotations_by_source.get(
        reference_source
    ):
        fallback = "motive" if reference_source == "biobuddy" else ""
        if fallback and rotations_by_source.get(fallback):
            report["status"] = "fallback_missing_reference"
            report["effective_reference"] = fallback
            reference_source = fallback
        else:
            report["status"] = "missing_reference"
            return [], [], report
    else:
        report["effective_reference"] = reference_source
    reference_rotations = rotations_by_source[reference_source]
    reference_time = times_by_source[reference_source]
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for source, source_rotations in sorted(rotations_by_source.items()):
        if source == reference_source or not source_rotations:
            continue
        source_time = times_by_source[source]
        source_indices = nearest_time_indices(source_time, reference_time)
        common_segments = sorted(
            set(reference_rotations).intersection(source_rotations)
        )
        for segment in common_segments:
            values: list[dict[str, Any]] = []
            for frame, source_frame in enumerate(source_indices):
                vector_rad = rotation_deviation_vector(
                    reference_rotations[segment][:, :, frame],
                    source_rotations[segment][:, :, source_frame],
                )
                vector_deg = np.degrees(vector_rad)
                global_deg = float(np.linalg.norm(vector_deg))
                row = {
                    "trial": trial,
                    "reference": reference_source,
                    "source": source,
                    "segment": segment,
                    "time": float(reference_time[frame]),
                    "global_deg": global_deg,
                    "x_deg": float(vector_deg[0]),
                    "y_deg": float(vector_deg[1]),
                    "z_deg": float(vector_deg[2]),
                    "abs_x_deg": float(abs(vector_deg[0])),
                    "abs_y_deg": float(abs(vector_deg[1])),
                    "abs_z_deg": float(abs(vector_deg[2])),
                }
                values.append(row)
                timeseries_rows.append(row)
            if not values:
                continue
            global_values = np.asarray([row["global_deg"] for row in values])
            abs_x = np.asarray([row["abs_x_deg"] for row in values])
            abs_y = np.asarray([row["abs_y_deg"] for row in values])
            abs_z = np.asarray([row["abs_z_deg"] for row in values])
            summary_rows.append(
                {
                    "trial": trial,
                    "reference": reference_source,
                    "source": source,
                    "segment": segment,
                    "median_global_deg": float(np.nanmedian(global_values)),
                    "p95_global_deg": float(np.nanpercentile(global_values, 95)),
                    "max_global_deg": float(np.nanmax(global_values)),
                    "median_abs_x_deg": float(np.nanmedian(abs_x)),
                    "median_abs_y_deg": float(np.nanmedian(abs_y)),
                    "median_abs_z_deg": float(np.nanmedian(abs_z)),
                    "p95_abs_x_deg": float(np.nanpercentile(abs_x, 95)),
                    "p95_abs_y_deg": float(np.nanpercentile(abs_y, 95)),
                    "p95_abs_z_deg": float(np.nanpercentile(abs_z, 95)),
                }
            )
    report["status"] = "ok" if summary_rows else "no_common_segments"
    return summary_rows, timeseries_rows, report


def root_alignment_score_mm(
    centres_c3d_mm: dict[str, np.ndarray],
    source_time: np.ndarray,
    c3d_markers_mm: np.ndarray,
    c3d_time: np.ndarray,
    max_frames: int = 120,
) -> float:
    if not centres_c3d_mm or c3d_markers_mm.size == 0:
        return float("inf")
    stacked = np.stack(list(centres_c3d_mm.values()), axis=1)
    centres_on_c3d = interpolate_array(stacked, source_time, c3d_time)
    n_frames = centres_on_c3d.shape[2]
    frame_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)
    frame_scores: list[float] = []
    for frame in frame_indices:
        centres = centres_on_c3d[:, :, frame].T
        markers = c3d_markers_mm[:, :, frame].T
        finite_centres = np.all(np.isfinite(centres), axis=1)
        finite_markers = np.all(np.isfinite(markers), axis=1)
        centres = centres[finite_centres]
        markers = markers[finite_markers]
        if centres.size == 0 or markers.size == 0:
            continue
        distances = np.linalg.norm(centres[:, None, :] - markers[None, :, :], axis=2)
        frame_scores.append(float(np.nanmedian(np.nanmin(distances, axis=1))))
    return float(np.nanmedian(frame_scores)) if frame_scores else float("inf")


def choose_root_offset_policy_in_c3d(
    source_name: str,
    biomod_path: Path,
    corrected_q: np.ndarray,
    uncorrected_q: np.ndarray,
    q_names: list[str],
    time: np.ndarray,
    joint_names: list[str],
    unit_scale_to_m: float,
    model_to_c3d_axis: str,
    c3d_markers_mm: np.ndarray,
    c3d_time: np.ndarray,
    requested_mode: str,
    out_dir: Path,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    corrected_centres = compute_model_joint_centres_native(
        biomod_path, corrected_q, set(joint_names)
    )
    uncorrected_centres = compute_model_joint_centres_native(
        biomod_path, uncorrected_q, set(joint_names)
    )
    corrected_score = root_alignment_score_mm(
        centres_to_c3d_mm(corrected_centres, unit_scale_to_m, model_to_c3d_axis),
        time,
        c3d_markers_mm,
        c3d_time,
    )
    uncorrected_score = root_alignment_score_mm(
        centres_to_c3d_mm(uncorrected_centres, unit_scale_to_m, model_to_c3d_axis),
        time,
        c3d_markers_mm,
        c3d_time,
    )
    if requested_mode == "subtract":
        use_correction = True
    elif requested_mode == "keep":
        use_correction = False
    else:
        use_correction = corrected_score <= uncorrected_score
    report = {
        "source": source_name,
        "requested_mode": requested_mode,
        "selected_mode": (
            "subtract_static_offset_from_root_q"
            if use_correction
            else "keep_root_q_as_file"
        ),
        "score_mm_subtract_static_offset": corrected_score,
        "score_mm_keep_file_translation": uncorrected_score,
        "score_frame": "c3d_mm_after_model_to_c3d_axis",
        "model_to_c3d_axis": model_to_c3d_axis,
        "q_names": q_names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{source_name}_root_translation_policy.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    selected_centres = corrected_centres if use_correction else uncorrected_centres
    return use_correction, report, selected_centres


def kabsch_rows(
    reference: np.ndarray, moving: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ref_mean = np.mean(reference, axis=0)
    moving_mean = np.mean(moving, axis=0)
    ref_centered = reference - ref_mean
    moving_centered = moving - moving_mean
    h = moving_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(h)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = ref_mean - moving_mean @ rotation
    return rotation, translation


def static_alignment(
    captury_centres_mm: dict[str, np.ndarray],
    motive_centres_mm: dict[str, np.ndarray],
    min_points: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    common = sorted(set(captury_centres_mm).intersection(motive_centres_mm))
    moving_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    used: list[str] = []
    for name in common:
        cap = np.nanmean(captury_centres_mm[name], axis=1)
        mot = np.nanmean(motive_centres_mm[name], axis=1)
        if np.all(np.isfinite(cap)) and np.all(np.isfinite(mot)):
            moving_rows.append(cap)
            reference_rows.append(mot)
            used.append(name)
    if len(used) < min_points:
        return (
            np.eye(3),
            np.zeros(3),
            {"status": "not_enough_common_centres", "used_centres": used},
        )
    reference = np.vstack(reference_rows)
    moving = np.vstack(moving_rows)
    rotation, translation = kabsch_rows(reference, moving)
    aligned = moving @ rotation + translation
    residuals = np.linalg.norm(aligned - reference, axis=1)
    return (
        rotation,
        translation,
        {
            "status": "ok",
            "used_centres": used,
            "median_static_residual_mm": float(np.nanmedian(residuals)),
            "p95_static_residual_mm": float(np.nanpercentile(residuals, 95)),
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    )


def apply_alignment(
    centres_mm: dict[str, np.ndarray], rotation: np.ndarray, translation: np.ndarray
) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for name, values in centres_mm.items():
        rows = values.T @ rotation + translation
        aligned[name] = rows.T
    return aligned


def yaw_rotation_rows(angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.asarray(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def yaw_alignment_rows(
    reference_rows: np.ndarray,
    moving_rows: np.ndarray,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = np.asarray(reference_rows, dtype=float)
    moving = np.asarray(moving_rows, dtype=float)
    valid = np.all(np.isfinite(reference), axis=1) & np.all(np.isfinite(moving), axis=1)
    reference = reference[valid]
    moving = moving[valid]
    if reference.shape[0] < min_points:
        return (
            np.eye(3),
            np.zeros(3),
            {
                "status": "not_enough_points",
                "n_points": int(reference.shape[0]),
            },
        )

    reference_center = np.nanmedian(reference, axis=0)
    moving_center = np.nanmedian(moving, axis=0)
    reference_xy = reference[:, :2] - reference_center[:2]
    moving_xy = moving[:, :2] - moving_center[:2]
    numerator = float(
        np.nansum(moving_xy[:, 0] * reference_xy[:, 1])
        - np.nansum(moving_xy[:, 1] * reference_xy[:, 0])
    )
    denominator = float(
        np.nansum(moving_xy[:, 0] * reference_xy[:, 0])
        + np.nansum(moving_xy[:, 1] * reference_xy[:, 1])
    )
    angle_rad = float(np.arctan2(numerator, denominator))
    rotation = yaw_rotation_rows(angle_rad)
    rotated = moving @ rotation
    translation = np.nanmedian(reference - rotated, axis=0)
    residuals = np.linalg.norm(rotated + translation - reference, axis=1)
    return (
        rotation,
        translation,
        {
            "status": "ok",
            "n_points": int(reference.shape[0]),
            "yaw_deg": float(np.degrees(angle_rad)),
            "median_residual_mm": float(np.nanmedian(residuals)),
            "p95_residual_mm": float(np.nanpercentile(residuals, 95)),
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    )


def horizontal_principal_axis_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] < 3:
        return np.asarray((1.0, 0.0), dtype=float)
    centered = values[:, :2] - np.nanmean(values[:, :2], axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    _values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.asarray((1.0, 0.0), dtype=float)
    return axis / norm


def nearest_distance_score(
    reference_rows: np.ndarray, moving_rows: np.ndarray
) -> float:
    reference = np.asarray(reference_rows, dtype=float)
    moving = np.asarray(moving_rows, dtype=float)
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    moving = moving[np.all(np.isfinite(moving), axis=1)]
    if reference.size == 0 or moving.size == 0:
        return float("inf")
    distances = np.linalg.norm(moving[:, None, :] - reference[None, :, :], axis=2)
    return float(np.nanmean(np.nanmin(distances, axis=1)))


def pca_yaw_alignment_rows(
    reference_rows: np.ndarray,
    moving_rows: np.ndarray,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = np.asarray(reference_rows, dtype=float)
    moving = np.asarray(moving_rows, dtype=float)
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    moving = moving[np.all(np.isfinite(moving), axis=1)]
    if reference.shape[0] < min_points or moving.shape[0] < min_points:
        return (
            np.eye(3),
            np.zeros(3),
            {
                "status": "not_enough_points",
                "reference_points": int(reference.shape[0]),
                "moving_points": int(moving.shape[0]),
            },
        )
    reference_axis = horizontal_principal_axis_rows(reference)
    moving_axis = horizontal_principal_axis_rows(moving)
    base_angle = float(
        np.arctan2(
            moving_axis[0] * reference_axis[1] - moving_axis[1] * reference_axis[0],
            np.dot(moving_axis, reference_axis),
        )
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for angle_rad in (base_angle, base_angle + np.pi):
        rotation = yaw_rotation_rows(angle_rad)
        rotated = moving @ rotation
        translation = np.nanmedian(reference, axis=0) - np.nanmedian(rotated, axis=0)
        score = nearest_distance_score(reference, rotated + translation)
        candidates.append((score, rotation, translation, angle_rad))
    score, rotation, translation, angle_rad = min(candidates, key=lambda item: item[0])
    return (
        rotation,
        translation,
        {
            "status": "ok_pca_fallback",
            "reference_points": int(reference.shape[0]),
            "moving_points": int(moving.shape[0]),
            "yaw_deg": float(np.degrees(angle_rad)),
            "nearest_distance_mm": score,
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    )


def interpolate_centres_to_time(
    centres_mm: dict[str, np.ndarray], source_time: np.ndarray, target_time: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: interpolate_array(values, source_time, target_time)
        for name, values in centres_mm.items()
    }


def time_window_mask(
    time: np.ndarray, start_s: float | None, end_s: float | None
) -> np.ndarray:
    mask = np.ones(time.shape[0], dtype=bool)
    if start_s is not None:
        mask &= time >= float(start_s)
    if end_s is not None:
        mask &= time <= float(end_s)
    if not np.any(mask):
        raise ValueError(
            f"Empty time window start={start_s!r}, end={end_s!r} for "
            f"signal spanning {float(time[0]) if time.size else np.nan:.3f} to "
            f"{float(time[-1]) if time.size else np.nan:.3f} s."
        )
    return mask


def trim_centres(
    centres: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[:, mask] for name, values in centres.items()}


def trim_model_run(
    run: ModelRun, start_s: float | None, end_s: float | None
) -> ModelRun:
    if start_s is None and end_s is None:
        return run
    mask = time_window_mask(run.time, start_s, end_s)
    return replace(
        run,
        q=run.q[:, mask],
        time=run.time[mask],
        centres_native=trim_centres(run.centres_native, mask),
        rotations_native=trim_rotations(run.rotations_native, mask),
    )


def resolve_cut_window(
    cut_mode: str,
    manual_start_s: float | None,
    manual_end_s: float | None,
    event_report: dict[str, Any],
) -> tuple[float | None, float | None, str]:
    if cut_mode == "full":
        return None, None, "full"
    if cut_mode == "movement":
        return (
            float(event_report["movement_start_time"]),
            float(event_report["movement_end_time"]),
            "movement",
        )
    if manual_start_s is not None and manual_end_s is not None:
        if manual_start_s > manual_end_s:
            raise ValueError(
                f"Manual time window start ({manual_start_s}) is after end ({manual_end_s})."
            )
    if manual_start_s is None and manual_end_s is None:
        return None, None, "full"
    return manual_start_s, manual_end_s, "manual"


def append_centres_to_motive_c3d(
    motive_c3d_path: Path,
    output_path: Path,
    captury_centres_mm: dict[str, np.ndarray],
    motive_centres_mm: dict[str, np.ndarray],
    captury_time: np.ndarray,
    motive_time: np.ndarray,
    angle_label_regex: str,
) -> Path:
    split = split_c3d_points(
        motive_c3d_path, bvh_unit_scale_to_m=0.01, angle_label_regex=angle_label_regex
    )
    c3d_copy = clone_c3d_dict(split.c3d)
    cap_on_motive = interpolate_centres_to_time(
        captury_centres_mm, captury_time, split.time
    )
    mot_on_motive = interpolate_centres_to_time(
        motive_centres_mm, motive_time, split.time
    )
    old_points = np.asarray(c3d_copy["data"]["points"], dtype=float)
    old_labels = as_str_list(get_c3d_param(c3d_copy, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d_copy, "POINT", "UNITS", [""]))[0]
    )

    labels: list[str] = []
    blocks: list[np.ndarray] = []
    for prefix, centres in (("CAPJC_", cap_on_motive), ("MOTJC_", mot_on_motive)):
        for name in sorted(centres):
            labels.append(f"{prefix}{name}")
            blocks.append(centres[name] / unit_mm)
    if blocks:
        xyz = np.stack(blocks, axis=1)
        residuals = np.zeros((1, xyz.shape[1], xyz.shape[2]), dtype=float)
        old_points = np.concatenate(
            (old_points, np.concatenate((xyz, residuals), axis=0)), axis=1
        )
    c3d_copy["data"]["points"] = old_points
    c3d_copy["parameters"]["POINT"]["LABELS"]["value"] = old_labels + labels
    descriptions = as_str_list(get_c3d_param(c3d_copy, "POINT", "DESCRIPTIONS", []))
    if len(descriptions) < len(old_labels):
        descriptions += [""] * (len(old_labels) - len(descriptions))
    descriptions += [
        "Joint centre generated from BioBuddy model and transformed to Motive C3D axes"
    ] * len(labels)
    c3d_copy["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = descriptions
    c3d_copy["parameters"]["POINT"]["USED"]["value"] = [len(old_labels) + len(labels)]
    if "meta_points" in c3d_copy.get("data", {}):
        del c3d_copy["data"]["meta_points"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c3d_copy.write(str(output_path))
    return output_path


def centre_metric_rows(
    trial: str,
    captury_centres_mm: dict[str, np.ndarray],
    motive_centres_mm: dict[str, np.ndarray],
    captury_time: np.ndarray,
    motive_time: np.ndarray,
    joint_filters: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    cap_on_motive = interpolate_centres_to_time(
        captury_centres_mm, captury_time, motive_time
    )
    common = sorted(set(cap_on_motive).intersection(motive_centres_mm))
    metric_joints = common
    if joint_filters:
        import re

        regexes = [re.compile(pattern) for pattern in joint_filters]
        metric_joints = [
            joint for joint in common if any(regex.search(joint) for regex in regexes)
        ]
    metric_joint_set = set(metric_joints)
    for joint in common:
        cap = cap_on_motive[joint].T
        mot = motive_centres_mm[joint].T
        valid = np.all(np.isfinite(cap), axis=1) & np.all(np.isfinite(mot), axis=1)
        errors = np.linalg.norm(cap - mot, axis=1)
        if joint in metric_joint_set:
            summary_rows.append(
                {
                    "trial": trial,
                    "joint": joint,
                    "n_frames": int(valid.sum()),
                    "median_error_mm": (
                        float(np.nanmedian(errors[valid])) if valid.any() else np.nan
                    ),
                    "p95_error_mm": (
                        float(np.nanpercentile(errors[valid], 95))
                        if valid.any()
                        else np.nan
                    ),
                    "max_error_mm": (
                        float(np.nanmax(errors[valid])) if valid.any() else np.nan
                    ),
                    **joint_center_error_xyz(mot, cap),
                }
            )
        for i, time_value in enumerate(motive_time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "joint": joint,
                    "captury_x_mm": cap[i, 0],
                    "captury_y_mm": cap[i, 1],
                    "captury_z_mm": cap[i, 2],
                    "motive_x_mm": mot[i, 0],
                    "motive_y_mm": mot[i, 1],
                    "motive_z_mm": mot[i, 2],
                    "distance_mm": errors[i],
                }
            )
    return summary_rows, timeseries_rows


def q_metric_rows(
    trial: str, captury: ModelRun, motive: ModelRun
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    cap_q = {name: captury.q[i] for i, name in enumerate(captury.q_names)}
    mot_q = {name: motive.q[i] for i, name in enumerate(motive.q_names)}
    for q_name in sorted(set(cap_q).intersection(mot_q)):
        cap_curve = interpolate_array(
            cap_q[q_name][None, :], captury.time, motive.time
        )[0]
        mot_curve = mot_q[q_name]
        unit = "rad" if "rot" in q_name.lower() else "native"
        summary_rows.append(
            {
                "trial": trial,
                "q_name": q_name,
                "unit": unit,
                **waveform_metrics(mot_curve, cap_curve, unit),
            }
        )
        for i, time_value in enumerate(motive.time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "q_name": q_name,
                    "motive": mot_curve[i],
                    "captury": cap_curve[i],
                    "difference": cap_curve[i] - mot_curve[i],
                }
            )
    return summary_rows, timeseries_rows


def c3d_angle_scale_to_deg(unit: str) -> float:
    normalized = str(unit).strip().lower()
    if normalized in {"rad", "radian", "radians"}:
        return 180.0 / np.pi
    return 1.0


def sanitize_channel_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"\W+", "_", str(name).strip()).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def captury_c3d_angle_rows(
    trial: str,
    captury_c3d: Path,
    angle_label_regex: str,
    c3d_angle_unit: str,
    cut_start_s: float | None,
    cut_end_s: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = split_c3d_points(
        captury_c3d,
        bvh_unit_scale_to_m=0.001,
        angle_label_regex=angle_label_regex,
    )
    if not split.angle_labels or split.angle_data.size == 0:
        return [], []
    mask = time_window_mask(split.time, cut_start_s, cut_end_s)
    time = split.time[mask]
    angle_deg = split.angle_data[:, :, mask] * c3d_angle_scale_to_deg(c3d_angle_unit)
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    axis_names = ("X", "Y", "Z")
    for angle_index, angle_label in enumerate(split.angle_labels):
        safe_label = sanitize_channel_name(angle_label, f"angle_{angle_index}")
        for axis_index, axis_name in enumerate(axis_names):
            values = angle_deg[axis_index, angle_index, :]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            q_name = f"CapturyC3D_{safe_label}_{axis_name}"
            summary_rows.append(
                {
                    "trial": trial,
                    "q_name": q_name,
                    "unit": "deg",
                    "source": "captury_c3d",
                    "c3d_angle_label": angle_label,
                    "c3d_angle_axis": axis_name,
                    "c3d_mean_deg": float(np.mean(finite)),
                    "c3d_sd_deg": float(np.std(finite)),
                    "c3d_min_deg": float(np.min(finite)),
                    "c3d_max_deg": float(np.max(finite)),
                }
            )
            for frame_index, time_value in enumerate(time):
                timeseries_rows.append(
                    {
                        "trial": trial,
                        "time": float(time_value),
                        "q_name": q_name,
                        "captury_c3d": float(values[frame_index]),
                    }
                )
    return summary_rows, timeseries_rows


def c3d_angle_inventory(path: Path, angle_label_regex: str) -> dict[str, Any]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    angle_indices = detect_angle_indices(c3d, labels, angle_label_regex)
    return {
        "path": str(path),
        "angle_count": len(angle_indices),
        "angles": {name: labels[index] for name, index in angle_indices.items()},
    }


def duplicate_label_inventory(path: Path) -> dict[str, Any]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    duplicates = {label: count for label, count in counts.items() if count > 1}
    return {
        "path": str(path),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }


def read_c3d_points_mm(
    path: Path,
    angle_label_regex: str = ANGLE_LABEL_REGEX,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    angle_indices = set(detect_angle_indices(c3d, labels, angle_label_regex).values())
    marker_indices = [
        index for index in range(len(labels)) if index not in angle_indices
    ]
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points = np.asarray(c3d["data"]["points"], dtype=float)
    xyz_mm = points[:3, marker_indices, :] * unit_mm
    residuals = (
        points[3, marker_indices, :]
        if points.shape[0] > 3
        else np.zeros((len(marker_indices), points.shape[2]))
    )
    xyz_mm[:, residuals < 0] = np.nan
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [120])
    rate = float(
        rate_value[0]
        if isinstance(rate_value, (list, tuple, np.ndarray))
        else rate_value
    )
    time = np.arange(xyz_mm.shape[2], dtype=float) / rate
    return [labels[index] for index in marker_indices], xyz_mm, residuals, time


def clean_marker_label(label: str) -> str:
    return label.replace("Skeleton_001_", "").strip()


def marker_indices_by_clean_label(labels: list[str]) -> dict[str, list[int]]:
    lookup: dict[str, list[int]] = {}
    clean_labels = [clean_marker_label(label) for label in labels]
    totals: dict[str, int] = {}
    for label in clean_labels:
        totals[label] = totals.get(label, 0) + 1
    seen: dict[str, int] = {}
    for i, label in enumerate(clean_labels):
        seen[label] = seen.get(label, 0) + 1
        lookup.setdefault(label, []).append(i)
        if totals[label] > 1:
            lookup.setdefault(f"{label}#{seen[label]}", []).append(i)
    return lookup


def average_marker_group(
    points_mm: np.ndarray, indices: list[int]
) -> np.ndarray | None:
    if not indices:
        return None
    values = points_mm[:, indices, :]
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=1)


def marker_proxy_centres_from_c3d(
    labels: list[str], points_mm: np.ndarray
) -> dict[str, np.ndarray]:
    lookup = marker_indices_by_clean_label(labels)
    proxies: dict[str, np.ndarray] = {}
    for joint, marker_labels in MODEL_JOINT_MARKER_PROXIES.items():
        indices = [
            index
            for marker_label in marker_labels
            for index in lookup.get(marker_label, [])
        ]
        signal = average_marker_group(points_mm, indices)
        if signal is not None:
            proxies[joint] = signal
    return proxies


def paired_model_marker_rows(
    model_centres_mm: dict[str, np.ndarray],
    model_time: np.ndarray,
    marker_proxy_centres_mm: dict[str, np.ndarray],
    marker_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    moving_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    used_joints: list[str] = []
    for joint in sorted(set(model_centres_mm).intersection(marker_proxy_centres_mm)):
        model_signal = interpolate_array(
            model_centres_mm[joint], model_time, marker_time
        ).T
        marker_signal = marker_proxy_centres_mm[joint].T
        valid = np.all(np.isfinite(model_signal), axis=1) & np.all(
            np.isfinite(marker_signal), axis=1
        )
        if not np.any(valid):
            continue
        moving_rows.append(model_signal[valid])
        reference_rows.append(marker_signal[valid])
        used_joints.append(joint)
    if not moving_rows:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.vstack(reference_rows), np.vstack(moving_rows), used_joints


def stacked_finite_rows_from_centres(
    centres_mm: dict[str, np.ndarray],
    time: np.ndarray,
    reference_time: np.ndarray,
    max_rows: int = 2000,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for values in centres_mm.values():
        interpolated = interpolate_array(values, time, reference_time).T
        interpolated = interpolated[np.all(np.isfinite(interpolated), axis=1)]
        if interpolated.size:
            rows.append(interpolated)
    if not rows:
        return np.empty((0, 3))
    stacked = np.vstack(rows)
    if stacked.shape[0] > max_rows:
        step = int(np.ceil(stacked.shape[0] / max_rows))
        stacked = stacked[::step]
    return stacked


def stacked_finite_rows_from_marker_points(
    points_mm: np.ndarray, max_rows: int = 4000
) -> np.ndarray:
    rows = np.asarray(points_mm, dtype=float).transpose(2, 1, 0).reshape(-1, 3)
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    if rows.shape[0] > max_rows:
        step = int(np.ceil(rows.shape[0] / max_rows))
        rows = rows[::step]
    return rows


def model_to_motive_marker_alignment(
    model_centres_mm: dict[str, np.ndarray],
    model_time: np.ndarray,
    motive_c3d: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels, marker_points_mm, residuals, marker_time = read_c3d_points_mm(motive_c3d)
    if residuals.shape == marker_points_mm.shape[1:]:
        marker_points_mm = marker_points_mm.copy()
        marker_points_mm[:, residuals < 0] = np.nan
    marker_proxies = marker_proxy_centres_from_c3d(labels, marker_points_mm)
    reference_rows, moving_rows, used_joints = paired_model_marker_rows(
        model_centres_mm, model_time, marker_proxies, marker_time
    )
    rotation, translation, report = yaw_alignment_rows(reference_rows, moving_rows)
    report["used_proxy_joints"] = used_joints
    report["method"] = "motive_57_marker_proxies"
    if report.get("status") == "ok":
        return rotation, translation, report

    marker_rows = stacked_finite_rows_from_marker_points(marker_points_mm)
    model_rows = stacked_finite_rows_from_centres(
        model_centres_mm, model_time, marker_time
    )
    rotation, translation, fallback_report = pca_yaw_alignment_rows(
        marker_rows, model_rows
    )
    fallback_report["used_proxy_joints"] = used_joints
    fallback_report["method"] = "horizontal_pca_fallback"
    fallback_report["proxy_status"] = report
    return rotation, translation, fallback_report


def occlusion_rows_from_points(
    trial: str, labels: list[str], points_mm: np.ndarray, residuals: np.ndarray
) -> list[dict[str, Any]]:
    finite_xyz = np.all(np.isfinite(points_mm), axis=0)
    missing = ~finite_xyz
    if residuals.shape == missing.shape:
        missing = missing | (residuals < 0)
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        marker_missing = missing[i]
        rows.append(
            {
                "trial": trial,
                "marker_order": i,
                "marker": clean_marker_label(label),
                "raw_marker": label,
                "missing_frames": int(np.sum(marker_missing)),
                "total_frames": int(marker_missing.shape[0]),
                "missing_percent": float(100.0 * np.mean(marker_missing)),
            }
        )
    return rows


def analyze_motive_occlusions(
    motive_c3d: Path,
    trial_dir: Path,
    trial: str,
    generate_figure: bool = True,
) -> tuple[list[dict[str, Any]], Path | None]:
    labels, points_mm, residuals, _ = read_c3d_points_mm(motive_c3d)
    rows = occlusion_rows_from_points(trial, labels, points_mm, residuals)
    csv_path = trial_dir / "motive_marker_occlusions.csv"
    write_rows(csv_path, rows)
    if not generate_figure:
        return rows, None
    fig_path = plot_metric_barh(
        pd.DataFrame(rows),
        category="marker",
        metric="missing_percent",
        output_path=trial_dir / "figures" / "occlusions" / "motive_missing_percent.png",
        title="Motive marker occlusions",
        xlabel="missing_percent",
    )
    return rows, fig_path


def detect_trial_events_and_contacts(
    motive_c3d: Path,
    trial_dir: Path,
    trial: str,
    foot_marker_regex: str = FOOT_MARKER_PATTERN,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import re

    labels, points_mm, _residuals, time = read_c3d_points_mm(motive_c3d)
    finite = np.all(np.isfinite(points_mm), axis=0)
    dt = float(np.nanmedian(np.diff(time))) if time.shape[0] > 1 else 1.0 / 120.0
    velocity = np.gradient(points_mm, dt, axis=2)
    speed = np.linalg.norm(velocity, axis=0)
    speed[~finite] = np.nan
    median_speed = np.nanmedian(speed, axis=0)
    baseline = float(np.nanpercentile(median_speed, 10))
    high = float(np.nanpercentile(median_speed, 95))
    threshold = baseline + 0.15 * (high - baseline)
    moving = median_speed > threshold
    moving_indices = np.flatnonzero(moving)
    start_index = int(moving_indices[0]) if moving_indices.size else 0
    end_index = (
        int(moving_indices[-1]) if moving_indices.size else int(time.shape[0] - 1)
    )

    regex = re.compile(foot_marker_regex, re.IGNORECASE)
    left_indices = [
        i
        for i, label in enumerate(labels)
        if regex.search(clean_marker_label(label))
        and clean_marker_label(label).startswith("L")
    ]
    right_indices = [
        i
        for i, label in enumerate(labels)
        if regex.search(clean_marker_label(label))
        and clean_marker_label(label).startswith("R")
    ]

    def foot_contact(indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        foot = average_marker_group(points_mm, indices)
        if foot is None:
            nan = np.full(time.shape[0], np.nan)
            return nan.astype(bool), nan, nan
        z = foot[2]
        foot_speed = np.linalg.norm(np.gradient(foot, dt, axis=1), axis=0)
        z_limit = float(np.nanpercentile(z, 35))
        speed_limit = float(np.nanpercentile(foot_speed, 35))
        return (z <= z_limit) & (foot_speed <= speed_limit), z, foot_speed

    left_contact, left_z, left_speed = foot_contact(left_indices)
    right_contact, right_z, right_speed = foot_contact(right_indices)
    rows: list[dict[str, Any]] = []
    contact_mask = time_window_mask(time, time_start_s, time_end_s)
    for i, time_value in enumerate(time):
        if not contact_mask[i]:
            continue
        rows.append(
            {
                "trial": trial,
                "time": float(time_value),
                "movement_speed_mm_s": float(median_speed[i]),
                "left_foot_z_mm": (
                    float(left_z[i]) if np.isfinite(left_z[i]) else np.nan
                ),
                "right_foot_z_mm": (
                    float(right_z[i]) if np.isfinite(right_z[i]) else np.nan
                ),
                "left_foot_speed_mm_s": (
                    float(left_speed[i]) if np.isfinite(left_speed[i]) else np.nan
                ),
                "right_foot_speed_mm_s": (
                    float(right_speed[i]) if np.isfinite(right_speed[i]) else np.nan
                ),
                "left_contact": bool(left_contact[i]),
                "right_contact": bool(right_contact[i]),
            }
        )
    write_rows(trial_dir / "trial_events_contacts.csv", rows)
    report = {
        "trial": trial,
        "movement_start_index": start_index,
        "movement_end_index": end_index,
        "movement_start_time": float(time[start_index]),
        "movement_end_time": float(time[end_index]),
        "movement_speed_threshold_mm_s": threshold,
        "manual_time_start_s": time_start_s,
        "manual_time_end_s": time_end_s,
        "used_start_time": (
            float(time[contact_mask][0]) if np.any(contact_mask) else np.nan
        ),
        "used_end_time": (
            float(time[contact_mask][-1]) if np.any(contact_mask) else np.nan
        ),
        "left_foot_markers": [labels[i] for i in left_indices],
        "right_foot_markers": [labels[i] for i in right_indices],
    }
    (trial_dir / "trial_events.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, rows


SEGMENT_LENGTH_PAIR_CANDIDATES = [
    (
        "pelvis_to_spine",
        [("Hips", "Spine"), ("pelvis", "spine_01"), ("Pelvis", "Thorax")],
    ),
    (
        "left_thigh",
        [("LeftUpLeg", "LeftLeg"), ("thigh_l", "calf_l"), ("LThigh", "LShank")],
    ),
    (
        "left_shank",
        [("LeftLeg", "LeftFoot"), ("calf_l", "foot_l"), ("LShank", "LFoot")],
    ),
    ("left_foot", [("LeftFoot", "LeftToeBase"), ("foot_l", "ball_l")]),
    (
        "right_thigh",
        [("RightUpLeg", "RightLeg"), ("thigh_r", "calf_r"), ("RThigh", "RShank")],
    ),
    (
        "right_shank",
        [("RightLeg", "RightFoot"), ("calf_r", "foot_r"), ("RShank", "RFoot")],
    ),
    ("right_foot", [("RightFoot", "RightToeBase"), ("foot_r", "ball_r")]),
    (
        "left_upper_arm",
        [
            ("LeftArm", "LeftForeArm"),
            ("upperarm_l", "lowerarm_l"),
            ("LUpperArm", "LForearm"),
        ],
    ),
    (
        "left_forearm",
        [("LeftForeArm", "LeftHand"), ("lowerarm_l", "hand_l"), ("LForearm", "LHand")],
    ),
    (
        "right_upper_arm",
        [
            ("RightArm", "RightForeArm"),
            ("upperarm_r", "lowerarm_r"),
            ("RUpperArm", "RForearm"),
        ],
    ),
    (
        "right_forearm",
        [
            ("RightForeArm", "RightHand"),
            ("lowerarm_r", "hand_r"),
            ("RForearm", "RHand"),
        ],
    ),
]
SEGMENT_LENGTH_PAIRS = [
    (name, candidates[0][0], candidates[0][1])
    for name, candidates in SEGMENT_LENGTH_PAIR_CANDIDATES
]


def segment_length_pair_for_centres(
    centres_mm: Mapping[str, np.ndarray], candidates: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """Choose the first segment-name pair available in a centre dictionary."""

    for proximal, distal in candidates:
        if proximal in centres_mm and distal in centres_mm:
            return proximal, distal
    return None


def dimension_rows_from_centres(
    trial: str,
    system: str,
    source_kind: str,
    centres_mm: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Summarize segment lengths from joint-centre positions in millimetres.

    The comparison GUI presents model dimensions as one row per anatomical
    segment and source. Captury/Motive dimensions come from animated model
    centres over time, while the BioBuddy template currently contributes a
    neutral-pose model. Both cases share this helper: arrays can contain one
    frame or many frames, and the median/standard deviation are computed over
    the available samples.
    """

    rows: list[dict[str, Any]] = []
    for name, candidates in SEGMENT_LENGTH_PAIR_CANDIDATES:
        pair = segment_length_pair_for_centres(centres_mm, candidates)
        if pair is None:
            continue
        proximal, distal = pair
        length = np.linalg.norm(centres_mm[distal] - centres_mm[proximal], axis=0)
        rows.append(
            {
                "trial": trial,
                "system": system,
                "source_kind": source_kind,
                "dimension": name,
                "median_length_mm": float(np.nanmedian(length)),
                "sd_length_mm": float(np.nanstd(length)),
            }
        )
    return rows


def model_dimension_rows(trial: str, runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        centres_mm = centres_to_c3d_mm(
            run.centres_native, run.unit_scale_to_m, "identity"
        )
        rows.extend(
            dimension_rows_from_centres(trial, run.system, run.source_kind, centres_mm)
        )
    return rows


def biomod_neutral_centres_mm(
    biomod_path: Path, unit_scale_to_m: float
) -> dict[str, np.ndarray]:
    """Return neutral-pose segment origins from a biorbd/BioBuddy model.

    BioBuddy models generated from the Motive 57 template do not yet provide an
    analysed q(t) in this pipeline. For model dimensions, the neutral segment
    origins are enough because the template lengths are static. Values are
    returned in millimetres with shape ``(3, 1)`` to match animated centre
    arrays used by Captury and Motive.
    """

    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    q = np.zeros(model.nbQ())
    centres_mm: dict[str, np.ndarray] = {}
    for index, name in enumerate(biorbd_segment_names(model)):
        rt = np.asarray(model.globalJCS(q, index).to_array(), dtype=float)
        centres_mm[name] = (rt[:3, 3] * unit_scale_to_m * 1000.0).reshape(3, 1)
    return centres_mm


def biobuddy_dimension_rows(
    trial: str,
    biomod_path: Path | None,
    unit_scale_to_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an optional BioBuddy model and expose it as a dimensions source."""

    if biomod_path is None:
        return [], {
            "status": "missing",
            "reason": "no_biobuddy_biomod_argument",
        }
    if not biomod_path.exists():
        return [], {
            "status": "missing",
            "path": str(biomod_path),
            "reason": "biobuddy_biomod_not_found",
        }
    centres_mm = biomod_neutral_centres_mm(biomod_path, unit_scale_to_m)
    rows = dimension_rows_from_centres(trial, "biobuddy", "motive_57", centres_mm)
    return rows, {
        "status": "ok",
        "path": str(biomod_path),
        "unit_scale_to_m": unit_scale_to_m,
        "segments": len(centres_mm),
        "dimensions": len(rows),
    }


def marker_correspondence_rows(
    trial: str,
    motive_c3d: Path,
    captury_c3d: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    landmark_map: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    motive_labels, motive_points, _motive_residuals, motive_time = read_c3d_points_mm(
        motive_c3d
    )
    captury_labels, captury_points, _captury_residuals, captury_time = (
        read_c3d_points_mm(captury_c3d)
    )
    motive_lookup = marker_indices_by_clean_label(motive_labels)
    captury_lookup = marker_indices_by_clean_label(captury_labels)
    rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for item in landmark_map:
        name = str(item["name"])
        motive_indices = [
            index
            for label in item["reference"]
            for index in motive_lookup.get(label, [])
        ]
        captury_indices = [
            index for label in item["test"] for index in captury_lookup.get(label, [])
        ]
        motive_signal = average_marker_group(motive_points, motive_indices)
        captury_signal = average_marker_group(captury_points, captury_indices)
        if motive_signal is None or captury_signal is None:
            continue
        captury_on_motive = (
            interpolate_array(captury_signal, captury_time, motive_time).T @ rotation
            + translation
        )
        motive_rows = motive_signal.T
        valid = np.all(np.isfinite(captury_on_motive), axis=1) & np.all(
            np.isfinite(motive_rows), axis=1
        )
        distance = np.linalg.norm(captury_on_motive - motive_rows, axis=1)
        rows.append(
            {
                "trial": trial,
                "landmark": name,
                "motive_labels": ";".join(item["reference"]),
                "captury_labels": ";".join(item["test"]),
                "n_frames": int(np.sum(valid)),
                "median_error_mm": (
                    float(np.nanmedian(distance[valid])) if valid.any() else np.nan
                ),
                "p95_error_mm": (
                    float(np.nanpercentile(distance[valid], 95))
                    if valid.any()
                    else np.nan
                ),
                "rmse_error_mm": (
                    float(np.sqrt(np.nanmean(distance[valid] ** 2)))
                    if valid.any()
                    else np.nan
                ),
            }
        )
        difference = captury_on_motive - motive_rows
        for frame, time_value in enumerate(motive_time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "landmark": name,
                    "motive_labels": ";".join(item["reference"]),
                    "captury_labels": ";".join(item["test"]),
                    "error_x_mm": float(difference[frame, 0]),
                    "error_y_mm": float(difference[frame, 1]),
                    "error_z_mm": float(difference[frame, 2]),
                    "distance_mm": float(distance[frame]),
                }
            )
    return rows, timeseries_rows


def vertical_amplitude_report(enriched_c3d: Path) -> dict[str, Any]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(enriched_c3d))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points_mm = np.asarray(c3d["data"]["points"][:3], dtype=float) * unit_mm
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        if not (label.startswith("CAPJC_") or label.startswith("MOTJC_")):
            continue
        values = points_mm[:, index, :]
        rows.append(
            {
                "label": label,
                "x_range_mm": float(np.nanmax(values[0]) - np.nanmin(values[0])),
                "y_range_mm": float(np.nanmax(values[1]) - np.nanmin(values[1])),
                "z_range_mm": float(np.nanmax(values[2]) - np.nanmin(values[2])),
            }
        )
    joint_indices = [labels.index(row["label"]) for row in rows]
    if joint_indices:
        joint_points = points_mm[:, joint_indices, :]
        spatial_ranges = np.nanmax(joint_points, axis=1) - np.nanmin(
            joint_points, axis=1
        )
        median_spatial_range = np.nanmedian(spatial_ranges, axis=1)
        max_spatial_range = np.nanmax(spatial_ranges, axis=1)
    else:
        median_spatial_range = np.full(3, np.nan)
        max_spatial_range = np.full(3, np.nan)
    return {
        "labels": len(rows),
        "median_z_range_mm": (
            float(np.nanmedian([row["z_range_mm"] for row in rows])) if rows else np.nan
        ),
        "median_xy_range_mm": (
            float(
                np.nanmedian(
                    [max(row["x_range_mm"], row["y_range_mm"]) for row in rows]
                )
            )
            if rows
            else np.nan
        ),
        "median_spatial_extent_x_mm": float(median_spatial_range[0]),
        "median_spatial_extent_y_mm": float(median_spatial_range[1]),
        "median_spatial_extent_z_mm": float(median_spatial_range[2]),
        "max_spatial_extent_x_mm": float(max_spatial_range[0]),
        "max_spatial_extent_y_mm": float(max_spatial_range[1]),
        "max_spatial_extent_z_mm": float(max_spatial_range[2]),
        "rows": rows,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    columns = np.asarray(list(dataframe.columns), dtype=str)
    payload: dict[str, np.ndarray] = {"columns": columns}
    for index, column in enumerate(columns):
        series = dataframe[str(column)]
        if pd.api.types.is_numeric_dtype(series):
            payload[f"col_{index}"] = series.to_numpy()
        else:
            payload[f"col_{index}"] = series.fillna("").astype(str).to_numpy(dtype=str)
    np.savez_compressed(path, **payload)


def metric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def plot_metric_barh(
    df: pd.DataFrame,
    category: str,
    metric: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> Path | None:
    if (
        df.empty
        or category not in df.columns
        or "trial" not in df.columns
        or metric not in df.columns
    ):
        return None
    values = df[[category, "trial", metric]].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna(subset=[metric])
    if values.empty:
        return None
    pivot = values.pivot_table(
        index=category, columns="trial", values=metric, aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
    height = min(24.0, max(5.0, 0.28 * max(1, len(pivot.index)) + 1.5))
    width = min(18.0, max(8.0, 1.8 * max(1, len(pivot.columns)) + 6.0))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    pivot.plot(kind="barh", ax=axis)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(category)
    axis.grid(axis="x", alpha=0.3)
    axis.legend(title="trial", loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_metric_figures(
    centre_rows: list[dict[str, Any]], q_rows: list[dict[str, Any]], out_dir: Path
) -> dict[str, list[str]]:
    figure_paths: dict[str, list[str]] = {"joint_centres": [], "kinematics_q": []}
    figures_dir = out_dir / "figures"
    if centre_rows:
        centre_df = pd.DataFrame(centre_rows)
        for metric in metric_columns(centre_df, {"trial", "joint", "n_frames"}):
            path = plot_metric_barh(
                centre_df,
                category="joint",
                metric=metric,
                output_path=figures_dir / "joint_centres" / f"{metric}.png",
                title=f"Joint centres - {metric}",
                xlabel=metric,
            )
            if path is not None:
                figure_paths["joint_centres"].append(str(path))
    if q_rows:
        q_df = pd.DataFrame(q_rows)
        for metric in metric_columns(q_df, {"trial", "q_name", "unit"}):
            path = plot_metric_barh(
                q_df,
                category="q_name",
                metric=metric,
                output_path=figures_dir / "kinematics_q" / f"{metric}.png",
                title=f"Kinematics q - {metric}",
                xlabel=metric,
            )
            if path is not None:
                figure_paths["kinematics_q"].append(str(path))
    return figure_paths


def visualize_enriched_c3d(
    enriched_c3d: Path, wait_seconds: float, headless: bool
) -> None:
    if headless:
        os.environ["PYORERUN_HEADLESS"] = "1"
    _, PhaseRerun, PyoMarkers = require_pyorerun()
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(enriched_c3d))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points = np.asarray(c3d["data"]["points"][:3], dtype=float) * unit_mm
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [120])
    rate = float(
        rate_value[0]
        if isinstance(rate_value, (list, tuple, np.ndarray))
        else rate_value
    )
    time = np.arange(points.shape[2], dtype=float) / rate
    keep = [
        i for i, label in enumerate(labels) if label.startswith(("CAPJC_", "MOTJC_"))
    ]
    if not keep:
        raise RuntimeError(f"No CAPJC_/MOTJC_ labels in {enriched_c3d}.")
    phase = PhaseRerun(time)
    phase.add_xp_markers(
        "p6_joint_centres",
        PyoMarkers(data=points[:, keep, :], channels=[labels[i] for i in keep]),
    )
    if headless:
        return
    phase.rerun("p6_motive_captury_joint_centres", notebook=False)
    if wait_seconds > 0:
        import time as time_module

        time_module.sleep(wait_seconds)


def run_ik_batch(
    bundle: TrialBundle, out_dir: Path, model_source: str, max_frames: int
) -> dict[str, Any]:
    source_kind, motive_model_path = select_model_file(bundle, "motive", model_source)
    command = [
        sys.executable,
        str(Path(__file__).with_name("bvh_c3d_biobuddy_pyorerun_compare.py")),
        "--bvh",
        str(motive_model_path if source_kind == "bvh" else bundle.motive_bvh),
        "--c3d",
        str(bundle.motive_c3d),
        "--out-dir",
        str(out_dir / "motive_ik_pipeline"),
        "--bvh-unit-scale-to-m",
        "0.01",
        "--inverse-kinematics",
        "--inverse-kinematics-max-frames",
        str(max_frames),
    ]
    if bundle.motive_fbx is not None:
        command.extend(
            ["--fbx", str(bundle.motive_fbx), "--fbx-unit-scale-to-m", "0.01"]
        )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def compare_trial(
    bundle: TrialBundle,
    out_root: Path,
    args: argparse.Namespace,
    static_alignment_transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    trial_dir = out_root / safe_name(bundle.name)
    trial_dir.mkdir(parents=True, exist_ok=True)
    cache_fingerprint = trial_cache_fingerprint(
        bundle, args, static_alignment_transform
    )
    cached_report = cached_trial_report(
        trial_dir, bundle, args, static_alignment_transform
    )
    if cached_report is not None:
        print(f"Using cached trial outputs: {bundle.name}")
        cached_report.setdefault("cache", {})["hit"] = True
        enriched_c3d = cached_report.get("outputs", {}).get("enriched_c3d")
        if (
            args.visualize
            and (args.visualize_trial is None or args.visualize_trial == bundle.name)
            and enriched_c3d
        ):
            visualize_enriched_c3d(
                Path(enriched_c3d), args.rerun_wait_seconds, args.headless
            )
        cached_transform = static_transform_from_report(cached_report)
        return cached_report, static_alignment_transform or cached_transform
    captury = build_model_run(
        bundle,
        "captury",
        args.model_source,
        trial_dir,
        include_mesh=not args.no_mesh,
        max_mesh_points=args.max_mesh_points,
        unit_scale_override=args.captury_unit_scale_to_m,
        root_offset_mode=args.root_offset_mode,
        model_to_c3d_axis=args.model_to_c3d_axis,
        angle_label_regex=args.angle_label_regex,
    )
    motive = build_model_run(
        bundle,
        "motive",
        args.model_source,
        trial_dir,
        include_mesh=not args.no_mesh,
        max_mesh_points=args.max_mesh_points,
        unit_scale_override=args.motive_unit_scale_to_m,
        root_offset_mode=args.root_offset_mode,
        model_to_c3d_axis=args.model_to_c3d_axis,
        angle_label_regex=args.angle_label_regex,
    )
    cap_c3d_mm = centres_to_c3d_mm(
        captury.centres_native, captury.unit_scale_to_m, args.model_to_c3d_axis
    )
    mot_c3d_mm = centres_to_c3d_mm(
        motive.centres_native, motive.unit_scale_to_m, args.model_to_c3d_axis
    )
    alignment_report: dict[str, Any]
    if args.disable_static_model_alignment:
        rotation = np.eye(3)
        translation = np.zeros(3)
        alignment_report = {
            "status": "disabled_static_model_alignment",
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
            "note": "Captury centres are kept in their converted C3D frame without Captury -> Motive model alignment.",
        }
        static_alignment_transform = (rotation, translation)
    elif static_alignment_transform is None:
        rotation, translation, alignment_report = static_alignment(
            cap_c3d_mm, mot_c3d_mm
        )
        static_alignment_transform = (rotation, translation)
    else:
        rotation, translation = static_alignment_transform
        alignment_report = {
            "status": "reused_static_alignment",
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        }
    cap_aligned_mm = apply_alignment(cap_c3d_mm, rotation, translation)
    if args.disable_motive_marker_alignment:
        model_marker_rotation = np.eye(3)
        model_marker_translation = np.zeros(3)
        model_marker_report = {
            "status": "disabled_motive_marker_alignment",
            "method": "identity",
            "rotation": model_marker_rotation.tolist(),
            "translation_mm": model_marker_translation.tolist(),
            "note": "Motive model centres are not yaw/translation-aligned to Motive C3D marker proxies.",
        }
    else:
        model_marker_rotation, model_marker_translation, model_marker_report = (
            model_to_motive_marker_alignment(mot_c3d_mm, motive.time, bundle.motive_c3d)
        )
    cap_aligned_mm = apply_alignment(
        cap_aligned_mm, model_marker_rotation, model_marker_translation
    )
    mot_c3d_mm = apply_alignment(
        mot_c3d_mm, model_marker_rotation, model_marker_translation
    )
    cap_rotations_c3d = rotations_to_c3d(
        captury.rotations_native,
        args.model_to_c3d_axis,
        rotation @ model_marker_rotation,
    )
    mot_rotations_c3d = rotations_to_c3d(
        motive.rotations_native,
        args.model_to_c3d_axis,
        model_marker_rotation,
    )
    segment_orientation_report: dict[str, Any] = {
        "captury_reorient_thigh_y_from_cor": bool(
            args.captury_reorient_thigh_y_from_cor
        ),
        "rotate_body_segments_180_x": bool(args.rotate_body_segments_180_x),
        "applied": [],
    }
    if args.captury_reorient_thigh_y_from_cor:
        cap_rotations_c3d = correct_captury_thigh_y_from_cor(
            cap_rotations_c3d, cap_aligned_mm
        )
        segment_orientation_report["applied"].append(
            "captury_thigh_y_axis_from_hip_to_knee_cor"
        )
    if args.rotate_body_segments_180_x:
        cap_rotations_c3d = rotate_segment_frames_180_x(cap_rotations_c3d)
        mot_rotations_c3d = rotate_segment_frames_180_x(mot_rotations_c3d)
        segment_orientation_report["applied"].append(
            "captury_and_motive_segment_frames_rotated_180_deg_about_local_x"
        )
    alignment_report["motive_model_to_c3d_markers"] = model_marker_report
    enriched_c3d = append_centres_to_motive_c3d(
        bundle.motive_c3d,
        trial_dir / f"{safe_name(bundle.name)}_motive_with_capjc_motjc.c3d",
        cap_aligned_mm,
        mot_c3d_mm,
        captury.time,
        motive.time,
        args.angle_label_regex,
    )
    detected_event_report, _ = detect_trial_events_and_contacts(
        bundle.motive_c3d, trial_dir, bundle.name
    )
    cut_start_s, cut_end_s, effective_cut_mode = resolve_cut_window(
        args.cut_mode, args.time_start, args.time_end, detected_event_report
    )
    captury_metrics = trim_model_run(captury, cut_start_s, cut_end_s)
    motive_metrics = trim_model_run(motive, cut_start_s, cut_end_s)
    cap_aligned_metrics_mm = trim_centres(
        cap_aligned_mm, time_window_mask(captury.time, cut_start_s, cut_end_s)
    )
    mot_metrics_mm = trim_centres(
        mot_c3d_mm, time_window_mask(motive.time, cut_start_s, cut_end_s)
    )
    centre_rows, centre_ts_rows = centre_metric_rows(
        bundle.name,
        cap_aligned_metrics_mm,
        mot_metrics_mm,
        captury_metrics.time,
        motive_metrics.time,
        args.joint_filter,
    )
    q_rows, q_ts_rows = q_metric_rows(bundle.name, captury_metrics, motive_metrics)
    c3d_angle_rows, c3d_angle_ts_rows = captury_c3d_angle_rows(
        bundle.name,
        bundle.captury_c3d,
        args.angle_label_regex,
        args.c3d_angle_unit,
        cut_start_s,
        cut_end_s,
    )
    q_rows.extend(c3d_angle_rows)
    q_ts_rows.extend(c3d_angle_ts_rows)
    cap_rotation_metrics = trim_rotations(
        cap_rotations_c3d, time_window_mask(captury.time, cut_start_s, cut_end_s)
    )
    mot_rotation_metrics = trim_rotations(
        mot_rotations_c3d, time_window_mask(motive.time, cut_start_s, cut_end_s)
    )
    segment_q_rows, segment_q_ts_rows = segment_relative_q_metric_rows(
        bundle.name,
        cap_rotation_metrics,
        mot_rotation_metrics,
        captury_metrics.time,
        motive_metrics.time,
    )
    q_rows.extend(segment_q_rows)
    q_ts_rows.extend(segment_q_ts_rows)
    segment_rows, segment_ts_rows, segment_report = segment_rotation_metric_rows(
        bundle.name,
        {
            "captury": cap_rotation_metrics,
            "motive": mot_rotation_metrics,
            "biobuddy": {},
        },
        {
            "captury": captury_metrics.time,
            "motive": motive_metrics.time,
            "biobuddy": np.asarray([], dtype=float),
        },
        args.segment_reference,
    )
    occlusion_rows, occlusion_figure = analyze_motive_occlusions(
        bundle.motive_c3d,
        trial_dir,
        bundle.name,
        generate_figure=not args.no_figures,
    )
    event_report, _contact_rows = detect_trial_events_and_contacts(
        bundle.motive_c3d,
        trial_dir,
        bundle.name,
        time_start_s=cut_start_s,
        time_end_s=cut_end_s,
    )
    dimension_rows = model_dimension_rows(bundle.name, [captury, motive])
    biobuddy_dimension_extra_rows, biobuddy_dimension_report = biobuddy_dimension_rows(
        bundle.name, args.biobuddy_biomod, args.biobuddy_unit_scale_to_m
    )
    dimension_rows.extend(biobuddy_dimension_extra_rows)
    landmark_map = load_landmark_map(args.landmark_map)
    marker_rows, marker_ts_rows = marker_correspondence_rows(
        bundle.name,
        bundle.motive_c3d,
        bundle.captury_c3d,
        rotation,
        translation,
        landmark_map,
    )
    write_rows(trial_dir / "joint_centre_metrics.csv", centre_rows)
    write_table_npz(trial_dir / "joint_centre_timeseries.npz", centre_ts_rows)
    write_rows(trial_dir / "kinematics_q_metrics.csv", q_rows)
    write_table_npz(trial_dir / "kinematics_q_timeseries.npz", q_ts_rows)
    write_rows(trial_dir / "captury_c3d_angle_metrics.csv", c3d_angle_rows)
    write_table_npz(trial_dir / "captury_c3d_angle_timeseries.npz", c3d_angle_ts_rows)
    write_rows(trial_dir / "segment_rotation_metrics.csv", segment_rows)
    write_table_npz(trial_dir / "segment_rotation_timeseries.npz", segment_ts_rows)
    write_rows(trial_dir / "model_dimensions.csv", dimension_rows)
    write_rows(trial_dir / "skin_marker_correspondence_metrics.csv", marker_rows)
    write_table_npz(
        trial_dir / "skin_marker_correspondence_timeseries.npz", marker_ts_rows
    )
    plot_metric_barh(
        pd.DataFrame(dimension_rows),
        category="dimension",
        metric="median_length_mm",
        output_path=trial_dir / "figures" / "model_dimensions" / "median_length_mm.png",
        title="Model dimensions - median_length_mm",
        xlabel="median_length_mm",
    )
    plot_metric_barh(
        pd.DataFrame(marker_rows),
        category="landmark",
        metric="median_error_mm",
        output_path=trial_dir / "figures" / "skin_markers" / "median_error_mm.png",
        title="Skin marker correspondences - median_error_mm",
        xlabel="median_error_mm",
    )
    vertical_report = vertical_amplitude_report(enriched_c3d)
    report: dict[str, Any] = {
        "trial": bundle.name,
        "files": {
            "captury_c3d": str(bundle.captury_c3d),
            "captury_bvh": str(bundle.captury_bvh) if bundle.captury_bvh else None,
            "captury_fbx": str(bundle.captury_fbx) if bundle.captury_fbx else None,
            "motive_c3d": str(bundle.motive_c3d),
            "motive_bvh": str(bundle.motive_bvh) if bundle.motive_bvh else None,
            "motive_fbx": str(bundle.motive_fbx) if bundle.motive_fbx else None,
        },
        "models": {
            "captury": {
                "source_kind": captury.source_kind,
                "biomod": str(captury.biomod_path),
                "unit_scale_to_m": captury.unit_scale_to_m,
                "mesh": captury.mesh_report,
                "n_q": int(captury.q.shape[0]),
                "n_frames": int(captury.q.shape[1]),
                "root_offset_policy": captury.root_offset_policy,
            },
            "motive": {
                "source_kind": motive.source_kind,
                "biomod": str(motive.biomod_path),
                "unit_scale_to_m": motive.unit_scale_to_m,
                "mesh": motive.mesh_report,
                "n_q": int(motive.q.shape[0]),
                "n_frames": int(motive.q.shape[1]),
                "root_offset_policy": motive.root_offset_policy,
            },
            "biobuddy": biobuddy_dimension_report,
        },
        "axis_conversion": args.model_to_c3d_axis,
        "time_window": {
            "cut_mode": args.cut_mode,
            "effective_cut_mode": effective_cut_mode,
            "manual_start_s": args.time_start,
            "manual_end_s": args.time_end,
            "used_start_s": cut_start_s,
            "used_end_s": cut_end_s,
            "captury_frames": int(captury_metrics.time.shape[0]),
            "motive_frames": int(motive_metrics.time.shape[0]),
        },
        "alignment": alignment_report,
        "outputs": {
            "enriched_c3d": str(enriched_c3d),
            "joint_centre_metrics": str(trial_dir / "joint_centre_metrics.csv"),
            "kinematics_q_metrics": str(trial_dir / "kinematics_q_metrics.csv"),
            "captury_c3d_angle_metrics": str(
                trial_dir / "captury_c3d_angle_metrics.csv"
            ),
            "captury_c3d_angle_timeseries": str(
                trial_dir / "captury_c3d_angle_timeseries.npz"
            ),
            "segment_rotation_metrics": str(trial_dir / "segment_rotation_metrics.csv"),
            "segment_rotation_timeseries": str(
                trial_dir / "segment_rotation_timeseries.npz"
            ),
            "motive_marker_occlusions": str(trial_dir / "motive_marker_occlusions.csv"),
            "trial_events_contacts": str(trial_dir / "trial_events_contacts.csv"),
            "model_dimensions": str(trial_dir / "model_dimensions.csv"),
            "skin_marker_correspondence_metrics": str(
                trial_dir / "skin_marker_correspondence_metrics.csv"
            ),
            "skin_marker_correspondence_timeseries": str(
                trial_dir / "skin_marker_correspondence_timeseries.npz"
            ),
        },
        "trial_events": event_report,
        "segment_rotations": segment_report,
        "segment_orientation_corrections": segment_orientation_report,
        "occlusion_figure": str(occlusion_figure) if occlusion_figure else None,
        "occlusion_marker_count": len(occlusion_rows),
        "angle_inventory": {
            "captury": c3d_angle_inventory(bundle.captury_c3d, args.angle_label_regex),
            "motive": c3d_angle_inventory(bundle.motive_c3d, args.angle_label_regex),
        },
        "duplicate_label_inventory": {
            "captury": duplicate_label_inventory(bundle.captury_c3d),
            "motive": duplicate_label_inventory(bundle.motive_c3d),
        },
        "vertical_amplitude": vertical_report,
        "limitations": [
            "BVH/FBX generalized-coordinate comparisons use matching q names only.",
            "Euler angle values can differ because Captury and Motive may export different axis orders and local segment frames.",
            "Joint-centre distances are the primary spatial comparison after static rigid alignment.",
        ],
        "cache": {
            "version": CACHE_VERSION,
            "hit": False,
            "fingerprint": cache_fingerprint,
        },
    }
    if args.run_ik_batch:
        report["motive_ik_batch"] = run_ik_batch(
            bundle, trial_dir, args.model_source, args.ik_max_frames
        )
    if args.visualize and (
        args.visualize_trial is None or args.visualize_trial == bundle.name
    ):
        visualize_enriched_c3d(enriched_c3d, args.rerun_wait_seconds, args.headless)
    (trial_dir / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, static_alignment_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare P6 Captury and Motive BVH/FBX/C3D trials."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--trial", action="append", default=[], help="Trial name filter. Repeatable."
    )
    parser.add_argument(
        "--joint-filter",
        action="append",
        default=[],
        help="Regex filter for compared joint centres. Repeatable.",
    )
    parser.add_argument(
        "--static-trial",
        default="Static",
        help="Trial used to compute Captury -> Motive alignment.",
    )
    parser.add_argument(
        "--time-start",
        type=float,
        default=None,
        help="Manual analysis window start in seconds.",
    )
    parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="Manual analysis window end in seconds.",
    )
    parser.add_argument(
        "--cut-mode",
        choices=["manual", "movement", "full"],
        default="manual",
        help=(
            "Trial cutting mode: manual uses --time-start/--time-end when provided, "
            "movement uses detected movement bounds, full ignores both."
        ),
    )
    parser.add_argument("--model-source", choices=["auto", "bvh", "fbx"], default="bvh")
    parser.add_argument(
        "--root-offset-mode",
        choices=["auto", "subtract", "keep"],
        default="auto",
        help=(
            "How to handle static BVH/FBX root offsets: auto scores both "
            "subtract and keep conventions against the matching C3D marker cloud."
        ),
    )
    parser.add_argument(
        "--model-to-c3d-axis",
        choices=["auto", "y_up_to_z_up", "identity"],
        default="auto",
    )
    parser.add_argument("--captury-unit-scale-to-m", type=float, default=None)
    parser.add_argument("--motive-unit-scale-to-m", type=float, default=None)
    parser.add_argument(
        "--biobuddy-biomod",
        type=Path,
        default=None,
        help=(
            "Optional BioBuddy/Biorbd model to include as a third source in "
            "model-dimension comparisons."
        ),
    )
    parser.add_argument(
        "--biobuddy-unit-scale-to-m",
        type=float,
        default=1.0,
        help=(
            "Scale applied to neutral BioBuddy model coordinates before "
            "dimension reporting. Default assumes the bioMod is in metres."
        ),
    )
    parser.add_argument("--angle-label-regex", default=ANGLE_LABEL_REGEX)
    parser.add_argument(
        "--landmark-map",
        type=Path,
        default=None,
        help="Optional JSON map for non-joint-centre Motive/Captury marker pairs.",
    )
    parser.add_argument(
        "--c3d-angle-unit",
        choices=["deg", "rad"],
        default="deg",
        help="Unit used by Captury C3D angle channels stored in POINT.",
    )
    parser.add_argument(
        "--segment-reference",
        choices=["biobuddy", "motive", "captury"],
        default="biobuddy",
        help="Reference source for segment rotation deviation metrics.",
    )
    parser.add_argument(
        "--captury-reorient-thigh-y-from-cor",
        action="store_true",
        help=(
            "Correct Captury thigh segment frames by orienting the local Y axis "
            "from hip CoR to knee CoR before segment-angle metrics."
        ),
    )
    parser.add_argument(
        "--rotate-body-segments-180-x",
        action="store_true",
        help=(
            "Rotate Captury and Motive segment frames by 180 degrees around each "
            "segment local X axis before segment-angle metrics."
        ),
    )
    parser.add_argument(
        "--no-mesh", action="store_true", help="Skip FBX visual mesh extraction."
    )
    parser.add_argument("--max-mesh-points", type=int, default=0)
    parser.add_argument(
        "--disable-static-model-alignment",
        action="store_true",
        help="Disable the static Captury model -> Motive model rigid alignment.",
    )
    parser.add_argument(
        "--disable-motive-marker-alignment",
        action="store_true",
        help="Disable the Motive model -> Motive C3D marker-proxy yaw/translation alignment.",
    )
    parser.add_argument("--run-ik-batch", action="store_true")
    parser.add_argument("--ik-max-frames", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize-trial", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rerun-wait-seconds", type=float, default=1.0)
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip PNG metric figure generation."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation even when trial outputs match the cache.",
    )
    parser.add_argument(
        "--occlusions-only",
        action="store_true",
        help="Only compute Motive marker occlusion CSV outputs.",
    )
    parser.add_argument("--list-trials", action="store_true")
    return parser.parse_args()


def run_occlusions_only(
    trials: list[TrialBundle], args: argparse.Namespace
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for bundle in trials:
        trial_dir = args.out_dir / safe_name(bundle.name)
        trial_dir.mkdir(parents=True, exist_ok=True)
        rows, figure = analyze_motive_occlusions(
            bundle.motive_c3d,
            trial_dir,
            bundle.name,
            generate_figure=not args.no_figures,
        )
        all_rows.extend(rows)
        reports.append(
            {
                "trial": bundle.name,
                "motive_c3d": str(bundle.motive_c3d),
                "rows": len(rows),
                "output": str(trial_dir / "motive_marker_occlusions.csv"),
                "figure": str(figure) if figure else None,
            }
        )
    write_rows(args.out_dir / "all_motive_marker_occlusions.csv", all_rows)
    (args.out_dir / "run_report_occlusions.json").write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "out_dir": str(args.out_dir),
                "n_trials": len(trials),
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Computed occlusions for {len(trials)} trial(s).")
    print(f"Occlusions: {args.out_dir / 'all_motive_marker_occlusions.csv'}")
    return all_rows


def main() -> None:
    args = parse_args()
    trials = discover_trials(args.data_root)
    if args.list_trials:
        for bundle in trials:
            print(bundle.name)
        return
    if args.trial:
        requested = set(args.trial)
        trials = [bundle for bundle in trials if bundle.name in requested]
    if not trials:
        raise RuntimeError(f"No trials found in {args.data_root}.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.occlusions_only:
        run_occlusions_only(trials, args)
        return

    static_bundle = next(
        (
            bundle
            for bundle in discover_trials(args.data_root)
            if bundle.name == args.static_trial
        ),
        None,
    )
    static_transform: tuple[np.ndarray, np.ndarray] | None = None
    reports: list[dict[str, Any]] = []
    if static_bundle is not None and not args.trial:
        static_report, static_transform = compare_trial(
            static_bundle, args.out_dir, args, static_alignment_transform=None
        )
        reports.append(static_report)
        trials = [bundle for bundle in trials if bundle.name != static_bundle.name]
    elif static_bundle is not None and all(
        bundle.name != static_bundle.name for bundle in trials
    ):
        _, static_transform = compare_trial(
            static_bundle,
            args.out_dir / "_static_alignment",
            args,
            static_alignment_transform=None,
        )
    for bundle in trials:
        report, static_transform = compare_trial(
            bundle, args.out_dir, args, static_alignment_transform=static_transform
        )
        reports.append(report)

    all_centre_rows: list[dict[str, Any]] = []
    all_q_rows: list[dict[str, Any]] = []
    all_occlusion_rows: list[dict[str, Any]] = []
    all_dimension_rows: list[dict[str, Any]] = []
    all_marker_rows: list[dict[str, Any]] = []
    all_segment_rows: list[dict[str, Any]] = []
    for report in reports:
        trial_dir = args.out_dir / safe_name(report["trial"])
        centre_csv = trial_dir / "joint_centre_metrics.csv"
        q_csv = trial_dir / "kinematics_q_metrics.csv"
        occlusion_csv = trial_dir / "motive_marker_occlusions.csv"
        dimension_csv = trial_dir / "model_dimensions.csv"
        marker_csv = trial_dir / "skin_marker_correspondence_metrics.csv"
        segment_csv = trial_dir / "segment_rotation_metrics.csv"
        if centre_csv.exists() and centre_csv.stat().st_size:
            all_centre_rows.extend(pd.read_csv(centre_csv).to_dict("records"))
        if q_csv.exists() and q_csv.stat().st_size:
            all_q_rows.extend(pd.read_csv(q_csv).to_dict("records"))
        if occlusion_csv.exists() and occlusion_csv.stat().st_size:
            all_occlusion_rows.extend(pd.read_csv(occlusion_csv).to_dict("records"))
        if dimension_csv.exists() and dimension_csv.stat().st_size:
            all_dimension_rows.extend(pd.read_csv(dimension_csv).to_dict("records"))
        if marker_csv.exists() and marker_csv.stat().st_size:
            all_marker_rows.extend(pd.read_csv(marker_csv).to_dict("records"))
        if segment_csv.exists() and segment_csv.stat().st_size:
            all_segment_rows.extend(pd.read_csv(segment_csv).to_dict("records"))
    write_rows(args.out_dir / "all_joint_centre_metrics.csv", all_centre_rows)
    write_rows(args.out_dir / "all_kinematics_q_metrics.csv", all_q_rows)
    write_rows(args.out_dir / "all_motive_marker_occlusions.csv", all_occlusion_rows)
    write_rows(args.out_dir / "all_model_dimensions.csv", all_dimension_rows)
    write_rows(args.out_dir / "all_segment_rotation_metrics.csv", all_segment_rows)
    write_rows(
        args.out_dir / "all_skin_marker_correspondence_metrics.csv", all_marker_rows
    )
    figures = (
        {
            "joint_centres": [],
            "kinematics_q": [],
            "occlusions": [],
            "model_dimensions": [],
            "skin_markers": [],
        }
        if args.no_figures
        else generate_metric_figures(all_centre_rows, all_q_rows, args.out_dir)
    )
    if not args.no_figures:
        figures.setdefault("occlusions", [])
        figures.setdefault("model_dimensions", [])
        figures.setdefault("skin_markers", [])
        if all_occlusion_rows:
            path = plot_metric_barh(
                pd.DataFrame(all_occlusion_rows),
                category="marker",
                metric="missing_percent",
                output_path=args.out_dir
                / "figures"
                / "occlusions"
                / "missing_percent.png",
                title="Motive marker occlusions - missing_percent",
                xlabel="missing_percent",
            )
            if path:
                figures["occlusions"].append(str(path))
        if all_dimension_rows:
            path = plot_metric_barh(
                pd.DataFrame(all_dimension_rows),
                category="dimension",
                metric="median_length_mm",
                output_path=args.out_dir
                / "figures"
                / "model_dimensions"
                / "median_length_mm.png",
                title="Model dimensions - median_length_mm",
                xlabel="median_length_mm",
            )
            if path:
                figures["model_dimensions"].append(str(path))
        if all_marker_rows:
            for metric in ("median_error_mm", "p95_error_mm", "rmse_error_mm"):
                path = plot_metric_barh(
                    pd.DataFrame(all_marker_rows),
                    category="landmark",
                    metric=metric,
                    output_path=args.out_dir
                    / "figures"
                    / "skin_markers"
                    / f"{metric}.png",
                    title=f"Skin marker correspondences - {metric}",
                    xlabel=metric,
                )
                if path:
                    figures["skin_markers"].append(str(path))
    (args.out_dir / "run_report.json").write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "out_dir": str(args.out_dir),
                "n_trials": len(reports),
                "static_trial": args.static_trial,
                "figures": figures,
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Compared {len(reports)} trial(s).")
    print(f"Joint-centre metrics: {args.out_dir / 'all_joint_centre_metrics.csv'}")
    print(f"Kinematics metrics: {args.out_dir / 'all_kinematics_q_metrics.csv'}")
    if not args.no_figures:
        print(f"Figures: {args.out_dir / 'figures'}")
    print(f"Report: {args.out_dir / 'run_report.json'}")


if __name__ == "__main__":
    main()
