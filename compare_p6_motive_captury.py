#!/usr/bin/env python3
"""Batch comparison for Captury/Motive kinematic datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    require_ezc3d,
    require_pyorerun,
    save_model_joint_centres,
    save_q_outputs,
    split_c3d_points,
)
from compare_capture_systems import (
    DEFAULT_LANDMARK_MAP,
    detect_angle_indices,
    unit_scale_to_mm,
)
from model_comparison_metrics import joint_center_error_xyz, waveform_metrics

DEFAULT_DATA_ROOT = Path("local_trials/2026-06-30_P6_flat")
DEFAULT_OUTPUT_ROOT = Path("out_p6_motive_captury_comparison")
ANGLE_LABEL_REGEX = r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)"
FOOT_MARKER_PATTERN = r"(LFCC|RFCC|LFM|RFM|LDP|RDP|Foot|Toe|Heel)"


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
    unit_scale_to_m: float
    mesh_report: dict[str, Any]


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


def build_model_run(
    bundle: TrialBundle,
    system: str,
    model_source: str,
    out_dir: Path,
    include_mesh: bool,
    max_mesh_points: int,
    unit_scale_override: float | None,
) -> ModelRun:
    source_kind, source_path = select_model_file(bundle, system, model_source)
    source_dir = out_dir / system / source_kind
    source_dir.mkdir(parents=True, exist_ok=True)
    biomod_path = source_dir / f"{system}_{source_kind}_biobuddy.bioMod"
    mesh_report: dict[str, Any] = {
        "mesh_file_count": 0,
        "mesh_vertices": 0,
        "mesh_faces": 0,
    }
    if source_kind == "bvh":
        _, parser = build_biomod_from_bvh_with_biobuddy(
            source_path, biomod_path, add_joint_centre_markers=True
        )
        runtime = extract_q_from_biobuddy_bvh_parser(
            parser, apply_root_offset_correction=True
        )
        joint_names = runtime.joint_names
    else:
        _, parser, mesh_report = build_biomod_from_fbx_with_biobuddy(
            source_path,
            biomod_path,
            add_joint_centre_markers=True,
            include_mesh=include_mesh,
            max_mesh_points=max_mesh_points,
        )
        runtime = extract_q_from_fbx_parser(parser, apply_root_offset_correction=True)
        joint_names = collect_fbx_joint_names_depth_first(parser)
        mesh_dir = biomod_path.parent / "meshes"
        if include_mesh and mesh_dir.exists():
            try:
                mesh_report = convert_biobuddy_ply_meshes_to_vtp(mesh_dir)
            except Exception:
                pass
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
    centres_native = compute_model_joint_centres_native(
        biomod_path, runtime.q, set(joint_names)
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
        unit_scale_to_m=native_unit_scale_to_m(
            system, source_kind, unit_scale_override
        ),
        mesh_report=mesh_report,
    )


def model_to_c3d_matrix(axis_mode: str) -> np.ndarray:
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


def interpolate_centres_to_time(
    centres_mm: dict[str, np.ndarray], source_time: np.ndarray, target_time: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: interpolate_array(values, source_time, target_time)
        for name, values in centres_mm.items()
    }


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
    if joint_filters:
        import re

        regexes = [re.compile(pattern) for pattern in joint_filters]
        common = [
            joint for joint in common if any(regex.search(joint) for regex in regexes)
        ]
    for joint in common:
        cap = cap_on_motive[joint].T
        mot = motive_centres_mm[joint].T
        valid = np.all(np.isfinite(cap), axis=1) & np.all(np.isfinite(mot), axis=1)
        errors = np.linalg.norm(cap - mot, axis=1)
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
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points = np.asarray(c3d["data"]["points"], dtype=float)
    xyz_mm = points[:3] * unit_mm
    residuals = (
        points[3]
        if points.shape[0] > 3
        else np.zeros((points.shape[1], points.shape[2]))
    )
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [120])
    rate = float(
        rate_value[0]
        if isinstance(rate_value, (list, tuple, np.ndarray))
        else rate_value
    )
    time = np.arange(xyz_mm.shape[2], dtype=float) / rate
    return labels, xyz_mm, residuals, time


def clean_marker_label(label: str) -> str:
    return label.replace("Skeleton_001_", "").strip()


def marker_indices_by_clean_label(labels: list[str]) -> dict[str, list[int]]:
    lookup: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        lookup.setdefault(clean_marker_label(label), []).append(i)
    return lookup


def average_marker_group(
    points_mm: np.ndarray, indices: list[int]
) -> np.ndarray | None:
    if not indices:
        return None
    values = points_mm[:, indices, :]
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=1)


def analyze_motive_occlusions(
    motive_c3d: Path, trial_dir: Path, trial: str
) -> tuple[list[dict[str, Any]], Path | None]:
    labels, points_mm, residuals, _ = read_c3d_points_mm(motive_c3d)
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        xyz = points_mm[:, i, :]
        missing = ~np.all(np.isfinite(xyz), axis=0)
        if residuals.shape[0] > i:
            missing = missing | (residuals[i, :] < 0)
        rows.append(
            {
                "trial": trial,
                "marker": label,
                "missing_frames": int(np.sum(missing)),
                "total_frames": int(missing.shape[0]),
                "missing_percent": float(100.0 * np.mean(missing)),
            }
        )
    csv_path = trial_dir / "motive_marker_occlusions.csv"
    write_rows(csv_path, rows)
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
    for i, time_value in enumerate(time):
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
        "left_foot_markers": [labels[i] for i in left_indices],
        "right_foot_markers": [labels[i] for i in right_indices],
    }
    (trial_dir / "trial_events.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, rows


SEGMENT_LENGTH_PAIRS = [
    ("pelvis_to_spine", "Hips", "Spine"),
    ("left_thigh", "LeftUpLeg", "LeftLeg"),
    ("left_shank", "LeftLeg", "LeftFoot"),
    ("left_foot", "LeftFoot", "LeftToeBase"),
    ("right_thigh", "RightUpLeg", "RightLeg"),
    ("right_shank", "RightLeg", "RightFoot"),
    ("right_foot", "RightFoot", "RightToeBase"),
    ("left_upper_arm", "LeftArm", "LeftForeArm"),
    ("left_forearm", "LeftForeArm", "LeftHand"),
    ("right_upper_arm", "RightArm", "RightForeArm"),
    ("right_forearm", "RightForeArm", "RightHand"),
]


def model_dimension_rows(trial: str, runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        centres_mm = centres_to_c3d_mm(
            run.centres_native, run.unit_scale_to_m, "identity"
        )
        for name, proximal, distal in SEGMENT_LENGTH_PAIRS:
            if proximal not in centres_mm or distal not in centres_mm:
                continue
            length = np.linalg.norm(centres_mm[distal] - centres_mm[proximal], axis=0)
            rows.append(
                {
                    "trial": trial,
                    "system": run.system,
                    "source_kind": run.source_kind,
                    "dimension": name,
                    "median_length_mm": float(np.nanmedian(length)),
                    "sd_length_mm": float(np.nanstd(length)),
                }
            )
    return rows


def marker_correspondence_rows(
    trial: str,
    motive_c3d: Path,
    captury_c3d: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[dict[str, Any]]:
    motive_labels, motive_points, _motive_residuals, motive_time = read_c3d_points_mm(
        motive_c3d
    )
    captury_labels, captury_points, _captury_residuals, captury_time = (
        read_c3d_points_mm(captury_c3d)
    )
    motive_lookup = marker_indices_by_clean_label(motive_labels)
    captury_lookup = marker_indices_by_clean_label(captury_labels)
    rows: list[dict[str, Any]] = []
    for item in DEFAULT_LANDMARK_MAP:
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
    return rows


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
    captury = build_model_run(
        bundle,
        "captury",
        args.model_source,
        trial_dir,
        include_mesh=not args.no_mesh,
        max_mesh_points=args.max_mesh_points,
        unit_scale_override=args.captury_unit_scale_to_m,
    )
    motive = build_model_run(
        bundle,
        "motive",
        args.model_source,
        trial_dir,
        include_mesh=not args.no_mesh,
        max_mesh_points=args.max_mesh_points,
        unit_scale_override=args.motive_unit_scale_to_m,
    )
    cap_c3d_mm = centres_to_c3d_mm(
        captury.centres_native, captury.unit_scale_to_m, args.model_to_c3d_axis
    )
    mot_c3d_mm = centres_to_c3d_mm(
        motive.centres_native, motive.unit_scale_to_m, args.model_to_c3d_axis
    )
    alignment_report: dict[str, Any]
    if static_alignment_transform is None:
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
    enriched_c3d = append_centres_to_motive_c3d(
        bundle.motive_c3d,
        trial_dir / f"{safe_name(bundle.name)}_motive_with_capjc_motjc.c3d",
        cap_aligned_mm,
        mot_c3d_mm,
        captury.time,
        motive.time,
        args.angle_label_regex,
    )
    centre_rows, centre_ts_rows = centre_metric_rows(
        bundle.name,
        cap_aligned_mm,
        mot_c3d_mm,
        captury.time,
        motive.time,
        args.joint_filter,
    )
    q_rows, q_ts_rows = q_metric_rows(bundle.name, captury, motive)
    occlusion_rows, occlusion_figure = analyze_motive_occlusions(
        bundle.motive_c3d, trial_dir, bundle.name
    )
    event_report, _contact_rows = detect_trial_events_and_contacts(
        bundle.motive_c3d, trial_dir, bundle.name
    )
    dimension_rows = model_dimension_rows(bundle.name, [captury, motive])
    marker_rows = marker_correspondence_rows(
        bundle.name, bundle.motive_c3d, bundle.captury_c3d, rotation, translation
    )
    write_rows(trial_dir / "joint_centre_metrics.csv", centre_rows)
    write_rows(trial_dir / "joint_centre_timeseries.csv", centre_ts_rows)
    write_rows(trial_dir / "kinematics_q_metrics.csv", q_rows)
    write_rows(trial_dir / "kinematics_q_timeseries.csv", q_ts_rows)
    write_rows(trial_dir / "model_dimensions.csv", dimension_rows)
    write_rows(trial_dir / "skin_marker_correspondence_metrics.csv", marker_rows)
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
            },
            "motive": {
                "source_kind": motive.source_kind,
                "biomod": str(motive.biomod_path),
                "unit_scale_to_m": motive.unit_scale_to_m,
                "mesh": motive.mesh_report,
                "n_q": int(motive.q.shape[0]),
                "n_frames": int(motive.q.shape[1]),
            },
        },
        "axis_conversion": args.model_to_c3d_axis,
        "alignment": alignment_report,
        "outputs": {
            "enriched_c3d": str(enriched_c3d),
            "joint_centre_metrics": str(trial_dir / "joint_centre_metrics.csv"),
            "kinematics_q_metrics": str(trial_dir / "kinematics_q_metrics.csv"),
            "motive_marker_occlusions": str(trial_dir / "motive_marker_occlusions.csv"),
            "trial_events_contacts": str(trial_dir / "trial_events_contacts.csv"),
            "model_dimensions": str(trial_dir / "model_dimensions.csv"),
            "skin_marker_correspondence_metrics": str(
                trial_dir / "skin_marker_correspondence_metrics.csv"
            ),
        },
        "trial_events": event_report,
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
    parser.add_argument("--model-source", choices=["auto", "bvh", "fbx"], default="bvh")
    parser.add_argument(
        "--model-to-c3d-axis",
        choices=["y_up_to_z_up", "identity"],
        default="y_up_to_z_up",
    )
    parser.add_argument("--captury-unit-scale-to-m", type=float, default=None)
    parser.add_argument("--motive-unit-scale-to-m", type=float, default=None)
    parser.add_argument("--angle-label-regex", default=ANGLE_LABEL_REGEX)
    parser.add_argument(
        "--no-mesh", action="store_true", help="Skip FBX visual mesh extraction."
    )
    parser.add_argument("--max-mesh-points", type=int, default=0)
    parser.add_argument("--run-ik-batch", action="store_true")
    parser.add_argument("--ik-max-frames", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize-trial", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rerun-wait-seconds", type=float, default=1.0)
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip PNG metric figure generation."
    )
    parser.add_argument("--list-trials", action="store_true")
    return parser.parse_args()


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
    for report in reports:
        trial_dir = args.out_dir / safe_name(report["trial"])
        centre_csv = trial_dir / "joint_centre_metrics.csv"
        q_csv = trial_dir / "kinematics_q_metrics.csv"
        occlusion_csv = trial_dir / "motive_marker_occlusions.csv"
        dimension_csv = trial_dir / "model_dimensions.csv"
        marker_csv = trial_dir / "skin_marker_correspondence_metrics.csv"
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
    write_rows(args.out_dir / "all_joint_centre_metrics.csv", all_centre_rows)
    write_rows(args.out_dir / "all_kinematics_q_metrics.csv", all_q_rows)
    write_rows(args.out_dir / "all_motive_marker_occlusions.csv", all_occlusion_rows)
    write_rows(args.out_dir / "all_model_dimensions.csv", all_dimension_rows)
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
