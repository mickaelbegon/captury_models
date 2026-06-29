"""Compare marker-based and markerless C3D/model outputs across capture systems."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_comparison_metrics import joint_center_error_xyz, waveform_metrics


DEFAULT_CAPTURY_ANGLE_LABELS = {
    "RHip",
    "LHip",
    "RKne",
    "LKne",
    "RAnk",
    "LAnk",
    "RSho",
    "LSho",
    "RElb",
    "LElb",
    "RWri",
    "LWri",
    "Neck",
}

DEFAULT_LANDMARK_MAP = [
    {"name": "pelvis_center", "reference": ["LIAS", "RIAS", "LIPS", "RIPS"], "test": ["Q_Wa"]},
    {"name": "left_hip_region", "reference": ["LFTC"], "test": ["Q_LT"]},
    {"name": "right_hip_region", "reference": ["RFTC"], "test": ["Q_RT"]},
    {"name": "left_knee_center", "reference": ["LFLE", "LFME"], "test": ["Q_LK"]},
    {"name": "right_knee_center", "reference": ["RFLE", "RFME"], "test": ["Q_RK"]},
    {"name": "left_ankle_center", "reference": ["LFAL", "LTAM"], "test": ["Q_LA"]},
    {"name": "right_ankle_center", "reference": ["RFAL", "RTAM"], "test": ["Q_RA"]},
    {"name": "left_foot_center", "reference": ["LFM1", "LFM2", "LFM5", "LFCC", "LDP1"], "test": ["Q_LF"]},
    {"name": "right_foot_center", "reference": ["RFM1", "RFM2", "RFM5", "RFCC", "RDP1"], "test": ["Q_RF"]},
    {"name": "chest_center", "reference": ["SJN", "SXS"], "test": ["Q_Ch"]},
    {"name": "upper_spine_center", "reference": ["TV2", "TV7", "CV7"], "test": ["Q_Sp"]},
    {"name": "head_center", "reference": ["LAH", "RAH", "LPH", "RPH"], "test": ["Q_He"]},
    {"name": "left_shoulder_region", "reference": ["LCAJ"], "test": ["Q_LS"]},
    {"name": "right_shoulder_region", "reference": ["RCAJ"], "test": ["Q_RS"]},
    {"name": "left_elbow_center", "reference": ["LHLE", "LHME"], "test": ["Q_LE"]},
    {"name": "right_elbow_center", "reference": ["RHLE", "RHME"], "test": ["Q_RE"]},
    {"name": "left_wrist_center", "reference": ["LUSP", "LRSP"], "test": ["Q_LW"]},
    {"name": "right_wrist_center", "reference": ["RUSP", "RRSP"], "test": ["Q_RW"]},
    {"name": "left_hand_region", "reference": ["LHM2"], "test": ["Q_LH"]},
    {"name": "right_hand_region", "reference": ["RHM2"], "test": ["Q_RH"]},
]


@dataclass
class C3dData:
    path: Path
    labels: list[str]
    points_mm: np.ndarray
    time: np.ndarray
    rate: float
    unit: str
    angle_units: str
    angle_indices: dict[str, int]


@dataclass
class TrialFiles:
    participant: str
    name: str
    system: str
    root: Path
    c3d: Path | None = None
    bvh: Path | None = None
    fbx: Path | None = None


@dataclass
class TrialPair:
    participant: str
    name: str
    reference: TrialFiles
    test: TrialFiles


def require_ezc3d():
    try:
        import ezc3d  # type: ignore
    except ImportError as exc:
        raise ImportError("ezc3d is required. Install it in the captury_biobuddy environment.") from exc
    return ezc3d


def get_c3d_param(c3d: dict, group: str, name: str, default: Any = None) -> Any:
    try:
        return c3d["parameters"][group][name]["value"]
    except KeyError:
        return default


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [str(value).strip()]
    return [str(v).strip() for v in value]


def unit_scale_to_mm(unit: str) -> float:
    unit = unit.strip().lower()
    if unit in {"m", "meter", "meters", "metre", "metres"}:
        return 1000.0
    if unit in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 1.0
    if unit in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 10.0
    return 1.0


def canonical_angle_name(name: str) -> str:
    value = name.strip()
    value = re.sub(r"(?i)angles?$", "", value)
    replacements = {
        "RHip": "right_hip",
        "LHip": "left_hip",
        "RKne": "right_knee",
        "LKne": "left_knee",
        "RKnee": "right_knee",
        "LKnee": "left_knee",
        "RAnk": "right_ankle",
        "LAnk": "left_ankle",
        "RAnkle": "right_ankle",
        "LAnkle": "left_ankle",
        "RSho": "right_shoulder",
        "LSho": "left_shoulder",
        "RShoulder": "right_shoulder",
        "LShoulder": "left_shoulder",
        "RElb": "right_elbow",
        "LElb": "left_elbow",
        "RElbow": "right_elbow",
        "LElbow": "left_elbow",
        "RWri": "right_wrist",
        "LWri": "left_wrist",
        "RWrist": "right_wrist",
        "LWrist": "left_wrist",
        "Neck": "neck",
    }
    if value in replacements:
        return replacements[value]
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()


def read_c3d(path: Path, angle_label_regex: str) -> C3dData:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    unit = as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [0])
    rate = float(rate_value[0] if isinstance(rate_value, (list, tuple, np.ndarray)) else rate_value)
    points_mm = np.asarray(c3d["data"]["points"], dtype=float)[:3, :, :] * unit_scale_to_mm(unit)
    time = np.arange(points_mm.shape[2], dtype=float) / rate
    angle_units = as_str_list(get_c3d_param(c3d, "POINT", "ANGLE_UNITS", ["deg"]))[0] or "deg"
    angle_indices = detect_angle_indices(c3d, labels, angle_label_regex)
    return C3dData(
        path=path,
        labels=labels,
        points_mm=points_mm,
        time=time,
        rate=rate,
        unit=unit,
        angle_units=angle_units,
        angle_indices=angle_indices,
    )


def detect_angle_indices(c3d: dict, labels: list[str], angle_label_regex: str) -> dict[str, int]:
    angle_indices: dict[str, int] = {}
    regex = re.compile(angle_label_regex)
    point_angles = as_str_list(get_c3d_param(c3d, "POINT", "ANGLES", []))
    if point_angles and len(point_angles) <= len(labels):
        start = len(labels) - len(point_angles)
        for i, angle_name in enumerate(point_angles):
            angle_indices[canonical_angle_name(angle_name)] = start + i
    for i, label in enumerate(labels):
        if label in DEFAULT_CAPTURY_ANGLE_LABELS or regex.search(label):
            angle_indices.setdefault(canonical_angle_name(label), i)
    return angle_indices


def duplicate_label_indices(labels: list[str], label: str) -> list[int]:
    return [i for i, current in enumerate(labels) if current == label]


def extract_label_average(c3d: C3dData, label_names: list[str]) -> tuple[np.ndarray | None, list[str], list[str]]:
    selected_indices: list[int] = []
    used: list[str] = []
    missing: list[str] = []
    for label in label_names:
        indices = duplicate_label_indices(c3d.labels, label)
        if indices:
            selected_indices.extend(indices)
            used.append(label)
        else:
            missing.append(label)
    if not selected_indices:
        return None, used, missing
    values = c3d.points_mm[:, selected_indices, :]
    with np.errstate(invalid="ignore"):
        summed = np.nansum(values, axis=1)
        counts = np.sum(np.isfinite(values), axis=1)
        averaged = np.divide(summed, counts, out=np.full_like(summed, np.nan), where=counts > 0)
    return averaged, used, missing


def resample_xyz(signal_3_by_t: np.ndarray, n_points: int) -> np.ndarray:
    x_old = np.linspace(0, 1, signal_3_by_t.shape[-1])
    x_new = np.linspace(0, 1, n_points)
    output = np.empty((n_points, 3), dtype=float)
    for component in range(3):
        values = signal_3_by_t[component, :]
        finite = np.isfinite(values)
        output[:, component] = np.interp(x_new, x_old[finite], values[finite]) if finite.sum() >= 2 else np.nan
    return output


def resample_1d(signal: np.ndarray, n_points: int) -> np.ndarray:
    x_old = np.linspace(0, 1, signal.shape[-1])
    x_new = np.linspace(0, 1, n_points)
    finite = np.isfinite(signal)
    return np.interp(x_new, x_old[finite], signal[finite]) if finite.sum() >= 2 else np.full(n_points, np.nan)


def load_landmark_map(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_LANDMARK_MAP
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, list):
        raise ValueError("The landmark map must be a JSON list.")
    return mapping


def extract_landmarks(
    reference: C3dData,
    test: C3dData,
    landmark_map: list[dict[str, Any]],
    n_points: int,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    names: list[str] = []
    reference_curves: list[np.ndarray] = []
    test_curves: list[np.ndarray] = []
    report: list[dict[str, Any]] = []
    for item in landmark_map:
        name = str(item["name"])
        ref_labels = list(item["reference"])
        test_labels = list(item["test"])
        ref_signal, ref_used, ref_missing = extract_label_average(reference, ref_labels)
        test_signal, test_used, test_missing = extract_label_average(test, test_labels)
        available = ref_signal is not None and test_signal is not None
        report.append(
            {
                "landmark": name,
                "available": available,
                "reference_labels": ref_labels,
                "reference_used": ref_used,
                "reference_missing": ref_missing,
                "test_labels": test_labels,
                "test_used": test_used,
                "test_missing": test_missing,
            }
        )
        if not available:
            continue
        names.append(name)
        reference_curves.append(resample_xyz(ref_signal, n_points))
        test_curves.append(resample_xyz(test_signal, n_points))
    if not names:
        return names, np.empty((0, n_points, 3)), np.empty((0, n_points, 3)), report
    return names, np.stack(reference_curves), np.stack(test_curves), report


def kabsch_transform(reference_points: np.ndarray, test_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_mean = np.mean(reference_points, axis=0)
    test_mean = np.mean(test_points, axis=0)
    ref_centered = reference_points - ref_mean
    test_centered = test_points - test_mean
    h = test_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(h)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = ref_mean - test_mean @ rotation
    return rotation, translation


def apply_global_rigid_alignment(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ref_flat = reference.reshape((-1, 3))
    test_flat = test.reshape((-1, 3))
    mask = np.all(np.isfinite(ref_flat), axis=1) & np.all(np.isfinite(test_flat), axis=1)
    if mask.sum() < 3:
        return test.copy(), {"alignment": "global_rigid", "n_points": int(mask.sum()), "status": "not_enough_points"}
    rotation, translation = kabsch_transform(ref_flat[mask], test_flat[mask])
    aligned = test.reshape((-1, 3)) @ rotation + translation
    return aligned.reshape(test.shape), {
        "alignment": "global_rigid",
        "n_points": int(mask.sum()),
        "rotation": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "status": "ok",
    }


def apply_per_frame_rigid_alignment(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    aligned = test.copy()
    ok_frames = 0
    for frame in range(reference.shape[1]):
        ref_frame = reference[:, frame, :]
        test_frame = test[:, frame, :]
        mask = np.all(np.isfinite(ref_frame), axis=1) & np.all(np.isfinite(test_frame), axis=1)
        if mask.sum() < 3:
            continue
        rotation, translation = kabsch_transform(ref_frame[mask], test_frame[mask])
        aligned[:, frame, :] = test_frame @ rotation + translation
        ok_frames += 1
    return aligned, {"alignment": "per_frame_rigid", "frames_aligned": ok_frames, "status": "ok" if ok_frames else "not_enough_points"}


def align_landmarks(reference: np.ndarray, test: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return test.copy(), {"alignment": "none", "status": "ok"}
    if mode == "global_rigid":
        return apply_global_rigid_alignment(reference, test)
    if mode == "per_frame_rigid":
        return apply_per_frame_rigid_alignment(reference, test)
    raise ValueError(f"Unsupported alignment mode: {mode}")


def landmark_metrics_rows(
    participant: str,
    trial_name: str,
    names: list[str],
    reference: np.ndarray,
    test: np.ndarray,
    aligned: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        for variant, candidate in (("raw", test[i]), ("aligned", aligned[i])):
            error = np.linalg.norm(candidate - reference[i], axis=1)
            valid = np.isfinite(error)
            xyz_metrics = joint_center_error_xyz(reference[i], candidate)
            rows.append(
                {
                    "participant": participant,
                    "trial": trial_name,
                    "landmark": name,
                    "variant": variant,
                    "n_points": int(valid.sum()),
                    "median_error_mm": float(np.nanmedian(error)),
                    "p95_error_mm": float(np.nanpercentile(error, 95)),
                    "max_error_mm": float(np.nanmax(error)),
                    **xyz_metrics,
                }
            )
    return rows


def landmark_timeseries_rows(
    participant: str,
    trial_name: str,
    names: list[str],
    reference: np.ndarray,
    test: np.ndarray,
    aligned: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_points = reference.shape[1]
    for i, name in enumerate(names):
        for frame in range(n_points):
            raw_error = float(np.linalg.norm(test[i, frame] - reference[i, frame]))
            aligned_error = float(np.linalg.norm(aligned[i, frame] - reference[i, frame]))
            rows.append(
                {
                    "participant": participant,
                    "trial": trial_name,
                    "landmark": name,
                    "percent": frame * 100.0 / (n_points - 1) if n_points > 1 else 0.0,
                    "reference_x_mm": reference[i, frame, 0],
                    "reference_y_mm": reference[i, frame, 1],
                    "reference_z_mm": reference[i, frame, 2],
                    "test_raw_x_mm": test[i, frame, 0],
                    "test_raw_y_mm": test[i, frame, 1],
                    "test_raw_z_mm": test[i, frame, 2],
                    "test_aligned_x_mm": aligned[i, frame, 0],
                    "test_aligned_y_mm": aligned[i, frame, 1],
                    "test_aligned_z_mm": aligned[i, frame, 2],
                    "raw_error_mm": raw_error,
                    "aligned_error_mm": aligned_error,
                }
            )
    return rows


def angle_metrics_and_timeseries(
    participant: str,
    trial_name: str,
    reference: C3dData,
    test: C3dData,
    n_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    common = sorted(set(reference.angle_indices).intersection(test.angle_indices))
    for angle_name in common:
        ref_idx = reference.angle_indices[angle_name]
        test_idx = test.angle_indices[angle_name]
        for component, axis_name in enumerate(("x", "y", "z")):
            ref_curve = resample_1d(reference.points_mm[component, ref_idx, :], n_points)
            test_curve = resample_1d(test.points_mm[component, test_idx, :], n_points)
            metric_rows.append(
                {
                    "participant": participant,
                    "trial": trial_name,
                    "angle": angle_name,
                    "component": axis_name,
                    "reference_label": reference.labels[ref_idx],
                    "test_label": test.labels[test_idx],
                    **waveform_metrics(ref_curve, test_curve, unit=reference.angle_units or "deg"),
                }
            )
            for frame in range(n_points):
                timeseries_rows.append(
                    {
                        "participant": participant,
                        "trial": trial_name,
                        "angle": angle_name,
                        "component": axis_name,
                        "percent": frame * 100.0 / (n_points - 1) if n_points > 1 else 0.0,
                        "reference": ref_curve[frame],
                        "test": test_curve[frame],
                    }
                )
    return metric_rows, timeseries_rows


def c3d_inventory(c3d: C3dData) -> dict[str, Any]:
    marker_labels = [label for i, label in enumerate(c3d.labels) if i not in set(c3d.angle_indices.values())]
    return {
        "path": str(c3d.path),
        "frames": int(c3d.points_mm.shape[2]),
        "rate_hz": c3d.rate,
        "unit": c3d.unit,
        "labels_count": len(c3d.labels),
        "marker_labels_count": len(marker_labels),
        "marker_labels": marker_labels,
        "angle_labels_count": len(c3d.angle_indices),
        "angle_labels": {name: c3d.labels[index] for name, index in c3d.angle_indices.items()},
        "angle_units": c3d.angle_units,
    }


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_") or "trial"


def trial_files_dict(trial: TrialFiles) -> dict[str, Any]:
    return {
        "participant": trial.participant,
        "name": trial.name,
        "system": trial.system,
        "root": str(trial.root),
        "c3d": None if trial.c3d is None else str(trial.c3d),
        "bvh": None if trial.bvh is None else str(trial.bvh),
        "fbx": None if trial.fbx is None else str(trial.fbx),
    }


def model_file_inventory(trial: TrialFiles) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("c3d", "bvh", "fbx"):
        path = getattr(trial, kind)
        rows.append(
            {
                "participant": trial.participant,
                "system": trial.system,
                "trial": trial.name,
                "kind": kind,
                "available": path is not None,
                "path": None if path is None else str(path),
                "size_bytes": None if path is None or not path.exists() else path.stat().st_size,
            }
        )
    return rows


def choose_preferred_file(files: list[Path], suffix: str) -> Path | None:
    if not files:
        return None
    unknown = [path for path in files if path.name.lower() == f"unknown{suffix}"]
    return sorted(unknown or files)[0]


def discover_trial_files(system_dir: Path, system_name: str, participant: str) -> dict[str, TrialFiles]:
    trials: dict[str, TrialFiles] = {}
    if not system_dir.exists():
        return trials

    for child in sorted(system_dir.iterdir()):
        if child.is_dir():
            c3d_files = sorted(child.glob("*.c3d"))
            bvh_files = sorted(child.glob("*.bvh"))
            fbx_files = sorted(child.glob("*.fbx"))
            if not (c3d_files or bvh_files or fbx_files):
                continue
            trials[child.name] = TrialFiles(
                participant=participant,
                name=child.name,
                system=system_name,
                root=child,
                c3d=choose_preferred_file(c3d_files, ".c3d"),
                bvh=choose_preferred_file(bvh_files, ".bvh"),
                fbx=choose_preferred_file(fbx_files, ".fbx"),
            )

    grouped_files: dict[str, dict[str, Path]] = {}
    for path in sorted(system_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".c3d", ".bvh", ".fbx"}:
            continue
        grouped_files.setdefault(path.stem, {})[path.suffix.lower().lstrip(".")] = path
    for name, files in grouped_files.items():
        existing = trials.get(name)
        if existing is None:
            trials[name] = TrialFiles(
                participant=participant,
                name=name,
                system=system_name,
                root=system_dir,
                c3d=files.get("c3d"),
                bvh=files.get("bvh"),
                fbx=files.get("fbx"),
            )
        else:
            existing.c3d = existing.c3d or files.get("c3d")
            existing.bvh = existing.bvh or files.get("bvh")
            existing.fbx = existing.fbx or files.get("fbx")

    return trials


def trial_name_aliases(name: str) -> list[str]:
    aliases = [name]
    if "_Func_" not in name and "Marche" not in name and "_Calib_" not in name:
        parts = name.split("_", 1)
        if len(parts) == 2:
            aliases.append(f"{parts[0]}_Calib_Func_{parts[1]}")
    if "_Func_" in name and "_Calib_" not in name:
        aliases.append(name.replace("_Func_", "_Calib_Func_"))
    if "_Calib_Func_" in name:
        aliases.append(name.replace("_Calib_Func_", "_Func_"))
        parts = name.split("_Calib_Func_", 1)
        if len(parts) == 2:
            aliases.append(f"{parts[0]}_{parts[1]}")
    return list(dict.fromkeys(aliases))


def infer_participant_label(trial_name: str, fallback: str) -> str:
    match = re.match(r"^([A-Za-z]*\d+)(?:[_-]|$)", trial_name)
    if match:
        return match.group(1)
    return fallback


def discover_participant_roots(data_root: Path, reference_system: str, test_system: str) -> list[tuple[str, Path, bool]]:
    if (data_root / reference_system).is_dir() and (data_root / test_system).is_dir():
        return [("auto", data_root, True)]

    roots: list[tuple[str, Path, bool]] = []
    if not data_root.exists():
        return roots
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / reference_system).is_dir() and (child / test_system).is_dir():
            roots.append((child.name, child, False))
    return roots


def matches_any_filter(value: str, filters: list[re.Pattern[str]]) -> bool:
    return not filters or any(regex.search(value) for regex in filters)


def aggregate_population_metrics(
    rows: list[dict[str, Any]],
    group_specs: list[tuple[str, list[str]]],
    metric_names: list[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for scope, group_cols in group_specs:
        available_group_cols = [col for col in group_cols if col in df.columns]
        if not available_group_cols:
            continue
        for keys, group in df.groupby(available_group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            summary: dict[str, Any] = {"summary_scope": scope}
            summary.update(dict(zip(available_group_cols, keys, strict=False)))
            summary["n_rows"] = int(len(group))
            if "participant" in group.columns:
                summary["n_participants"] = int(group["participant"].nunique(dropna=True))
            if "trial" in group.columns:
                summary["n_trials"] = int(group["trial"].nunique(dropna=True))
            for metric in metric_names:
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                summary[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                summary[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
                summary[f"{metric}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
            summaries.append(summary)
    return pd.DataFrame(summaries)


def landmark_population_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return aggregate_population_metrics(
        rows,
        group_specs=[
            ("population_by_landmark", ["variant", "landmark"]),
            ("population_by_trial_landmark", ["trial", "variant", "landmark"]),
            ("participant_by_landmark", ["participant", "variant", "landmark"]),
        ],
        metric_names=[
            "median_error_mm",
            "p95_error_mm",
            "max_error_mm",
            "mae_x",
            "mae_y",
            "mae_z",
            "rmse_x",
            "rmse_y",
            "rmse_z",
            "mae_euclidean",
            "rmse_euclidean",
        ],
    )


def angle_population_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return aggregate_population_metrics(
        rows,
        group_specs=[
            ("population_by_angle", ["angle", "component"]),
            ("population_by_trial_angle", ["trial", "angle", "component"]),
            ("participant_by_angle", ["participant", "angle", "component"]),
        ],
        metric_names=["mae", "rmse", "bias", "pearson_r", "ccc", "nrmse", "mape_range"],
    )


def compare_pair(
    pair: TrialPair,
    out_dir: Path,
    landmark_map: list[dict[str, Any]],
    n_points: int,
    alignment: str,
    angle_label_regex: str,
) -> dict[str, Any]:
    trial_dir = out_dir / safe_name(pair.participant) / safe_name(pair.name)
    trial_dir.mkdir(parents=True, exist_ok=True)
    if pair.reference.c3d is None or pair.test.c3d is None:
        raise ValueError(f"Trial {pair.name} requires one C3D per system for C3D-based comparison.")
    reference = read_c3d(pair.reference.c3d, angle_label_regex=angle_label_regex)
    test = read_c3d(pair.test.c3d, angle_label_regex=angle_label_regex)
    names, ref_landmarks, test_landmarks, landmark_report = extract_landmarks(reference, test, landmark_map, n_points)
    aligned_landmarks, alignment_report = align_landmarks(ref_landmarks, test_landmarks, alignment)

    landmark_metrics = landmark_metrics_rows(
        pair.participant, pair.name, names, ref_landmarks, test_landmarks, aligned_landmarks
    )
    landmark_timeseries = landmark_timeseries_rows(
        pair.participant, pair.name, names, ref_landmarks, test_landmarks, aligned_landmarks
    )
    angle_metrics, angle_timeseries = angle_metrics_and_timeseries(pair.participant, pair.name, reference, test, n_points)

    pd.DataFrame(landmark_metrics).to_csv(trial_dir / "landmark_metrics.csv", index=False)
    pd.DataFrame(landmark_timeseries).to_csv(trial_dir / "landmark_timeseries.csv", index=False)
    pd.DataFrame(angle_metrics).to_csv(trial_dir / "angle_metrics.csv", index=False)
    pd.DataFrame(angle_timeseries).to_csv(trial_dir / "angle_timeseries.csv", index=False)

    inventory = {"reference": c3d_inventory(reference), "test": c3d_inventory(test)}
    file_inventory = {
        "reference": trial_files_dict(pair.reference),
        "test": trial_files_dict(pair.test),
    }
    report = {
        "participant": pair.participant,
        "trial": pair.name,
        "reference": trial_files_dict(pair.reference),
        "test": trial_files_dict(pair.test),
        "resample_points": n_points,
        "alignment": alignment_report,
        "landmarks": landmark_report,
        "n_available_landmarks": len(names),
        "n_landmark_metric_rows": len(landmark_metrics),
        "n_shared_angle_channels": len(set(reference.angle_indices).intersection(test.angle_indices)),
        "inventory": inventory,
        "file_inventory": file_inventory,
    }
    with (trial_dir / "c3d_inventory.json").open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    with (trial_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return {
        "report": report,
        "landmark_metrics": landmark_metrics,
        "angle_metrics": angle_metrics,
        "model_inventory": model_file_inventory(pair.reference) + model_file_inventory(pair.test),
    }


def discover_trial_pairs(
    data_root: Path,
    reference_system: str,
    test_system: str,
    participant_filters: list[str] | None = None,
) -> list[TrialPair]:
    participant_regexes = [re.compile(pattern) for pattern in participant_filters or []]
    pairs: list[TrialPair] = []
    for participant_label, participant_root, infer_from_trial in discover_participant_roots(
        data_root, reference_system, test_system
    ):
        reference_trials = discover_trial_files(participant_root / reference_system, reference_system, participant_label)
        test_trials = discover_trial_files(participant_root / test_system, test_system, participant_label)
        for test_name, test_trial in sorted(test_trials.items()):
            if test_trial.c3d is None:
                continue
            participant = (
                infer_participant_label(test_name, data_root.name or "single") if infer_from_trial else participant_label
            )
            if not matches_any_filter(participant, participant_regexes):
                continue
            reference_trial = next(
                (reference_trials[alias] for alias in trial_name_aliases(test_name) if alias in reference_trials),
                None,
            )
            if reference_trial is not None and reference_trial.c3d is not None:
                pairs.append(
                    TrialPair(
                        participant=participant,
                        name=test_name,
                        reference=replace(reference_trial, participant=participant),
                        test=replace(test_trial, participant=participant),
                    )
                )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Motive marker-based C3D data with Captury markerless C3D data.")
    parser.add_argument("--data-root", type=Path, default=Path("local_trials/data"), help="Root containing Motive/ and Captury/.")
    parser.add_argument("--reference-system", default="Motive", help="Reference system directory name inside --data-root.")
    parser.add_argument("--test-system", default="Captury", help="Test system directory name inside --data-root.")
    parser.add_argument("--reference-c3d", type=Path, default=None, help="Reference C3D, typically Motive.")
    parser.add_argument("--reference-bvh", type=Path, default=None, help="Optional reference BVH for inventory/model workflows.")
    parser.add_argument("--reference-fbx", type=Path, default=None, help="Optional reference FBX for inventory/model workflows.")
    parser.add_argument("--test-c3d", type=Path, default=None, help="Test C3D, typically Captury.")
    parser.add_argument("--test-bvh", type=Path, default=None, help="Optional test BVH for inventory/model workflows.")
    parser.add_argument("--test-fbx", type=Path, default=None, help="Optional test FBX for inventory/model workflows.")
    parser.add_argument("--trial-name", default=None, help="Trial name for single-pair mode.")
    parser.add_argument("--trial-filter", action="append", default=[], help="Regex filter for discovered trial names.")
    parser.add_argument(
        "--participant-filter",
        action="append",
        default=[],
        help="Regex filter for participant directory names or inferred participant labels.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out_capture_system_comparison"), help="Output directory.")
    parser.add_argument("--landmark-map", type=Path, default=None, help="Optional JSON landmark map.")
    parser.add_argument("--resample-points", type=int, default=101, help="Time-normalized points per trial.")
    parser.add_argument(
        "--alignment",
        choices=["none", "global_rigid", "per_frame_rigid"],
        default="global_rigid",
        help="How to align test landmarks to reference landmarks before aligned metrics.",
    )
    parser.add_argument(
        "--angle-label-regex",
        default=r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)",
        help="Regex used to identify C3D angle point labels.",
    )
    return parser.parse_args()


def selected_pairs(args: argparse.Namespace) -> list[TrialPair]:
    explicit_files = [
        args.reference_c3d,
        args.reference_bvh,
        args.reference_fbx,
        args.test_c3d,
        args.test_bvh,
        args.test_fbx,
    ]
    if any(path is not None for path in explicit_files):
        if args.reference_c3d is None or args.test_c3d is None:
            raise ValueError("--reference-c3d and --test-c3d are required in explicit pair mode.")
        reference_name = args.trial_name or args.reference_c3d.stem
        test_name = args.trial_name or args.test_c3d.parent.name or args.test_c3d.stem
        return [
            TrialPair(
                name=args.trial_name or test_name,
                participant=infer_participant_label(args.trial_name or test_name, "single"),
                reference=TrialFiles(
                    participant=infer_participant_label(args.trial_name or test_name, "single"),
                    name=reference_name,
                    system=args.reference_system,
                    root=args.reference_c3d.parent,
                    c3d=args.reference_c3d,
                    bvh=args.reference_bvh,
                    fbx=args.reference_fbx,
                ),
                test=TrialFiles(
                    participant=infer_participant_label(args.trial_name or test_name, "single"),
                    name=test_name,
                    system=args.test_system,
                    root=args.test_c3d.parent,
                    c3d=args.test_c3d,
                    bvh=args.test_bvh,
                    fbx=args.test_fbx,
                ),
            )
        ]
    pairs = discover_trial_pairs(args.data_root, args.reference_system, args.test_system, args.participant_filter)
    if args.trial_filter:
        regexes = [re.compile(pattern) for pattern in args.trial_filter]
        pairs = [pair for pair in pairs if any(regex.search(pair.name) for regex in regexes)]
    return pairs


def main() -> None:
    args = parse_args()
    pairs = selected_pairs(args)
    if not pairs:
        raise RuntimeError("No C3D pairs found. Provide --reference-c3d/--test-c3d or check --data-root.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    landmark_map = load_landmark_map(args.landmark_map)

    all_landmark_metrics: list[dict[str, Any]] = []
    all_angle_metrics: list[dict[str, Any]] = []
    all_model_inventory: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for pair in pairs:
        result = compare_pair(
            pair=pair,
            out_dir=args.out_dir,
            landmark_map=landmark_map,
            n_points=args.resample_points,
            alignment=args.alignment,
            angle_label_regex=args.angle_label_regex,
        )
        reports.append(result["report"])
        all_landmark_metrics.extend(result["landmark_metrics"])
        all_angle_metrics.extend(result["angle_metrics"])
        all_model_inventory.extend(result["model_inventory"])

    pd.DataFrame(all_landmark_metrics).to_csv(args.out_dir / "all_landmark_metrics.csv", index=False)
    pd.DataFrame(all_angle_metrics).to_csv(args.out_dir / "all_angle_metrics.csv", index=False)
    pd.DataFrame(all_model_inventory).to_csv(args.out_dir / "all_model_inventory.csv", index=False)
    landmark_population_summary(all_landmark_metrics).to_csv(args.out_dir / "population_landmark_summary.csv", index=False)
    angle_population_summary(all_angle_metrics).to_csv(args.out_dir / "population_angle_summary.csv", index=False)
    report = {
        "n_pairs": len(pairs),
        "n_participants": len({pair.participant for pair in pairs}),
        "reference_system": args.reference_system,
        "test_system": args.test_system,
        "pairs": [
            {
                "participant": pair.participant,
                "name": pair.name,
                "reference": trial_files_dict(pair.reference),
                "test": trial_files_dict(pair.test),
            }
            for pair in pairs
        ],
        "out_dir": str(args.out_dir),
        "alignment": args.alignment,
        "resample_points": args.resample_points,
        "landmark_map": landmark_map,
        "trial_reports": reports,
    }
    with (args.out_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Compared {len(pairs)} pair(s).")
    print(f"Landmark metrics: {args.out_dir / 'all_landmark_metrics.csv'}")
    print(f"Angle metrics: {args.out_dir / 'all_angle_metrics.csv'}")
    print(f"Model inventory: {args.out_dir / 'all_model_inventory.csv'}")
    print(f"Population landmark summary: {args.out_dir / 'population_landmark_summary.csv'}")
    print(f"Population angle summary: {args.out_dir / 'population_angle_summary.csv'}")
    print(f"Report: {args.out_dir / 'run_report.json'}")


if __name__ == "__main__":
    main()
