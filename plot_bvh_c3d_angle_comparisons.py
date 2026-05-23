#!/usr/bin/env python3
"""
Generate figures comparing BVH generalized coordinates to C3D angle channels.

Expected inputs (from the BVH/C3D comparison pipeline):
- results_dir/bvh_q_biorbd_order.npz
- results_dir/q_bvh_vs_c3d_angles_explicit_mapping.csv (preferred) OR
  results_dir/q_bvh_vs_c3d_angles_best_matches.csv
- original C3D file

Outputs:
- figures_angle_comparison/summary_rmse_deg.png
- figures_angle_comparison/summary_corr.png
- figures_angle_comparison/<angle_name>.png  (time series + error + scatter)
- figures_angle_comparison/metrics_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AXIS_TO_INDEX = {"X": 0, "Y": 1, "Z": 2, "x": 0, "y": 1, "z": 2}


def require_ezc3d():
    try:
        import ezc3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ezc3d is required to read the C3D for plotting. Install it in the same environment."
        ) from exc
    return ezc3d


def get_c3d_param(c3d: dict, group: str, name: str, default=None):
    try:
        return c3d["parameters"][group][name]["value"]
    except KeyError:
        return default


def as_str_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [str(value).strip()]
    return [str(v).strip() for v in value]


def c3d_time_vector(c3d: dict) -> np.ndarray:
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [0])
    rate = float(rate_value[0] if isinstance(rate_value, (list, tuple, np.ndarray)) else rate_value)
    if rate <= 0:
        raise ValueError("Invalid or missing C3D POINT:RATE.")
    n_frames = int(c3d["data"]["points"].shape[2])
    return np.arange(n_frames, dtype=float) / rate


def interpolate_array(data: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    if source_time.shape == target_time.shape and np.allclose(source_time, target_time):
        return data.copy()
    flat = data.reshape((-1, data.shape[-1]))
    out = np.empty((flat.shape[0], target_time.shape[0]), dtype=float)
    for i, y in enumerate(flat):
        finite = np.isfinite(y)
        if finite.sum() < 2:
            out[i, :] = np.nan
        else:
            out[i, :] = np.interp(target_time, source_time[finite], y[finite])
    return out.reshape((*data.shape[:-1], target_time.shape[0]))


def get_angle_label_set_from_c3d_parameters(c3d: dict) -> set[str]:
    candidates: set[str] = set()
    for param_name in ("ANGLES", "ANGLE_LABELS"):
        for label in as_str_list(get_c3d_param(c3d, "POINT", param_name, [])):
            if label:
                candidates.add(label)
                candidates.add(label.replace(" ", ""))
    return candidates


def split_c3d_points(c3d_path: Path, angle_label_regex: str = r"Angles?$", extra_angle_labels: list[str] | None = None):
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(c3d_path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    descriptions = as_str_list(get_c3d_param(c3d, "POINT", "DESCRIPTIONS", []))
    if len(descriptions) < len(labels):
        descriptions += [""] * (len(labels) - len(descriptions))

    points = np.asarray(c3d["data"]["points"], dtype=float)[:3, :, :]
    time = c3d_time_vector(c3d)
    regex = re.compile(angle_label_regex) if angle_label_regex else None
    c3d_angle_param_labels = get_angle_label_set_from_c3d_parameters(c3d)
    extra_angle_label_set = {label.strip() for label in (extra_angle_labels or [])}

    def is_angle_point(i: int) -> bool:
        label = labels[i]
        compact_label = label.replace(" ", "")
        description = descriptions[i]
        if label in c3d_angle_param_labels or compact_label in c3d_angle_param_labels:
            return True
        if label in extra_angle_label_set:
            return True
        if regex is not None and (regex.search(label) or regex.search(description)):
            return True
        return False

    angle_indices = [i for i in range(len(labels)) if is_angle_point(i)]
    angle_data = points[:, angle_indices, :]
    angle_labels = [labels[i] for i in angle_indices]
    angle_unit = as_str_list(get_c3d_param(c3d, "POINT", "ANGLE_UNITS", ["deg"]))
    angle_unit = angle_unit[0] if angle_unit else "deg"
    return time, angle_labels, angle_data, angle_unit


def c3d_angles_to_deg(angle_data: np.ndarray, unit: str) -> np.ndarray:
    unit = str(unit).lower()
    if unit in {"deg", "degree", "degrees"}:
        return angle_data.copy()
    if unit in {"rad", "radian", "radians"}:
        return np.rad2deg(angle_data)
    raise ValueError(f"Unsupported C3D angle unit: {unit}")


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    aa = a[finite] - np.nanmean(a[finite])
    bb = b[finite] - np.nanmean(b[finite])
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def compute_metrics(bvh_deg: np.ndarray, c3d_deg: np.ndarray) -> dict:
    finite = np.isfinite(bvh_deg) & np.isfinite(c3d_deg)
    if finite.sum() == 0:
        return {"n": 0, "bias_deg": np.nan, "rmse_deg": np.nan, "mae_deg": np.nan, "corr": np.nan}
    diff = bvh_deg[finite] - c3d_deg[finite]
    return {
        "n": int(finite.sum()),
        "bias_deg": float(np.mean(diff)),
        "rmse_deg": float(np.sqrt(np.mean(diff**2))),
        "mae_deg": float(np.mean(np.abs(diff))),
        "corr": pearson_corr(bvh_deg, c3d_deg),
    }


def sanitize_filename(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return clean.strip("_") or "comparison"


def pick_comparison_csv(results_dir: Path, explicit_csv: Path | None = None) -> Path:
    if explicit_csv is not None:
        return explicit_csv
    candidates = [
        results_dir / "q_bvh_vs_c3d_angles_explicit_mapping.csv",
        results_dir / "q_bvh_vs_c3d_angles_best_matches.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "No comparison CSV found. Expected q_bvh_vs_c3d_angles_explicit_mapping.csv or q_bvh_vs_c3d_angles_best_matches.csv."
    )


def load_q_npz(results_dir: Path):
    npz_path = results_dir / "bvh_q_biorbd_order.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    q = np.asarray(data["q"], dtype=float)
    q_names = [str(x) for x in data["q_names"].tolist()]
    time = np.asarray(data["time"], dtype=float)
    return q, q_names, time


def resolve_rows(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"bvh_q_name", "c3d_angle_label", "c3d_component"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Comparison CSV is missing required columns: {sorted(missing)}")
    if "name" not in df.columns:
        df = df.copy()
        df["name"] = [
            f"{row['bvh_q_name']}__{row['c3d_angle_label']}_{row['c3d_component']}" for _, row in df.iterrows()
        ]
    return df


def make_summary_figures(metrics_df: pd.DataFrame, out_dir: Path):
    metrics_df = metrics_df.sort_values("rmse_deg", ascending=True)
    labels = metrics_df["name"].tolist()

    plt.figure(figsize=(max(8, 0.45 * len(labels)), 6))
    plt.bar(range(len(labels)), metrics_df["rmse_deg"].to_numpy(dtype=float))
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("RMSE (deg)")
    plt.title("BVH vs C3D angle comparison: RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_rmse_deg.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, 0.45 * len(labels)), 6))
    plt.bar(range(len(labels)), metrics_df["corr"].to_numpy(dtype=float))
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("Correlation")
    plt.title("BVH vs C3D angle comparison: Pearson correlation")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_corr.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, 0.45 * len(labels)), 6))
    plt.bar(range(len(labels)), metrics_df["bias_deg"].to_numpy(dtype=float))
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("Bias (deg)")
    plt.title("BVH vs C3D angle comparison: mean signed error")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bias_deg.png", dpi=200)
    plt.close()


def make_single_comparison_figure(name: str, t: np.ndarray, bvh_deg: np.ndarray, c3d_deg: np.ndarray, out_path: Path, metrics: dict):
    diff = bvh_deg - c3d_deg
    finite_scatter = np.isfinite(bvh_deg) & np.isfinite(c3d_deg)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    axes[0].plot(t, bvh_deg, label="BVH q")
    axes[0].plot(t, c3d_deg, label="C3D angle")
    axes[0].set_title(name)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, diff)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_ylabel("BVH - C3D (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    text = (
        f"n={metrics['n']}\n"
        f"bias={metrics['bias_deg']:.2f} deg\n"
        f"RMSE={metrics['rmse_deg']:.2f} deg\n"
        f"MAE={metrics['mae_deg']:.2f} deg\n"
        f"corr={metrics['corr']:.3f}"
    )
    axes[1].text(1.01, 0.98, text, transform=axes[1].transAxes, va="top")

    if finite_scatter.sum() > 0:
        axes[2].scatter(c3d_deg[finite_scatter], bvh_deg[finite_scatter], s=8)
        lo = float(np.nanmin([np.nanmin(c3d_deg[finite_scatter]), np.nanmin(bvh_deg[finite_scatter])]))
        hi = float(np.nanmax([np.nanmax(c3d_deg[finite_scatter]), np.nanmax(bvh_deg[finite_scatter])]))
        if math.isfinite(lo) and math.isfinite(hi):
            axes[2].plot([lo, hi], [lo, hi], linewidth=1)
            axes[2].set_xlim(lo, hi)
            axes[2].set_ylim(lo, hi)
    axes[2].set_xlabel("C3D angle (deg)")
    axes[2].set_ylabel("BVH q (deg)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate figures comparing BVH q to C3D angles.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Output folder from the BVH/C3D comparison pipeline.")
    parser.add_argument("--c3d", type=Path, required=True, help="Original C3D file.")
    parser.add_argument("--comparison-csv", type=Path, default=None, help="Optional CSV to use instead of the default explicit/best mapping CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where figures will be saved.")
    parser.add_argument("--angle-label-regex", type=str, default=r"Angles?$", help="Regex used to help detect angle point channels in the C3D.")
    parser.add_argument("--extra-angle-label", action="append", default=[], help="Extra point labels to force as angle channels.")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or (results_dir / "figures_angle_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    q, q_names, bvh_time = load_q_npz(results_dir)
    comparison_csv = pick_comparison_csv(results_dir, args.comparison_csv)
    comparison_df = resolve_rows(pd.read_csv(comparison_csv))

    c3d_time, c3d_angle_labels, c3d_angle_data, c3d_angle_unit = split_c3d_points(
        args.c3d, angle_label_regex=args.angle_label_regex, extra_angle_labels=args.extra_angle_label
    )
    c3d_angle_data_deg = c3d_angles_to_deg(c3d_angle_data, c3d_angle_unit)
    c3d_angles_on_bvh_time = interpolate_array(c3d_angle_data_deg, c3d_time, bvh_time)

    metrics_rows = []
    for _, row in comparison_df.iterrows():
        name = str(row["name"])
        bvh_q_name = str(row["bvh_q_name"])
        c3d_angle_label = str(row["c3d_angle_label"])
        c3d_component = str(row["c3d_component"])

        if bvh_q_name not in q_names:
            print(f"[WARNING] Skipping {name}: BVH q not found: {bvh_q_name}")
            continue
        if c3d_angle_label not in c3d_angle_labels:
            print(f"[WARNING] Skipping {name}: C3D angle label not found: {c3d_angle_label}")
            continue
        if c3d_component not in AXIS_TO_INDEX:
            print(f"[WARNING] Skipping {name}: invalid C3D component: {c3d_component}")
            continue

        q_idx = q_names.index(bvh_q_name)
        angle_idx = c3d_angle_labels.index(c3d_angle_label)
        comp_idx = AXIS_TO_INDEX[c3d_component]

        bvh_deg = np.rad2deg(q[q_idx, :])
        c3d_deg = c3d_angles_on_bvh_time[comp_idx, angle_idx, :]
        metrics = compute_metrics(bvh_deg, c3d_deg)
        metrics_rows.append(
            {
                "name": name,
                "bvh_q_name": bvh_q_name,
                "c3d_angle_label": c3d_angle_label,
                "c3d_component": c3d_component,
                **metrics,
            }
        )
        out_name = sanitize_filename(name) + ".png"
        make_single_comparison_figure(name, bvh_time, bvh_deg, c3d_deg, output_dir / out_name, metrics)

    if not metrics_rows:
        raise RuntimeError("No figures were generated. Check the mapping CSV and detected angle labels.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    make_summary_figures(metrics_df, output_dir)

    print(f"Figures written to: {output_dir}")
    print(f"Metrics summary: {output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
