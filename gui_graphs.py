"""Graph configuration and drawing helpers for the Captury/BioBuddy GUI."""

from __future__ import annotations

from typing import Iterable
from pathlib import Path

import numpy as np

from gui_trial_viewer import COR_LAYER_LABELS, data_source_color

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional plotting dependency
    pd = None

GRAPH_CONFIGS = {
    "occlusions": {
        "csv": "all_motive_marker_occlusions.csv",
        "groups": ("marker",),
        "metrics": ("missing_percent", "missing_frames"),
        "title": "Occlusions Motive",
    },
    "dimensions": {
        "csv": "all_model_dimensions.csv",
        "groups": ("system", "dimension"),
        "metrics": ("median_length_mm", "sd_length_mm"),
        "title": "Dimensions modèles",
    },
    "centres": {
        "csv": "all_joint_centre_metrics.csv",
        "groups": ("joint",),
        "metrics": (
            "median_error_mm",
            "p95_error_mm",
            "max_error_mm",
            "mae_x",
            "mae_y",
            "mae_z",
            "mae_euclidean",
            "rmse_euclidean",
        ),
        "title": "Centres articulaires",
    },
    "skin_markers": {
        "csv": "all_skin_marker_correspondence_metrics.csv",
        "groups": ("landmark",),
        "metrics": ("median_error_mm", "p95_error_mm", "rmse_error_mm"),
        "title": "Marqueurs cutanés",
    },
    "kinematics": {
        "csv": "all_kinematics_q_metrics.csv",
        "groups": ("q_name",),
        "metrics": (
            "bias_rad",
            "mae_rad",
            "rmse_rad",
            "nrmse_range",
            "pearson_r_waveform",
            "lin_ccc_waveform",
            "bias_native",
            "mae_native",
            "rmse_native",
            "c3d_mean_deg",
            "c3d_sd_deg",
            "c3d_min_deg",
            "c3d_max_deg",
        ),
        "title": "Cinématiques",
    },
}


def read_table_npz(path: str | Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required to read GUI table npz files.")
    with np.load(Path(path), allow_pickle=False) as data:
        columns = [str(column) for column in data["columns"]]
        values = {column: data[f"col_{index}"] for index, column in enumerate(columns)}
    return pd.DataFrame(values)


EVENT_METRICS = (
    "movement_speed_mm_s",
    "left_foot_z_mm",
    "right_foot_z_mm",
    "left_foot_speed_mm_s",
    "right_foot_speed_mm_s",
    "left_contact",
    "right_contact",
)
KINEMATIC_RAD_METRICS = {"bias_rad", "mae_rad", "rmse_rad"}
KINEMATIC_TIMESERIES_COLUMNS = ("motive", "captury", "captury_c3d", "difference")


def graph_metric_columns(
    dataframe: "pd.DataFrame", requested_metrics: Iterable[str]
) -> list[str]:
    if pd is None:
        return []
    return [
        metric
        for metric in requested_metrics
        if metric in dataframe.columns
        and pd.api.types.is_numeric_dtype(dataframe[metric])
    ]


def is_rotation_q_name(q_name: str) -> bool:
    return "_rot" in str(q_name).lower()


def metric_display_name(metric: str, *, q_name: str | None = None) -> str:
    if metric in KINEMATIC_RAD_METRICS:
        return metric.replace("_rad", "_deg")
    if metric in KINEMATIC_TIMESERIES_COLUMNS and q_name is not None:
        return f"{metric} ({'deg' if is_rotation_q_name(q_name) else 'native'})"
    return metric


def values_for_display(
    values: "pd.Series | np.ndarray", metric: str, *, q_name: str | None = None
) -> "pd.Series | np.ndarray":
    if metric in KINEMATIC_RAD_METRICS or (
        metric in KINEMATIC_TIMESERIES_COLUMNS
        and q_name is not None
        and is_rotation_q_name(q_name)
    ):
        return values * (180.0 / np.pi)
    return values


def joint_centre_error_timeseries(
    dataframe: "pd.DataFrame", joint: str
) -> "pd.DataFrame":
    required = {
        "time",
        "joint",
        "captury_x_mm",
        "captury_y_mm",
        "captury_z_mm",
        "motive_x_mm",
        "motive_y_mm",
        "motive_z_mm",
    }
    if not required.issubset(dataframe.columns):
        return pd.DataFrame() if pd is not None else dataframe
    values = dataframe[dataframe["joint"].astype(str) == str(joint)].copy()
    if values.empty:
        return values
    for axis in ("x", "y", "z"):
        values[f"error_{axis}_mm"] = values[f"captury_{axis}_mm"].astype(
            float
        ) - values[f"motive_{axis}_mm"].astype(float)
        values[f"abs_error_{axis}_mm"] = values[f"error_{axis}_mm"].abs()
    if "distance_mm" not in values.columns:
        values["distance_mm"] = np.linalg.norm(
            values[["error_x_mm", "error_y_mm", "error_z_mm"]].to_numpy(dtype=float),
            axis=1,
        )
    values["time"] = values["time"].astype(float)
    return values.sort_values("time")


def draw_joint_centre_error_timeseries(
    axes: object, dataframe: "pd.DataFrame", trial: str, joint: str
) -> bool:
    values = joint_centre_error_timeseries(dataframe, joint)
    if values.empty:
        axes.set_title(f"Aucune erreur temporelle: {joint}")
        return False
    curves = (
        ("distance_mm", "distance", "#111827"),
        ("abs_error_x_mm", "|x|", "#ef4444"),
        ("abs_error_y_mm", "|y|", "#22c55e"),
        ("abs_error_z_mm", "|z|", "#3b82f6"),
    )
    for column, label, color in curves:
        if column in values.columns:
            axes.plot(
                values["time"], values[column].astype(float), label=label, color=color
            )
    axes.set_title(f"{trial} - {joint} - erreur centres")
    axes.set_xlabel("Temps (s)")
    axes.set_ylabel("Erreur (mm)")
    axes.legend()
    axes.grid(alpha=0.3)
    return True


def joint_centre_error_boxplot_series(
    dataframe: "pd.DataFrame",
    metric: str,
    *,
    trial: str | None = None,
    joints: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    if pd is None or dataframe.empty:
        return []
    values = dataframe
    if trial and "trial" in values.columns:
        values = values[values["trial"].astype(str) == str(trial)]
    if values.empty or "joint" not in values.columns:
        return []
    requested_joints = {str(joint) for joint in joints or () if str(joint)}
    metric_column = {
        "mae_x": "abs_error_x_mm",
        "mae_y": "abs_error_y_mm",
        "mae_z": "abs_error_z_mm",
    }.get(metric, "distance_mm")
    series: list[dict[str, object]] = []
    for joint in sorted(values["joint"].astype(str).unique()):
        if requested_joints and joint not in requested_joints:
            continue
        joint_values = joint_centre_error_timeseries(values, joint)
        if joint_values.empty or metric_column not in joint_values.columns:
            continue
        joint_errors = (
            joint_values[metric_column].astype(float).replace([np.inf, -np.inf], np.nan)
        )
        joint_errors = joint_errors.dropna().to_numpy()
        if joint_errors.size == 0:
            continue
        series.append(
            {
                "metric": metric,
                "label": joint,
                "dataframe": joint_values,
                "values": joint_errors,
            }
        )
    return series


def draw_metric_boxplot(
    axes: object, series: list[dict[str, object]], title: str
) -> None:
    labels = [str(item["label"]) for item in series]
    values = [item["values"] for item in series]
    positions = np.arange(1, len(values) + 1, dtype=float)
    boxes = axes.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#111827",
            "markeredgecolor": "#111827",
            "markersize": 4,
        },
        medianprops={"color": "#111827", "linewidth": 1.6},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#64748b",
            "markersize": 3,
            "alpha": 0.7,
        },
    )
    for index, patch in enumerate(boxes["boxes"]):
        metric = str(series[index]["metric"])
        color = data_source_color(metric)
        if color == "#64748b":
            color = f"C{index % 10}"
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for index, item in enumerate(series):
        item_values = np.asarray(item["values"], dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=item_values.shape[0])
        axes.scatter(
            np.full(item_values.shape[0], positions[index]) + jitter,
            item_values,
            s=16,
            alpha=0.65,
            color=boxes["boxes"][index].get_edgecolor(),
            edgecolors="none",
        )
    axes.set_title(title)
    axes.set_ylabel("valeur")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.grid(axis="y", alpha=0.3)


def draw_dimension_metric_graph(
    axes: object, dataframe: "pd.DataFrame", metric: str
) -> None:
    if "dimension" not in dataframe.columns or "system" not in dataframe.columns:
        axes.set_title("Aucune donnée dimensionnelle")
        return
    values = dataframe.copy()
    values[metric] = values[metric].astype(float)
    dimensions = sorted(values["dimension"].dropna().astype(str).unique())
    systems = [
        system
        for system in ("captury", "motive", "biobuddy")
        if system in set(values["system"].dropna().astype(str).str.lower())
    ]
    for system in sorted(values["system"].dropna().astype(str).str.lower().unique()):
        if system not in systems:
            systems.append(system)
    if not dimensions or not systems:
        axes.set_title("Aucune donnée dimensionnelle")
        return
    x = np.arange(len(dimensions), dtype=float)
    width = min(0.8 / max(1, len(systems)), 0.28)
    offsets = (np.arange(len(systems)) - (len(systems) - 1) / 2.0) * width
    for index, system in enumerate(systems):
        system_rows = values[values["system"].astype(str).str.lower() == system]
        grouped = system_rows.groupby("dimension", sort=False)[metric].median()
        y = [float(grouped.get(dimension, np.nan)) for dimension in dimensions]
        axes.bar(
            x + offsets[index],
            y,
            width=width,
            label=COR_LAYER_LABELS.get(system, system),
            color=data_source_color(system),
        )
    axes.set_title(f"Dimensions modèles - {metric}")
    axes.set_ylabel(metric)
    axes.set_xticks(x)
    axes.set_xticklabels(dimensions, rotation=45, ha="right")
    axes.legend()
    axes.grid(axis="y", alpha=0.3)
