"""Agreement metrics for marker-based versus markerless model comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

try:
    from scipy.stats import pearsonr
except ImportError:  # pragma: no cover - scipy is available in the project env
    pearsonr = None


def _as_array(x):
    return np.asarray(x, dtype=float)


def _paired_clean_1d(ref, test):
    ref = _as_array(ref).ravel()
    test = _as_array(test).ravel()
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {test.shape}")
    mask = np.isfinite(ref) & np.isfinite(test)
    return ref[mask], test[mask]


def waveform_amplitude_threshold(unit: str) -> float:
    """Return the minimum range required for normalized/shape metrics."""

    normalized = str(unit).strip().lower()
    if normalized in {"deg", "degree", "degrees"}:
        return 1.0
    if normalized in {"rad", "radian", "radians"}:
        return float(np.deg2rad(1.0))
    return np.nan


def resample_1d(signal, n_points: int = 101):
    signal = _as_array(signal)
    x_old = np.linspace(0, 1, signal.shape[-1])
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, signal)


def mae(ref, test):
    ref, test = _paired_clean_1d(ref, test)
    return float(np.mean(np.abs(test - ref))) if ref.size else np.nan


def rmse(ref, test):
    ref, test = _paired_clean_1d(ref, test)
    return float(np.sqrt(np.mean((test - ref) ** 2))) if ref.size else np.nan


def bias(ref, test):
    ref, test = _paired_clean_1d(ref, test)
    return float(np.mean(test - ref)) if ref.size else np.nan


def pearson_r(ref, test):
    ref, test = _paired_clean_1d(ref, test)
    if len(ref) < 2:
        return np.nan
    if np.nanstd(ref) == 0 or np.nanstd(test) == 0:
        return np.nan
    if pearsonr is not None:
        return float(pearsonr(ref, test)[0])
    return float(np.corrcoef(ref, test)[0, 1])


def lin_ccc(ref, test):
    ref, test = _paired_clean_1d(ref, test)
    if len(ref) < 2:
        return np.nan
    mean_ref = np.mean(ref)
    mean_test = np.mean(test)
    var_ref = np.var(ref, ddof=1)
    var_test = np.var(test, ddof=1)
    cov = np.cov(ref, test, ddof=1)[0, 1]
    denom = var_ref + var_test + (mean_ref - mean_test) ** 2
    return float((2 * cov) / denom) if denom != 0 else np.nan


def bland_altman(ref_values, test_values):
    ref, test = _paired_clean_1d(ref_values, test_values)
    if len(ref) < 2:
        return {
            "bias": np.nan,
            "loa_lower": np.nan,
            "loa_upper": np.nan,
            "sd_diff": np.nan,
        }
    diff = test - ref
    b = np.mean(diff)
    sd = np.std(diff, ddof=1)
    return {
        "bias": float(b),
        "loa_lower": float(b - 1.96 * sd),
        "loa_upper": float(b + 1.96 * sd),
        "sd_diff": float(sd),
    }


def nrmse(ref, test, method: str = "range"):
    ref, test = _paired_clean_1d(ref, test)
    if not ref.size:
        return np.nan
    e = np.sqrt(np.mean((test - ref) ** 2))
    if method == "range":
        denom = np.max(ref) - np.min(ref)
    elif method == "sd":
        denom = np.std(ref, ddof=1)
    elif method == "mean_abs":
        denom = np.mean(np.abs(ref))
    else:
        raise ValueError("method must be 'range', 'sd', or 'mean_abs'")
    return float(e / denom) if denom != 0 else np.nan


def mape_range(ref, test):
    ref, test = _paired_clean_1d(ref, test)
    if not ref.size:
        return np.nan
    denom = np.max(ref) - np.min(ref)
    return float(np.mean(np.abs(test - ref)) / denom * 100) if denom != 0 else np.nan


def waveform_metrics(
    ref_curve,
    test_curve,
    unit: str = "deg",
    *,
    time=None,
    minimum_amplitude: float | None = None,
    minimum_paired_samples: int = 10,
    minimum_paired_coverage: float = 0.8,
):
    """Return guarded within-trial waveform agreement metrics.

    Absolute errors remain available for low-amplitude curves. Correlation,
    concordance and range-normalized RMSE are returned only when both paired
    curves exceed the declared amplitude threshold. Limits of agreement here
    describe paired frames within one trial; they are not population-level
    confidence intervals because frames are temporally autocorrelated.
    """

    reference_all = _as_array(ref_curve).ravel()
    test_all = _as_array(test_curve).ravel()
    if reference_all.shape != test_all.shape:
        raise ValueError(f"Shape mismatch: {reference_all.shape} vs {test_all.shape}")
    valid = np.isfinite(reference_all) & np.isfinite(test_all)
    reference = reference_all[valid]
    test = test_all[valid]
    paired_samples = int(reference.size)
    paired_coverage = (
        float(paired_samples / reference_all.size) if reference_all.size else 0.0
    )
    threshold = (
        waveform_amplitude_threshold(unit)
        if minimum_amplitude is None
        else float(minimum_amplitude)
    )
    reference_range = (
        float(np.max(reference) - np.min(reference)) if paired_samples else np.nan
    )
    test_range = float(np.max(test) - np.min(test)) if paired_samples else np.nan
    status = "ok"
    if paired_samples < int(minimum_paired_samples):
        status = "insufficient_pairs"
    elif paired_coverage < float(minimum_paired_coverage):
        status = "insufficient_coverage"
    elif not np.isfinite(threshold):
        status = "unknown_native_scale"
    elif reference_range < threshold:
        status = "low_reference_amplitude"
    elif test_range < threshold:
        status = "low_test_amplitude"
    shape_eligible = status == "ok"
    agreement = bland_altman(reference, test)
    if shape_eligible and paired_samples >= 2 and np.var(reference) > 0:
        gain, offset = np.polyfit(reference, test, 1)
        linear_gain = float(gain)
        linear_offset = float(offset)
    else:
        linear_gain = np.nan
        linear_offset = np.nan

    max_reference_index = int(np.argmax(reference)) if shape_eligible else -1
    max_test_index = int(np.argmax(test)) if shape_eligible else -1
    min_reference_index = int(np.argmin(reference)) if shape_eligible else -1
    min_test_index = int(np.argmin(test)) if shape_eligible else -1
    max_index_difference = (
        max_test_index - max_reference_index if shape_eligible else np.nan
    )
    min_index_difference = (
        min_test_index - min_reference_index if shape_eligible else np.nan
    )
    max_time_difference = np.nan
    min_time_difference = np.nan
    if time is not None:
        time_all = _as_array(time).ravel()
        if time_all.shape != reference_all.shape:
            raise ValueError(
                f"Time shape mismatch: {time_all.shape} vs {reference_all.shape}"
            )
        paired_time = time_all[valid]
        if shape_eligible:
            max_time_difference = float(
                paired_time[max_test_index] - paired_time[max_reference_index]
            )
            min_time_difference = float(
                paired_time[min_test_index] - paired_time[min_reference_index]
            )

    return {
        f"bias_{unit}": bias(reference, test),
        f"mae_{unit}": mae(reference, test),
        f"rmse_{unit}": rmse(reference, test),
        f"loa_lower_{unit}": agreement["loa_lower"],
        f"loa_upper_{unit}": agreement["loa_upper"],
        f"sd_difference_{unit}": agreement["sd_diff"],
        "limits_of_agreement_scope": (
            "descriptive_within_trial_frames_not_population_ci"
        ),
        "nrmse_range": (
            nrmse(reference, test, method="range") if shape_eligible else np.nan
        ),
        "pearson_r_waveform": pearson_r(reference, test) if shape_eligible else np.nan,
        "lin_ccc_waveform": lin_ccc(reference, test) if shape_eligible else np.nan,
        "linear_gain": linear_gain,
        f"linear_offset_{unit}": linear_offset,
        f"reference_rom_{unit}": reference_range,
        f"test_rom_{unit}": test_range,
        f"rom_difference_{unit}": test_range - reference_range,
        "max_index_difference_samples": max_index_difference,
        "min_index_difference_samples": min_index_difference,
        "max_time_difference_s": max_time_difference,
        "min_time_difference_s": min_time_difference,
        "paired_samples": paired_samples,
        "paired_coverage": paired_coverage,
        "minimum_paired_samples": int(minimum_paired_samples),
        "minimum_paired_coverage": float(minimum_paired_coverage),
        f"minimum_amplitude_{unit}": threshold,
        "shape_metrics_eligible": shape_eligible,
        "waveform_status": status,
    }


def joint_center_error_xyz(ref_xyz, test_xyz):
    ref_xyz = _as_array(ref_xyz)
    test_xyz = _as_array(test_xyz)
    if ref_xyz.shape != test_xyz.shape:
        raise ValueError(f"Shape mismatch: {ref_xyz.shape} vs {test_xyz.shape}")
    if ref_xyz.ndim != 2 or ref_xyz.shape[1] != 3:
        raise ValueError("Expected shape = (n_frames, 3)")
    valid = np.all(np.isfinite(ref_xyz), axis=1) & np.all(np.isfinite(test_xyz), axis=1)
    diff = test_xyz[valid] - ref_xyz[valid]
    euclidean = np.linalg.norm(diff, axis=1)
    result: dict[str, float | int | str] = {
        "paired_frames": int(valid.sum()),
        "paired_coverage": float(valid.mean()) if valid.size else 0.0,
        "limits_of_agreement_scope": (
            "descriptive_within_trial_frames_not_population_ci"
        ),
    }
    for axis_index, axis in enumerate(("x", "y", "z")):
        values = diff[:, axis_index]
        agreement = bland_altman(np.zeros(values.shape), values)
        result[f"bias_{axis}"] = float(np.mean(values)) if values.size else np.nan
        result[f"mae_{axis}"] = (
            float(np.mean(np.abs(values))) if values.size else np.nan
        )
        result[f"loa_lower_{axis}"] = agreement["loa_lower"]
        result[f"loa_upper_{axis}"] = agreement["loa_upper"]
    result["mae_euclidean"] = float(np.mean(euclidean)) if euclidean.size else np.nan
    result["rmse_euclidean"] = (
        float(np.sqrt(np.mean(euclidean**2))) if euclidean.size else np.nan
    )
    return result


def aggregate_trial_metrics_by_participant(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_keys: tuple[str, ...],
    group_keys: tuple[str, ...],
    participant_key: str = "participant",
    trial_key: str = "trial",
    bootstrap_samples: int = 2000,
    random_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate trial summaries to participant, then population summaries."""

    records = [dict(row) for row in rows]

    def missing_participant(row: Mapping[str, Any]) -> bool:
        value = row.get(participant_key, "")
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return True
        return not str(value).strip()

    if not records or any(missing_participant(row) for row in records):
        return [], {
            "status": "missing_participant_identifier",
            "statistical_unit": "participant",
            "trial_rows": len(records),
            "participant_rows": 0,
        }
    participant_values: dict[tuple[str, ...], dict[str, list[float]]] = {}
    participant_trials: set[tuple[str, str]] = set()
    for row in records:
        participant = str(row[participant_key])
        participant_trials.add((participant, str(row.get(trial_key, ""))))
        group = tuple(str(row.get(key, "")) for key in group_keys)
        key = (participant, *group)
        metrics = participant_values.setdefault(
            key, {metric: [] for metric in metric_keys}
        )
        for metric in metric_keys:
            try:
                value = float(row.get(metric, np.nan))
            except (TypeError, ValueError):
                value = np.nan
            if np.isfinite(value):
                metrics[metric].append(value)
    by_group: dict[tuple[str, ...], dict[str, list[float]]] = {}
    participants_by_group: dict[tuple[str, ...], set[str]] = {}
    participant_rows = 0
    for (_participant, *group_values), metrics in participant_values.items():
        group = tuple(group_values)
        destination = by_group.setdefault(group, {metric: [] for metric in metric_keys})
        participants_by_group.setdefault(group, set()).add(_participant)
        participant_rows += 1
        for metric, values in metrics.items():
            if values:
                destination[metric].append(float(np.mean(values)))
    summary: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    for group, metrics in sorted(by_group.items()):
        if not any(values for values in metrics.values()):
            continue
        row: dict[str, Any] = {
            key: group[index] for index, key in enumerate(group_keys)
        }
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=float)
            row[f"participants_{metric}"] = int(array.size)
            row[f"mean_{metric}"] = float(np.mean(array)) if array.size else np.nan
            row[f"sd_{metric}"] = (
                float(np.std(array, ddof=1)) if array.size >= 2 else np.nan
            )
            row[f"median_{metric}"] = float(np.median(array)) if array.size else np.nan
            row[f"p25_{metric}"] = (
                float(np.percentile(array, 25)) if array.size else np.nan
            )
            row[f"p75_{metric}"] = (
                float(np.percentile(array, 75)) if array.size else np.nan
            )
            if array.size >= 2 and bootstrap_samples > 0:
                samples = rng.choice(
                    array,
                    size=(int(bootstrap_samples), array.size),
                    replace=True,
                )
                bootstrap_means = np.mean(samples, axis=1)
                row[f"ci95_lower_{metric}"] = float(np.percentile(bootstrap_means, 2.5))
                row[f"ci95_upper_{metric}"] = float(
                    np.percentile(bootstrap_means, 97.5)
                )
            else:
                row[f"ci95_lower_{metric}"] = np.nan
                row[f"ci95_upper_{metric}"] = np.nan
        row["participants"] = len(participants_by_group.get(group, set()))
        summary.append(row)
    maximum_participants = max((int(row["participants"]) for row in summary), default=0)
    if not summary:
        status = "no_finite_metrics"
    elif maximum_participants < 2:
        status = "insufficient_participants"
    else:
        status = "ok"
    return summary, {
        "status": status,
        "statistical_unit": "participant",
        "trial_rows": len(participant_trials),
        "participant_rows": participant_rows,
        "group_keys": list(group_keys),
        "metric_keys": list(metric_keys),
        "uncertainty_method": "participant_cluster_bootstrap_mean",
        "bootstrap_samples": int(bootstrap_samples),
        "random_seed": int(random_seed),
        "minimum_participants_for_inference": 2,
        "maximum_participants_per_group": maximum_participants,
    }


def discrete_metrics(ref_values, test_values, unit: str = "deg"):
    ba = bland_altman(ref_values, test_values)
    return {
        f"bias_{unit}": bias(ref_values, test_values),
        f"mae_{unit}": mae(ref_values, test_values),
        f"rmse_{unit}": rmse(ref_values, test_values),
        "pearson_r_across_observations": pearson_r(ref_values, test_values),
        "lin_ccc_across_observations": lin_ccc(ref_values, test_values),
        f"loa_lower_{unit}": ba["loa_lower"],
        f"loa_upper_{unit}": ba["loa_upper"],
    }


def coefficient_multiple_correlation(ref_curves, test_curves):
    ref = _as_array(ref_curves)
    test = _as_array(test_curves)
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {test.shape}")
    if ref.ndim != 2:
        raise ValueError("Expected shape = (n_trials, n_time)")
    data = np.stack([ref, test], axis=0)
    mean_time = np.nanmean(data, axis=(0, 1), keepdims=True)
    mean_system_trial = np.nanmean(data, axis=2, keepdims=True)
    numerator = np.nansum((data - mean_system_trial) ** 2)
    denominator = np.nansum((data - mean_time) ** 2)
    if denominator == 0:
        return np.nan
    return float(np.sqrt(max(0.0, 1.0 - numerator / denominator)))


def icc_2_1(data):
    x = _as_array(data)
    if x.ndim != 2:
        raise ValueError("Expected shape = (n_subjects, n_raters)")
    if np.isnan(x).any():
        raise ValueError("Remove NaN before ICC")
    n, k = x.shape
    mean_subject = np.mean(x, axis=1, keepdims=True)
    mean_rater = np.mean(x, axis=0, keepdims=True)
    grand_mean = np.mean(x)
    ss_subject = k * np.sum((mean_subject - grand_mean) ** 2)
    ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
    ss_error = np.sum((x - mean_subject - mean_rater + grand_mean) ** 2)
    ms_subject = ss_subject / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    return float(
        (ms_subject - ms_error)
        / (ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)
    )


def sem_from_icc(ref_values, icc):
    ref_values = _as_array(ref_values)
    return float(np.nanstd(ref_values, ddof=1) * np.sqrt(1 - icc))


def mdc95(sem):
    return float(1.96 * np.sqrt(2) * sem)
