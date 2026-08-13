"""Temporal synchronization contracts for Captury/Motive comparisons.

Motive is the reference clock.  A positive lag means that a Captury sample
must be placed later on the Motive timeline::

    captury_corrected_time = captury_original_time + lag_s

The automatic estimator compares translation-invariant speeds from common
joint-centre trajectories.  It estimates only one constant clock offset; it
does not compensate clock drift, deform time, or alter spatial samples.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

LAG_CONVENTION = "captury_corrected_time = captury_original_time + lag_s"


def apply_time_offset(time: np.ndarray, lag_s: float) -> np.ndarray:
    """Return a shifted copy of a time vector according to ``LAG_CONVENTION``."""

    return np.asarray(time, dtype=float).copy() + float(lag_s)


def _filled_coordinate(values: np.ndarray, time: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(values) & np.isfinite(time)
    if np.count_nonzero(finite) < 3:
        return None
    return np.interp(time, time[finite], values[finite])


def _trajectory_speed(values: np.ndarray, time: np.ndarray) -> np.ndarray | None:
    trajectory = np.asarray(values, dtype=float)
    timestamps = np.asarray(time, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[0] != 3:
        raise ValueError("Joint-centre trajectories must have shape 3 x frames.")
    if trajectory.shape[1] != timestamps.size or timestamps.size < 3:
        return None
    coordinates = [_filled_coordinate(axis, timestamps) for axis in trajectory]
    if any(axis is None for axis in coordinates):
        return None
    filled = np.vstack(coordinates)
    velocity = np.gradient(filled, timestamps, axis=1)
    speed = np.linalg.norm(velocity, axis=0)
    originally_finite = np.all(np.isfinite(trajectory), axis=0)
    speed[~originally_finite] = np.nan
    return speed


def composite_joint_centre_speed(
    centres: Mapping[str, np.ndarray], time: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Build a robust dimensionless speed signal from available joint centres."""

    normalized: list[np.ndarray] = []
    used: list[str] = []
    for name in sorted(centres):
        speed = _trajectory_speed(centres[name], time)
        if speed is None:
            continue
        finite = speed[np.isfinite(speed)]
        if finite.size < 8:
            continue
        lower, upper = np.percentile(finite, [10.0, 90.0])
        scale = float(upper - lower)
        if not np.isfinite(scale) or scale <= 1e-9:
            continue
        normalized.append((speed - float(np.median(finite))) / scale)
        used.append(name)
    if not normalized:
        return np.full(np.asarray(time).shape, np.nan, dtype=float), []
    stacked = np.vstack(normalized)
    finite_count = np.sum(np.isfinite(stacked), axis=0)
    composite = np.full(stacked.shape[1], np.nan, dtype=float)
    available = finite_count > 0
    composite[available] = np.nanmedian(stacked[:, available], axis=0)
    return composite, used


def interpolate_finite_signal(
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
    *,
    max_gap_s: float | None = None,
) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(source_time)
    if np.count_nonzero(finite) < 2:
        return np.full(target_time.shape, np.nan, dtype=float)
    result = np.interp(target_time, source_time[finite], values[finite])
    outside = (target_time < source_time[finite][0]) | (
        target_time > source_time[finite][-1]
    )
    result[outside] = np.nan
    finite_time = source_time[finite]
    gap_limit = (
        float(max_gap_s)
        if max_gap_s is not None
        else 1.5 * float(np.median(np.diff(source_time)))
    )
    if finite_time.size >= 2 and np.isfinite(gap_limit):
        for left, right in zip(finite_time[:-1], finite_time[1:], strict=True):
            if right - left > gap_limit:
                result[(target_time > left) & (target_time < right)] = np.nan
    return result


def interpolate_finite_array(
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
    *,
    max_gap_s: float | None = None,
) -> np.ndarray:
    """Interpolate the last axis and return NaN outside temporal overlap."""

    array = np.asarray(values, dtype=float)
    source = np.asarray(source_time, dtype=float)
    target = np.asarray(target_time, dtype=float)
    if array.shape[-1] != source.size:
        raise ValueError(
            f"Time length {source.size} does not match values axis {array.shape[-1]}."
        )
    flat = array.reshape(-1, array.shape[-1])
    interpolated = np.vstack(
        [
            interpolate_finite_signal(row, source, target, max_gap_s=max_gap_s)
            for row in flat
        ]
    )
    return interpolated.reshape(*array.shape[:-1], target.size)


def _peak_prominence(
    candidates: np.ndarray,
    correlations: np.ndarray,
    best_index: int,
    exclusion_width_s: float,
) -> tuple[float | None, float | None]:
    separated = (
        np.abs(candidates - candidates[best_index]) >= float(exclusion_width_s)
    ) & np.isfinite(correlations)
    if not np.any(separated):
        return None, None
    second = float(np.nanmax(correlations[separated]))
    return second, float(correlations[best_index] - second)


def _agreement_for_lag(
    reference_signal: np.ndarray,
    reference_time: np.ndarray,
    moving_signal: np.ndarray,
    moving_time: np.ndarray,
    lag_s: float,
    sample_period_s: float,
) -> tuple[float, float, int, float]:
    corrected_time = apply_time_offset(moving_time, lag_s)
    overlap_start = max(float(reference_time[0]), float(corrected_time[0]))
    overlap_end = min(float(reference_time[-1]), float(corrected_time[-1]))
    if overlap_end <= overlap_start:
        return np.nan, np.nan, 0, 0.0
    common_time = np.arange(
        overlap_start, overlap_end + 0.5 * sample_period_s, sample_period_s
    )
    reference = interpolate_finite_signal(reference_signal, reference_time, common_time)
    moving = interpolate_finite_signal(moving_signal, corrected_time, common_time)
    valid = np.isfinite(reference) & np.isfinite(moving)
    if np.count_nonzero(valid) < 20:
        return (
            np.nan,
            np.nan,
            int(np.count_nonzero(valid)),
            overlap_end - overlap_start,
        )
    reference = reference[valid]
    moving = moving[valid]
    if np.std(reference) <= 1e-9 or np.std(moving) <= 1e-9:
        return np.nan, np.nan, int(valid.sum()), overlap_end - overlap_start
    reference_z = (reference - np.mean(reference)) / np.std(reference)
    moving_z = (moving - np.mean(moving)) / np.std(moving)
    return (
        float(np.corrcoef(reference, moving)[0, 1]),
        float(np.sqrt(np.mean((moving_z - reference_z) ** 2))),
        int(valid.sum()),
        overlap_end - overlap_start,
    )


def estimate_constant_lag(
    reference_centres: Mapping[str, np.ndarray],
    reference_time: np.ndarray,
    moving_centres: Mapping[str, np.ndarray],
    moving_time: np.ndarray,
    *,
    max_lag_s: float = 0.5,
    min_correlation: float = 0.5,
    min_correlation_gain: float = 0.01,
    min_peak_prominence: float = 0.02,
) -> dict[str, Any]:
    """Estimate one constant Captury-to-Motive clock offset from common centres."""

    common = sorted(set(reference_centres).intersection(moving_centres))
    reference_signal, reference_used = composite_joint_centre_speed(
        {name: reference_centres[name] for name in common}, reference_time
    )
    moving_signal, moving_used = composite_joint_centre_speed(
        {name: moving_centres[name] for name in common}, moving_time
    )
    used = sorted(set(reference_used).intersection(moving_used))
    if used and (set(reference_used) != set(used) or set(moving_used) != set(used)):
        reference_signal, _ = composite_joint_centre_speed(
            {name: reference_centres[name] for name in used}, reference_time
        )
        moving_signal, _ = composite_joint_centre_speed(
            {name: moving_centres[name] for name in used}, moving_time
        )
    base = {
        "method": "common_joint_centre_composite_speed_cross_correlation",
        "reference_clock": "motive",
        "moving_clock": "captury",
        "lag_convention": LAG_CONVENTION,
        "max_lag_s": float(max_lag_s),
        "minimum_correlation_for_application": float(min_correlation),
        "minimum_correlation_gain_for_application": float(min_correlation_gain),
        "minimum_peak_prominence_for_application": float(min_peak_prominence),
        "common_centres": common,
        "used_centres": used,
    }
    if not used:
        return {
            **base,
            "status": "insufficient_motion",
            "applied": False,
            "lag_s": 0.0,
            "reason": "no_common_centre_with_variable_speed",
        }
    reference_dt = np.diff(np.asarray(reference_time, dtype=float))
    moving_dt = np.diff(np.asarray(moving_time, dtype=float))
    finite_dt = np.concatenate(
        (reference_dt[np.isfinite(reference_dt)], moving_dt[np.isfinite(moving_dt)])
    )
    finite_dt = finite_dt[finite_dt > 0]
    if finite_dt.size == 0:
        return {
            **base,
            "status": "invalid_time",
            "applied": False,
            "lag_s": 0.0,
        }
    sample_period_s = float(
        min(
            np.median(reference_dt[np.isfinite(reference_dt) & (reference_dt > 0)]),
            np.median(moving_dt[np.isfinite(moving_dt) & (moving_dt > 0)]),
        )
    )
    search_step_s = max(sample_period_s, 0.001)
    candidates = np.arange(
        -abs(max_lag_s), abs(max_lag_s) + 0.5 * search_step_s, search_step_s
    )
    evaluations = [
        _agreement_for_lag(
            reference_signal,
            np.asarray(reference_time, dtype=float),
            moving_signal,
            np.asarray(moving_time, dtype=float),
            float(lag),
            sample_period_s,
        )
        for lag in candidates
    ]
    correlations = np.asarray([item[0] for item in evaluations], dtype=float)
    if not np.any(np.isfinite(correlations)):
        return {
            **base,
            "status": "insufficient_motion",
            "applied": False,
            "lag_s": 0.0,
            "reason": "no_finite_cross_correlation",
        }
    best_index = int(np.nanargmax(correlations))
    best_lag = float(candidates[best_index])
    if abs(best_lag) < 0.5 * search_step_s:
        best_lag = 0.0
    correlation_before, rmse_before, before_samples, _ = _agreement_for_lag(
        reference_signal,
        np.asarray(reference_time, dtype=float),
        moving_signal,
        np.asarray(moving_time, dtype=float),
        0.0,
        sample_period_s,
    )
    correlation_after, rmse_after, overlap_samples, overlap_duration_s = evaluations[
        best_index
    ]
    corrected_time = apply_time_offset(moving_time, best_lag)
    overlap_start_s = max(float(reference_time[0]), float(corrected_time[0]))
    overlap_end_s = min(float(reference_time[-1]), float(corrected_time[-1]))
    at_boundary = bool(abs(best_lag) >= abs(max_lag_s) - search_step_s)
    second_peak_correlation, peak_prominence = _peak_prominence(
        candidates,
        correlations,
        best_index,
        max(0.1, 5.0 * search_step_s),
    )
    correlation_gain = (
        float(correlation_after - correlation_before)
        if np.isfinite(correlation_before)
        else None
    )
    low_correlation = bool(correlation_after < min_correlation)
    negligible_gain = bool(
        correlation_gain is not None
        and abs(best_lag) >= 0.5 * search_step_s
        and correlation_gain < min_correlation_gain
    )
    ambiguous_peak = bool(
        peak_prominence is not None and peak_prominence < min_peak_prominence
    )
    status = (
        "boundary_peak"
        if at_boundary
        else (
            "low_confidence"
            if low_correlation
            else (
                "negligible_gain"
                if negligible_gain
                else "ambiguous_peak" if ambiguous_peak else "ok"
            )
        )
    )
    return {
        **base,
        "status": status,
        "applied": status == "ok" and best_lag != 0.0,
        "lag_s": best_lag if status == "ok" else 0.0,
        "estimated_lag_s": best_lag,
        "search_step_s": search_step_s,
        "sample_period_s": sample_period_s,
        "correlation_before": (
            float(correlation_before) if np.isfinite(correlation_before) else None
        ),
        "correlation_after": float(correlation_after),
        "correlation_gain": correlation_gain,
        "second_peak_correlation": second_peak_correlation,
        "peak_prominence": peak_prominence,
        "lag_uncertainty_s": float(search_step_s),
        "normalized_rmse_before": (
            float(rmse_before) if np.isfinite(rmse_before) else None
        ),
        "normalized_rmse_after": float(rmse_after),
        "overlap_samples": int(overlap_samples),
        "overlap_duration_s": float(overlap_duration_s),
        "estimation_window_s": {
            "start": overlap_start_s,
            "end": overlap_end_s,
        },
        "samples_at_zero_lag": int(before_samples),
    }


def resolve_temporal_synchronization(
    mode: str,
    reference_centres: Mapping[str, np.ndarray],
    reference_time: np.ndarray,
    moving_centres: Mapping[str, np.ndarray],
    moving_time: np.ndarray,
    *,
    manual_lag_s: float | None = None,
    max_lag_s: float = 0.5,
    min_correlation: float = 0.5,
    min_correlation_gain: float = 0.01,
    min_peak_prominence: float = 0.02,
) -> dict[str, Any]:
    """Resolve ``none``, ``manual`` or ``auto`` synchronization policies."""

    if mode == "none":
        return {
            "status": "disabled",
            "method": "none",
            "applied": False,
            "lag_s": 0.0,
            "lag_convention": LAG_CONVENTION,
            "reference_clock": "motive",
            "moving_clock": "captury",
        }
    if mode == "manual":
        if manual_lag_s is None or not np.isfinite(manual_lag_s):
            raise ValueError("Manual temporal synchronization requires --manual-lag-s.")
        return {
            "status": "manual",
            "method": "manual_constant_lag",
            "applied": bool(float(manual_lag_s) != 0.0),
            "lag_s": float(manual_lag_s),
            "lag_convention": LAG_CONVENTION,
            "reference_clock": "motive",
            "moving_clock": "captury",
        }
    if mode != "auto":
        raise ValueError(f"Unknown synchronization mode: {mode!r}.")
    return estimate_constant_lag(
        reference_centres,
        reference_time,
        moving_centres,
        moving_time,
        max_lag_s=max_lag_s,
        min_correlation=min_correlation,
        min_correlation_gain=min_correlation_gain,
        min_peak_prominence=min_peak_prominence,
    )


def normalize_selected_phase(
    reference_time: np.ndarray,
    reference_signal: np.ndarray,
    moving_time: np.ndarray,
    moving_signal: np.ndarray,
    *,
    lag_s: float,
    start_s: float | None,
    end_s: float | None,
    n_points: int = 101,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Normalize two scalar signals over one selected Motive-time phase."""

    corrected_time = apply_time_offset(moving_time, lag_s)
    overlap_start = max(float(reference_time[0]), float(corrected_time[0]))
    overlap_end = min(float(reference_time[-1]), float(corrected_time[-1]))
    used_start = (
        overlap_start if start_s is None else max(overlap_start, float(start_s))
    )
    used_end = overlap_end if end_s is None else min(overlap_end, float(end_s))
    if n_points < 2:
        raise ValueError("Phase normalization requires at least two points.")
    if used_end <= used_start:
        return [], {
            "status": "empty_overlap",
            "start_s": used_start,
            "end_s": used_end,
            "points": int(n_points),
        }
    normalized_time = np.linspace(used_start, used_end, int(n_points))
    reference = interpolate_finite_signal(
        np.asarray(reference_signal, dtype=float),
        np.asarray(reference_time, dtype=float),
        normalized_time,
    )
    moving = interpolate_finite_signal(
        np.asarray(moving_signal, dtype=float), corrected_time, normalized_time
    )
    valid = np.isfinite(reference) & np.isfinite(moving)
    if np.count_nonzero(valid) < 3:
        return [], {
            "status": "insufficient_signal",
            "start_s": used_start,
            "end_s": used_end,
            "points": int(n_points),
            "finite_pairs": int(np.count_nonzero(valid)),
            "lag_s": float(lag_s),
            "lag_convention": LAG_CONVENTION,
        }
    phase_percent = np.linspace(0.0, 100.0, int(n_points))
    rows = [
        {
            "phase_percent": float(phase),
            "time_s": float(time_value),
            "reference": float(reference[index]),
            "moving": float(moving[index]),
            "difference": float(moving[index] - reference[index]),
        }
        for index, (phase, time_value) in enumerate(zip(phase_percent, normalized_time))
    ]
    return rows, {
        "status": "ok",
        "start_s": used_start,
        "end_s": used_end,
        "points": int(n_points),
        "lag_s": float(lag_s),
        "lag_convention": LAG_CONVENTION,
    }


def normalize_timeseries_groups(
    rows: list[dict[str, Any]],
    *,
    time_key: str,
    group_keys: tuple[str, ...],
    value_keys: tuple[str, ...],
    start_s: float | None,
    end_s: float | None,
    n_points: int,
) -> list[dict[str, Any]]:
    """Normalize grouped long-form timeseries over one selected time window."""

    if n_points < 2:
        raise ValueError("Phase normalization requires at least two points.")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            time_value = float(row[time_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(time_value):
            continue
        key = tuple(str(row.get(group, "")) for group in group_keys)
        groups.setdefault(key, []).append(row)
    normalized_rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=lambda row: float(row[time_key]))
        times = np.asarray([float(row[time_key]) for row in ordered], dtype=float)
        used_start = (
            float(times[0]) if start_s is None else max(float(times[0]), start_s)
        )
        used_end = float(times[-1]) if end_s is None else min(float(times[-1]), end_s)
        if used_end <= used_start:
            continue
        target_time = np.linspace(used_start, used_end, int(n_points))
        phase = np.linspace(0.0, 100.0, int(n_points))
        interpolated: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        duration_s = float(used_end - used_start)
        for value_key in value_keys:
            source_values = np.asarray(
                [float(row.get(value_key, np.nan)) for row in ordered], dtype=float
            )
            in_window = (times >= used_start) & (times <= used_end)
            finite = np.isfinite(source_values)
            finite_times = times[finite]
            finite_count = int(finite_times.size)
            finite_fraction = float(finite_count / max(1, times.size))
            positive_steps = np.diff(finite_times)
            positive_steps = positive_steps[positive_steps > 0]
            gap_starts = finite_times[:-1]
            gap_ends = finite_times[1:]
            relevant_gaps = positive_steps[
                (gap_ends > used_start) & (gap_starts < used_end)
            ]
            largest_gap_s = (
                float(np.max(relevant_gaps)) if relevant_gaps.size else np.nan
            )
            all_steps = np.diff(times)
            all_steps = all_steps[all_steps > 0]
            nominal_step_s = float(np.median(all_steps)) if all_steps.size else np.nan
            maximum_gap_s = max(
                5.0 * nominal_step_s if np.isfinite(nominal_step_s) else 0.0,
                0.1 * duration_s,
            )
            status = "ok"
            if finite_count < 2 or (
                np.isfinite(largest_gap_s) and largest_gap_s > maximum_gap_s
            ):
                status = "unavailable"
                values = np.full(target_time.shape, np.nan, dtype=float)
            else:
                values = interpolate_finite_signal(source_values, times, target_time)
            interpolated[value_key] = values
            diagnostics[value_key] = {
                "status": status,
                "finite_fraction": finite_fraction,
                "largest_gap_s": largest_gap_s,
                "maximum_gap_s": maximum_gap_s,
            }
        statuses = {item["status"] for item in diagnostics.values()}
        group_status = (
            "ok"
            if statuses == {"ok"}
            else "unavailable" if statuses == {"unavailable"} else "partial"
        )
        for index in range(int(n_points)):
            output: dict[str, Any] = {
                group_key: group[group_index]
                for group_index, group_key in enumerate(group_keys)
            }
            output.update(
                {
                    "phase_percent": float(phase[index]),
                    "time_s": float(target_time[index]),
                    "normalization_status": group_status,
                }
            )
            output.update(
                {
                    value_key: float(values[index])
                    for value_key, values in interpolated.items()
                }
            )
            for value_key, diagnostic in diagnostics.items():
                output[f"{value_key}_normalization_status"] = diagnostic["status"]
                output[f"{value_key}_finite_fraction"] = diagnostic["finite_fraction"]
                output[f"{value_key}_largest_gap_s"] = diagnostic["largest_gap_s"]
                output[f"{value_key}_maximum_gap_s"] = diagnostic["maximum_gap_s"]
            normalized_rows.append(output)
    return normalized_rows
