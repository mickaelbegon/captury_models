"""Agreement metrics for marker-based versus markerless model comparisons."""

from __future__ import annotations

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
        return {"bias": np.nan, "loa_lower": np.nan, "loa_upper": np.nan, "sd_diff": np.nan}
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


def waveform_metrics(ref_curve, test_curve, unit: str = "deg"):
    return {
        f"bias_{unit}": bias(ref_curve, test_curve),
        f"mae_{unit}": mae(ref_curve, test_curve),
        f"rmse_{unit}": rmse(ref_curve, test_curve),
        "nrmse_range": nrmse(ref_curve, test_curve, method="range"),
        "pearson_r_waveform": pearson_r(ref_curve, test_curve),
        "lin_ccc_waveform": lin_ccc(ref_curve, test_curve),
    }


def joint_center_error_xyz(ref_xyz, test_xyz):
    ref_xyz = _as_array(ref_xyz)
    test_xyz = _as_array(test_xyz)
    if ref_xyz.shape != test_xyz.shape:
        raise ValueError(f"Shape mismatch: {ref_xyz.shape} vs {test_xyz.shape}")
    if ref_xyz.ndim != 2 or ref_xyz.shape[1] != 3:
        raise ValueError("Expected shape = (n_frames, 3)")
    diff = test_xyz - ref_xyz
    euclidean = np.linalg.norm(diff, axis=1)
    return {
        "mae_x": float(np.nanmean(np.abs(diff[:, 0]))),
        "mae_y": float(np.nanmean(np.abs(diff[:, 1]))),
        "mae_z": float(np.nanmean(np.abs(diff[:, 2]))),
        "mae_euclidean": float(np.nanmean(euclidean)),
        "rmse_euclidean": float(np.sqrt(np.nanmean(euclidean**2))),
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
    return float((ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n))


def sem_from_icc(ref_values, icc):
    ref_values = _as_array(ref_values)
    return float(np.nanstd(ref_values, ddof=1) * np.sqrt(1 - icc))


def mdc95(sem):
    return float(1.96 * np.sqrt(2) * sem)
