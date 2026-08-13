#!/usr/bin/env python3
"""Run cached nonlinear TRF inverse kinematics for a BioBuddy model.

The solver consumes one existing ``.bioMod`` and one marker C3D.  It matches
the C3D trajectories directly to ``model.technicalMarkerNames()`` and solves a
bounded nonlinear least-squares problem frame by frame with SciPy's Trust
Region Reflective (TRF) algorithm.  The previous biorbd wrapper mixed indices
from all model markers with indices from technical markers when anatomical
markers were interleaved; this module deliberately avoids that ambiguity.

Results are cached from the content hashes of the BioMod and C3D plus every
solver/mapping parameter.  Progress lines include elapsed time, per-frame
``nfev``, marker error and estimated remaining time so the CLI and Tk GUI can
show useful live feedback without embedding scientific logic in the GUI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time as time_module
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy.optimize import least_squares

from bvh_c3d_biobuddy_pyorerun_compare import (
    biorbd_strings_to_list,
    finite_difference_by_time,
    require_biorbd,
    split_c3d_points,
)
from mocap_labels import (
    DEFAULT_MARKER_PREFIXES_TO_STRIP,
    marker_name_index,
    stripped_marker_label,
)

ANGLE_LABEL_REGEX = r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)"
IK_CACHE_VERSION = 2
IK_SOLVER_NAME = "scipy.optimize.least_squares:trf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Hash this solver module so implementation changes invalidate the cache."""

    return _sha256(Path(__file__).resolve())


def ik_cache_fingerprint(
    biomod: Path,
    c3d: Path,
    *,
    method: str,
    max_frames: int,
    biomod_unit_scale_to_m: float,
    marker_prefixes_to_strip: tuple[str, ...],
    angle_label_regex: str,
    xtol: float,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
) -> dict[str, Any]:
    """Return a content-addressed cache key for one IK computation."""

    payload = {
        "version": IK_CACHE_VERSION,
        "biomod": {"path": str(biomod.resolve()), "sha256": _sha256(biomod)},
        "c3d": {"path": str(c3d.resolve()), "sha256": _sha256(c3d)},
        "solver": IK_SOLVER_NAME,
        "implementation_sha256": implementation_sha256(),
        "scipy_version": scipy.__version__,
        "solver_parameters": {
            "method": method,
            "xtol": float(xtol),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "max_nfev": max_nfev,
        },
        "max_frames": int(max_frames),
        "biomod_unit_scale_to_m": float(biomod_unit_scale_to_m),
        "marker_prefixes_to_strip": list(marker_prefixes_to_strip),
        "angle_label_regex": angle_label_regex,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": IK_CACHE_VERSION,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def build_direct_marker_data(
    model: Any,
    c3d_path: Path,
    *,
    biomod_unit_scale_to_m: float,
    marker_prefixes_to_strip: tuple[str, ...],
    angle_label_regex: str,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Build ``3 x nbTechnicalMarkers x frames`` data in BioMod units."""

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
    technical_marker_names = biorbd_strings_to_list(model.technicalMarkerNames())
    marker_data = np.full(
        (3, len(technical_marker_names), n_frames), np.nan, dtype=float
    )
    used: list[str] = []
    missing: list[str] = []
    c3d_markers_model_units = split.marker_data_native * (
        split.c3d_unit_scale_to_m / biomod_unit_scale_to_m
    )
    for model_index, marker_name in enumerate(technical_marker_names):
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
            "Aucun marqueur C3D ne correspond aux marqueurs techniques BioBuddy."
        )
    report = {
        "c3d": str(c3d_path),
        "c3d_unit_scale_to_m": float(split.c3d_unit_scale_to_m),
        "biomod_unit_scale_to_m": biomod_unit_scale_to_m,
        "marker_prefixes_to_strip": list(marker_prefixes_to_strip),
        "model_markers": int(model.nbMarkers()),
        "technical_model_markers": len(technical_marker_names),
        "markers_used": len(used),
        "used_marker_names": used,
        "missing_model_markers": missing,
        "angle_channels_ignored": split.angle_labels,
        "marker_indexing": "technicalMarkerNames_direct",
    }
    return marker_data, split.time[frame_indices], technical_marker_names, report


def _technical_marker_positions(model: Any, q: np.ndarray) -> np.ndarray:
    return np.asarray(
        [marker.to_array() for marker in model.technicalMarkers(q)], dtype=float
    )


def _technical_marker_jacobians(model: Any, q: np.ndarray) -> np.ndarray:
    return np.asarray(
        [jacobian.to_array() for jacobian in model.technicalMarkersJacobian(q)],
        dtype=float,
    )


def solve_inverse_kinematics_trf(
    model: Any,
    marker_data: np.ndarray,
    *,
    biomod_unit_scale_to_m: float,
    xtol: float = 1e-6,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve bounded TRF IK using direct technical-marker indexing."""

    biorbd = require_biorbd()
    lower, upper = (
        np.asarray(value, dtype=float) for value in biorbd.get_range_q(model)
    )
    q = np.zeros((int(model.nbQ()), marker_data.shape[2]), dtype=float)
    residuals_mm = np.full(
        (marker_data.shape[1], marker_data.shape[2]), np.nan, dtype=float
    )
    nfev = np.zeros(marker_data.shape[2], dtype=int)
    success = np.zeros(marker_data.shape[2], dtype=bool)
    statuses = np.zeros(marker_data.shape[2], dtype=int)
    messages: list[str] = []
    elapsed_by_frame = np.zeros(marker_data.shape[2], dtype=float)
    midpoint = np.where(
        np.isfinite(lower) & np.isfinite(upper), (lower + upper) / 2.0, 0.0
    )
    x0 = np.clip(midpoint, lower, upper)
    started = time_module.perf_counter()

    for frame in range(marker_data.shape[2]):
        finite = np.all(np.isfinite(marker_data[:, :, frame]), axis=0)
        indices = np.flatnonzero(finite)
        frame_started = time_module.perf_counter()
        if indices.size == 0:
            q[:, frame] = x0
            messages.append("No finite technical marker for this frame.")
            continue
        target = marker_data[:, indices, frame].T

        def residual(current_q: np.ndarray) -> np.ndarray:
            positions = _technical_marker_positions(model, current_q)
            return (positions[indices] - target).reshape(-1)

        def jacobian(current_q: np.ndarray) -> np.ndarray:
            jacobians = _technical_marker_jacobians(model, current_q)
            return jacobians[indices].reshape(-1, int(model.nbQ()))

        solution = least_squares(
            residual,
            x0,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        q[:, frame] = solution.x
        x0 = np.clip(solution.x, lower, upper)
        marker_residuals = np.linalg.norm(solution.fun.reshape(-1, 3), axis=1)
        residuals_mm[indices, frame] = (
            marker_residuals * biomod_unit_scale_to_m * 1000.0
        )
        nfev[frame] = int(solution.nfev)
        success[frame] = bool(solution.success)
        statuses[frame] = int(solution.status)
        messages.append(str(solution.message))
        elapsed_by_frame[frame] = time_module.perf_counter() - frame_started
        elapsed = time_module.perf_counter() - started
        completed = frame + 1
        eta = elapsed / completed * (marker_data.shape[2] - completed)
        event = {
            "frame": frame,
            "frames": marker_data.shape[2],
            "elapsed_s": elapsed,
            "frame_s": elapsed_by_frame[frame],
            "eta_s": eta,
            "nfev": nfev[frame],
            "marker_error_mean_mm": float(np.nanmean(residuals_mm[:, frame])),
            "marker_error_rmse_mm": float(
                np.sqrt(np.nanmean(residuals_mm[:, frame] ** 2))
            ),
        }
        if progress is not None and (
            frame == 0 or completed == marker_data.shape[2] or completed % 25 == 0
        ):
            progress(event)

    return q, {
        "residuals_mm": residuals_mm,
        "nfev": nfev,
        "success": success,
        "status": statuses,
        "message": messages,
        "elapsed_by_frame_s": elapsed_by_frame,
        "elapsed_s": time_module.perf_counter() - started,
    }


def _residual_stats(values: np.ndarray) -> dict[str, Any]:
    """Return descriptive statistics for finite residual norms in millimetres."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "samples": 0,
            "mean_mm": None,
            "median_mm": None,
            "rmse_mm": None,
            "p95_mm": None,
            "max_mm": None,
        }
    return {
        "samples": int(finite.size),
        "mean_mm": float(np.mean(finite)),
        "median_mm": float(np.median(finite)),
        "rmse_mm": float(np.sqrt(np.mean(finite**2))),
        "p95_mm": float(np.percentile(finite, 95)),
        "max_mm": float(np.max(finite)),
    }


def marker_residual_summary(
    residuals_mm: np.ndarray, marker_names: list[str]
) -> dict[str, Any]:
    """Summarize finite marker residual norms in millimetres."""

    return {
        "global": _residual_stats(residuals_mm),
        "per_marker": {
            name: _residual_stats(values)
            for name, values in zip(marker_names, residuals_mm)
        },
    }


def marker_residuals_by_segment(
    model: Any, residuals_mm: np.ndarray, marker_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Group technical-marker residuals by their parent BioMod segment."""

    markers_by_name = {
        model.marker(index).name().to_string(): model.marker(index)
        for index in range(int(model.nbMarkers()))
    }
    segment_indices: dict[str, list[int]] = {}
    for index, marker_name in enumerate(marker_names):
        marker = markers_by_name.get(marker_name)
        if marker is None:
            continue
        segment_indices.setdefault(marker.parent().to_string(), []).append(index)
    return {
        segment: _residual_stats(residuals_mm[indices])
        for segment, indices in sorted(segment_indices.items())
    }


def cached_ik_is_valid(report: dict[str, Any], npz_path: Path) -> bool:
    """Validate the cached NPZ digest and its minimal readable data contract."""

    expected_sha256 = report.get("artifacts", {}).get("npz_sha256")
    if not expected_sha256 or not npz_path.is_file():
        return False
    if _sha256(npz_path) != expected_sha256:
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            required = {"time", "q", "q_names"}
            return required.issubset(data.files) and data["q"].ndim == 2
    except (OSError, ValueError, EOFError):
        return False


def _print_progress(event: dict[str, Any]) -> None:
    print(
        "[BioBuddy IK] "
        f"frame {event['frame'] + 1}/{event['frames']} | "
        f"temps {event['elapsed_s']:.1f} s | nfev {event['nfev']} | "
        f"erreur {event['marker_error_mean_mm']:.2f} mm | "
        f"reste ~{event['eta_s']:.1f} s",
        flush=True,
    )


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
    xtol: float = 1e-6,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Solve or reuse nonlinear TRF IK and store compact NPZ/JSON outputs."""

    biomod = Path(biomod)
    c3d = Path(c3d)
    if method != "trf":
        raise ValueError("BioBuddy IK supports only the bounded nonlinear TRF solver.")
    fingerprint = ik_cache_fingerprint(
        biomod,
        c3d,
        method=method,
        max_frames=max_frames,
        biomod_unit_scale_to_m=biomod_unit_scale_to_m,
        marker_prefixes_to_strip=marker_prefixes_to_strip,
        angle_label_regex=angle_label_regex,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        max_nfev=max_nfev,
    )
    result_dir = (
        Path(cache_dir) / fingerprint["digest"]
        if cache_dir is not None
        else Path(out_dir)
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "inverse_kinematics_trf"
        if cache_dir is not None
        else f"{source_name}_inverse_kinematics_trf"
    )
    npz_path = result_dir / f"{stem}.npz"
    summary_path = result_dir / f"{stem}_summary.json"
    if not force and npz_path.exists() and summary_path.exists():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("cache", {}).get("digest") == fingerprint[
            "digest"
        ] and cached_ik_is_valid(cached, npz_path):
            cached["cache"]["hit"] = True
            print(f"[BioBuddy IK] cache utilisé: {fingerprint['digest']}", flush=True)
            return cached

    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod))
    marker_data, time, marker_names, report = build_direct_marker_data(
        model,
        c3d,
        biomod_unit_scale_to_m=biomod_unit_scale_to_m,
        marker_prefixes_to_strip=marker_prefixes_to_strip,
        angle_label_regex=angle_label_regex,
        max_frames=max_frames,
    )
    q, diagnostics = solve_inverse_kinematics_trf(
        model,
        marker_data,
        biomod_unit_scale_to_m=biomod_unit_scale_to_m,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        max_nfev=max_nfev,
        progress=_print_progress,
    )
    qdot = finite_difference_by_time(q, time)
    qddot = finite_difference_by_time(qdot, time)
    q_names = biorbd_strings_to_list(model.nameDof())
    residual_summary = marker_residual_summary(
        diagnostics["residuals_mm"], marker_names
    )
    residuals_by_segment = marker_residuals_by_segment(
        model, diagnostics["residuals_mm"], marker_names
    )
    np.savez_compressed(
        npz_path,
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        q_names=np.asarray(q_names, dtype=object),
        marker_names=np.asarray(marker_names, dtype=object),
        marker_residuals_mm=diagnostics["residuals_mm"],
        nfev=diagnostics["nfev"],
        success=diagnostics["success"],
        status=diagnostics["status"],
        elapsed_by_frame_s=diagnostics["elapsed_by_frame_s"],
        solver=IK_SOLVER_NAME,
        least_squares_method=method,
    )
    npz_sha256 = _sha256(npz_path)
    report.update(
        {
            "status": "ok",
            "biomod": str(biomod),
            "solver": IK_SOLVER_NAME,
            "least_squares_method": method,
            "solver_parameters": {
                "xtol": xtol,
                "ftol": ftol,
                "gtol": gtol,
                "max_nfev": max_nfev,
            },
            "frames": int(time.shape[0]),
            "nb_q": int(q.shape[0]),
            "elapsed_s": float(diagnostics["elapsed_s"]),
            "nfev": {
                "total": int(np.sum(diagnostics["nfev"])),
                "median_per_frame": float(np.median(diagnostics["nfev"])),
                "max_per_frame": int(np.max(diagnostics["nfev"])),
            },
            "successful_frames": int(np.count_nonzero(diagnostics["success"])),
            "marker_error": residual_summary,
            "marker_error_by_segment": residuals_by_segment,
            "scientific_interpretation": {
                "solver_convergence_is_not_model_validity": True,
                "acceptance_threshold_defined": False,
                "status": "diagnostic_only",
                "note": (
                    "Successful TRF termination does not establish biomechanical "
                    "validity. Inspect global, per-marker and per-segment residuals "
                    "before using reconstructed kinematics."
                ),
            },
            "cache": {**fingerprint, "hit": False},
            "artifacts": {"npz_sha256": npz_sha256},
            "outputs": {"npz": str(npz_path), "summary": str(summary_path)},
        }
    )
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached nonlinear TRF BioBuddy IK from a bioMod and C3D."
    )
    parser.add_argument("--biomod", type=Path, required=True)
    parser.add_argument("--c3d", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--source-name", default="biobuddy")
    parser.add_argument("--biomod-unit-scale-to-m", type=float, default=1.0)
    parser.add_argument("--strip-marker-prefix", action="append", default=[])
    parser.add_argument("--angle-label-regex", default=ANGLE_LABEL_REGEX)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--method", choices=["trf"], default="trf")
    parser.add_argument("--xtol", type=float, default=1e-6)
    parser.add_argument("--ftol", type=float, default=1e-8)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument("--max-nfev", type=int, default=None)
    parser.add_argument("--force", action="store_true")
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
        xtol=args.xtol,
        ftol=args.ftol,
        gtol=args.gtol,
        max_nfev=args.max_nfev,
        cache_dir=args.cache_dir,
        force=args.force,
    )
    marker_error = report["marker_error"]["global"]
    print(f"BioBuddy IK non linéaire TRF: {report['outputs']['npz']}")
    print(
        f"Marqueurs utilisés: {report['markers_used']}/{report['technical_model_markers']}"
    )
    print(
        f"Temps: {report['elapsed_s']:.2f} s | nfev total: {report['nfev']['total']} | "
        f"erreur moyenne: {marker_error['mean_mm']:.3f} mm"
    )


if __name__ == "__main__":
    main()
