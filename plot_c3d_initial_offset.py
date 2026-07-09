#!/usr/bin/env python3
"""Plot raw Motive and Captury C3D marker clouds to inspect initial offsets.

The script intentionally does not perform anatomical registration, yaw correction,
root-offset correction, or model-based alignment.  It only reads marker points from
two C3D files, converts them to millimetres through the existing C3D loader, removes
C3D angle channels from the marker cloud, and overlays the two raw point clouds for
one frame or a short median window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from c3d_trial_viewer import ANGLE_LABEL_REGEX, C3DMarkerData, load_c3d_marker_data

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "local_trials" / "2026-06-30_P6_flat"
DEFAULT_MOTIVE_C3D = DEFAULT_DATA_ROOT / "Motive" / "P6_Static.c3d"
DEFAULT_CAPTURY_C3D = DEFAULT_DATA_ROOT / "Captury" / "Static_P6.c3d"
DEFAULT_OUTPUT = PROJECT_DIR / "out_c3d_initial_offset" / "static_initial_offset.png"

SOURCE_COLORS = {"Motive": "#0ea5e9", "Captury": "#f97316"}


@dataclass(frozen=True)
class OffsetSummary:
    motive_centroid_mm: np.ndarray
    captury_centroid_mm: np.ndarray
    delta_captury_minus_motive_mm: np.ndarray
    euclidean_mm: float
    frame_start: int
    frame_stop: int
    motive_valid_markers: int
    captury_valid_markers: int


def selected_frame_window(n_frames: int, frame: int, window: int) -> tuple[int, int]:
    """Return a clipped half-open frame window around ``frame``."""

    if n_frames <= 0:
        raise ValueError("C3D data contains no frame.")
    frame = int(np.clip(frame, 0, n_frames - 1))
    window = max(1, int(window))
    half = window // 2
    start = max(0, frame - half)
    stop = min(n_frames, start + window)
    start = max(0, stop - window)
    return start, stop


def representative_points(points: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Return marker positions as ``(n_markers, 3)`` for one frame/window."""

    if stop <= start:
        raise ValueError("Frame window is empty.")
    window = points[:, :, start:stop]
    if window.shape[2] == 1:
        return window[:, :, 0].T
    with np.errstate(all="ignore"):
        return np.nanmedian(window, axis=2).T


def finite_marker_points(points: np.ndarray) -> np.ndarray:
    """Keep markers whose XYZ coordinates are finite."""

    mask = np.all(np.isfinite(points), axis=1)
    return points[mask]


def centroid(points: np.ndarray) -> np.ndarray:
    """Return a centroid for finite marker points."""

    finite = finite_marker_points(points)
    if finite.size == 0:
        return np.full(3, np.nan)
    return np.nanmean(finite, axis=0)


def offset_summary(
    motive: C3DMarkerData, captury: C3DMarkerData, frame: int, window: int
) -> OffsetSummary:
    """Compute raw centroid offset for the same frame index in two C3D files."""

    n_frames = min(motive.n_frames, captury.n_frames)
    start, stop = selected_frame_window(n_frames, frame, window)
    motive_points = representative_points(motive.points, start, stop)
    captury_points = representative_points(captury.points, start, stop)
    motive_centroid = centroid(motive_points)
    captury_centroid = centroid(captury_points)
    delta = captury_centroid - motive_centroid
    return OffsetSummary(
        motive_centroid_mm=motive_centroid,
        captury_centroid_mm=captury_centroid,
        delta_captury_minus_motive_mm=delta,
        euclidean_mm=float(np.linalg.norm(delta)),
        frame_start=start,
        frame_stop=stop,
        motive_valid_markers=int(finite_marker_points(motive_points).shape[0]),
        captury_valid_markers=int(finite_marker_points(captury_points).shape[0]),
    )


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    finite = finite_marker_points(points)
    if finite.size == 0:
        return
    mins = np.nanmin(finite, axis=0)
    maxs = np.nanmax(finite, axis=0)
    centre = (mins + maxs) / 2.0
    radius = max(float(np.nanmax(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)


def set_axes_equal_2d(ax, points: np.ndarray, dims: tuple[int, int]) -> None:
    finite = finite_marker_points(points)
    if finite.size == 0:
        return
    subset = finite[:, dims]
    mins = np.nanmin(subset, axis=0)
    maxs = np.nanmax(subset, axis=0)
    centre = (mins + maxs) / 2.0
    radius = max(float(np.nanmax(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_clouds(
    motive_points: np.ndarray,
    captury_points: np.ndarray,
    summary: OffsetSummary,
    motive_path: Path,
    captury_path: Path,
    output: Path | None,
    show: bool,
) -> None:
    all_points = np.vstack(
        [finite_marker_points(motive_points), finite_marker_points(captury_points)]
    )
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Offset initial brut C3D Motive vs Captury", fontsize=14)

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    for name, points in (("Motive", motive_points), ("Captury", captury_points)):
        finite = finite_marker_points(points)
        ax3d.scatter(
            finite[:, 0],
            finite[:, 1],
            finite[:, 2],
            s=22,
            alpha=0.78,
            label=f"{name} ({finite.shape[0]})",
            color=SOURCE_COLORS[name],
        )
    ax3d.scatter(
        *summary.motive_centroid_mm, marker="x", s=110, color=SOURCE_COLORS["Motive"]
    )
    ax3d.scatter(
        *summary.captury_centroid_mm, marker="x", s=110, color=SOURCE_COLORS["Captury"]
    )
    ax3d.plot(
        [summary.motive_centroid_mm[0], summary.captury_centroid_mm[0]],
        [summary.motive_centroid_mm[1], summary.captury_centroid_mm[1]],
        [summary.motive_centroid_mm[2], summary.captury_centroid_mm[2]],
        color="#111827",
        linewidth=2.0,
        label="Delta centroïdes",
    )
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.legend(loc="best")
    set_axes_equal_3d(ax3d, all_points)

    projection_specs = [
        ("XY", (0, 1), ("X (mm)", "Y (mm)")),
        ("XZ", (0, 2), ("X (mm)", "Z (mm)")),
        ("YZ", (1, 2), ("Y (mm)", "Z (mm)")),
    ]
    for index, (title, dims, labels) in enumerate(projection_specs, start=2):
        ax = fig.add_subplot(2, 2, index)
        for name, points in (("Motive", motive_points), ("Captury", captury_points)):
            finite = finite_marker_points(points)
            ax.scatter(
                finite[:, dims[0]],
                finite[:, dims[1]],
                s=20,
                alpha=0.72,
                label=name,
                color=SOURCE_COLORS[name],
            )
        motive_centroid = summary.motive_centroid_mm[list(dims)]
        captury_centroid = summary.captury_centroid_mm[list(dims)]
        ax.scatter(*motive_centroid, marker="x", s=100, color=SOURCE_COLORS["Motive"])
        ax.scatter(*captury_centroid, marker="x", s=100, color=SOURCE_COLORS["Captury"])
        ax.plot(
            [motive_centroid[0], captury_centroid[0]],
            [motive_centroid[1], captury_centroid[1]],
            color="#111827",
            linewidth=1.8,
        )
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True, alpha=0.25)
        set_axes_equal_2d(ax, all_points, dims)

    delta = summary.delta_captury_minus_motive_mm
    text = (
        f"Motive: {motive_path}\n"
        f"Captury: {captury_path}\n"
        f"Frames utilisées: {summary.frame_start + 1}-{summary.frame_stop}\n"
        f"Delta centroïdes Captury - Motive (mm): "
        f"X={delta[0]:.1f}, Y={delta[1]:.1f}, Z={delta[2]:.1f}; "
        f"|delta|={summary.euclidean_mm:.1f} mm"
    )
    fig.text(0.02, 0.02, text, ha="left", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(summary: OffsetSummary) -> None:
    delta = summary.delta_captury_minus_motive_mm
    print(f"Frames utilisées: {summary.frame_start + 1}-{summary.frame_stop}")
    print(f"Marqueurs valides Motive: {summary.motive_valid_markers}")
    print(f"Marqueurs valides Captury: {summary.captury_valid_markers}")
    print(
        "Centroïde Motive (mm): "
        f"X={summary.motive_centroid_mm[0]:.3f}, "
        f"Y={summary.motive_centroid_mm[1]:.3f}, "
        f"Z={summary.motive_centroid_mm[2]:.3f}"
    )
    print(
        "Centroïde Captury (mm): "
        f"X={summary.captury_centroid_mm[0]:.3f}, "
        f"Y={summary.captury_centroid_mm[1]:.3f}, "
        f"Z={summary.captury_centroid_mm[2]:.3f}"
    )
    print(
        "Delta Captury - Motive (mm): "
        f"X={delta[0]:.3f}, Y={delta[1]:.3f}, Z={delta[2]:.3f}, "
        f"|delta|={summary.euclidean_mm:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw Motive and Captury C3D marker clouds to inspect an initial offset."
    )
    parser.add_argument("--motive-c3d", type=Path, default=DEFAULT_MOTIVE_C3D)
    parser.add_argument("--captury-c3d", type=Path, default=DEFAULT_CAPTURY_C3D)
    parser.add_argument("--frame", type=int, default=0, help="0-based frame index.")
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Number of frames around --frame to summarize with a median.",
    )
    parser.add_argument(
        "--angle-label-regex",
        default=ANGLE_LABEL_REGEX,
        help="Regex used to exclude C3D angle channels from marker clouds.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true", help="Open a matplotlib window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motive = load_c3d_marker_data(
        args.motive_c3d, angle_label_regex=args.angle_label_regex
    )
    captury = load_c3d_marker_data(
        args.captury_c3d, angle_label_regex=args.angle_label_regex
    )
    n_frames = min(motive.n_frames, captury.n_frames)
    start, stop = selected_frame_window(n_frames, args.frame, args.window)
    motive_points = representative_points(motive.points, start, stop)
    captury_points = representative_points(captury.points, start, stop)
    summary = offset_summary(motive, captury, args.frame, args.window)
    print_summary(summary)
    plot_clouds(
        motive_points,
        captury_points,
        summary,
        args.motive_c3d,
        args.captury_c3d,
        args.output,
        args.show,
    )
    if args.output is not None:
        print(f"Figure écrite: {args.output}")


if __name__ == "__main__":
    main()
