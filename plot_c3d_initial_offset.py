#!/usr/bin/env python3
"""Plot Motive and Captury C3D marker clouds to inspect initial offsets.

The script intentionally does not perform anatomical registration, yaw correction,
or model-based alignment.  It reads marker points from two C3D files, converts them
to millimetres through the existing C3D loader, removes C3D angle channels from the
marker cloud, and overlays the two point clouds for one frame or a short median
window.

Each source can be prepared independently.  Optional root-translation offsets are
subtracted in the original C3D/source coordinate system first, then any diagnostic
axis transform is applied.  This order mirrors the model pipeline question being
debugged: root translations are interpreted before expressing the data in the
comparison frame.  For raw Captury C3D markers, the default root offset is zero and
the subtraction flag is off because the FBX/BVH static root offset belongs to q
interpretation rather than to marker coordinates.
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
DEFAULT_MOTIVE_ROOT_OFFSET_MM = (0.0, 0.0, 0.0)
DEFAULT_CAPTURY_ROOT_OFFSET_MM = (0.0, 0.0, 0.0)

SOURCE_COLORS = {"Motive": "#0ea5e9", "Captury": "#f97316"}
POINT_TRANSFORM_CHOICES = ("none", "rx_plus_90")
CAPTURY_TRANSFORM_CHOICES = POINT_TRANSFORM_CHOICES


@dataclass(frozen=True)
class SourcePreparationConfig:
    """Preparation options applied independently to one C3D source.

    ``root_offset_mm`` is expressed in the source C3D coordinate system.  When
    ``subtract_root_offset`` is true, it is subtracted before applying
    ``transform``.  ``transform`` is an active diagnostic rotation used to express
    one source in a candidate comparison frame.
    """

    root_offset_mm: np.ndarray
    subtract_root_offset: bool = False
    transform: str = "none"


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


def rotation_x_degrees(angle_degrees: float) -> np.ndarray:
    """Return an active 3D rotation matrix around the X axis."""

    angle = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=float,
    )


def apply_point_transform(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a 3D rotation to C3D points stored as ``(3, n_markers, n_frames)``."""

    return np.einsum("ij,jkf->ikf", rotation, points)


def subtract_root_offset(points: np.ndarray, root_offset_mm: np.ndarray) -> np.ndarray:
    """Subtract a root offset from C3D points before axis conversion."""

    offset = np.asarray(root_offset_mm, dtype=float).reshape(3, 1, 1)
    return np.asarray(points, dtype=float) - offset


def transform_points(points: np.ndarray, mode: str) -> np.ndarray:
    """Apply an optional diagnostic point transform."""

    if mode == "none":
        return points.copy()
    if mode == "rx_plus_90":
        return apply_point_transform(points, rotation_x_degrees(90.0))
    raise ValueError(
        f"Unsupported point transform {mode!r}. Expected one of {POINT_TRANSFORM_CHOICES}."
    )


def prepare_source_points(
    points: np.ndarray,
    config: SourcePreparationConfig,
) -> np.ndarray:
    """Apply one source's diagnostic preparation in pipeline order."""

    prepared = np.asarray(points, dtype=float)
    if config.subtract_root_offset:
        prepared = subtract_root_offset(prepared, config.root_offset_mm)
    return transform_points(prepared, config.transform)


def transform_captury_points(points: np.ndarray, mode: str) -> np.ndarray:
    """Backward-compatible wrapper for older tests/imports."""

    return transform_points(points, mode)


def prepare_c3d_points_for_offset_test(
    points: np.ndarray,
    *,
    root_offset_mm: np.ndarray,
    subtract_offset: bool,
    captury_transform: str = "none",
) -> np.ndarray:
    """Backward-compatible wrapper using Captury-named transform arguments."""

    return prepare_source_points(
        points,
        SourcePreparationConfig(
            root_offset_mm=np.asarray(root_offset_mm, dtype=float),
            subtract_root_offset=subtract_offset,
            transform=captury_transform,
        ),
    )


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
    motive: C3DMarkerData,
    captury: C3DMarkerData,
    frame: int,
    window: int,
    *,
    motive_transform: str = "none",
    captury_transform: str = "none",
    subtract_root_offsets: bool = False,
    motive_subtract_root_offset: bool | None = None,
    captury_subtract_root_offset: bool | None = None,
    motive_root_offset_mm: np.ndarray | None = None,
    captury_root_offset_mm: np.ndarray | None = None,
) -> OffsetSummary:
    """Compute raw centroid offset for the same frame index in two C3D files."""

    n_frames = min(motive.n_frames, captury.n_frames)
    start, stop = selected_frame_window(n_frames, frame, window)
    if motive_subtract_root_offset is None:
        motive_subtract_root_offset = subtract_root_offsets
    if captury_subtract_root_offset is None:
        captury_subtract_root_offset = subtract_root_offsets
    motive_prepared = prepare_source_points(
        motive.points,
        SourcePreparationConfig(
            root_offset_mm=(
                np.zeros(3)
                if motive_root_offset_mm is None
                else np.asarray(motive_root_offset_mm, dtype=float)
            ),
            subtract_root_offset=motive_subtract_root_offset,
            transform=motive_transform,
        ),
    )
    captury_prepared = prepare_source_points(
        captury.points,
        SourcePreparationConfig(
            root_offset_mm=(
                np.zeros(3)
                if captury_root_offset_mm is None
                else np.asarray(captury_root_offset_mm, dtype=float)
            ),
            subtract_root_offset=captury_subtract_root_offset,
            transform=captury_transform,
        ),
    )
    motive_points = representative_points(motive_prepared, start, stop)
    captury_points = representative_points(
        captury_prepared,
        start,
        stop,
    )
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
    motive_transform: str = "none",
    captury_transform: str = "none",
    motive_subtract_root_offset: bool = False,
    captury_subtract_root_offset: bool = False,
    motive_root_offset_mm: np.ndarray | None = None,
    captury_root_offset_mm: np.ndarray | None = None,
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
        f"Transforms: Motive={motive_transform}, Captury={captury_transform}\n"
        f"Offsets racine soustraits avant rotation: "
        f"Motive={motive_subtract_root_offset} "
        f"{np.asarray(motive_root_offset_mm if motive_root_offset_mm is not None else np.zeros(3)).tolist()}, "
        f"Captury={captury_subtract_root_offset} "
        f"{np.asarray(captury_root_offset_mm if captury_root_offset_mm is not None else np.zeros(3)).tolist()}\n"
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


def effective_root_offset_subtractions(args: argparse.Namespace) -> tuple[bool, bool]:
    """Return effective Motive/Captury root-offset flags from parsed CLI args."""

    legacy_global_flag = bool(getattr(args, "legacy_subtract_root_offsets", False))
    return (
        bool(getattr(args, "motive_subtract_root_offset", False) or legacy_global_flag),
        bool(
            getattr(args, "captury_subtract_root_offset", False) or legacy_global_flag
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Motive and Captury C3D marker clouds with independent source preparation to inspect an initial offset."
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
    parser.add_argument(
        "--motive-transform",
        choices=POINT_TRANSFORM_CHOICES,
        default="none",
        help=(
            "Optional Motive diagnostic transform before computing the offset. "
            "The root offset, when enabled, is subtracted before this transform."
        ),
    )
    parser.add_argument(
        "--captury-transform",
        choices=POINT_TRANSFORM_CHOICES,
        default="none",
        help=(
            "Optional Captury diagnostic transform before computing the offset. "
            "Use rx_plus_90 to apply active R(x, +90 deg) after any enabled "
            "Captury root-offset subtraction."
        ),
    )
    parser.add_argument(
        "--motive-subtract-root-offset",
        action="store_true",
        help=(
            "Subtract --motive-root-offset-mm from Motive C3D markers before any "
            "Motive transform."
        ),
    )
    parser.add_argument(
        "--captury-subtract-root-offset",
        action="store_true",
        help=(
            "Subtract --captury-root-offset-mm from Captury C3D markers before any "
            "Captury transform. Keep this disabled for raw Captury C3D marker "
            "diagnostics unless you are explicitly testing a marker-space offset."
        ),
    )
    parser.add_argument(
        "--subtract-root-offsets",
        action="store_true",
        dest="legacy_subtract_root_offsets",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--motive-root-offset-mm",
        nargs=3,
        type=float,
        default=DEFAULT_MOTIVE_ROOT_OFFSET_MM,
        metavar=("X", "Y", "Z"),
        help="Motive root offset to subtract before transforms, in millimetres.",
    )
    parser.add_argument(
        "--captury-root-offset-mm",
        nargs=3,
        type=float,
        default=DEFAULT_CAPTURY_ROOT_OFFSET_MM,
        metavar=("X", "Y", "Z"),
        help=(
            "Captury root offset to subtract before transforms, in millimetres. "
            "Default is zero because raw Captury C3D markers should not receive "
            "the FBX/BVH root offset correction."
        ),
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
    motive_root_offset = np.asarray(args.motive_root_offset_mm, dtype=float)
    captury_root_offset = np.asarray(args.captury_root_offset_mm, dtype=float)
    (
        motive_subtract_root_offset,
        captury_subtract_root_offset,
    ) = effective_root_offset_subtractions(args)
    motive_prepared_points = prepare_source_points(
        motive.points,
        SourcePreparationConfig(
            root_offset_mm=motive_root_offset,
            subtract_root_offset=motive_subtract_root_offset,
            transform=args.motive_transform,
        ),
    )
    transformed_captury_points = prepare_source_points(
        captury.points,
        SourcePreparationConfig(
            root_offset_mm=captury_root_offset,
            subtract_root_offset=captury_subtract_root_offset,
            transform=args.captury_transform,
        ),
    )
    motive_points = representative_points(motive_prepared_points, start, stop)
    captury_points = representative_points(transformed_captury_points, start, stop)
    summary = offset_summary(
        motive,
        captury,
        args.frame,
        args.window,
        motive_transform=args.motive_transform,
        captury_transform=args.captury_transform,
        motive_subtract_root_offset=motive_subtract_root_offset,
        captury_subtract_root_offset=captury_subtract_root_offset,
        motive_root_offset_mm=motive_root_offset,
        captury_root_offset_mm=captury_root_offset,
    )
    if motive_subtract_root_offset or captury_subtract_root_offset:
        print(
            "Offsets racine soustraits avant rotation: "
            f"Motive={motive_subtract_root_offset} {motive_root_offset.tolist()}, "
            f"Captury={captury_subtract_root_offset} {captury_root_offset.tolist()}"
        )
    if args.motive_transform != "none":
        print(f"Transformation Motive appliquée: {args.motive_transform}")
    if args.captury_transform != "none":
        print(f"Transformation Captury appliquée: {args.captury_transform}")
    print_summary(summary)
    plot_clouds(
        motive_points,
        captury_points,
        summary,
        args.motive_c3d,
        args.captury_c3d,
        args.output,
        args.show,
        args.motive_transform,
        args.captury_transform,
        motive_subtract_root_offset,
        captury_subtract_root_offset,
        motive_root_offset,
        captury_root_offset,
    )
    if args.output is not None:
        print(f"Figure écrite: {args.output}")


if __name__ == "__main__":
    main()
