#!/usr/bin/env python3
"""Flatten a trial-folder capture dataset into Captury/ and Motive/ folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("/Users/mickaelbegon/Downloads/2026-06-30_P6")
DEFAULT_OUTPUT = Path("local_trials/2026-06-30_P6_flat")


def copy_if_exists(
    source: Path, destination: Path, rows: list[dict[str, str]], trial: str, system: str
) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rows.append(
        {
            "trial": trial,
            "system": system,
            "kind": source.suffix.lower().lstrip("."),
            "source": str(source),
            "destination": str(destination),
        }
    )


def flatten_dataset(source_root: Path, output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    captury_out = output_root / "Captury"
    motive_out = output_root / "Motive"
    captury_out.mkdir(parents=True, exist_ok=True)
    motive_out.mkdir(parents=True, exist_ok=True)

    for trial_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        trial = trial_dir.name
        captury_dir = trial_dir / "captury"
        motive_dir = trial_dir / "squelettes"
        for suffix in ("bvh", "fbx", "c3d"):
            copy_if_exists(
                captury_dir / f"P6.{suffix}",
                captury_out / f"{trial}_P6.{suffix}",
                rows,
                trial,
                "Captury",
            )
        if motive_dir.is_dir():
            for source_file in sorted(motive_dir.glob("*")):
                if source_file.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
                    continue
                copy_if_exists(
                    source_file, motive_out / source_file.name, rows, trial, "Motive"
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a flat Captury/Motive dataset folder."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = flatten_dataset(args.source_root, args.output_root)
    manifest = args.output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trial", "system", "kind", "source", "destination"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Copied {len(rows)} file(s) into {args.output_root}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
