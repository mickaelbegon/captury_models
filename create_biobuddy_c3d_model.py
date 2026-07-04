"""CLI wrapper for BioBuddy C3D-driven model creation.

The GUI calls this script instead of importing BioBuddy directly from Tkinter so
model creation can run in a subprocess with live logs and a copyable command.
The scientific/model-building work stays in BioBuddy's
``biobuddy.gui.c3d_model_creation`` module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from motive57_c3d_mapping import prepared_motive57_c3d_folder

DEFAULT_C3D_FOLDER = Path("/Users/mickaelbegon/Downloads/data/Motive")
DEFAULT_OUTPUT = Path("/tmp/motive_57.bioMod")
DEFAULT_PRESET = "motive_57"


def _load_biobuddy_c3d_api() -> dict[str, Any]:
    try:
        from biobuddy.gui.c3d_model_creation import (  # type: ignore
            C3dModelPreset,
            create_model_from_c3d_folder,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BioBuddy ne fournit pas encore biobuddy.gui.c3d_model_creation. "
            "Mets l'environnement captury_biobuddy à jour avec la branche BioBuddy "
            "qui contient l'interface de création de modèle C3D."
        ) from exc

    try:
        from biobuddy.gui.c3d_model_creation import (  # type: ignore
            c3d_model_preset_from_cli_value,
        )
    except ImportError:

        def c3d_model_preset_from_cli_value(value: str | Any) -> Any:
            if isinstance(value, C3dModelPreset):
                return value
            normalized = str(value).strip().lower().replace("-", "_")
            for preset in C3dModelPreset:
                if normalized in {preset.name.lower(), str(preset.value).lower()}:
                    return preset
            raise ValueError(f"Preset C3D BioBuddy non supporté: {value}")

    try:
        from biobuddy.gui.c3d_model_creation import (  # type: ignore
            default_static_virtual_points_for_c3d_model_preset,
        )
    except ImportError:

        def default_static_virtual_points_for_c3d_model_preset(_preset: Any) -> tuple:
            return ()

    return {
        "preset_from_cli": c3d_model_preset_from_cli_value,
        "create_model": create_model_from_c3d_folder,
        "default_static_virtual_points": default_static_virtual_points_for_c3d_model_preset,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a BioBuddy/biorbd model from a folder of C3D files."
    )
    parser.add_argument(
        "c3d_folder",
        nargs="?",
        default=str(DEFAULT_C3D_FOLDER),
        help=f"Folder containing calibration C3D files. Default: {DEFAULT_C3D_FOLDER}",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help="BioBuddy C3D preset, for example motive_57, full_body, lower_limbs.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .bioMod path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--motive-57-mapping-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON role mapping for the Motive 57 template. When provided, "
            "the selected C3D files are linked into a temporary folder with the "
            "filenames expected by BioBuddy."
        ),
    )
    parser.add_argument(
        "--no-default-virtual-points",
        action="store_true",
        help="Do not add preset default virtual points before model generation.",
    )
    parser.add_argument(
        "--with-mesh",
        action="store_true",
        help="Write mesh entries in the generated bioMod.",
    )
    return parser


def create_biobuddy_c3d_model(
    c3d_folder: str | Path,
    *,
    preset: str = DEFAULT_PRESET,
    output: str | Path = DEFAULT_OUTPUT,
    add_default_virtual_points: bool = True,
    with_mesh: bool = False,
    motive_57_mapping_json: str | Path | None = None,
) -> Path:
    """Create a BioBuddy model from calibration C3D files and write a bioMod.

    Parameters mirror the GUI fields and intentionally stay close to the
    BioBuddy API:

    * ``c3d_folder`` is passed to ``create_model_from_c3d_folder``;
    * ``preset`` is resolved with BioBuddy's preset parser when available;
    * ``output`` receives ``result.model.to_biomod(...)``;
    * ``with_mesh`` maps directly to the ``to_biomod`` option.
    * ``motive_57_mapping_json`` can explicitly map Motive 57 static and
      functional trials when the source files do not match BioBuddy glob names.
    """
    api = _load_biobuddy_c3d_api()
    folder_path = Path(c3d_folder).expanduser()
    output_path = Path(output).expanduser()
    if not folder_path.exists():
        raise FileNotFoundError(f"Dossier C3D introuvable: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Le chemin C3D n'est pas un dossier: {folder_path}")

    preset_value = api["preset_from_cli"](preset)
    effective_folder = folder_path
    mapping_context = (
        prepared_motive57_c3d_folder(folder_path, motive_57_mapping_json)
        if motive_57_mapping_json is not None
        else None
    )
    if mapping_context is not None:
        effective_folder = mapping_context.__enter__()
        print(
            f"Mapping Motive 57: {Path(motive_57_mapping_json).expanduser()}",
            flush=True,
        )
        print(f"Dossier C3D préparé: {effective_folder}", flush=True)
    static_virtual_points = (
        api["default_static_virtual_points"](preset_value)
        if add_default_virtual_points
        else ()
    )
    try:
        result = api["create_model"](
            effective_folder,
            preset=preset_value,
            static_virtual_points=static_virtual_points,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.model.to_biomod(str(output_path), with_mesh=with_mesh)
        print(f"Preset: {getattr(result.preset, 'value', result.preset)}", flush=True)
        print(f"bioMod: {output_path}", flush=True)
        return output_path
    finally:
        if mapping_context is not None:
            mapping_context.__exit__(None, None, None)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        create_biobuddy_c3d_model(
            args.c3d_folder,
            preset=args.preset,
            output=args.output,
            add_default_virtual_points=not args.no_default_virtual_points,
            with_mesh=args.with_mesh,
            motive_57_mapping_json=args.motive_57_mapping_json,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
