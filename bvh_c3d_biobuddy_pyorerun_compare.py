#!/usr/bin/env python3
"""
BVH/FBX -> bioMod with BioBuddy, q export, C3D marker/angle comparison, and pyorerun animation.

What this script does
---------------------
1. Uses BioBuddy's BVH/FBX parsers from the branch codex/add-fbx-segment-meshes to translate Captury
   skeleton files into biorbd-compatible .bioMod models.
2. Exports generalized coordinates in the same DOF order as the generated biorbd model:
   translations first, then rotations, for each joint/segment.
3. Reads the C3D with ezc3d. C3D point channels detected as angles are kept for comparison, but
   excluded from the marker cloud used for animation.
4. Computes BVH/FBX joint-centre trajectories and appends them to copies of the C3D.
5. Compares BVH rotational generalized coordinates against C3D angle point channels.
6. Expresses C3D markers in local segment frames and writes those local markers back to each bioMod.
7. Optionally animates the generated bioMod models with pyorerun and overlays the C3D markers.

Typical usage
-------------
python bvh_c3d_biobuddy_pyorerun_compare.py \
    --bvh data/unknown.bvh \
    --fbx data/unknown.fbx \
    --c3d data/unknown.c3d \
    --out-dir out_biobuddy_bvh_c3d \
    --animate

Important unit convention
-------------------------
BioBuddy currently writes BVH offsets in the native BVH unit. In many Captury BVH exports,
the ROOT OFFSET is a static rest-pose offset, while ROOT X/Y/Zposition channels are already
absolute laboratory coordinates. If the ROOT OFFSET is also kept in the generated bioMod, it
must be subtracted from the root translational q before animation; otherwise the model is shifted
relative to the C3D markers. This correction is enabled by default.

Beginner reading notes
----------------------
The code uses a few recurring conventions:

- A trajectory is usually a NumPy array. For markers, the shape is
  ``3 x n_markers x n_frames``: the first axis is X/Y/Z, the second is the
  marker number, and the third is time.
- Generalized coordinates ``q`` have shape ``n_q x n_frames``: one row per
  biorbd degree of freedom, one column per time frame.
- Rotations sent to biorbd are always in radians. BVH/FBX and many C3D angle
  exports store angles in degrees, so the script converts them explicitly.
- BVH, FBX and C3D may not use metres internally. The script keeps BVH/FBX q in
  the model native unit, then converts C3D marker positions only when comparing
  or overlaying data.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import re
import struct
import time
import warnings
import zlib
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# These labels are Captury angle channels, not marker trajectories. Keeping this
# list close to the top makes the filtering rule easy to find and edit.
DEFAULT_C3D_ANGLE_LABELS = {
    "RHip",
    "LHip",
    "RKne",
    "LKne",
    "RAnk",
    "LAnk",
    "RSho",
    "LSho",
    "RElb",
    "LElb",
    "RWri",
    "LWri",
}

DEFAULT_RERUN_MARKER_RADIUS_NATIVE = 15.0
DEFAULT_RERUN_WAIT_SECONDS = 2.0
DEFAULT_RERUN_UP_AXIS = "y"
CAPTURY_HAND_MARKER_LABELS = {"Q_LH", "Q_RH", "Q_LW", "Q_RW", "LH", "RH", "LW", "RW"}
CAPTURY_FOOT_MARKER_LABELS = {"Q_LF", "Q_RF", "Q_LA", "Q_RA", "LF", "RF", "LA", "RA"}
HAND_NAME_PATTERN = re.compile(r"(hand|wrist|thumb|index|middle|ring|pinky)", re.IGNORECASE)
FOOT_NAME_PATTERN = re.compile(r"(foot|toe|ankle|heel)", re.IGNORECASE)


# =============================================================================
# Small utilities
# =============================================================================


def require_ezc3d():
    try:
        import ezc3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ezc3d is required. Install it with `conda install -c conda-forge ezc3d`."
        ) from exc
    return ezc3d


def require_biobuddy():
    try:
        from biobuddy import BiomechanicalModelReal  # type: ignore
        from biobuddy.model_parser.bvh import BvhModelParser  # type: ignore
        from biobuddy.model_parser.fbx import FbxModelParser  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "BioBuddy with BVH/FBX support is required. Install the branch with:\n"
            "pip install git+https://github.com/mickaelbegon/biobuddy.git@codex/add-fbx-segment-meshes"
        ) from exc
    return BiomechanicalModelReal, BvhModelParser, FbxModelParser


def require_biorbd():
    try:
        import biorbd  # type: ignore
    except ImportError as exc:
        raise ImportError("biorbd is required for segment transforms and local marker tests.") from exc
    return biorbd


def require_pyorerun():
    try:
        from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyorerun is required only for --animate. Install with `conda install -c conda-forge pyorerun rerun-sdk`."
        ) from exc
    return BiorbdModel, PhaseRerun, PyoMarkers


def get_c3d_param(c3d: dict, group: str, name: str, default: Any = None) -> Any:
    try:
        return c3d["parameters"][group][name]["value"]
    except KeyError:
        return default


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [str(value).strip()]
    return [str(v).strip() for v in value]


def c3d_point_unit_scale_to_m(c3d: dict) -> float:
    # C3D files store their unit as metadata. We convert through metres so BVH,
    # FBX and C3D can be compared even when their native units differ.
    units = as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))
    unit = units[0].strip().lower() if units else ""
    if unit in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 0.001
    if unit in {"m", "meter", "meters", "metre", "metres"}:
        return 1.0
    # Most mocap/Captury C3D marker trajectories are in mm. This fallback is explicit in the report.
    return 0.001


def c3d_time_vector(c3d: dict) -> np.ndarray:
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [0])
    rate = float(rate_value[0] if isinstance(rate_value, (list, tuple, np.ndarray)) else rate_value)
    if rate <= 0:
        raise ValueError("Invalid or missing C3D POINT:RATE.")
    n_frames = int(c3d["data"]["points"].shape[2])
    return np.arange(n_frames, dtype=float) / rate


def interpolate_array(data: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Interpolate arrays shaped (..., n_source_frames) to (..., n_target_frames)."""
    if source_time.shape == target_time.shape and np.allclose(source_time, target_time):
        return data.copy()

    # NumPy interpolation is one-dimensional. Flatten every non-time dimension,
    # interpolate each signal, then reshape back to the original marker/q layout.
    flat = data.reshape((-1, data.shape[-1]))
    out = np.empty((flat.shape[0], target_time.shape[0]), dtype=float)
    for i, y in enumerate(flat):
        finite = np.isfinite(y)
        if finite.sum() < 2:
            out[i, :] = np.nan
        else:
            out[i, :] = np.interp(target_time, source_time[finite], y[finite])
    return out.reshape((*data.shape[:-1], target_time.shape[0]))


def rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    axis = axis.lower()[0]
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unsupported rotation axis: {axis}")


def sanitize_biomod_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned


def strip_known_marker_prefix(name: str) -> str:
    for prefix in ("BVHJC_", "FBXJC_", "C3D_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def is_hand_display_name(name: str) -> bool:
    base = strip_known_marker_prefix(str(name)).strip()
    if base in CAPTURY_HAND_MARKER_LABELS:
        return True
    return bool(HAND_NAME_PATTERN.search(base))


def is_foot_display_name(name: str) -> bool:
    base = strip_known_marker_prefix(str(name)).strip()
    if base in CAPTURY_FOOT_MARKER_LABELS:
        return True
    return bool(FOOT_NAME_PATTERN.search(base))


def is_hidden_display_name(name: str, hide_hands: bool, hide_feet: bool) -> bool:
    return (hide_hands and is_hand_display_name(name)) or (hide_feet and is_foot_display_name(name))


def is_rotation_q_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith("rotation") or bool(re.search(r"_rot[xyz]$", lower))


def is_translation_q_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith("position") or bool(re.search(r"_trans[xyz]$", lower))


def q_channel_units(q_names: list[str]) -> list[str]:
    """Return the biorbd unit used by each generalized coordinate channel."""
    units: list[str] = []
    for name in q_names:
        if is_rotation_q_name(name):
            units.append("rad")
        elif is_translation_q_name(name):
            units.append("native_length_unit")
        else:
            units.append("unknown")
    return units


def rotation_q_indices(q_names: list[str]) -> list[int]:
    return [i for i, name in enumerate(q_names) if is_rotation_q_name(name)]


def subtract_static_root_offset_from_q(
    q: np.ndarray,
    q_names: list[str],
    root_name: str | None,
    root_offset: np.ndarray,
    apply_root_offset_correction: bool,
) -> np.ndarray:
    """Subtract the static root RT offset from root translation q channels when requested."""
    corrected = q.copy()
    if not apply_root_offset_correction or root_name is None:
        return corrected

    root_offset = np.asarray(root_offset, dtype=float).reshape(3)
    # BioBuddy's newer parsers name translations Hips_transX, while the legacy
    # fallback used names like Hips_Xposition. Support both so old outputs can
    # still be inspected.
    axis_to_index = {"X": 0, "Y": 1, "Z": 2}
    for axis, axis_index in axis_to_index.items():
        supported_names = {
            f"{root_name}_trans{axis}",
            f"{root_name}_{axis}position",
        }
        for q_index, q_name in enumerate(q_names):
            if q_name in supported_names:
                corrected[q_index, :] = corrected[q_index, :] - root_offset[axis_index]
    return corrected


def unwrap_rotation_q(q: np.ndarray, q_names: list[str]) -> np.ndarray:
    """Unwrap rotational generalized coordinates in radians, channel by channel.

    biorbd expects rotational q in radians. BVH and FBX store Euler channels in
    degrees and may wrap at +/-180 or 0/360 degrees. Unwrapping after conversion
    to radians preserves the intended continuous trajectory without changing the
    represented pose modulo 2*pi.
    """
    q_unwrapped = q.copy()
    for idx in rotation_q_indices(q_names):
        values = q_unwrapped[idx, :]
        finite = np.isfinite(values)
        if finite.sum() > 1:
            q_unwrapped[idx, finite] = np.unwrap(values[finite])
    return q_unwrapped


def q_unwrap_summary(q_before: np.ndarray, q_after: np.ndarray, q_names: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx in rotation_q_indices(q_names):
        delta = q_after[idx, :] - q_before[idx, :]
        finite = np.isfinite(delta)
        if not finite.any():
            continue
        max_abs_delta = float(np.nanmax(np.abs(delta[finite])))
        if max_abs_delta > 1e-12:
            rows.append(
                {
                    "q_name": q_names[idx],
                    "max_abs_unwrap_delta_rad": max_abs_delta,
                    "max_abs_unwrap_delta_deg": math.degrees(max_abs_delta),
                }
            )
    return {
        "applied": True,
        "rotation_channels": len(rotation_q_indices(q_names)),
        "channels_changed": len(rows),
        "changed_channels": rows,
    }


# =============================================================================
# BVH through BioBuddy
# =============================================================================


@dataclass
class BvhRuntimeData:
    # Small container for all BVH animation data used later in the pipeline.
    # Dataclasses avoid passing long tuples where the meaning of each element is
    # easy to forget.
    parser: Any
    joint_names: list[str]
    channel_entries_file_order: list[tuple[str, str]]
    channel_entries_q_order: list[tuple[str, str]]
    q: np.ndarray
    q_names: list[str]
    time: np.ndarray
    root_offset_correction_applied: bool = False
    root_offset_native: np.ndarray | None = None
    q_units: list[str] = field(default_factory=list)
    unwrap_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class FbxRecord:
    name: str
    properties: list[Any] = field(default_factory=list)
    children: list["FbxRecord"] = field(default_factory=list)


@dataclass
class FbxRuntimeData:
    # Same idea as BvhRuntimeData, but for the FBX parser and animation.
    parser: Any
    joint_names: list[str]
    q: np.ndarray
    q_names: list[str]
    time: np.ndarray
    root_offset_correction_applied: bool = False
    root_offset_native: np.ndarray | None = None
    q_units: list[str] = field(default_factory=list)
    unwrap_summary: dict[str, Any] = field(default_factory=dict)


def iter_bvh_joints_depth_first(joint: Any) -> Iterable[Any]:
    yield joint
    for child in joint.children:
        yield from iter_bvh_joints_depth_first(child)


def collect_bvh_channels(parser: Any) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return joint names, raw file channel order, and BioBuddy/biorbd q order.

    BioBuddy converts each joint channels to a translation sequence and a rotation sequence.
    In biorbd, translations are listed before rotations for a segment. Therefore, q_order is
    positions first, then rotations, for each joint.
    """
    if parser.root is None:
        raise ValueError("BioBuddy BVH parser has no root.")

    joint_names: list[str] = []
    file_order: list[tuple[str, str]] = []
    q_order: list[tuple[str, str]] = []

    for joint in iter_bvh_joints_depth_first(parser.root):
        joint_names.append(joint.name)
        for channel in joint.channels:
            file_order.append((joint.name, channel))
        for channel in joint.channels:
            if channel.lower().endswith("position"):
                q_order.append((joint.name, channel))
        for channel in joint.channels:
            if channel.lower().endswith("rotation"):
                q_order.append((joint.name, channel))

    return joint_names, file_order, q_order


def collect_fbx_joint_names_depth_first(parser: Any) -> list[str]:
    names: list[str] = []

    def recurse(node_id: int) -> None:
        node = parser.skeleton_nodes[node_id]
        names.append(node.name)
        for child_id in node.children_ids:
            recurse(child_id)

    for root_id in parser.root_ids:
        recurse(root_id)
    return names


def runtime_q_from_parsed_animation(
    parsed_animation: Any,
    q_names: list[str],
    root_name: str | None,
    root_offset: np.ndarray,
    apply_root_offset_correction: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # BioBuddy gives q in biorbd order. This helper applies the two project-level
    # corrections that are not BioBuddy's job: root offset policy and angle unwrap.
    q_raw = np.asarray(parsed_animation.q, dtype=float)
    q_root_corrected = subtract_static_root_offset_from_q(
        q=q_raw,
        q_names=q_names,
        root_name=root_name,
        root_offset=root_offset,
        apply_root_offset_correction=apply_root_offset_correction,
    )
    q = unwrap_rotation_q(q_root_corrected, q_names)
    return q, q_root_corrected, q_unwrap_summary(q_root_corrected, q, q_names)


def parse_fbx_records(fbx_path: Path) -> dict[str, Any]:
    """Parse enough binary FBX structure to read animation curves and mesh vertices."""
    data = fbx_path.read_bytes()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError("Only binary FBX files are supported.")

    version = struct.unpack_from("<I", data, 23)[0]

    def parse_property(cursor: int) -> tuple[Any, int]:
        property_type = chr(data[cursor])
        cursor += 1
        if property_type == "S":
            length = struct.unpack_from("<I", data, cursor)[0]
            cursor += 4
            value = data[cursor : cursor + length].decode("utf-8", errors="ignore")
            return value, cursor + length
        if property_type == "R":
            length = struct.unpack_from("<I", data, cursor)[0]
            cursor += 4
            return data[cursor : cursor + length], cursor + length
        if property_type == "L":
            return struct.unpack_from("<q", data, cursor)[0], cursor + 8
        if property_type == "I":
            return struct.unpack_from("<i", data, cursor)[0], cursor + 4
        if property_type == "D":
            return struct.unpack_from("<d", data, cursor)[0], cursor + 8
        if property_type == "F":
            return struct.unpack_from("<f", data, cursor)[0], cursor + 4
        if property_type == "C":
            return bool(data[cursor]), cursor + 1
        if property_type == "Y":
            return struct.unpack_from("<h", data, cursor)[0], cursor + 2
        if property_type in "fdiilbc":
            array_length, encoding, compressed_length = struct.unpack_from("<III", data, cursor)
            cursor += 12
            raw = data[cursor : cursor + compressed_length]
            cursor += compressed_length
            if encoding == 1:
                raw = zlib.decompress(raw)
            elif encoding != 0:
                raise NotImplementedError(f"Unsupported FBX array encoding {encoding}.")
            dtype_by_type = {
                "f": np.dtype("<f4"),
                "d": np.dtype("<f8"),
                "i": np.dtype("<i4"),
                "l": np.dtype("<i8"),
                "b": np.dtype("?"),
                "c": np.dtype("?"),
            }
            dtype = dtype_by_type[property_type]
            return np.frombuffer(raw, dtype=dtype, count=array_length).copy(), cursor
        raise NotImplementedError(f"Unsupported FBX property type '{property_type}'.")

    def parse_record(start_offset: int) -> tuple[FbxRecord | None, int]:
        if version >= 7500:
            end_offset, prop_count, _ = struct.unpack_from("<QQQ", data, start_offset)
            cursor = start_offset + 24
            null_record_size = 25
        else:
            end_offset, prop_count, _ = struct.unpack_from("<III", data, start_offset)
            cursor = start_offset + 12
            null_record_size = 13
        name_length = data[cursor]
        cursor += 1
        if end_offset == 0:
            return None, start_offset + null_record_size
        name = data[cursor : cursor + name_length].decode("utf-8", errors="ignore")
        cursor += name_length
        properties = []
        for _ in range(prop_count):
            value, cursor = parse_property(cursor)
            properties.append(value)
        children = []
        while cursor < end_offset - null_record_size:
            child, cursor = parse_record(cursor)
            if child is not None:
                children.append(child)
        return FbxRecord(name=name, properties=properties, children=children), end_offset

    offset = 27
    records = []
    while offset < len(data):
        record, offset = parse_record(offset)
        if record is None:
            break
        records.append(record)
    return {"version": version, "records": records}


def fbx_top_record(fbx_tree: dict[str, Any], name: str) -> FbxRecord | None:
    return next((record for record in fbx_tree["records"] if record.name == name), None)


def fbx_properties70_dict(record: FbxRecord) -> dict[str, list[Any]]:
    properties_record = next((child for child in record.children if child.name == "Properties70"), None)
    if properties_record is None:
        return {}
    out: dict[str, list[Any]] = {}
    for child in properties_record.children:
        if child.name == "P" and len(child.properties) >= 5:
            out[str(child.properties[0])] = child.properties[4:]
    return out


def fbx_vector3(properties: dict[str, list[Any]], property_name: str) -> np.ndarray:
    values = properties.get(property_name, [0.0, 0.0, 0.0])
    return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)


def clean_fbx_name(raw_name: str) -> str:
    return str(raw_name).split("\x00", maxsplit=1)[0].split(":", maxsplit=1)[-1]


def extract_fbx_model_defaults(fbx_tree: dict[str, Any]) -> dict[int, dict[str, Any]]:
    objects = fbx_top_record(fbx_tree, "Objects")
    if objects is None:
        return {}
    defaults: dict[int, dict[str, Any]] = {}
    for record in objects.children:
        if record.name != "Model" or len(record.properties) < 3:
            continue
        props = fbx_properties70_dict(record)
        node_id = int(record.properties[0])
        defaults[node_id] = {
            "name": clean_fbx_name(str(record.properties[1])),
            "node_type": str(record.properties[2]),
            "translation": fbx_vector3(props, "Lcl Translation"),
            "rotation": fbx_vector3(props, "Lcl Rotation"),
            "pre_rotation": fbx_vector3(props, "PreRotation"),
        }
    return defaults


def extract_fbx_animation_curves(fbx_tree: dict[str, Any]) -> tuple[dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]], np.ndarray]:
    objects = fbx_top_record(fbx_tree, "Objects")
    connections = fbx_top_record(fbx_tree, "Connections")
    if objects is None or connections is None:
        return {}, np.zeros(1)

    curves: dict[int, dict[str, np.ndarray]] = {}
    curve_nodes: set[int] = set()
    for record in objects.children:
        if record.name == "AnimationCurve" and record.properties:
            curve_id = int(record.properties[0])
            key_time = np.array([], dtype=np.int64)
            key_value = np.array([], dtype=float)
            for child in record.children:
                if child.name == "KeyTime" and child.properties:
                    key_time = np.asarray(child.properties[0], dtype=np.int64)
                elif child.name == "KeyValueFloat" and child.properties:
                    key_value = np.asarray(child.properties[0], dtype=float)
            curves[curve_id] = {"time": key_time, "value": key_value}
        elif record.name == "AnimationCurveNode" and record.properties:
            curve_nodes.add(int(record.properties[0]))

    curve_to_node: dict[int, tuple[int, str]] = {}
    node_to_model: dict[int, tuple[int, str]] = {}
    for connection in connections.children:
        if connection.name != "C" or len(connection.properties) < 3:
            continue
        child_id = int(connection.properties[1])
        parent_id = int(connection.properties[2])
        prop = str(connection.properties[3]) if len(connection.properties) >= 4 else ""
        if child_id in curves and parent_id in curve_nodes and prop.startswith("d|"):
            component = prop.split("|")[-1].upper() if prop else ""
            curve_to_node[child_id] = (parent_id, component)
        elif child_id in curve_nodes and prop in {"Lcl Rotation", "Lcl Translation"}:
            node_to_model[child_id] = (parent_id, prop)

    channel_curves: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    all_times: list[np.ndarray] = []
    for curve_id, (curve_node_id, component) in curve_to_node.items():
        if curve_node_id not in node_to_model or component not in {"X", "Y", "Z"}:
            continue
        model_id, property_name = node_to_model[curve_node_id]
        curve = curves[curve_id]
        if curve["time"].size == 0 or curve["value"].size == 0:
            continue
        n = min(curve["time"].shape[0], curve["value"].shape[0])
        times = curve["time"][:n]
        values = curve["value"][:n]
        channel_curves[(model_id, property_name, component)] = (times, values)
        all_times.append(times)

    if not all_times:
        return channel_curves, np.zeros(1)
    unique_times = np.unique(np.concatenate(all_times))
    return channel_curves, unique_times


def interpolate_fbx_curve(
    curve: tuple[np.ndarray, np.ndarray] | None,
    target_ticks: np.ndarray,
    default_value: float,
) -> np.ndarray:
    if curve is None:
        return np.full(target_ticks.shape, default_value, dtype=float)
    ticks, values = curve
    if ticks.size == 1:
        return np.full(target_ticks.shape, float(values[0]), dtype=float)
    return np.interp(target_ticks.astype(float), ticks.astype(float), values.astype(float))


def extract_q_from_fbx_parser(
    parser: Any,
    fbx_path: Path,
    apply_root_offset_correction: bool = True,
) -> FbxRuntimeData:
    """Extract FBX generalized coordinates through BioBuddy, with a legacy parser fallback."""
    if hasattr(parser, "to_q"):
        # Preferred path with the current BioBuddy branch: no custom FBX curve
        # mapping here, just ask BioBuddy for the animation in model q order.
        parsed_animation = parser.to_q()
        q_names = [str(name) for name in parsed_animation.dof_names]
        joint_names = collect_fbx_joint_names_depth_first(parser)
        root_node = parser.skeleton_nodes[parser.root_ids[0]] if parser.root_ids else None
        root_name = root_node.name if root_node is not None else None
        root_offset = np.asarray(root_node.translation, dtype=float) if root_node is not None else np.zeros(3)
        q, _, unwrap_summary = runtime_q_from_parsed_animation(
            parsed_animation=parsed_animation,
            q_names=q_names,
            root_name=root_name,
            root_offset=root_offset,
            apply_root_offset_correction=apply_root_offset_correction,
        )
        return FbxRuntimeData(
            parser=parser,
            joint_names=joint_names,
            q=q,
            q_names=q_names,
            time=np.asarray(parsed_animation.time, dtype=float),
            root_offset_correction_applied=apply_root_offset_correction,
            root_offset_native=root_offset,
            q_units=q_channel_units(q_names),
            unwrap_summary=unwrap_summary,
        )

    fbx_tree = parse_fbx_records(fbx_path)
    defaults = extract_fbx_model_defaults(fbx_tree)
    curves, ticks = extract_fbx_animation_curves(fbx_tree)
    ticks_per_second = 46186158000.0
    time = (ticks.astype(float) - float(ticks[0])) / ticks_per_second if ticks.size else np.zeros(1)

    q_rows: list[np.ndarray] = []
    q_names: list[str] = []
    joint_names: list[str] = []
    root_ids = set(parser.root_ids)
    axis_index = {"X": 0, "Y": 1, "Z": 2}

    def append_node(node_id: int) -> None:
        node = parser.skeleton_nodes[node_id]
        joint_names.append(node.name)
        node_defaults = defaults.get(
            node_id,
            {
                "translation": np.asarray(node.translation, dtype=float),
                "rotation": np.zeros(3),
            },
        )
        translation_default = np.asarray(node_defaults["translation"], dtype=float)
        rotation_default = np.asarray(node_defaults["rotation"], dtype=float)

        if node_id in root_ids:
            for axis in "XYZ":
                values = interpolate_fbx_curve(
                    curves.get((node_id, "Lcl Translation", axis)),
                    ticks,
                    translation_default[axis_index[axis]],
                )
                if apply_root_offset_correction:
                    values = values - translation_default[axis_index[axis]]
                q_rows.append(values)
                q_names.append(f"{node.name}_{axis}position")

        for axis in "XYZ":
            values_deg = interpolate_fbx_curve(
                curves.get((node_id, "Lcl Rotation", axis)),
                ticks,
                rotation_default[axis_index[axis]],
            )
            values_deg = values_deg - rotation_default[axis_index[axis]]
            q_rows.append(np.deg2rad(values_deg))
            q_names.append(f"{node.name}_{axis}rotation")

        for child_id in node.children_ids:
            append_node(child_id)

    for root_id in parser.root_ids:
        append_node(root_id)

    q_raw_units = np.vstack(q_rows) if q_rows else np.zeros((0, time.shape[0]))
    q = unwrap_rotation_q(q_raw_units, q_names)
    root_offset = (
        np.asarray(parser.skeleton_nodes[parser.root_ids[0]].translation, dtype=float)
        if parser.root_ids
        else np.zeros(3)
    )
    return FbxRuntimeData(
        parser=parser,
        joint_names=joint_names,
        q=q,
        q_names=q_names,
        time=time,
        root_offset_correction_applied=apply_root_offset_correction,
        root_offset_native=root_offset,
        q_units=q_channel_units(q_names),
        unwrap_summary=q_unwrap_summary(q_raw_units, q, q_names),
    )


def polygon_vertex_indices_to_faces(polygon_vertex_indices: np.ndarray) -> list[list[int]]:
    """Decode FBX polygon indices into zero-based faces.

    FBX encodes the final vertex of each polygon as ``-index - 1``.
    """
    faces: list[list[int]] = []
    current: list[int] = []
    for raw_index in polygon_vertex_indices.astype(int).tolist():
        if raw_index < 0:
            current.append(-raw_index - 1)
            if len(current) >= 3:
                faces.append(current)
            current = []
        else:
            current.append(raw_index)
    return faces


def triangulate_faces(faces: list[list[int]]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        if len(face) == 3:
            triangles.append((face[0], face[1], face[2]))
        elif len(face) > 3:
            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))
    return triangles


def extract_fbx_meshes(fbx_tree: dict[str, Any]) -> list[dict[str, Any]]:
    objects = fbx_top_record(fbx_tree, "Objects")
    if objects is None:
        return []
    meshes: list[dict[str, Any]] = []
    for record in objects.children:
        if record.name != "Geometry" or len(record.properties) < 3 or str(record.properties[2]) != "Mesh":
            continue
        vertices_record = next((child for child in record.children if child.name == "Vertices"), None)
        polygon_record = next((child for child in record.children if child.name == "PolygonVertexIndex"), None)
        if vertices_record is None or polygon_record is None or not vertices_record.properties or not polygon_record.properties:
            continue
        vertex_values = np.asarray(vertices_record.properties[0], dtype=float)
        polygon_values = np.asarray(polygon_record.properties[0], dtype=int)
        if vertex_values.size < 3 or polygon_values.size < 3:
            continue
        vertices = vertex_values.reshape((-1, 3)).T
        faces = polygon_vertex_indices_to_faces(polygon_values)
        triangles = triangulate_faces(faces)
        if triangles:
            meshes.append(
                {
                    "name": clean_fbx_name(str(record.properties[1])),
                    "vertices": vertices,
                    "triangles": triangles,
                }
            )
    return meshes


def write_fbx_mesh_obj(
    fbx_tree: dict[str, Any],
    obj_path: Path,
    root_translation: np.ndarray,
    max_vertices: int,
) -> dict[str, Any]:
    meshes = extract_fbx_meshes(fbx_tree)
    if not meshes:
        return {"mesh_file": None, "mesh_vertices": 0, "mesh_faces": 0}

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offset = 0
    kept_vertices_total = 0
    kept_faces_total = 0
    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# Generated from FBX Geometry/Vertices and PolygonVertexIndex\n")
        for mesh in meshes:
            vertices = np.asarray(mesh["vertices"], dtype=float) - root_translation.reshape(3, 1)
            triangles = list(mesh["triangles"])
            if vertices.shape[1] > max_vertices > 0:
                # Keep a deterministic subset of vertices and only faces fully
                # contained in that subset. Use 0 for all vertices/faces.
                kept = np.linspace(0, vertices.shape[1] - 1, max_vertices, dtype=int)
                kept_set = set(int(i) for i in kept)
                remap = {int(old): new for new, old in enumerate(kept)}
                vertices = vertices[:, kept]
                triangles = [
                    (remap[a], remap[b], remap[c])
                    for a, b, c in triangles
                    if a in kept_set and b in kept_set and c in kept_set
                ]
            f.write(f"o {sanitize_biomod_name(str(mesh['name']), 'fbx_mesh')}\n")
            for vertex in vertices.T:
                f.write(f"v {vertex[0]:0.8f} {vertex[1]:0.8f} {vertex[2]:0.8f}\n")
            for tri in triangles:
                a, b, c = (vertex_offset + tri[0] + 1, vertex_offset + tri[1] + 1, vertex_offset + tri[2] + 1)
                f.write(f"f {a} {b} {c}\n")
            vertex_offset += vertices.shape[1]
            kept_vertices_total += int(vertices.shape[1])
            kept_faces_total += len(triangles)
    return {
        "mesh_file": str(obj_path),
        "mesh_vertices": kept_vertices_total,
        "mesh_faces": kept_faces_total,
    }


def append_fbx_mesh_file_to_biomod(
    biomod_path: Path,
    fbx_tree: dict[str, Any],
    parent_name: str | None,
    root_translation: np.ndarray,
    max_vertices: int,
) -> dict[str, Any]:
    if parent_name is None:
        return {"mesh_file": None, "mesh_vertices": 0, "mesh_faces": 0, "mesh_parent": parent_name}

    mesh_dir = biomod_path.parent / "meshes"
    obj_path = mesh_dir / "unknown_fbx_mesh.obj"
    mesh_report = write_fbx_mesh_obj(
        fbx_tree=fbx_tree,
        obj_path=obj_path,
        root_translation=root_translation,
        max_vertices=max_vertices,
    )
    if mesh_report["mesh_vertices"] == 0 or mesh_report["mesh_faces"] == 0:
        mesh_report["mesh_parent"] = parent_name
        return mesh_report

    text = biomod_path.read_text(encoding="utf-8")
    segment_header = f"segment\t{parent_name}"
    start = text.find(segment_header)
    if start < 0:
        segment_header = f"segment {parent_name}"
        start = text.find(segment_header)
    if start < 0:
        mesh_report["mesh_parent"] = parent_name
        return mesh_report
    end = text.find("endsegment", start)
    if end < 0:
        mesh_report["mesh_parent"] = parent_name
        return mesh_report
    meshfile_line = "\tmeshfile\tmeshes/unknown_fbx_mesh.obj\n"
    meshscale_line = "\tmeshscale\t1\t1\t1\n"
    text = text[:end] + meshfile_line + meshscale_line + text[end:]
    biomod_path.write_text(text, encoding="utf-8")
    mesh_report["mesh_parent"] = parent_name
    return mesh_report


def read_ascii_ply_mesh(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Read the simple ASCII PLY files emitted by BioBuddy for segment meshes."""
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line != "ply":
            raise ValueError(f"{path} is not a PLY file.")

        vertex_count = 0
        face_count = 0
        is_ascii = False
        for line in f:
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                is_ascii = True
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("element face "):
                face_count = int(stripped.split()[-1])
            elif stripped == "end_header":
                break

        if not is_ascii:
            raise ValueError(f"{path} is not an ASCII PLY file.")

        vertices = np.zeros((vertex_count, 3), dtype=float)
        for i in range(vertex_count):
            parts = f.readline().split()
            if len(parts) < 3:
                raise ValueError(f"{path} has an invalid vertex row at index {i}.")
            vertices[i, :] = [float(parts[0]), float(parts[1]), float(parts[2])]

        faces: list[list[int]] = []
        for i in range(face_count):
            parts = f.readline().split()
            if not parts:
                raise ValueError(f"{path} has an invalid face row at index {i}.")
            n_vertices = int(parts[0])
            if len(parts) < n_vertices + 1:
                raise ValueError(f"{path} has an incomplete face row at index {i}.")
            faces.append([int(index) for index in parts[1 : n_vertices + 1]])

    return vertices, faces


def face_normal(vertices: np.ndarray, triangle: tuple[int, int, int]) -> np.ndarray:
    a, b, c = (vertices[index] for index in triangle)
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros(3)
    return normal / norm


def triangulate_mesh_faces(faces: list[list[int]]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
    return triangles


def vertex_normals(vertices: np.ndarray, triangles: list[tuple[int, int, int]]) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=float)
    for triangle in triangles:
        normal = face_normal(vertices, triangle)
        for index in triangle:
            normals[index] += normal
    norms = np.linalg.norm(normals, axis=1)
    nonzero = norms > 0
    normals[nonzero] /= norms[nonzero, None]
    return normals


def write_ascii_vtp_mesh(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> int:
    """Write a triangular ASCII VTP mesh in the shape expected by pyorerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    triangles = triangulate_mesh_faces(faces)
    normals = vertex_normals(vertices, triangles)
    offsets = [3 * (i + 1) for i in range(len(triangles))]

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{vertices.shape[0]}" NumberOfPolys="{len(triangles)}">\n')
        f.write('      <PointData Normals="Normals">\n')
        f.write('        <DataArray type="Float32" Name="Normals" NumberOfComponents="3" format="ascii">\n')
        for normal in normals:
            f.write(f"          {normal[0]:.8g} {normal[1]:.8g} {normal[2]:.8g}\n")
        f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for vertex in vertices:
            f.write(f"          {vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Polys>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for triangle in triangles:
            f.write(f"          {triangle[0]} {triangle[1]} {triangle[2]}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        if offsets:
            f.write("          " + " ".join(str(offset) for offset in offsets) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Polys>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")
    return len(triangles)


def convert_biobuddy_ply_meshes_to_vtp(mesh_dir: Path) -> dict[str, Any]:
    """Convert BioBuddy's per-segment PLY meshes to VTP files accepted by pyorerun."""
    mesh_entries: list[dict[str, Any]] = []
    total_vertices = 0
    total_faces = 0
    total_triangles = 0
    for ply_file in sorted(mesh_dir.glob("*.ply")):
        vertices, faces = read_ascii_ply_mesh(ply_file)
        vtp_file = ply_file.with_suffix(".vtp")
        triangles = write_ascii_vtp_mesh(vtp_file, vertices=vertices, faces=faces)
        total_vertices += int(vertices.shape[0])
        total_faces += len(faces)
        total_triangles += triangles
        mesh_entries.append(
            {
                "source_mesh_file": str(ply_file),
                "mesh_file": str(vtp_file),
                "mesh_vertices": int(vertices.shape[0]),
                "mesh_faces": len(faces),
                "mesh_triangles": triangles,
            }
        )

    return {
        "mesh_source": "biobuddy_segment_vtp_from_ply",
        "mesh_directory": str(mesh_dir),
        "mesh_files": mesh_entries,
        "mesh_file_count": len(mesh_entries),
        "mesh_vertices": total_vertices,
        "mesh_faces": total_faces,
        "mesh_triangles": total_triangles,
        "mesh_parent": "per_segment",
    }


def normalize_biomod_meshfile_paths(biomod_path: Path) -> None:
    """Make generated meshfile paths relative to the bioMod location for biorbd."""
    text = biomod_path.read_text(encoding="utf-8")
    lines: list[str] = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("meshfile"):
            lines.append(line)
            continue

        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            lines.append(line)
            continue
        mesh_path = Path(parts[1])
        if mesh_path.is_absolute():
            try:
                mesh_path = Path(os.path.relpath(mesh_path, start=biomod_path.parent))
            except ValueError:
                lines.append(line)
                continue
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}meshfile\t{mesh_path.as_posix()}")
            changed = True
        else:
            lines.append(line)

    if changed:
        biomod_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_generated_fbx_meshes(mesh_dir: Path) -> None:
    if not mesh_dir.exists():
        return
    for pattern in ("*.ply", "*.stl", "*.vtp", "unknown_fbx_mesh.obj"):
        for mesh_file in mesh_dir.glob(pattern):
            if mesh_file.is_file():
                mesh_file.unlink()


def build_biomod_from_bvh_with_biobuddy(
    bvh_path: Path,
    biomod_path: Path,
    add_joint_centre_markers: bool = True,
) -> tuple[Any, Any]:
    """Create a bioMod model from a BVH file using BioBuddy's BVH parser."""
    BiomechanicalModelReal, BvhModelParser, _ = require_biobuddy()

    # Build the model using the public API described in BioBuddy's README.
    try:
        model = BiomechanicalModelReal().from_bvh(filepath=str(bvh_path))
    except AttributeError:
        # Fallback for development versions where the parser exists but the convenience method is not exposed yet.
        parser_for_model = BvhModelParser(str(bvh_path))
        model = parser_for_model.to_real()

    model.to_biomod(str(biomod_path), with_mesh=False)

    parser = BvhModelParser(str(bvh_path))
    if add_joint_centre_markers:
        append_joint_centre_markers_to_biomod(biomod_path, [j.name for j in iter_bvh_joints_depth_first(parser.root)])

    return model, parser


def fbx_visual_mesh_kwargs(callable_obj: Any, include_mesh: bool, mesh_dir: Path) -> dict[str, Any]:
    """Return FBX visual-mesh kwargs matching the installed BioBuddy API."""
    parameters = inspect.signature(callable_obj).parameters
    mesh_output_dir = str(mesh_dir) if include_mesh else None
    if "split_meshes_per_segment" in parameters:
        return {
            "split_meshes_per_segment": include_mesh,
            "mesh_output_dir": mesh_output_dir,
        }
    if "load_visual_meshes" in parameters:
        return {
            "load_visual_meshes": include_mesh,
            "mesh_output_dir": mesh_output_dir,
        }
    return {}


def build_biomod_from_fbx_with_biobuddy(
    fbx_path: Path,
    biomod_path: Path,
    add_joint_centre_markers: bool = True,
    include_mesh: bool = True,
    max_mesh_points: int = 0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Create a bioMod model from an FBX file and optionally attach per-segment FBX meshes."""
    BiomechanicalModelReal, _, FbxModelParser = require_biobuddy()

    mesh_dir = biomod_path.parent / "meshes"
    if include_mesh:
        # Mesh files are generated artifacts. Removing old ones keeps reports and
        # directory listings from mixing outputs from different parser versions.
        clean_generated_fbx_meshes(mesh_dir)

    try:
        model_kwargs = fbx_visual_mesh_kwargs(BiomechanicalModelReal.from_fbx, include_mesh, mesh_dir)
        parser_kwargs = fbx_visual_mesh_kwargs(FbxModelParser, include_mesh, mesh_dir)
        model = BiomechanicalModelReal().from_fbx(
            filepath=str(fbx_path),
            **model_kwargs,
        )
        parser = FbxModelParser(str(fbx_path), **parser_kwargs)
    except TypeError:
        parser_for_model = FbxModelParser(str(fbx_path))
        model = parser_for_model.to_real()
        parser = FbxModelParser(str(fbx_path))
    except AttributeError:
        parser_for_model = FbxModelParser(str(fbx_path))
        model = parser_for_model.to_real()
        parser = FbxModelParser(str(fbx_path))

    model.to_biomod(str(biomod_path), with_mesh=include_mesh)
    normalize_biomod_meshfile_paths(biomod_path)
    joint_names = collect_fbx_joint_names_depth_first(parser)
    if add_joint_centre_markers:
        append_joint_centre_markers_to_biomod(biomod_path, joint_names, marker_prefix="FBXJC_")

    mesh_report: dict[str, Any] = {"mesh_file": None, "mesh_vertices": 0, "mesh_faces": 0, "mesh_parent": None}
    if include_mesh and mesh_dir.exists() and any(mesh_dir.glob("*.ply")):
        mesh_report = convert_biobuddy_ply_meshes_to_vtp(mesh_dir)
    elif include_mesh:
        fbx_tree = parse_fbx_records(fbx_path)
        root_name = joint_names[0] if joint_names else None
        root_translation = (
            np.asarray(parser.skeleton_nodes[parser.root_ids[0]].translation, dtype=float)
            if parser.root_ids
            else np.zeros(3)
        )
        mesh_report = append_fbx_mesh_file_to_biomod(
            biomod_path=biomod_path,
            fbx_tree=fbx_tree,
            parent_name=root_name,
            root_translation=root_translation,
            max_vertices=max_mesh_points,
        )
        mesh_report["mesh_source"] = "legacy_whole_body_obj"

    return model, parser, mesh_report


def append_joint_centre_markers_to_biomod(
    biomod_path: Path, joint_names: list[str], marker_prefix: str = "BVHJC_"
) -> None:
    """Append model markers at each BVH segment origin so the skeleton is visible in pyorerun."""
    text = biomod_path.read_text(encoding="utf-8")
    marker_blocks: list[str] = []
    for name in joint_names:
        marker_name = f"{marker_prefix}{name}"
        if f"marker {marker_name}" in text:
            continue
        marker_blocks.extend(
            [
                "",
                f"marker {marker_name}",
                f"    parent {name}",
                "    position 0 0 0",
                "endmarker",
            ]
        )
    if marker_blocks:
        biomod_path.write_text(text.rstrip() + "\n" + "\n".join(marker_blocks) + "\n", encoding="utf-8")


def extract_q_from_biobuddy_bvh_parser(
    parser: Any,
    apply_root_offset_correction: bool = True,
) -> BvhRuntimeData:
    """Extract q from BioBuddy's BvhModelParser.

    BioBuddy's public ``to_q`` API converts rotations to radians and orders the
    channels like the generated biorbd model. Translation channels are kept in
    native BVH units. By default, the ROOT OFFSET is subtracted from the root translation
    channels because BioBuddy also writes this OFFSET in the generated bioMod. Without this
    correction, the biorbd model is translated by q_root + root_offset instead of q_root.
    """
    if hasattr(parser, "to_q"):
        # Preferred path with the current BioBuddy branch. The fallback below is
        # kept only to make the script easier to run with older development builds.
        parsed_animation = parser.to_q()
        joint_names, file_order, q_order = collect_bvh_channels(parser)
        q_names = [str(name) for name in parsed_animation.dof_names]
        root_name = parser.root.name if parser.root is not None else None
        root_offset = np.asarray(parser.root.offset, dtype=float) if parser.root is not None else np.zeros(3)
        q, _, unwrap_summary = runtime_q_from_parsed_animation(
            parsed_animation=parsed_animation,
            q_names=q_names,
            root_name=root_name,
            root_offset=root_offset,
            apply_root_offset_correction=apply_root_offset_correction,
        )
        return BvhRuntimeData(
            parser=parser,
            joint_names=joint_names,
            channel_entries_file_order=file_order,
            channel_entries_q_order=q_order,
            q=q,
            q_names=q_names,
            time=np.asarray(parsed_animation.time, dtype=float),
            root_offset_correction_applied=apply_root_offset_correction,
            root_offset_native=root_offset,
            q_units=q_channel_units(q_names),
            unwrap_summary=unwrap_summary,
        )

    if parser.motion_data is None or parser.frame_time is None or parser.frame_count is None:
        raise ValueError("The BVH file contains no MOTION block.")

    joint_names, file_order, q_order = collect_bvh_channels(parser)
    file_col = {entry: i for i, entry in enumerate(file_order)}

    q_rows: list[np.ndarray] = []
    q_names: list[str] = []
    root_name = parser.root.name if parser.root is not None else None
    root_offset = np.asarray(parser.root.offset, dtype=float) if parser.root is not None else np.zeros(3)
    axis_to_index = {"x": 0, "y": 1, "z": 2}

    for joint_name, channel in q_order:
        raw = parser.motion_data[:, file_col[(joint_name, channel)]].astype(float)
        if channel.lower().endswith("rotation"):
            values = np.deg2rad(raw)
        elif channel.lower().endswith("position"):
            values = raw.copy()
            if apply_root_offset_correction and joint_name == root_name:
                axis_index = axis_to_index[channel[0].lower()]
                values = values - root_offset[axis_index]
        else:
            raise ValueError(f"Unsupported BVH channel: {joint_name} {channel}")
        q_rows.append(values)
        q_names.append(f"{joint_name}_{channel}")

    q_raw_units = np.vstack(q_rows) if q_rows else np.zeros((0, int(parser.frame_count)))
    q = unwrap_rotation_q(q_raw_units, q_names)
    time = np.arange(int(parser.frame_count), dtype=float) * float(parser.frame_time)
    return BvhRuntimeData(
        parser=parser,
        joint_names=joint_names,
        channel_entries_file_order=file_order,
        channel_entries_q_order=q_order,
        q=q,
        q_names=q_names,
        time=time,
        root_offset_correction_applied=apply_root_offset_correction,
        root_offset_native=root_offset,
        q_units=q_channel_units(q_names),
        unwrap_summary=q_unwrap_summary(q_raw_units, q, q_names),
    )


def save_q_outputs(
    q: np.ndarray,
    q_names: list[str],
    time: np.ndarray,
    out_dir: Path,
    source_name: str = "bvh",
    q_units: list[str] | None = None,
) -> tuple[Path, Path]:
    npz_path = out_dir / f"{source_name}_q_biorbd_order.npz"
    csv_path = out_dir / f"{source_name}_q_biorbd_order.csv"
    q_units = q_units if q_units is not None else q_channel_units(q_names)
    np.savez(
        npz_path,
        q=q,
        q_names=np.asarray(q_names, dtype=object),
        q_units=np.asarray(q_units, dtype=object),
        time=time,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *q_names])
        for frame in range(q.shape[1]):
            writer.writerow([time[frame], *q[:, frame].tolist()])

    return npz_path, csv_path


def compute_bvh_joint_centres_native(
    parser: Any,
    ignore_root_offset_for_position_channels: bool = True,
) -> dict[str, np.ndarray]:
    """Compute BVH joint centres in native BVH units, using the BioBuddy parser tree.

    In Captury BVH files, the root translation channels are already in the laboratory
    coordinate system. Therefore, when the root has position channels, its OFFSET should not
    be added again to the root global position.
    """
    if parser.motion_data is None:
        raise ValueError("The BVH file contains no MOTION block.")

    joint_names, file_order, _ = collect_bvh_channels(parser)
    file_col = {entry: i for i, entry in enumerate(file_order)}
    centres = {name: np.zeros((3, parser.motion_data.shape[0]), dtype=float) for name in joint_names}

    def recurse(joint: Any, frame: int, parent_rotation: np.ndarray, parent_position: np.ndarray) -> None:
        is_root = joint is parser.root
        has_position_channels = any(channel.lower().endswith("position") for channel in joint.channels)
        if ignore_root_offset_for_position_channels and is_root and has_position_channels:
            translation = np.zeros(3, dtype=float)
        else:
            translation = np.asarray(joint.offset, dtype=float).copy()
        local_rotation = np.eye(3)

        for channel in joint.channels:
            value = float(parser.motion_data[frame, file_col[(joint.name, channel)]])
            lower = channel.lower()
            axis_index = {"x": 0, "y": 1, "z": 2}[channel[0].lower()]
            if lower.endswith("position"):
                translation[axis_index] += value
            elif lower.endswith("rotation"):
                local_rotation = local_rotation @ rotation_matrix(channel[0], math.radians(value))

        position = parent_position + parent_rotation @ translation
        rotation = parent_rotation @ local_rotation
        centres[joint.name][:, frame] = position

        for child in joint.children:
            recurse(child, frame, rotation, position)

    for frame in range(parser.motion_data.shape[0]):
        recurse(parser.root, frame, np.eye(3), np.zeros(3))

    return centres


def save_joint_centres(centres: dict[str, np.ndarray], time: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / "bvh_joint_centres_native_units.npz"
    np.savez(path, time=time, joint_names=np.asarray(list(centres.keys()), dtype=object), **centres)
    return path


def save_model_joint_centres(
    centres: dict[str, np.ndarray], time: np.ndarray, out_dir: Path, source_name: str
) -> Path:
    path = out_dir / f"{source_name}_joint_centres_native_units.npz"
    np.savez(path, time=time, joint_names=np.asarray(list(centres.keys()), dtype=object), **centres)
    return path


# =============================================================================
# C3D: markers versus angle point channels
# =============================================================================


@dataclass
class C3dSplitData:
    c3d: dict
    time: np.ndarray
    labels: list[str]
    marker_labels: list[str]
    marker_data_native: np.ndarray  # 3 x n_markers x n_frames in C3D native units
    marker_data_bvh_units: np.ndarray  # 3 x n_markers x n_frames in BVH native units, for pyorerun overlay
    angle_labels: list[str]
    angle_data: np.ndarray  # 3 x n_angles x n_frames, degrees by default unless specified otherwise
    angle_indices: list[int]
    marker_indices: list[int]
    c3d_unit_scale_to_m: float


def get_angle_label_set_from_c3d_parameters(c3d: dict) -> set[str]:
    candidates: set[str] = set()
    for param_name in ("ANGLES", "ANGLE_LABELS"):
        for label in as_str_list(get_c3d_param(c3d, "POINT", param_name, [])):
            if label:
                candidates.add(label)
                candidates.add(label.replace(" ", ""))
    return candidates


def split_c3d_points(
    c3d_path: Path,
    bvh_unit_scale_to_m: float,
    angle_label_regex: str,
    extra_angle_labels: list[str] | None = None,
) -> C3dSplitData:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(c3d_path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    descriptions = as_str_list(get_c3d_param(c3d, "POINT", "DESCRIPTIONS", []))
    if len(descriptions) < len(labels):
        descriptions += [""] * (len(labels) - len(descriptions))

    # ezc3d stores point data as 4 x n_points x n_frames. Rows 0, 1 and 2 are
    # X/Y/Z. Row 3 is a residual/confidence value, so it is not a coordinate.
    points = np.asarray(c3d["data"]["points"], dtype=float)[:3, :, :]
    time = c3d_time_vector(c3d)
    c3d_unit_scale = c3d_point_unit_scale_to_m(c3d)

    regex = re.compile(angle_label_regex) if angle_label_regex else None
    c3d_angle_param_labels = get_angle_label_set_from_c3d_parameters(c3d)
    extra_angle_label_set = DEFAULT_C3D_ANGLE_LABELS | {label.strip() for label in (extra_angle_labels or [])}

    def is_angle_point(i: int) -> bool:
        label = labels[i]
        compact_label = label.replace(" ", "")
        description = descriptions[i]
        if label in c3d_angle_param_labels or compact_label in c3d_angle_param_labels:
            return True
        if label in extra_angle_label_set or compact_label in extra_angle_label_set:
            return True
        if regex is not None and (regex.search(label) or regex.search(description)):
            return True
        return False

    angle_indices = [i for i in range(len(labels)) if is_angle_point(i)]
    # Angles can live in the POINT section of a C3D, but they are not physical
    # markers. The rest of the script only animates and localizes marker_indices.
    marker_indices = [i for i in range(len(labels)) if i not in set(angle_indices)]

    marker_data_native = points[:, marker_indices, :]
    marker_data_bvh_units = marker_data_native * (c3d_unit_scale / bvh_unit_scale_to_m)
    angle_data = points[:, angle_indices, :]

    return C3dSplitData(
        c3d=c3d,
        time=time,
        labels=labels,
        marker_labels=[labels[i] for i in marker_indices],
        marker_data_native=marker_data_native,
        marker_data_bvh_units=marker_data_bvh_units,
        angle_labels=[labels[i] for i in angle_indices],
        angle_data=angle_data,
        angle_indices=angle_indices,
        marker_indices=marker_indices,
        c3d_unit_scale_to_m=c3d_unit_scale,
    )


def save_c3d_split_outputs(split: C3dSplitData, out_dir: Path) -> tuple[Path, Path, Path]:
    markers_path = out_dir / "c3d_markers_for_animation_bvh_units.npz"
    angles_path = out_dir / "c3d_angle_point_channels_raw.npz"
    labels_path = out_dir / "detected_c3d_angle_labels.txt"

    np.savez(
        markers_path,
        time=split.time,
        marker_labels=np.asarray(split.marker_labels, dtype=object),
        markers=split.marker_data_bvh_units,
    )
    np.savez(
        angles_path,
        time=split.time,
        angle_labels=np.asarray(split.angle_labels, dtype=object),
        angles=split.angle_data,
    )
    labels_path.write_text("\n".join(split.angle_labels) + ("\n" if split.angle_labels else ""), encoding="utf-8")
    return markers_path, angles_path, labels_path


def append_bvh_joint_centres_to_c3d(
    split: C3dSplitData,
    centres_native: dict[str, np.ndarray],
    bvh_time: np.ndarray,
    bvh_unit_scale_to_m: float,
    output_path: Path,
    label_prefix: str = "BVHJC_",
) -> Path:
    joint_names = list(centres_native.keys())
    if not joint_names:
        raise ValueError("No BVH joint centres to append.")

    stacked_native = np.stack([centres_native[name] for name in joint_names], axis=1)  # 3 x n_joints x n_bvh_frames
    stacked_on_c3d_time = interpolate_array(stacked_native, bvh_time, split.time)
    stacked_c3d_units = stacked_on_c3d_time * (bvh_unit_scale_to_m / split.c3d_unit_scale_to_m)

    old_points = np.asarray(split.c3d["data"]["points"], dtype=float)
    residuals = np.zeros((1, stacked_c3d_units.shape[1], stacked_c3d_units.shape[2]), dtype=float)
    new_points = np.concatenate((stacked_c3d_units, residuals), axis=0)
    split.c3d["data"]["points"] = np.concatenate((old_points, new_points), axis=1)

    old_labels = as_str_list(get_c3d_param(split.c3d, "POINT", "LABELS", []))
    new_labels = [f"{label_prefix}{name}" for name in joint_names]
    split.c3d["parameters"]["POINT"]["LABELS"]["value"] = old_labels + new_labels

    descriptions = as_str_list(get_c3d_param(split.c3d, "POINT", "DESCRIPTIONS", []))
    if len(descriptions) < len(old_labels):
        descriptions += [""] * (len(old_labels) - len(descriptions))
    descriptions += ["BVH joint centre generated from BioBuddy BVH parser"] * len(new_labels)
    split.c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = descriptions
    split.c3d["parameters"]["POINT"]["USED"]["value"] = [len(old_labels) + len(new_labels)]

    # ezc3d stores per-point metadata such as residuals and camera masks in
    # c3d["data"]["meta_points"]. After appending new POINT trajectories, these
    # arrays still have the old number of points. If they are left untouched,
    # ezc3d.write() raises:
    #   c3d['data']['meta_points']['residuals'] must have its second dimension's
    #   shape equal to the number of points.
    # Deleting the block is the intended ezc3d workflow: it is rebuilt at write
    # time with dimensions consistent with c3d["data"]["points"].
    if "meta_points" in split.c3d.get("data", {}):
        del split.c3d["data"]["meta_points"]

    split.c3d.write(str(output_path))
    return output_path


def clone_c3d_dict(c3d: dict) -> dict:
    import copy

    return copy.deepcopy(c3d)


def append_joint_centres_to_c3d(
    split: C3dSplitData,
    centres_native: dict[str, np.ndarray],
    source_time: np.ndarray,
    source_unit_scale_to_m: float,
    output_path: Path,
    label_prefix: str,
    description: str,
) -> Path:
    split_copy = C3dSplitData(
        c3d=clone_c3d_dict(split.c3d),
        time=split.time,
        labels=split.labels,
        marker_labels=split.marker_labels,
        marker_data_native=split.marker_data_native,
        marker_data_bvh_units=split.marker_data_bvh_units,
        angle_labels=split.angle_labels,
        angle_data=split.angle_data,
        angle_indices=split.angle_indices,
        marker_indices=split.marker_indices,
        c3d_unit_scale_to_m=split.c3d_unit_scale_to_m,
    )
    joint_names = list(centres_native.keys())
    stacked_native = np.stack([centres_native[name] for name in joint_names], axis=1)
    stacked_on_c3d_time = interpolate_array(stacked_native, source_time, split.time)
    stacked_c3d_units = stacked_on_c3d_time * (source_unit_scale_to_m / split.c3d_unit_scale_to_m)

    old_points = np.asarray(split_copy.c3d["data"]["points"], dtype=float)
    residuals = np.zeros((1, stacked_c3d_units.shape[1], stacked_c3d_units.shape[2]), dtype=float)
    split_copy.c3d["data"]["points"] = np.concatenate(
        (old_points, np.concatenate((stacked_c3d_units, residuals), axis=0)), axis=1
    )
    old_labels = as_str_list(get_c3d_param(split_copy.c3d, "POINT", "LABELS", []))
    new_labels = [f"{label_prefix}{name}" for name in joint_names]
    split_copy.c3d["parameters"]["POINT"]["LABELS"]["value"] = old_labels + new_labels
    descriptions = as_str_list(get_c3d_param(split_copy.c3d, "POINT", "DESCRIPTIONS", []))
    if len(descriptions) < len(old_labels):
        descriptions += [""] * (len(old_labels) - len(descriptions))
    descriptions += [description] * len(new_labels)
    split_copy.c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = descriptions
    split_copy.c3d["parameters"]["POINT"]["USED"]["value"] = [len(old_labels) + len(new_labels)]
    if "meta_points" in split_copy.c3d.get("data", {}):
        del split_copy.c3d["data"]["meta_points"]
    split_copy.c3d.write(str(output_path))
    return output_path


# =============================================================================
# biorbd transforms: root policy, joint centres, local marker test
# =============================================================================


def biorbd_segment_names(model: Any) -> list[str]:
    return [model.segment(i).name().to_string() for i in range(model.nbSegment())]


def biorbd_strings_to_list(values: Iterable[Any]) -> list[str]:
    """Convert biorbd's SWIG string objects to normal Python strings."""
    return [value.to_string() if hasattr(value, "to_string") else str(value) for value in values]


def compute_model_joint_centres_native(
    biomod_path: Path,
    q: np.ndarray,
    keep_segment_names: set[str] | None = None,
) -> dict[str, np.ndarray]:
    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    if q.shape[0] != model.nbQ():
        raise RuntimeError(f"{biomod_path} expects {model.nbQ()} q, got {q.shape[0]}.")
    names = biorbd_segment_names(model)
    centres: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if name == "root":
            continue
        if keep_segment_names is not None and name not in keep_segment_names:
            continue
        centres[name] = np.zeros((3, q.shape[1]), dtype=float)

    for frame in range(q.shape[1]):
        q_frame = np.ascontiguousarray(q[:, frame], dtype=float)
        for i, name in enumerate(names):
            if name not in centres:
                continue
            # biorbd.globalJCS gives a segment pose as a 4x4 matrix. The last
            # column is the global position of that segment origin.
            rt = np.asarray(model.globalJCS(q_frame, i).to_array(), dtype=float)
            centres[name][:, frame] = rt[:3, 3]
    return centres


def root_alignment_score(
    centres_native: dict[str, np.ndarray],
    source_time: np.ndarray,
    c3d_markers_source_units: np.ndarray,
    c3d_time: np.ndarray,
    max_frames: int = 120,
) -> float:
    if not centres_native or c3d_markers_source_units.size == 0:
        return float("inf")
    stacked = np.stack(list(centres_native.values()), axis=1)
    centres_on_c3d = interpolate_array(stacked, source_time, c3d_time)
    n_frames = centres_on_c3d.shape[2]
    frame_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)
    frame_scores: list[float] = []
    for frame in frame_indices:
        centres = centres_on_c3d[:, :, frame].T
        markers = c3d_markers_source_units[:, :, frame].T
        finite_centres = np.all(np.isfinite(centres), axis=1)
        finite_markers = np.all(np.isfinite(markers), axis=1)
        centres = centres[finite_centres]
        markers = markers[finite_markers]
        if centres.size == 0 or markers.size == 0:
            continue
        distances = np.linalg.norm(centres[:, None, :] - markers[None, :, :], axis=2)
        frame_scores.append(float(np.nanmedian(np.nanmin(distances, axis=1))))
    return float(np.nanmedian(frame_scores)) if frame_scores else float("inf")


def choose_root_offset_policy(
    source_name: str,
    biomod_path: Path,
    corrected_q: np.ndarray,
    uncorrected_q: np.ndarray,
    q_names: list[str],
    time: np.ndarray,
    joint_names: list[str],
    c3d_markers_source_units: np.ndarray,
    c3d_time: np.ndarray,
    requested_mode: str,
    out_dir: Path,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    # Compare both root-translation conventions against the marker cloud. The
    # lower median nearest-centre distance is the better visual overlay.
    corrected_centres = compute_model_joint_centres_native(biomod_path, corrected_q, set(joint_names))
    uncorrected_centres = compute_model_joint_centres_native(biomod_path, uncorrected_q, set(joint_names))
    corrected_score = root_alignment_score(corrected_centres, time, c3d_markers_source_units, c3d_time)
    uncorrected_score = root_alignment_score(uncorrected_centres, time, c3d_markers_source_units, c3d_time)

    if requested_mode == "subtract":
        use_correction = True
    elif requested_mode == "keep":
        use_correction = False
    else:
        use_correction = corrected_score <= uncorrected_score
    selected_centres = corrected_centres if use_correction else uncorrected_centres
    report = {
        "source": source_name,
        "requested_mode": requested_mode,
        "selected_mode": "subtract_static_offset_from_root_q" if use_correction else "keep_root_q_as_file",
        "score_native_units_subtract_static_offset": corrected_score,
        "score_native_units_keep_file_translation": uncorrected_score,
        "q_names": q_names,
    }
    (out_dir / f"{source_name}_root_translation_policy.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return use_correction, report, selected_centres


def marker_data_in_source_units(split: C3dSplitData, source_unit_scale_to_m: float) -> np.ndarray:
    return split.marker_data_native * (split.c3d_unit_scale_to_m / source_unit_scale_to_m)


def build_animation_markers_with_joint_centres(
    split: C3dSplitData,
    centres_native: dict[str, np.ndarray],
    source_time: np.ndarray,
    source_unit_scale_to_m: float,
    label_prefix: str,
    out_dir: Path,
    source_name: str,
) -> tuple[np.ndarray, list[str], Path]:
    """Build animation markers from C3D markers plus generated joint centres.

    The input C3D contains marker points and angle point channels. ``split`` has
    already removed the angle channels, so this overlay contains only true C3D
    markers and the model joint centres appended by this pipeline.
    """
    marker_data = marker_data_in_source_units(split, source_unit_scale_to_m)
    marker_labels = list(split.marker_labels)
    if centres_native:
        joint_names = list(centres_native.keys())
        joint_data = np.stack([centres_native[name] for name in joint_names], axis=1)
        joint_data_on_c3d_time = interpolate_array(joint_data, source_time, split.time)
        marker_data = np.concatenate((marker_data, joint_data_on_c3d_time), axis=1)
        marker_labels.extend([f"{label_prefix}{name}" for name in joint_names])

    out_path = out_dir / f"{source_name}_animation_markers_no_angles_with_joint_centres.npz"
    np.savez(
        out_path,
        time=split.time,
        marker_labels=np.asarray(marker_labels, dtype=object),
        markers=marker_data,
        source_name=source_name,
    )
    return marker_data, marker_labels, out_path


def write_local_marker_csv(rows: list[dict[str, Any]], out_path: Path) -> Path:
    fieldnames = [
        "marker",
        "marker_index",
        "biomod_marker",
        "parent_segment",
        "x",
        "y",
        "z",
        "std_x",
        "std_y",
        "std_z",
        "rms_std",
        "max_abs_deviation",
        "n_frames",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def append_generated_local_markers_to_biomod(biomod_path: Path, rows: list[dict[str, Any]], source_name: str) -> None:
    text = biomod_path.read_text(encoding="utf-8")
    begin = f"// BEGIN GENERATED C3D LOCAL MARKERS {source_name}"
    end = f"// END GENERATED C3D LOCAL MARKERS {source_name}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", flags=re.DOTALL)
    text = pattern.sub("", text).rstrip()
    blocks = ["", begin]
    for row in rows:
        blocks.extend(
            [
                "",
                f"marker {row['biomod_marker']}",
                f"    parent {row['parent_segment']}",
                f"    position {row['x']:.8g} {row['y']:.8g} {row['z']:.8g}",
                "endmarker",
            ]
        )
    blocks.extend(["", end, ""])
    biomod_path.write_text(text + "\n" + "\n".join(blocks), encoding="utf-8")


def compute_and_append_c3d_local_markers(
    biomod_path: Path,
    q: np.ndarray,
    source_time: np.ndarray,
    split: C3dSplitData,
    source_unit_scale_to_m: float,
    source_name: str,
    out_dir: Path,
    max_assignment_frames: int = 180,
) -> tuple[Path, dict[str, Any]]:
    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    if q.shape[0] != model.nbQ():
        raise RuntimeError(f"{biomod_path} expects {model.nbQ()} q, got {q.shape[0]}.")

    markers = interpolate_array(marker_data_in_source_units(split, source_unit_scale_to_m), split.time, source_time)
    segment_names = biorbd_segment_names(model)
    segment_indices = [i for i, name in enumerate(segment_names) if name != "root"]
    if not segment_indices:
        raise RuntimeError(f"No usable segment found in {biomod_path}.")

    n_frames = q.shape[1]
    sample_frames = np.linspace(0, n_frames - 1, min(max_assignment_frames, n_frames), dtype=int)
    local_by_segment: dict[int, np.ndarray] = {}
    for seg_idx in segment_indices:
        local = np.full((3, markers.shape[1], sample_frames.shape[0]), np.nan)
        for out_i, frame in enumerate(sample_frames):
            q_frame = np.ascontiguousarray(q[:, frame], dtype=float)
            rt = np.asarray(model.globalJCS(q_frame, seg_idx).to_array(), dtype=float)
            rotation = rt[:3, :3]
            translation = rt[:3, 3:4]
            local[:, :, out_i] = rotation.T @ (markers[:, :, frame] - translation)
        local_by_segment[seg_idx] = local

    used_names: set[str] = set()
    rows: list[dict[str, Any]] = []
    for marker_idx, marker_label in enumerate(split.marker_labels):
        best_seg_idx: int | None = None
        best_score = float("inf")
        for seg_idx, local in local_by_segment.items():
            marker_local = local[:, marker_idx, :]
            finite = np.all(np.isfinite(marker_local), axis=0)
            if finite.sum() < 3:
                continue
            std = np.nanstd(marker_local[:, finite], axis=1)
            score = float(np.sqrt(np.mean(std**2)))
            if score < best_score:
                best_score = score
                best_seg_idx = seg_idx
        if best_seg_idx is None:
            continue

        full_local = np.full((3, n_frames), np.nan)
        for frame in range(n_frames):
            q_frame = np.ascontiguousarray(q[:, frame], dtype=float)
            rt = np.asarray(model.globalJCS(q_frame, best_seg_idx).to_array(), dtype=float)
            full_local[:, frame] = rt[:3, :3].T @ (markers[:, marker_idx, frame] - rt[:3, 3])
        finite = np.all(np.isfinite(full_local), axis=0)
        if finite.sum() == 0:
            continue
        mean_local = np.nanmean(full_local[:, finite], axis=1)
        std_local = np.nanstd(full_local[:, finite], axis=1)
        deviations = np.linalg.norm(full_local[:, finite] - mean_local.reshape(3, 1), axis=0)
        marker_name_base = f"C3D_{sanitize_biomod_name(marker_label, f'marker_{marker_idx}')}"
        marker_name = marker_name_base
        suffix = 2
        while marker_name in used_names:
            marker_name = f"{marker_name_base}_{suffix}"
            suffix += 1
        used_names.add(marker_name)
        rows.append(
            {
                "marker": marker_label,
                "marker_index": marker_idx,
                "biomod_marker": marker_name,
                "parent_segment": segment_names[best_seg_idx],
                "x": float(mean_local[0]),
                "y": float(mean_local[1]),
                "z": float(mean_local[2]),
                "std_x": float(std_local[0]),
                "std_y": float(std_local[1]),
                "std_z": float(std_local[2]),
                "rms_std": float(np.sqrt(np.mean(std_local**2))),
                "max_abs_deviation": float(np.nanmax(deviations)),
                "n_frames": int(finite.sum()),
            }
        )

    csv_path = write_local_marker_csv(rows, out_dir / f"{source_name}_c3d_local_markers.csv")
    append_generated_local_markers_to_biomod(biomod_path, rows, source_name)
    rms_values = [row["rms_std"] for row in rows]
    summary = {
        "source": source_name,
        "markers_written": len(rows),
        "median_rms_std_native_units": float(np.nanmedian(rms_values)) if rms_values else None,
        "max_rms_std_native_units": float(np.nanmax(rms_values)) if rms_values else None,
        "csv": str(csv_path),
    }
    (out_dir / f"{source_name}_c3d_local_marker_stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return csv_path, summary


# =============================================================================
# Inverse kinematics from C3D markers
# =============================================================================


def read_local_marker_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def c3d_index_from_local_marker_row(row: dict[str, str], split: C3dSplitData) -> int:
    """Get the original C3D channel index, even when labels are duplicated.

    Captury files may contain distinct physical markers sharing one visible
    label. Their channel indices therefore carry information that the label
    alone cannot preserve.
    """
    marker_label = row["marker"]
    marker_index = row.get("marker_index", "")
    if marker_index:
        index = int(marker_index)
        if index < 0 or index >= len(split.marker_labels) or split.marker_labels[index] != marker_label:
            raise RuntimeError(f"Invalid C3D marker index stored for {marker_label}: {marker_index}.")
        return index

    matching_indices = [i for i, label in enumerate(split.marker_labels) if label == marker_label]
    if len(matching_indices) != 1:
        raise RuntimeError(
            f"The local marker CSV does not identify duplicated C3D marker {marker_label}; regenerate it."
        )
    return matching_indices[0]


def finite_error_values(errors_by_marker: dict[str, np.ndarray]) -> np.ndarray:
    """Gather finite marker error samples from a marker-to-trajectory dictionary."""
    if not errors_by_marker:
        return np.zeros(0, dtype=float)
    values = np.concatenate([errors for errors in errors_by_marker.values()])
    return values[np.isfinite(values)]


def save_and_plot_c3d_marker_error_norms(
    biomod_path: Path,
    q: np.ndarray,
    source_time: np.ndarray,
    split: C3dSplitData,
    local_marker_csv: Path,
    source_unit_scale_to_m: float,
    source_name: str,
    out_dir: Path,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, np.ndarray]]:
    """Compute and boxplot the norm of model-to-C3D marker errors.

    The local marker positions have already been appended to the bioMod. For
    every frame this function animates those markers with ``q``, compares them
    to the matching measured C3D marker, and reports the Euclidean distance.
    Distances are converted to millimetres for readable biomechanics plots.
    """
    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    if q.shape[0] != model.nbQ():
        raise RuntimeError(f"{biomod_path} expects {model.nbQ()} q, got {q.shape[0]}.")

    local_marker_rows = read_local_marker_rows(local_marker_csv)
    model_marker_index = {
        name: i for i, name in enumerate(biorbd_strings_to_list(model.markerNames()))
    }
    measured = interpolate_array(
        marker_data_in_source_units(split, source_unit_scale_to_m),
        split.time,
        source_time,
    )
    native_to_mm = source_unit_scale_to_m * 1000.0

    marker_pairs: list[tuple[str, str, int, int]] = []
    for row in local_marker_rows:
        marker_label = row["marker"]
        biomod_marker = row["biomod_marker"]
        if biomod_marker in model_marker_index:
            marker_pairs.append(
                (
                    biomod_marker,
                    marker_label,
                    c3d_index_from_local_marker_row(row, split),
                    model_marker_index[biomod_marker],
                )
            )
    if not marker_pairs:
        raise RuntimeError(f"No C3D/local marker pair available to compute errors for {source_name}.")

    errors_by_marker = {
        biomod_marker: np.full(source_time.shape[0], np.nan, dtype=float)
        for biomod_marker, _, _, _ in marker_pairs
    }
    for frame in range(source_time.shape[0]):
        predicted = model.markers(np.ascontiguousarray(q[:, frame], dtype=float))
        for biomod_marker, _, c3d_index, model_index in marker_pairs:
            measured_position = measured[:, c3d_index, frame]
            if not np.all(np.isfinite(measured_position)):
                continue
            predicted_position = np.asarray(predicted[model_index].to_array(), dtype=float).ravel()
            errors_by_marker[biomod_marker][frame] = (
                np.linalg.norm(predicted_position - measured_position) * native_to_mm
            )

    csv_path = out_dir / f"{source_name}_c3d_marker_error_norm_mm.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "marker", "marker_index", "biomod_marker", "frame", "time", "error_norm_mm"])
        for biomod_marker, marker_label, c3d_index, _ in marker_pairs:
            errors = errors_by_marker[biomod_marker]
            for frame, error in enumerate(errors):
                writer.writerow(
                    [source_name, marker_label, c3d_index, biomod_marker, frame, source_time[frame], error]
                )

    finite_all = finite_error_values(errors_by_marker)
    per_marker_summary: dict[str, dict[str, Any]] = {}
    for biomod_marker, marker_label, c3d_index, _ in marker_pairs:
        errors = errors_by_marker[biomod_marker]
        finite = errors[np.isfinite(errors)]
        per_marker_summary[biomod_marker] = {
            "c3d_label": marker_label,
            "c3d_marker_index": c3d_index,
            "n": int(finite.size),
            "median_mm": float(np.nanmedian(finite)) if finite.size else None,
            "p95_mm": float(np.nanpercentile(finite, 95)) if finite.size else None,
            "max_mm": float(np.nanmax(finite)) if finite.size else None,
        }
    summary = {
        "source": source_name,
        "error_definition": "Euclidean norm of model marker position minus measured C3D marker position",
        "unit": "mm",
        "markers": len(errors_by_marker),
        "samples": int(finite_all.size),
        "median_mm": float(np.nanmedian(finite_all)) if finite_all.size else None,
        "p95_mm": float(np.nanpercentile(finite_all, 95)) if finite_all.size else None,
        "max_mm": float(np.nanmax(finite_all)) if finite_all.size else None,
        "per_marker": per_marker_summary,
        "csv": str(csv_path),
    }
    summary_path = out_dir / f"{source_name}_c3d_marker_error_norm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt  # Imported only when generating requested error figures.

    marker_labels = [biomod_marker.removeprefix("C3D_") for biomod_marker, _, _, _ in marker_pairs]
    box_values = [
        errors_by_marker[biomod_marker][np.isfinite(errors_by_marker[biomod_marker])]
        for biomod_marker, _, _, _ in marker_pairs
    ]
    flier_style = {
        "marker": ".",
        "markersize": 1.0,
        "markerfacecolor": "#314b57",
        "markeredgecolor": "none",
        "alpha": 0.12,
    }
    figure, axis = plt.subplots(figsize=(max(12.0, 0.38 * len(marker_labels)), 6.0), constrained_layout=True)
    axis.boxplot(box_values, tick_labels=marker_labels, showfliers=True, flierprops=flier_style)
    axis.set_title(f"{source_name.upper()} - erreur marqueur modele vs C3D")
    axis.set_xlabel("Marqueur C3D")
    axis.set_ylabel("Norme de l'erreur (mm)")
    axis.tick_params(axis="x", labelrotation=65)
    axis.grid(axis="y", alpha=0.3)
    figure_path = out_dir / f"{source_name}_c3d_marker_error_norm_boxplot.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    return csv_path, summary_path, figure_path, summary, errors_by_marker


def plot_overall_model_marker_error_boxplot(
    errors_by_source: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> Path:
    """Plot one error distribution per source model for quick comparison."""
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[np.ndarray] = []
    for source_name, errors_by_marker in errors_by_source.items():
        finite = finite_error_values(errors_by_marker)
        if finite.size:
            labels.append(source_name.upper())
            values.append(finite)
    if not values:
        raise RuntimeError("No finite marker error was available for the overall boxplot.")

    flier_style = {
        "marker": ".",
        "markersize": 1.0,
        "markerfacecolor": "#314b57",
        "markeredgecolor": "none",
        "alpha": 0.12,
    }
    figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    axis.boxplot(values, tick_labels=labels, showfliers=True, flierprops=flier_style)
    axis.set_title("Erreur globale des marqueurs C3D")
    axis.set_ylabel("Norme de l'erreur (mm)")
    axis.grid(axis="y", alpha=0.3)
    figure_path = out_dir / "bvh_fbx_c3d_marker_error_norm_overall_boxplot.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def finite_difference_by_time(data: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Differentiate each row of data with respect to a time vector."""
    if data.shape[1] != time.shape[0]:
        raise ValueError("data and time must have the same number of frames.")
    if time.shape[0] < 2:
        return np.zeros_like(data)
    edge_order = 2 if time.shape[0] > 2 else 1
    return np.gradient(data, time, axis=1, edge_order=edge_order)


def build_marker_data_for_biorbd_ik(
    model: Any,
    split: C3dSplitData,
    local_marker_rows: list[dict[str, str]],
    source_unit_scale_to_m: float,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build the marker array expected by biorbd.InverseKinematics.

    biorbd expects one marker slot per marker in the bioMod. We fill only the
    generated C3D local markers and leave all other model markers as NaN, which
    tells biorbd to ignore them during inverse kinematics.
    """
    model_marker_names = biorbd_strings_to_list(model.markerNames())
    model_marker_index = {name: i for i, name in enumerate(model_marker_names)}
    c3d_markers_source_units = marker_data_in_source_units(split, source_unit_scale_to_m)

    marker_data = np.full((3, len(model_marker_names), frame_indices.shape[0]), np.nan, dtype=float)
    used_marker_names: list[str] = []
    for row in local_marker_rows:
        biomod_marker = row["biomod_marker"]
        if biomod_marker not in model_marker_index:
            continue
        c3d_marker_index = c3d_index_from_local_marker_row(row, split)
        marker_data[:, model_marker_index[biomod_marker], :] = c3d_markers_source_units[
            :, c3d_marker_index, :
        ][:, frame_indices]
        used_marker_names.append(biomod_marker)

    if not used_marker_names:
        raise RuntimeError("No generated C3D local markers were found in the bioMod for inverse kinematics.")
    return marker_data, used_marker_names


def save_inverse_kinematics_outputs(
    out_dir: Path,
    source_name: str,
    time: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    q_names: list[str],
    marker_names: list[str],
    solver: str,
) -> tuple[Path, Path, Path]:
    npz_path = out_dir / f"{source_name}_inverse_kinematics_from_c3d_markers.npz"
    csv_path = out_dir / f"{source_name}_inverse_kinematics_from_c3d_markers.csv"
    summary_path = out_dir / f"{source_name}_inverse_kinematics_from_c3d_markers_summary.json"

    np.savez(
        npz_path,
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        q_names=np.asarray(q_names, dtype=object),
        marker_names=np.asarray(marker_names, dtype=object),
        solver=solver,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        fieldnames = (
            ["time"]
            + [f"q_{name}" for name in q_names]
            + [f"qdot_{name}" for name in q_names]
            + [f"qddot_{name}" for name in q_names]
        )
        writer.writerow(fieldnames)
        for frame in range(q.shape[1]):
            writer.writerow([time[frame], *q[:, frame].tolist(), *qdot[:, frame].tolist(), *qddot[:, frame].tolist()])

    summary = {
        "source": source_name,
        "solver": solver,
        "frames": int(time.shape[0]),
        "nb_q": int(q.shape[0]),
        "markers_used": len(marker_names),
        "marker_names": marker_names,
        "outputs": {
            "npz": str(npz_path),
            "csv": str(csv_path),
        },
        "important_note": (
            "Inverse kinematics uses only C3D marker channels; C3D angle channels are filtered out before solving."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return npz_path, csv_path, summary_path


def solve_inverse_kinematics_least_squares(model: Any, marker_data: np.ndarray, time: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve inverse kinematics with biorbd's nonlinear least-squares wrapper."""
    biorbd = require_biorbd()
    ik = biorbd.InverseKinematics(model, marker_data)
    q = np.asarray(ik.solve(method=method), dtype=float)
    if q.shape[0] != model.nbQ():
        q = q.T
    if q.shape != (model.nbQ(), time.shape[0]):
        raise RuntimeError(f"Inverse kinematics returned q with shape {q.shape}, expected {(model.nbQ(), time.shape[0])}.")
    qdot = finite_difference_by_time(q, time)
    qddot = finite_difference_by_time(qdot, time)
    return q, qdot, qddot


def solve_inverse_kinematics_kalman(
    model: Any,
    marker_data: np.ndarray,
    time: np.ndarray,
    noise_factor: float,
    error_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve inverse kinematics with biorbd's extended Kalman marker reconstructor."""
    biorbd = require_biorbd()
    if time.shape[0] > 1:
        frequency = 1.0 / float(np.nanmedian(np.diff(time)))
    else:
        frequency = 100.0
    params = biorbd.KalmanParam(frequency, noise_factor, error_factor)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    q_state = biorbd.GeneralizedCoordinates(model)
    qdot_state = biorbd.GeneralizedVelocity(model)
    qddot_state = biorbd.GeneralizedAcceleration(model)
    q = np.zeros((model.nbQ(), time.shape[0]), dtype=float)
    qdot = np.zeros((model.nbQdot(), time.shape[0]), dtype=float)
    qddot = np.zeros((model.nbQddot(), time.shape[0]), dtype=float)
    for frame in range(time.shape[0]):
        markers_flat = np.reshape(marker_data[:, :, frame].T, -1)
        kalman.reconstructFrame(model, markers_flat, q_state, qdot_state, qddot_state)
        q[:, frame] = np.asarray(q_state.to_array(), dtype=float).ravel()
        qdot[:, frame] = np.asarray(qdot_state.to_array(), dtype=float).ravel()
        qddot[:, frame] = np.asarray(qddot_state.to_array(), dtype=float).ravel()
    return q, qdot, qddot


def run_inverse_kinematics_from_c3d_markers(
    biomod_path: Path,
    split: C3dSplitData,
    local_marker_csv: Path,
    source_unit_scale_to_m: float,
    source_name: str,
    out_dir: Path,
    solver: str = "least_squares",
    least_squares_method: str = "trf",
    max_frames: int = 0,
    kalman_noise_factor: float = 1e-10,
    kalman_error_factor: float = 1e-5,
) -> dict[str, Any]:
    """Run marker-based inverse kinematics from C3D markers.

    The C3D angle channels are already excluded in ``split``. Only markers that
    were written into the bioMod by ``compute_and_append_c3d_local_markers`` are
    used for inverse kinematics.
    """
    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    local_marker_rows = read_local_marker_rows(local_marker_csv)

    n_c3d_frames = split.time.shape[0]
    n_frames = n_c3d_frames if max_frames <= 0 else min(max_frames, n_c3d_frames)
    frame_indices = np.arange(n_frames, dtype=int)
    time = split.time[frame_indices]

    marker_data, used_marker_names = build_marker_data_for_biorbd_ik(
        model=model,
        split=split,
        local_marker_rows=local_marker_rows,
        source_unit_scale_to_m=source_unit_scale_to_m,
        frame_indices=frame_indices,
    )

    if solver == "least_squares":
        q, qdot, qddot = solve_inverse_kinematics_least_squares(
            model=model,
            marker_data=marker_data,
            time=time,
            method=least_squares_method,
        )
        solver_label = f"least_squares:{least_squares_method}"
    elif solver == "kalman":
        q, qdot, qddot = solve_inverse_kinematics_kalman(
            model=model,
            marker_data=marker_data,
            time=time,
            noise_factor=kalman_noise_factor,
            error_factor=kalman_error_factor,
        )
        solver_label = "kalman"
    else:
        raise ValueError("solver must be 'least_squares' or 'kalman'.")

    q_names = biorbd_strings_to_list(model.nameDof())
    npz_path, csv_path, summary_path = save_inverse_kinematics_outputs(
        out_dir=out_dir,
        source_name=source_name,
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        q_names=q_names,
        marker_names=used_marker_names,
        solver=solver_label,
    )
    return {
        "npz": str(npz_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "frames": int(time.shape[0]),
        "markers_used": len(used_marker_names),
        "solver": solver_label,
    }


# =============================================================================
# q_BVH versus C3D angles
# =============================================================================


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
INDEX_TO_AXIS = {0: "X", 1: "Y", 2: "Z"}


def c3d_angles_to_rad(angle_data: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.lower()
    if unit in {"deg", "degree", "degrees"}:
        return np.deg2rad(angle_data)
    if unit in {"rad", "radian", "radians"}:
        return angle_data.copy()
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


def comparison_metrics(bvh_values_rad: np.ndarray, c3d_values_rad: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(bvh_values_rad) & np.isfinite(c3d_values_rad)
    if finite.sum() == 0:
        return {
            "n": 0,
            "bias_deg": float("nan"),
            "rmse_deg": float("nan"),
            "rmse_after_bias_removal_deg": float("nan"),
            "corr": float("nan"),
        }
    diff = bvh_values_rad[finite] - c3d_values_rad[finite]
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    rmse_unbiased = float(np.sqrt(np.mean((diff - bias) ** 2)))
    return {
        "n": int(finite.sum()),
        "bias_deg": math.degrees(bias),
        "rmse_deg": math.degrees(rmse),
        "rmse_after_bias_removal_deg": math.degrees(rmse_unbiased),
        "corr": pearson_corr(bvh_values_rad, c3d_values_rad),
    }


def bvh_rotation_q_indices(q_names: list[str]) -> list[int]:
    return rotation_q_indices(q_names)


def write_pairwise_q_vs_c3d_angle_comparison(
    q: np.ndarray,
    q_names: list[str],
    bvh_time: np.ndarray,
    c3d_angle_data: np.ndarray,
    c3d_angle_labels: list[str],
    c3d_time: np.ndarray,
    c3d_angle_unit: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Compare all BVH rotational q against all C3D angle components."""
    pairwise_path = out_dir / "q_bvh_vs_c3d_angles_pairwise.csv"
    best_path = out_dir / "q_bvh_vs_c3d_angles_best_matches.csv"

    if c3d_angle_data.size == 0 or not c3d_angle_labels:
        pairwise_path.write_text("No C3D angle point channels were detected.\n", encoding="utf-8")
        best_path.write_text("No C3D angle point channels were detected.\n", encoding="utf-8")
        return pairwise_path, best_path

    c3d_angles_rad = c3d_angles_to_rad(c3d_angle_data, c3d_angle_unit)
    c3d_angles_on_bvh_time = interpolate_array(c3d_angles_rad, c3d_time, bvh_time)

    rows: list[dict[str, Any]] = []
    for q_idx in bvh_rotation_q_indices(q_names):
        q_series = q[q_idx, :]
        for angle_idx, angle_label in enumerate(c3d_angle_labels):
            for component_idx in range(3):
                c3d_series = c3d_angles_on_bvh_time[component_idx, angle_idx, :]
                metrics = comparison_metrics(q_series, c3d_series)
                rows.append(
                    {
                        "bvh_q_name": q_names[q_idx],
                        "bvh_q_index": q_idx,
                        "c3d_angle_label": angle_label,
                        "c3d_component": INDEX_TO_AXIS[component_idx],
                        "c3d_component_index": component_idx,
                        **metrics,
                    }
                )

    fieldnames = [
        "bvh_q_name",
        "bvh_q_index",
        "c3d_angle_label",
        "c3d_component",
        "c3d_component_index",
        "n",
        "bias_deg",
        "rmse_deg",
        "rmse_after_bias_removal_deg",
        "corr",
    ]
    with pairwise_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_rows: list[dict[str, Any]] = []
    for q_name in [q_names[i] for i in bvh_rotation_q_indices(q_names)]:
        q_rows = [row for row in rows if row["bvh_q_name"] == q_name and row["n"] > 0]
        q_rows.sort(key=lambda r: (r["rmse_after_bias_removal_deg"], r["rmse_deg"]))
        if q_rows:
            best_rows.append(q_rows[0])

    with best_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    return pairwise_path, best_path


def write_explicit_q_vs_c3d_angle_comparison(
    mapping_path: Path | None,
    q: np.ndarray,
    q_names: list[str],
    bvh_time: np.ndarray,
    c3d_angle_data: np.ndarray,
    c3d_angle_labels: list[str],
    c3d_time: np.ndarray,
    c3d_angle_unit: str,
    out_dir: Path,
) -> Path | None:
    """Use a user-provided mapping JSON to compare selected pairs.

    Mapping format:
    [
      {"name": "right_knee_flexion", "bvh_q": "RightLeg_rotX", "c3d_label": "RKneeAngles", "c3d_component": "X"}
    ]
    """
    if mapping_path is None:
        return None
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, list):
        raise ValueError("The comparison mapping JSON must be a list of entries.")

    c3d_angles_rad = c3d_angles_to_rad(c3d_angle_data, c3d_angle_unit)
    c3d_angles_on_bvh_time = interpolate_array(c3d_angles_rad, c3d_time, bvh_time)
    out_path = out_dir / "q_bvh_vs_c3d_angles_explicit_mapping.csv"

    rows: list[dict[str, Any]] = []
    for entry in mapping:
        bvh_q_name = entry["bvh_q"]
        c3d_label = entry["c3d_label"]
        c3d_component = str(entry.get("c3d_component", "X")).lower()[0]
        component_idx = AXIS_TO_INDEX[c3d_component]

        if bvh_q_name not in q_names:
            raise ValueError(f"BVH q '{bvh_q_name}' from mapping is not in q_names.")
        if c3d_label not in c3d_angle_labels:
            raise ValueError(f"C3D angle label '{c3d_label}' from mapping was not detected.")

        q_idx = q_names.index(bvh_q_name)
        angle_idx = c3d_angle_labels.index(c3d_label)
        metrics = comparison_metrics(q[q_idx, :], c3d_angles_on_bvh_time[component_idx, angle_idx, :])
        rows.append(
            {
                "name": entry.get("name", f"{bvh_q_name}__{c3d_label}_{c3d_component.upper()}"),
                "bvh_q_name": bvh_q_name,
                "bvh_q_index": q_idx,
                "c3d_angle_label": c3d_label,
                "c3d_component": c3d_component.upper(),
                "c3d_component_index": component_idx,
                **metrics,
            }
        )

    fieldnames = [
        "name",
        "bvh_q_name",
        "bvh_q_index",
        "c3d_angle_label",
        "c3d_component",
        "c3d_component_index",
        "n",
        "bias_deg",
        "rmse_deg",
        "rmse_after_bias_removal_deg",
        "corr",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def write_mapping_template(q_names: list[str], c3d_angle_labels: list[str], out_dir: Path) -> Path:
    template_path = out_dir / "comparison_mapping_template.json"
    rotation_q = [name for name in q_names if is_rotation_q_name(name)]
    template = []
    for q_name in rotation_q[: min(12, len(rotation_q))]:
        template.append(
            {
                "name": q_name.replace("_", " "),
                "bvh_q": q_name,
                "c3d_label": c3d_angle_labels[0] if c3d_angle_labels else "REPLACE_WITH_C3D_ANGLE_LABEL",
                "c3d_component": "X",
            }
        )
    template_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return template_path


# =============================================================================
# Animation
# =============================================================================


def filter_display_markers_for_rerun(
    markers: np.ndarray,
    marker_labels: list[str],
    hide_hands: bool,
    hide_feet: bool,
) -> tuple[np.ndarray, list[str]]:
    if not (hide_hands or hide_feet) or not marker_labels:
        return markers, marker_labels
    keep_indices = [
        i for i, label in enumerate(marker_labels) if not is_hidden_display_name(label, hide_hands, hide_feet)
    ]
    return markers[:, keep_indices, :], [marker_labels[i] for i in keep_indices]


def marker_block_parent(marker_block: list[str]) -> str | None:
    for line in marker_block:
        stripped = line.strip()
        if stripped.startswith("parent"):
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def remove_display_markers_from_biomod_text(text: str, hide_hands: bool, hide_feet: bool) -> str:
    lines = text.splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped.startswith("marker"):
            output.append(lines[i])
            i += 1
            continue

        marker_block = [lines[i]]
        marker_name = stripped.split(maxsplit=1)[1].strip() if len(stripped.split(maxsplit=1)) == 2 else ""
        i += 1
        while i < len(lines):
            marker_block.append(lines[i])
            if lines[i].strip() == "endmarker":
                i += 1
                break
            i += 1

        parent_name = marker_block_parent(marker_block)
        if is_hidden_display_name(marker_name, hide_hands, hide_feet) or (
            parent_name is not None and is_hidden_display_name(parent_name, hide_hands, hide_feet)
        ):
            continue
        output.extend(marker_block)

    return "\n".join(output) + "\n"


def remove_display_meshes_from_biomod_text(text: str, hide_hands: bool, hide_feet: bool) -> str:
    output: list[str] = []
    current_segment: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("segment"):
            parts = stripped.split(maxsplit=1)
            current_segment = parts[1].strip() if len(parts) == 2 else None
            output.append(line)
            continue
        if stripped == "endsegment":
            current_segment = None
            output.append(line)
            continue
        if current_segment is not None and is_hidden_display_name(current_segment, hide_hands, hide_feet):
            if stripped.startswith(("meshfile", "meshscale", "meshrt")):
                continue
        output.append(line)
    return "\n".join(output) + "\n"


def pyorerun_display_biomod_path(biomod_path: Path, hide_hands: bool, hide_feet: bool) -> Path:
    if not (hide_hands or hide_feet):
        return biomod_path
    text = biomod_path.read_text(encoding="utf-8")
    text = remove_display_markers_from_biomod_text(text, hide_hands, hide_feet)
    text = remove_display_meshes_from_biomod_text(text, hide_hands, hide_feet)
    if hide_hands and hide_feet:
        suffix = "no_extremities"
    elif hide_hands:
        suffix = "no_hands"
    else:
        suffix = "no_feet"
    display_path = biomod_path.with_name(f"{biomod_path.stem}_pyorerun_{suffix}{biomod_path.suffix}")
    display_path.write_text(text, encoding="utf-8")
    return display_path


def use_vtp_meshes_for_pyorerun(model: Any) -> None:
    """Point pyorerun mesh paths to VTP siblings while keeping the bioMod PLY paths intact."""
    for segment in model.segments:
        mesh_paths: list[str] = []
        changed = False
        for mesh_path in segment.mesh_path:
            path = Path(mesh_path)
            if path.suffix.lower() == ".ply":
                vtp_path = path.with_suffix(".vtp")
                if not vtp_path.exists():
                    vertices, faces = read_ascii_ply_mesh(path)
                    write_ascii_vtp_mesh(vtp_path, vertices=vertices, faces=faces)
                mesh_paths.append(str(vtp_path))
                changed = True
            else:
                mesh_paths.append(mesh_path)
        if changed:
            segment.__dict__["mesh_path"] = mesh_paths


def set_pyorerun_marker_radius(phase: Any, marker_radius: float) -> None:
    """Make model and experimental markers visible when source units are millimetres."""
    for rerun_model in phase.models.rerun_models:
        if hasattr(rerun_model.markers, "marker_properties"):
            rerun_model.markers.marker_properties.radius = marker_radius
    for xp_data in phase.xp_data.xp_data:
        if hasattr(xp_data, "markers_properties"):
            xp_data.markers_properties.radius = marker_radius


def rerun_view_coordinates(up_axis: str) -> Any | None:
    if up_axis == "none":
        return None
    import rerun as rr

    return {
        "x": rr.ViewCoordinates.RIGHT_HAND_X_UP,
        "y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        "z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    }[up_axis]


def run_phase_with_rerun_view(
    phase: Any,
    name: str,
    up_axis: str,
    rerun_wait_seconds: float,
) -> None:
    import rerun as rr

    rr.init(f"{name}_{phase.phase}", spawn=True)
    coordinates = rerun_view_coordinates(up_axis)
    if coordinates is not None:
        rr.log(phase.name, coordinates, static=True)
    phase.rerun(name, init=False)
    if rerun_wait_seconds > 0:
        time.sleep(rerun_wait_seconds)


def animate_with_pyorerun(
    biomod_path: Path,
    q: np.ndarray,
    bvh_time: np.ndarray,
    c3d_markers_bvh_units: np.ndarray,
    c3d_marker_labels: list[str],
    c3d_time: np.ndarray,
    name: str = "bvh_biobuddy_c3d_comparison",
    display_q_in_rerun: bool = False,
    rerun_marker_radius: float = DEFAULT_RERUN_MARKER_RADIUS_NATIVE,
    rerun_wait_seconds: float = DEFAULT_RERUN_WAIT_SECONDS,
    rerun_up_axis: str = DEFAULT_RERUN_UP_AXIS,
    hide_hands_in_rerun: bool = False,
    hide_feet_in_rerun: bool = False,
) -> None:
    BiorbdModel, PhaseRerun, PyoMarkers = require_pyorerun()

    c3d_markers_on_bvh_time = interpolate_array(c3d_markers_bvh_units, c3d_time, bvh_time)
    c3d_markers_on_bvh_time, c3d_marker_labels = filter_display_markers_for_rerun(
        c3d_markers_on_bvh_time,
        c3d_marker_labels,
        hide_hands_in_rerun,
        hide_feet_in_rerun,
    )
    biomod_path = pyorerun_display_biomod_path(biomod_path, hide_hands_in_rerun, hide_feet_in_rerun)

    model = BiorbdModel(str(biomod_path))
    use_vtp_meshes_for_pyorerun(model)
    if q.shape[0] != model.nb_q:
        raise RuntimeError(
            f"q has {q.shape[0]} rows, but the generated pyorerun/biorbd model has {model.nb_q} DoFs. "
            "Check that the BioBuddy branch and q extraction order are consistent."
        )

    model.options.show_marker_labels = False
    model.options.show_center_of_mass_labels = False
    model.options.markers_radius = rerun_marker_radius
    model.options.centers_of_mass_radius = rerun_marker_radius

    def _build_phase(display_q: bool):
        phase_local = PhaseRerun(bvh_time)
        phase_local.add_animated_model(model, q, display_q=display_q)
        if c3d_markers_on_bvh_time.size and c3d_marker_labels:
            phase_local.add_xp_markers(
                "c3d_markers_no_angle_channels",
                PyoMarkers(data=c3d_markers_on_bvh_time, channels=c3d_marker_labels),
            )
        set_pyorerun_marker_radius(phase_local, rerun_marker_radius)
        return phase_local

    try:
        phase = _build_phase(display_q=display_q_in_rerun)
    except AttributeError as exc:
        # pyorerun versions that call rr.SeriesLine are incompatible with newer rerun-sdk
        # versions where the class was renamed/removed. Animation itself does not require
        # TimeSeriesQ, so fall back to model+marker animation and keep q in CSV/NPZ outputs.
        if display_q_in_rerun and "SeriesLine" in str(exc):
            warnings.warn(
                "pyorerun could not display q time-series because the installed rerun-sdk "
                "does not expose rerun.SeriesLine. Falling back to animation without display_q. "
                "The q values are still saved in bvh_q_biorbd_order.csv/.npz.",
                RuntimeWarning,
            )
            phase = _build_phase(display_q=False)
        else:
            raise

    run_phase_with_rerun_view(phase, name, rerun_up_axis, rerun_wait_seconds)


def animate_superposed_models_with_pyorerun(
    bvh_biomod_path: Path,
    bvh_q: np.ndarray,
    bvh_time: np.ndarray,
    fbx_biomod_path: Path,
    fbx_q: np.ndarray,
    fbx_time: np.ndarray,
    c3d_markers_bvh_units: np.ndarray,
    c3d_marker_labels: list[str],
    c3d_time: np.ndarray,
    name: str = "bvh_fbx_c3d_superposed",
    display_q_in_rerun: bool = False,
    rerun_marker_radius: float = DEFAULT_RERUN_MARKER_RADIUS_NATIVE,
    rerun_wait_seconds: float = DEFAULT_RERUN_WAIT_SECONDS,
    rerun_up_axis: str = DEFAULT_RERUN_UP_AXIS,
    hide_hands_in_rerun: bool = False,
    hide_feet_in_rerun: bool = False,
) -> None:
    """Animate the BVH model, the FBX model and C3D markers in one Rerun scene."""
    BiorbdModel, PhaseRerun, PyoMarkers = require_pyorerun()

    # Use the BVH sampling times as the shared timeline. Both FBX q and the C3D
    # points are interpolated to it so pyorerun can add every object to one phase.
    fbx_q_on_bvh_time = interpolate_array(fbx_q, fbx_time, bvh_time)
    c3d_markers_on_bvh_time = interpolate_array(c3d_markers_bvh_units, c3d_time, bvh_time)
    c3d_markers_on_bvh_time, c3d_marker_labels = filter_display_markers_for_rerun(
        c3d_markers_on_bvh_time,
        c3d_marker_labels,
        hide_hands_in_rerun,
        hide_feet_in_rerun,
    )
    bvh_biomod_path = pyorerun_display_biomod_path(bvh_biomod_path, hide_hands_in_rerun, hide_feet_in_rerun)
    fbx_biomod_path = pyorerun_display_biomod_path(fbx_biomod_path, hide_hands_in_rerun, hide_feet_in_rerun)

    bvh_model = BiorbdModel(str(bvh_biomod_path))
    fbx_model = BiorbdModel(str(fbx_biomod_path))
    use_vtp_meshes_for_pyorerun(bvh_model)
    use_vtp_meshes_for_pyorerun(fbx_model)
    if bvh_q.shape[0] != bvh_model.nb_q:
        raise RuntimeError(f"BVH q has {bvh_q.shape[0]} rows, but its model has {bvh_model.nb_q} DoFs.")
    if fbx_q_on_bvh_time.shape[0] != fbx_model.nb_q:
        raise RuntimeError(f"FBX q has {fbx_q_on_bvh_time.shape[0]} rows, but its model has {fbx_model.nb_q} DoFs.")

    for model in (bvh_model, fbx_model):
        model.options.show_marker_labels = False
        model.options.show_center_of_mass_labels = False
        model.options.markers_radius = rerun_marker_radius
        model.options.centers_of_mass_radius = rerun_marker_radius

    # The FBX model carries the body surfaces. Keep it opaque because pyorerun's
    # transparent mode draws a very thin wireframe that is almost invisible for
    # millimetre-scale models.
    fbx_model.options.mesh_color = (70, 178, 160)
    fbx_model.options.transparent_mesh = False

    def _build_phase(display_q: bool):
        phase_local = PhaseRerun(bvh_time)
        phase_local.add_animated_model(bvh_model, bvh_q, display_q=display_q)
        phase_local.add_animated_model(fbx_model, fbx_q_on_bvh_time, display_q=display_q)
        if c3d_markers_on_bvh_time.size and c3d_marker_labels:
            phase_local.add_xp_markers(
                "c3d_markers_no_angle_channels",
                PyoMarkers(
                    data=c3d_markers_on_bvh_time,
                    channels=c3d_marker_labels,
                    show_labels=False,
                ),
            )
        set_pyorerun_marker_radius(phase_local, rerun_marker_radius)
        return phase_local

    try:
        phase = _build_phase(display_q=display_q_in_rerun)
    except AttributeError as exc:
        if display_q_in_rerun and "SeriesLine" in str(exc):
            warnings.warn(
                "pyorerun could not display q time-series; displaying the superposed animation without q plots.",
                RuntimeWarning,
            )
            phase = _build_phase(display_q=False)
        else:
            raise

    run_phase_with_rerun_view(phase, name, rerun_up_axis, rerun_wait_seconds)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a biorbd bioMod from BVH using BioBuddy, export BVH q, overlay C3D markers, and compare q to C3D angles."
    )
    parser.add_argument("--bvh", default=Path("data/unknown.bvh"), type=Path, help="Input BVH file.")
    parser.add_argument("--fbx", default=None, type=Path, help="Optional input FBX file.")
    parser.add_argument("--c3d", default=Path("data/unknown.c3d"), type=Path, help="Input C3D file.")
    parser.add_argument("--out-dir", default=Path("out_biobuddy_bvh_c3d"), type=Path, help="Output directory.")
    parser.add_argument(
        "--bvh-unit-scale-to-m",
        default=0.001,
        type=float,
        help="Scale from native BVH units to metres. Used for C3D conversion only. Default: 0.001 for mm.",
    )
    parser.add_argument(
        "--fbx-unit-scale-to-m",
        default=0.001,
        type=float,
        help="Scale from native FBX units to metres. Used for C3D conversion only. Default: 0.001 for mm.",
    )
    parser.add_argument(
        "--c3d-angle-unit",
        default="deg",
        choices=["deg", "rad"],
        help="Unit of the C3D angle point components. Default: deg.",
    )
    parser.add_argument(
        "--angle-label-regex",
        default=r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)",
        help="Regex used on C3D point labels/descriptions to identify angle point channels.",
    )
    parser.add_argument(
        "--extra-angle-label",
        action="append",
        default=[],
        help="Additional C3D point label to treat as an angle channel. Can be repeated.",
    )
    parser.add_argument(
        "--comparison-map",
        default=None,
        type=Path,
        help="Optional JSON mapping between BVH q names and C3D angle labels/components.",
    )
    parser.add_argument(
        "--no-biomod-joint-centre-markers",
        action="store_true",
        help="Do not append joint-centre markers to the generated bioMod.",
    )
    parser.add_argument(
        "--no-root-offset-correction",
        action="store_true",
        help=(
            "Deprecated shortcut for --root-offset-mode keep."
        ),
    )
    parser.add_argument(
        "--root-offset-mode",
        default="auto",
        choices=["auto", "subtract", "keep"],
        help=(
            "How to handle the static root OFFSET/Lcl Translation written in the bioMod. "
            "'auto' compares both choices against the C3D marker cloud and keeps the best overlay."
        ),
    )
    parser.add_argument(
        "--no-fbx-mesh",
        action="store_true",
        help="Do not append FBX mesh vertices to the generated FBX bioMod.",
    )
    parser.add_argument(
        "--max-fbx-mesh-points",
        default=0,
        type=int,
        help="Maximum FBX mesh vertices to export to OBJ. Use 0 for all vertices/faces.",
    )
    parser.add_argument("--animate", action="store_true", help="Launch a pyorerun animation.")
    parser.add_argument(
        "--animate-superposed",
        action="store_true",
        help="Launch one pyorerun scene containing the BVH model, FBX model and filtered C3D markers.",
    )
    parser.add_argument(
        "--display-q-in-rerun",
        action="store_true",
        help=(
            "Also display q time-series inside rerun. Disabled by default because some "
            "pyorerun/rerun-sdk version combinations fail on rerun.SeriesLine."
        ),
    )
    parser.add_argument(
        "--rerun-marker-radius",
        default=DEFAULT_RERUN_MARKER_RADIUS_NATIVE,
        type=float,
        help="Marker radius used in pyorerun scenes, in the model native length unit. Default: 15 for mm models.",
    )
    parser.add_argument(
        "--rerun-wait-seconds",
        default=DEFAULT_RERUN_WAIT_SECONDS,
        type=float,
        help="Seconds to keep the Python process alive after sending data to Rerun. Default: 2.",
    )
    parser.add_argument(
        "--rerun-up-axis",
        default=DEFAULT_RERUN_UP_AXIS,
        choices=["x", "y", "z", "none"],
        help="Vertical axis declared to Rerun for pyorerun 3D views. Default: y for Captury/BioBuddy models.",
    )
    parser.add_argument(
        "--hide-hands-in-rerun",
        action="store_true",
        help=(
            "Hide hand/wrist/finger meshes and markers in pyorerun animations. "
            "Numerical outputs and generated bioMod files remain complete."
        ),
    )
    parser.add_argument(
        "--hide-feet-in-rerun",
        action="store_true",
        help=(
            "Hide foot/ankle/toe meshes and markers in pyorerun animations. "
            "Numerical outputs and generated bioMod files remain complete."
        ),
    )
    parser.add_argument(
        "--hide-extremities-in-rerun",
        action="store_true",
        help=(
            "Hide both hands and feet in pyorerun animations. This is a display-only alias for "
            "--hide-hands-in-rerun plus --hide-feet-in-rerun."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Set PYORERUN_HEADLESS=1 before animation. Useful for tests or servers.",
    )
    parser.add_argument(
        "--inverse-kinematics",
        action="store_true",
        help=(
            "Run marker-based inverse kinematics from the C3D markers. C3D angle channels are ignored."
        ),
    )
    parser.add_argument(
        "--inverse-kinematics-solver",
        default="least_squares",
        choices=["least_squares", "kalman"],
        help="Inverse kinematics solver. Default: least_squares.",
    )
    parser.add_argument(
        "--inverse-kinematics-method",
        default="trf",
        choices=["trf", "lm", "only_lm"],
        help="Least-squares method passed to biorbd.InverseKinematics.solve. Used only with --inverse-kinematics-solver least_squares.",
    )
    parser.add_argument(
        "--inverse-kinematics-max-frames",
        default=0,
        type=int,
        help="Limit inverse kinematics to the first N C3D frames. Use 0 for all frames.",
    )
    parser.add_argument(
        "--kalman-noise-factor",
        default=1e-10,
        type=float,
        help="Noise factor for biorbd.KalmanParam when using the Kalman IK solver.",
    )
    parser.add_argument(
        "--kalman-error-factor",
        default=1e-5,
        type=float,
        help="Error factor for biorbd.KalmanParam when using the Kalman IK solver.",
    )
    parser.add_argument(
        "--inverse-dynamics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inverse-dynamics-method",
        default=None,
        choices=["trf", "lm", "only_lm"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inverse-dynamics-max-frames",
        default=None,
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inverse_dynamics:
        warnings.warn(
            "--inverse-dynamics is deprecated; use --inverse-kinematics. The script now computes IK only.",
            DeprecationWarning,
        )
        args.inverse_kinematics = True
    if args.inverse_dynamics_method is not None:
        args.inverse_kinematics_method = args.inverse_dynamics_method
    if args.inverse_dynamics_max_frames is not None:
        args.inverse_kinematics_max_frames = args.inverse_dynamics_max_frames
    if args.animate_superposed and args.fbx is None:
        raise ValueError("--animate-superposed requires --fbx because both generated models are displayed.")
    hide_hands_in_rerun = args.hide_hands_in_rerun or args.hide_extremities_in_rerun
    hide_feet_in_rerun = args.hide_feet_in_rerun or args.hide_extremities_in_rerun
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    root_offset_mode = "keep" if args.no_root_offset_correction else args.root_offset_mode

    # 1. Build the BVH model. The bioMod is the biorbd model file used by all
    # later steps: joint centres, local marker coordinates and animation.
    biomod_path = out_dir / "model_from_bvh_biobuddy.bioMod"
    model, parser = build_biomod_from_bvh_with_biobuddy(
        bvh_path=args.bvh,
        biomod_path=biomod_path,
        add_joint_centre_markers=not args.no_biomod_joint_centre_markers,
    )

    # 2. Read the C3D once and split POINT channels into real markers vs angles.
    # This avoids accidentally animating Captury joint angles as if they were
    # marker positions.
    c3d_split = split_c3d_points(
        c3d_path=args.c3d,
        bvh_unit_scale_to_m=args.bvh_unit_scale_to_m,
        angle_label_regex=args.angle_label_regex,
        extra_angle_labels=args.extra_angle_label,
    )
    c3d_markers_npz, c3d_angles_npz, detected_angle_labels_txt = save_c3d_split_outputs(c3d_split, out_dir)

    # 3. Extract BVH q twice: one version subtracts the static root offset, the
    # other keeps root translation exactly as written in the BVH. The policy
    # selector below keeps the version that overlays better with the C3D markers.
    bvh_corrected_runtime = extract_q_from_biobuddy_bvh_parser(parser, apply_root_offset_correction=True)
    bvh_uncorrected_runtime = extract_q_from_biobuddy_bvh_parser(parser, apply_root_offset_correction=False)
    bvh_use_correction, bvh_policy_report, centres_native = choose_root_offset_policy(
        source_name="bvh",
        biomod_path=biomod_path,
        corrected_q=bvh_corrected_runtime.q,
        uncorrected_q=bvh_uncorrected_runtime.q,
        q_names=bvh_corrected_runtime.q_names,
        time=bvh_corrected_runtime.time,
        joint_names=bvh_corrected_runtime.joint_names,
        c3d_markers_source_units=marker_data_in_source_units(c3d_split, args.bvh_unit_scale_to_m),
        c3d_time=c3d_split.time,
        requested_mode=root_offset_mode,
        out_dir=out_dir,
    )
    bvh_runtime = bvh_corrected_runtime if bvh_use_correction else bvh_uncorrected_runtime
    # 4. Save q in both NPZ and CSV. NPZ is convenient for Python; CSV is easier
    # to inspect in a spreadsheet.
    q_npz, q_csv = save_q_outputs(
        bvh_runtime.q,
        bvh_runtime.q_names,
        bvh_runtime.time,
        out_dir,
        "bvh",
        bvh_runtime.q_units,
    )
    joint_centres_npz = save_model_joint_centres(centres_native, bvh_runtime.time, out_dir, "bvh")
    # 5. Use biorbd segment poses to express each C3D marker in local segment
    # frames. A true anatomical marker should be almost fixed in one local frame.
    bvh_local_markers_csv, bvh_local_marker_summary = compute_and_append_c3d_local_markers(
        biomod_path=biomod_path,
        q=bvh_runtime.q,
        source_time=bvh_runtime.time,
        split=c3d_split,
        source_unit_scale_to_m=args.bvh_unit_scale_to_m,
        source_name="bvh",
        out_dir=out_dir,
    )
    bvh_marker_error_csv, bvh_marker_error_summary_path, bvh_marker_error_boxplot, bvh_marker_error_summary, bvh_marker_errors = (
        save_and_plot_c3d_marker_error_norms(
            biomod_path=biomod_path,
            q=bvh_runtime.q,
            source_time=bvh_runtime.time,
            split=c3d_split,
            local_marker_csv=bvh_local_markers_csv,
            source_unit_scale_to_m=args.bvh_unit_scale_to_m,
            source_name="bvh",
            out_dir=out_dir,
        )
    )
    bvh_inverse_kinematics_report: dict[str, Any] | None = None
    if args.inverse_kinematics:
        bvh_inverse_kinematics_report = run_inverse_kinematics_from_c3d_markers(
            biomod_path=biomod_path,
            split=c3d_split,
            local_marker_csv=bvh_local_markers_csv,
            source_unit_scale_to_m=args.bvh_unit_scale_to_m,
            source_name="bvh",
            out_dir=out_dir,
            solver=args.inverse_kinematics_solver,
            least_squares_method=args.inverse_kinematics_method,
            max_frames=args.inverse_kinematics_max_frames,
            kalman_noise_factor=args.kalman_noise_factor,
            kalman_error_factor=args.kalman_error_factor,
        )

    # 6. Create a C3D copy with generated joint centres so external tools can
    # inspect markers and model centres on the same time base.
    augmented_c3d_path = append_joint_centres_to_c3d(
        split=c3d_split,
        centres_native=centres_native,
        source_time=bvh_runtime.time,
        source_unit_scale_to_m=args.bvh_unit_scale_to_m,
        output_path=out_dir / "c3d_with_bvh_joint_centres.c3d",
        label_prefix="BVHJC_",
        description="BVH joint centre generated from BioBuddy BVH parser",
    )
    bvh_animation_markers, bvh_animation_marker_labels, bvh_animation_markers_npz = (
        build_animation_markers_with_joint_centres(
            split=c3d_split,
            centres_native=centres_native,
            source_time=bvh_runtime.time,
            source_unit_scale_to_m=args.bvh_unit_scale_to_m,
            label_prefix="BVHJC_",
            out_dir=out_dir,
            source_name="bvh",
        )
    )

    # 7. Compare rotational q channels to the angle channels found in the C3D.
    pairwise_csv, best_csv = write_pairwise_q_vs_c3d_angle_comparison(
        q=bvh_runtime.q,
        q_names=bvh_runtime.q_names,
        bvh_time=bvh_runtime.time,
        c3d_angle_data=c3d_split.angle_data,
        c3d_angle_labels=c3d_split.angle_labels,
        c3d_time=c3d_split.time,
        c3d_angle_unit=args.c3d_angle_unit,
        out_dir=out_dir,
    )
    explicit_csv = write_explicit_q_vs_c3d_angle_comparison(
        mapping_path=args.comparison_map,
        q=bvh_runtime.q,
        q_names=bvh_runtime.q_names,
        bvh_time=bvh_runtime.time,
        c3d_angle_data=c3d_split.angle_data,
        c3d_angle_labels=c3d_split.angle_labels,
        c3d_time=c3d_split.time,
        c3d_angle_unit=args.c3d_angle_unit,
        out_dir=out_dir,
    )
    mapping_template = write_mapping_template(bvh_runtime.q_names, c3d_split.angle_labels, out_dir)

    if args.animate and not args.animate_superposed:
        # Animation is optional because it opens/runs rerun. The numeric outputs
        # above are produced even when --animate is not used.
        if args.headless:
            os.environ["PYORERUN_HEADLESS"] = "1"
        animate_with_pyorerun(
            biomod_path=biomod_path,
            q=bvh_runtime.q,
            bvh_time=bvh_runtime.time,
            c3d_markers_bvh_units=bvh_animation_markers,
            c3d_marker_labels=bvh_animation_marker_labels,
            c3d_time=c3d_split.time,
            display_q_in_rerun=args.display_q_in_rerun,
            rerun_marker_radius=args.rerun_marker_radius,
            rerun_wait_seconds=args.rerun_wait_seconds,
            rerun_up_axis=args.rerun_up_axis,
            hide_hands_in_rerun=hide_hands_in_rerun,
            hide_feet_in_rerun=hide_feet_in_rerun,
        )

    fbx_outputs: dict[str, Any] | None = None
    fbx_report: dict[str, Any] | None = None
    if args.fbx is not None:
        # Repeat the same model/q/local-marker workflow for the FBX file when it
        # is provided. The FBX branch also writes segment mesh surfaces.
        fbx_biomod_path = out_dir / "model_from_fbx_biobuddy.bioMod"
        _, fbx_parser, fbx_mesh_report = build_biomod_from_fbx_with_biobuddy(
            fbx_path=args.fbx,
            biomod_path=fbx_biomod_path,
            add_joint_centre_markers=not args.no_biomod_joint_centre_markers,
            include_mesh=not args.no_fbx_mesh,
            max_mesh_points=args.max_fbx_mesh_points,
        )
        fbx_corrected_runtime = extract_q_from_fbx_parser(
            fbx_parser, args.fbx, apply_root_offset_correction=True
        )
        fbx_uncorrected_runtime = extract_q_from_fbx_parser(
            fbx_parser, args.fbx, apply_root_offset_correction=False
        )
        fbx_use_correction, fbx_policy_report, fbx_centres_native = choose_root_offset_policy(
            source_name="fbx",
            biomod_path=fbx_biomod_path,
            corrected_q=fbx_corrected_runtime.q,
            uncorrected_q=fbx_uncorrected_runtime.q,
            q_names=fbx_corrected_runtime.q_names,
            time=fbx_corrected_runtime.time,
            joint_names=fbx_corrected_runtime.joint_names,
            c3d_markers_source_units=marker_data_in_source_units(c3d_split, args.fbx_unit_scale_to_m),
            c3d_time=c3d_split.time,
            requested_mode=root_offset_mode,
            out_dir=out_dir,
        )
        fbx_runtime = fbx_corrected_runtime if fbx_use_correction else fbx_uncorrected_runtime
        fbx_q_npz, fbx_q_csv = save_q_outputs(
            fbx_runtime.q,
            fbx_runtime.q_names,
            fbx_runtime.time,
            out_dir,
            "fbx",
            fbx_runtime.q_units,
        )
        fbx_joint_centres_npz = save_model_joint_centres(
            fbx_centres_native, fbx_runtime.time, out_dir, "fbx"
        )
        fbx_local_markers_csv, fbx_local_marker_summary = compute_and_append_c3d_local_markers(
            biomod_path=fbx_biomod_path,
            q=fbx_runtime.q,
            source_time=fbx_runtime.time,
            split=c3d_split,
            source_unit_scale_to_m=args.fbx_unit_scale_to_m,
            source_name="fbx",
            out_dir=out_dir,
        )
        fbx_marker_error_csv, fbx_marker_error_summary_path, fbx_marker_error_boxplot, fbx_marker_error_summary, fbx_marker_errors = (
            save_and_plot_c3d_marker_error_norms(
                biomod_path=fbx_biomod_path,
                q=fbx_runtime.q,
                source_time=fbx_runtime.time,
                split=c3d_split,
                local_marker_csv=fbx_local_markers_csv,
                source_unit_scale_to_m=args.fbx_unit_scale_to_m,
                source_name="fbx",
                out_dir=out_dir,
            )
        )
        fbx_inverse_kinematics_report: dict[str, Any] | None = None
        if args.inverse_kinematics:
            fbx_inverse_kinematics_report = run_inverse_kinematics_from_c3d_markers(
                biomod_path=fbx_biomod_path,
                split=c3d_split,
                local_marker_csv=fbx_local_markers_csv,
                source_unit_scale_to_m=args.fbx_unit_scale_to_m,
                source_name="fbx",
                out_dir=out_dir,
                solver=args.inverse_kinematics_solver,
                least_squares_method=args.inverse_kinematics_method,
                max_frames=args.inverse_kinematics_max_frames,
                kalman_noise_factor=args.kalman_noise_factor,
                kalman_error_factor=args.kalman_error_factor,
            )
        fbx_augmented_c3d_path = append_joint_centres_to_c3d(
            split=c3d_split,
            centres_native=fbx_centres_native,
            source_time=fbx_runtime.time,
            source_unit_scale_to_m=args.fbx_unit_scale_to_m,
            output_path=out_dir / "c3d_with_fbx_joint_centres.c3d",
            label_prefix="FBXJC_",
            description="FBX joint centre generated from BioBuddy FBX parser",
        )
        fbx_animation_markers, fbx_animation_marker_labels, fbx_animation_markers_npz = (
            build_animation_markers_with_joint_centres(
                split=c3d_split,
                centres_native=fbx_centres_native,
                source_time=fbx_runtime.time,
                source_unit_scale_to_m=args.fbx_unit_scale_to_m,
                label_prefix="FBXJC_",
                out_dir=out_dir,
                source_name="fbx",
            )
        )
        if args.animate and not args.animate_superposed:
            animate_with_pyorerun(
                biomod_path=fbx_biomod_path,
                q=fbx_runtime.q,
                bvh_time=fbx_runtime.time,
                c3d_markers_bvh_units=fbx_animation_markers,
                c3d_marker_labels=fbx_animation_marker_labels,
                c3d_time=c3d_split.time,
                name="fbx_biobuddy_c3d_comparison",
                display_q_in_rerun=args.display_q_in_rerun,
                rerun_marker_radius=args.rerun_marker_radius,
                rerun_wait_seconds=args.rerun_wait_seconds,
                rerun_up_axis=args.rerun_up_axis,
                hide_hands_in_rerun=hide_hands_in_rerun,
                hide_feet_in_rerun=hide_feet_in_rerun,
            )
        if args.animate_superposed:
            if args.headless:
                os.environ["PYORERUN_HEADLESS"] = "1"
            if not math.isclose(args.bvh_unit_scale_to_m, args.fbx_unit_scale_to_m):
                warnings.warn(
                    "BVH and FBX native unit scales differ; the shared overlay uses BVH units for C3D markers.",
                    RuntimeWarning,
                )
            animate_superposed_models_with_pyorerun(
                bvh_biomod_path=biomod_path,
                bvh_q=bvh_runtime.q,
                bvh_time=bvh_runtime.time,
                fbx_biomod_path=fbx_biomod_path,
                fbx_q=fbx_runtime.q,
                fbx_time=fbx_runtime.time,
                c3d_markers_bvh_units=marker_data_in_source_units(c3d_split, args.bvh_unit_scale_to_m),
                c3d_marker_labels=c3d_split.marker_labels,
                c3d_time=c3d_split.time,
                display_q_in_rerun=args.display_q_in_rerun,
                rerun_marker_radius=args.rerun_marker_radius,
                rerun_wait_seconds=args.rerun_wait_seconds,
                rerun_up_axis=args.rerun_up_axis,
                hide_hands_in_rerun=hide_hands_in_rerun,
                hide_feet_in_rerun=hide_feet_in_rerun,
            )
        fbx_outputs = {
            "biomod": str(fbx_biomod_path),
            "fbx_q_npz": str(fbx_q_npz),
            "fbx_q_csv": str(fbx_q_csv),
            "fbx_joint_centres_npz": str(fbx_joint_centres_npz),
            "augmented_c3d": str(fbx_augmented_c3d_path),
            "animation_markers_no_angles_with_joint_centres": str(fbx_animation_markers_npz),
            "local_markers_csv": str(fbx_local_markers_csv),
            "marker_error_norm_csv": str(fbx_marker_error_csv),
            "marker_error_norm_summary": str(fbx_marker_error_summary_path),
            "marker_error_norm_boxplot": str(fbx_marker_error_boxplot),
            "root_policy": str(out_dir / "fbx_root_translation_policy.json"),
            "inverse_kinematics": fbx_inverse_kinematics_report,
            "superposed_animation_requested": bool(args.animate_superposed),
        }
        fbx_report = {
            "input_fbx": str(args.fbx),
            "mesh": fbx_mesh_report,
            "root_offset_correction": {
                "applied": bool(fbx_runtime.root_offset_correction_applied),
                "root_offset_native": (
                    fbx_runtime.root_offset_native.tolist()
                    if fbx_runtime.root_offset_native is not None
                    else None
                ),
                "policy": fbx_policy_report,
            },
            "q_units": {
                "translations": "native FBX length unit, matching the generated bioMod RT values",
                "rotations": "radians, converted from FBX degrees and unwrapped per Euler channel",
            },
            "q_unwrap": fbx_runtime.unwrap_summary,
            "local_marker_stability": fbx_local_marker_summary,
            "marker_error_norm_mm": fbx_marker_error_summary,
            "inverse_kinematics_from_c3d_markers": fbx_inverse_kinematics_report,
            "counts": {
                "fbx_joints": len(fbx_runtime.joint_names),
                "fbx_q": int(fbx_runtime.q.shape[0]),
                "fbx_frames": int(fbx_runtime.q.shape[1]),
                "fbx_animation_points_with_joint_centres": len(fbx_animation_marker_labels),
            },
            "q_names": fbx_runtime.q_names,
        }

    marker_error_sets = {"bvh": bvh_marker_errors}
    if args.fbx is not None:
        marker_error_sets["fbx"] = fbx_marker_errors
    overall_marker_error_boxplot = plot_overall_model_marker_error_boxplot(marker_error_sets, out_dir)

    report = {
        "input_bvh": str(args.bvh),
        "input_fbx": str(args.fbx) if args.fbx else None,
        "input_c3d": str(args.c3d),
        "biobuddy_branch": "mickaelbegon/biobuddy@codex/add-fbx-segment-meshes",
        "outputs": {
            "biomod": str(biomod_path),
            "bvh_q_npz": str(q_npz),
            "bvh_q_csv": str(q_csv),
            "bvh_joint_centres_npz": str(joint_centres_npz),
            "c3d_markers_npz": str(c3d_markers_npz),
            "c3d_angles_npz": str(c3d_angles_npz),
            "detected_c3d_angle_labels": str(detected_angle_labels_txt),
            "augmented_c3d": str(augmented_c3d_path),
            "pairwise_comparison_csv": str(pairwise_csv),
            "best_matches_csv": str(best_csv),
            "explicit_mapping_csv": str(explicit_csv) if explicit_csv else None,
            "comparison_mapping_template": str(mapping_template),
            "bvh_local_markers_csv": str(bvh_local_markers_csv),
            "bvh_marker_error_norm_csv": str(bvh_marker_error_csv),
            "bvh_marker_error_norm_summary": str(bvh_marker_error_summary_path),
            "bvh_marker_error_norm_boxplot": str(bvh_marker_error_boxplot),
            "overall_marker_error_norm_boxplot": str(overall_marker_error_boxplot),
            "bvh_root_policy": str(out_dir / "bvh_root_translation_policy.json"),
            "bvh_animation_markers_no_angles_with_joint_centres": str(bvh_animation_markers_npz),
            "bvh_inverse_kinematics": bvh_inverse_kinematics_report,
            "fbx": fbx_outputs,
        },
        "counts": {
            "bvh_joints": len(bvh_runtime.joint_names),
            "bvh_q": int(bvh_runtime.q.shape[0]),
            "bvh_frames": int(bvh_runtime.q.shape[1]),
            "c3d_marker_points_used_for_animation": len(c3d_split.marker_labels),
            "bvh_animation_points_with_joint_centres": len(bvh_animation_marker_labels),
            "c3d_angle_point_channels_detected": len(c3d_split.angle_labels),
            "c3d_frames": int(c3d_split.marker_data_native.shape[2]),
        },
        "units": {
            "q_translations": "native BVH/FBX length unit, matching the generated bioMod RT values; root translations corrected for static root offset according to the selected policy",
            "q_rotations": "radians, converted from source degrees and unwrapped per Euler channel",
            "bvh_unit_scale_to_m": args.bvh_unit_scale_to_m,
            "fbx_unit_scale_to_m": args.fbx_unit_scale_to_m,
            "c3d_point_unit_scale_to_m": c3d_split.c3d_unit_scale_to_m,
            "c3d_angle_unit": args.c3d_angle_unit,
        },
        "root_offset_correction": {
            "applied": bool(bvh_runtime.root_offset_correction_applied),
            "root_offset_native": (
                bvh_runtime.root_offset_native.tolist()
                if bvh_runtime.root_offset_native is not None
                else None
            ),
            "policy": bvh_policy_report,
        },
        "q_units": {
            "translations": "native BVH length unit, matching the generated bioMod RT values",
            "rotations": "radians, converted from BVH degrees and unwrapped per Euler channel",
        },
        "q_unwrap": bvh_runtime.unwrap_summary,
        "local_marker_stability": bvh_local_marker_summary,
        "marker_error_norm_mm": bvh_marker_error_summary,
        "inverse_kinematics_from_c3d_markers": bvh_inverse_kinematics_report,
        "fbx_report": fbx_report,
        "q_names": bvh_runtime.q_names,
        "c3d_angle_labels": c3d_split.angle_labels,
    }
    report_path = out_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Done.")
    print(f"bioMod: {biomod_path}")
    print(f"BVH q: {q_npz}")
    print(f"Augmented C3D: {augmented_c3d_path}")
    print(f"BVH local C3D markers: {bvh_local_markers_csv}")
    print(f"BVH marker error boxplot: {bvh_marker_error_boxplot}")
    if bvh_inverse_kinematics_report is not None:
        print(f"BVH inverse kinematics: {bvh_inverse_kinematics_report['npz']}")
    if fbx_outputs:
        print(f"FBX bioMod: {fbx_outputs['biomod']}")
        print(f"FBX q: {fbx_outputs['fbx_q_npz']}")
        print(f"FBX augmented C3D: {fbx_outputs['augmented_c3d']}")
        print(f"FBX marker error boxplot: {fbx_outputs['marker_error_norm_boxplot']}")
        if fbx_outputs.get("inverse_kinematics") is not None:
            print(f"FBX inverse kinematics: {fbx_outputs['inverse_kinematics']['npz']}")
    print(f"Pairwise comparison: {pairwise_csv}")
    print(f"Best matches: {best_csv}")
    print(f"Overall marker error boxplot: {overall_marker_error_boxplot}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
