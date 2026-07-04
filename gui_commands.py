"""Command builders used by the Captury/BioBuddy Tk GUI.

The GUI stores options in Tk variables, but command construction itself is pure:
these helpers accept plain values and return argv lists. Keeping this layer out
of the Tk class makes it easier to test and keeps the GUI focused on user
interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_DIR / "bvh_c3d_biobuddy_pyorerun_compare.py"
MODEL_EDITOR_SCRIPT = PROJECT_DIR / "launch_biobuddy_model_editor.py"
COMPARISON_SCRIPT = PROJECT_DIR / "compare_capture_systems.py"
KINEMATIC_COMPARISON_SCRIPT = PROJECT_DIR / "compare_p6_motive_captury.py"
C3D_VIEWER_SCRIPT = PROJECT_DIR / "c3d_trial_viewer.py"
BIOBUDDY_C3D_MODEL_SCRIPT = PROJECT_DIR / "create_biobuddy_c3d_model.py"

COMMAND_MODES = {
    "kinematic": "Analyse Captury/Motive",
    "biobuddy_c3d_model": "Modèle BioBuddy C3D",
    "pipeline": "Pipeline BVH/FBX/C3D",
    "comparison": "Comparaison générique",
}


def value_of(values: Mapping[str, object], name: str) -> str:
    return str(values.get(name, "")).strip()


def bool_value(values: Mapping[str, object], name: str) -> bool:
    return bool(values.get(name, False))


def append_value(
    args: list[str],
    values: Mapping[str, object],
    option: str,
    name: str,
    *,
    required: bool = False,
) -> None:
    value = value_of(values, name)
    if value or required:
        args.extend([option, value])


def append_flag(
    args: list[str], values: Mapping[str, object], option: str, name: str
) -> None:
    if bool_value(values, name):
        args.append(option)


def split_lines(value: object) -> list[str]:
    return [
        part.strip()
        for part in str(value).replace("\n", ",").split(",")
        if part.strip()
    ]


def split_extra_labels(values: Mapping[str, object]) -> list[str]:
    raw = str(values.get("extra_angle_labels", ""))
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


def build_pipeline_args(values: Mapping[str, object]) -> list[str]:
    args = [sys.executable, str(PIPELINE_SCRIPT)]
    append_value(args, values, "--bvh", "bvh", required=True)
    append_value(args, values, "--fbx", "fbx")
    append_value(args, values, "--c3d", "c3d", required=True)
    append_value(args, values, "--out-dir", "out_dir", required=True)

    append_value(args, values, "--bvh-unit-scale-to-m", "bvh_unit_scale_to_m")
    append_value(args, values, "--fbx-unit-scale-to-m", "fbx_unit_scale_to_m")
    append_value(args, values, "--c3d-angle-unit", "c3d_angle_unit")
    append_value(args, values, "--angle-label-regex", "angle_label_regex")
    for label in split_extra_labels(values):
        args.extend(["--extra-angle-label", label])
    append_value(args, values, "--comparison-map", "comparison_map")

    append_flag(
        args,
        values,
        "--no-biomod-joint-centre-markers",
        "no_biomod_joint_centre_markers",
    )
    append_flag(
        args, values, "--no-root-offset-correction", "no_root_offset_correction"
    )
    append_value(args, values, "--root-offset-mode", "root_offset_mode")
    append_flag(args, values, "--no-fbx-mesh", "no_fbx_mesh")
    append_value(args, values, "--max-fbx-mesh-points", "max_fbx_mesh_points")

    append_flag(args, values, "--animate", "animate")
    append_flag(args, values, "--animate-superposed", "animate_superposed")
    append_flag(args, values, "--display-q-in-rerun", "display_q_in_rerun")
    append_value(args, values, "--rerun-marker-radius", "rerun_marker_radius")
    append_value(args, values, "--rerun-wait-seconds", "rerun_wait_seconds")
    append_value(args, values, "--rerun-up-axis", "rerun_up_axis")
    append_flag(args, values, "--hide-hands-in-rerun", "hide_hands_in_rerun")
    append_flag(args, values, "--hide-feet-in-rerun", "hide_feet_in_rerun")
    append_flag(
        args, values, "--hide-extremities-in-rerun", "hide_extremities_in_rerun"
    )
    append_flag(args, values, "--headless", "headless")

    append_flag(args, values, "--inverse-kinematics", "inverse_kinematics")
    append_value(
        args, values, "--inverse-kinematics-solver", "inverse_kinematics_solver"
    )
    append_value(
        args, values, "--inverse-kinematics-method", "inverse_kinematics_method"
    )
    append_value(
        args,
        values,
        "--inverse-kinematics-max-frames",
        "inverse_kinematics_max_frames",
    )
    append_value(args, values, "--kalman-noise-factor", "kalman_noise_factor")
    append_value(args, values, "--kalman-error-factor", "kalman_error_factor")

    append_flag(args, values, "--inverse-dynamics", "inverse_dynamics")
    append_value(args, values, "--inverse-dynamics-method", "inverse_dynamics_method")
    append_value(
        args, values, "--inverse-dynamics-max-frames", "inverse_dynamics_max_frames"
    )
    return args


def build_comparison_args(values: Mapping[str, object]) -> list[str]:
    args = [sys.executable, str(COMPARISON_SCRIPT)]
    reference_c3d = value_of(values, "compare_reference_c3d")
    test_c3d = value_of(values, "compare_test_c3d")
    append_value(args, values, "--reference-system", "compare_reference_system")
    append_value(args, values, "--test-system", "compare_test_system")
    if reference_c3d or test_c3d:
        args.extend(["--reference-c3d", reference_c3d, "--test-c3d", test_c3d])
        append_value(args, values, "--reference-bvh", "compare_reference_bvh")
        append_value(args, values, "--reference-fbx", "compare_reference_fbx")
        append_value(args, values, "--test-bvh", "compare_test_bvh")
        append_value(args, values, "--test-fbx", "compare_test_fbx")
        trial_name = value_of(values, "compare_trial_name")
        if trial_name:
            args.extend(["--trial-name", trial_name])
    else:
        args.extend(["--data-root", value_of(values, "compare_data_root")])
        for pattern in split_lines(values.get("compare_participant_filter", "")):
            args.extend(["--participant-filter", pattern])
        for pattern in split_lines(values.get("compare_trial_filter", "")):
            args.extend(["--trial-filter", pattern])
    append_value(args, values, "--out-dir", "compare_out_dir")
    append_value(args, values, "--landmark-map", "compare_landmark_map")
    append_value(args, values, "--resample-points", "compare_resample_points")
    append_value(args, values, "--alignment", "compare_alignment")
    return args


def build_p6_args(values: Mapping[str, object]) -> list[str]:
    args = [sys.executable, str(KINEMATIC_COMPARISON_SCRIPT)]
    append_value(args, values, "--data-root", "p6_data_root")
    append_value(args, values, "--out-dir", "p6_out_dir")
    for trial in split_lines(values.get("p6_trials", "")):
        args.extend(["--trial", trial])
    append_p6_common_args(args, values)
    append_flag(args, values, "--no-figures", "p6_no_figures")
    append_flag(args, values, "--no-cache", "p6_no_cache")
    append_flag(args, values, "--no-mesh", "p6_no_mesh")
    append_value(args, values, "--max-mesh-points", "p6_max_mesh_points")
    append_flag(args, values, "--run-ik-batch", "p6_run_ik_batch")
    append_value(args, values, "--ik-max-frames", "p6_ik_max_frames")
    append_flag(args, values, "--visualize", "p6_visualize")
    append_value(args, values, "--visualize-trial", "p6_visualize_trial")
    append_flag(args, values, "--headless", "p6_headless")
    append_value(args, values, "--rerun-wait-seconds", "p6_rerun_wait_seconds")
    return args


def append_p6_common_args(args: list[str], values: Mapping[str, object]) -> None:
    for pattern in split_lines(values.get("p6_joint_filter", "")):
        args.extend(["--joint-filter", pattern])
    append_value(args, values, "--static-trial", "p6_static_trial")
    append_value(args, values, "--cut-mode", "p6_cut_mode")
    append_value(args, values, "--time-start", "p6_time_start")
    append_value(args, values, "--time-end", "p6_time_end")
    append_value(args, values, "--model-source", "p6_model_source")
    append_value(args, values, "--root-offset-mode", "root_offset_mode")
    append_value(args, values, "--model-to-c3d-axis", "p6_model_to_c3d_axis")
    append_value(args, values, "--c3d-angle-unit", "c3d_angle_unit")


def build_p6_occlusions_args(values: Mapping[str, object], trial: str) -> list[str]:
    return [
        sys.executable,
        str(KINEMATIC_COMPARISON_SCRIPT),
        "--occlusions-only",
        "--no-figures",
        "--data-root",
        value_of(values, "p6_data_root"),
        "--out-dir",
        value_of(values, "p6_out_dir"),
        "--trial",
        trial,
    ]


def build_p6_auto_analysis_args(values: Mapping[str, object], trial: str) -> list[str]:
    args = [
        sys.executable,
        str(KINEMATIC_COMPARISON_SCRIPT),
        "--data-root",
        value_of(values, "p6_data_root"),
        "--out-dir",
        value_of(values, "p6_out_dir"),
        "--trial",
        trial,
        "--no-figures",
        "--no-mesh",
        "--max-mesh-points",
        "0",
    ]
    append_p6_common_args(args, values)
    append_flag(args, values, "--no-cache", "p6_no_cache")
    return args


def build_biobuddy_c3d_model_args(values: Mapping[str, object]) -> list[str]:
    args = [
        sys.executable,
        str(BIOBUDDY_C3D_MODEL_SCRIPT),
        value_of(values, "biobuddy_c3d_folder"),
    ]
    append_value(args, values, "--preset", "biobuddy_c3d_preset")
    append_value(args, values, "--output", "biobuddy_c3d_output")
    append_value(args, values, "--motive-57-mapping-json", "biobuddy_c3d_mapping_json")
    append_flag(
        args,
        values,
        "--no-default-virtual-points",
        "biobuddy_c3d_no_default_virtual_points",
    )
    append_flag(args, values, "--with-mesh", "biobuddy_c3d_with_mesh")
    return args
