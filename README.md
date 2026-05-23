# captury_models

Small workspace for comparing Captury BVH/FBX skeleton exports with C3D marker data, generating biorbd `bioMod` models through BioBuddy, and checking whether C3D markers are stable in local segment frames.

## Contents

- `bvh_c3d_biobuddy_pyorerun_compare.py`: main BVH/FBX to `bioMod` pipeline.
- `plot_bvh_c3d_angle_comparisons.py`: optional plotting helper for BVH q versus C3D angle channels.
- `environment_bvh_c3d_biobuddy.yml`: conda environment definition.
- `unknown.bvh`, `unknown.fbx`, `unknown.c3d`: expected local Captury input files. They are ignored by git because they are data files.

Generated outputs are written to `out_biobuddy_bvh_c3d/` and are intentionally ignored by git.

## Environment

Create or update the conda environment:

```bash
conda env create -f environment_bvh_c3d_biobuddy.yml
conda activate bvh-c3d-biobuddy
```

If the environment already exists:

```bash
conda env update -n bvh-c3d-biobuddy -f environment_bvh_c3d_biobuddy.yml --prune
conda activate bvh-c3d-biobuddy
```

The environment installs BioBuddy from:

```text
git+https://github.com/mickaelbegon/biobuddy.git@codex/add-fbx-segment-meshes
```

## C3D Point Classification

The C3D contains both marker trajectories and angle-like point channels. The pipeline excludes detected angle channels from the marker cloud used for animation and local marker placement. In addition to labels/descriptions matching the angle regex, these Captury joint labels are treated as angles by default:

```text
RHip LHip RKne LKne RAnk LAnk RSho LSho RElb LElb RWri LWri
```

During animation, the marker overlay contains the C3D marker points after this angle filtering plus the generated joint centres (`BVHJC_*` or `FBXJC_*`). Angle point channels are never sent to pyorerun as markers.

## Run

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh unknown.bvh \
  --fbx unknown.fbx \
  --c3d unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d
```

To also launch pyorerun animations:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh unknown.bvh \
  --fbx unknown.fbx \
  --c3d unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --animate
```

To compute inverse kinematics from the C3D markers with biorbd nonlinear least squares, add:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh unknown.bvh \
  --fbx unknown.fbx \
  --c3d unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-solver least_squares
```

To use biorbd's marker Kalman reconstruction instead:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh unknown.bvh \
  --fbx unknown.fbx \
  --c3d unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-solver kalman
```

For a quick smoke test, limit inverse kinematics to the first few frames:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh unknown.bvh \
  --fbx unknown.fbx \
  --c3d unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-max-frames 5
```

Useful generated files include:

- `model_from_bvh_biobuddy.bioMod`
- `model_from_fbx_biobuddy.bioMod`
- `meshes/*.ply`
- `bvh_q_biorbd_order.npz`
- `fbx_q_biorbd_order.npz`
- `bvh_c3d_local_markers.csv`
- `fbx_c3d_local_markers.csv`
- `bvh_animation_markers_no_angles_with_joint_centres.npz`
- `fbx_animation_markers_no_angles_with_joint_centres.npz`
- `bvh_inverse_kinematics_from_c3d_markers.npz`
- `fbx_inverse_kinematics_from_c3d_markers.npz`
- `run_report.json`

## Root Translation Policy

Captury exports may store a static root offset in the skeleton while also storing root position channels in laboratory coordinates. The script defaults to `--root-offset-mode auto`: it compares the C3D marker cloud against joint centres with and without subtracting the static root offset from the root translation q, then keeps the better overlay. The selected policy is written to:

- `bvh_root_translation_policy.json`
- `fbx_root_translation_policy.json`

Use `--root-offset-mode subtract` or `--root-offset-mode keep` to force either convention.

## Generalized Coordinate Units

The exported `*_q_biorbd_order.npz` files are now populated from BioBuddy's `to_q()` output for both BVH and FBX, so they follow the DOF order expected by the generated `bioMod` (`*_transX`, `*_rotZ`, etc.). Translation channels remain in the native length unit of the BVH/FBX file so they match the `RT` offsets written in the `bioMod`. Rotation channels are returned by BioBuddy in radians, then unwrapped per Euler channel before saving and animation. The `.npz` files include `q_units`, and `run_report.json` includes an unwrap summary.

## FBX Mesh

The FBX mesh is handled by the BioBuddy branch `codex/add-fbx-segment-meshes`: the skinned visual mesh is split into per-segment `.ply` files and referenced from the FBX `bioMod` through `meshfile`. This is required for pyorerun/biorbd to render surfaces. Writing raw `mesh x y z` points into a `bioMod` only provides vertices and typically appears as a line/point cloud in the viewer.

## Local Marker Test

For each C3D marker, the script uses biorbd segment rototranslations to express the marker in every segment's local frame. It assigns the marker to the segment where that local position varies least across frames, writes the local mean position into the corresponding `bioMod`, and reports stability statistics in the local marker CSV files.

## Inverse Kinematics

With `--inverse-kinematics`, the script uses only the C3D marker channels, never the C3D angle channels. The solver can be `least_squares`, which calls `biorbd.InverseKinematics`, or `kalman`, which calls `biorbd.KalmanReconsMarkers`. The outputs contain reconstructed `q`, `qdot`, and `qddot`; no inverse dynamics or generalized forces are computed in this step.
