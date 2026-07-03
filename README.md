# captury_models

Small workspace for comparing Captury BVH/FBX skeleton exports with C3D marker data, generating biorbd `bioMod` models through BioBuddy, and checking whether C3D markers are stable in local segment frames.

## Contents

- `bvh_c3d_biobuddy_pyorerun_compare.py`: main BVH/FBX to `bioMod` pipeline.
- `captury_biobuddy_gui.py`: graphical launcher for the pipeline options.
- `c3d_trial_viewer.py`: lightweight PySide/QPainter C3D marker viewer used from the trial-cutting tab.
- `compare_capture_systems.py`: compare Motive marker-based C3D trials against Captury markerless C3D trials.
- `compare_p6_motive_captury.py`: Captury/Motive model-centre comparison and C3D enrichment workflow for the `captury/` + `squelettes/` trial layout.
- `model_comparison_metrics.py`: agreement metrics used by the comparison script.
- `motive_captury_landmark_map.json`: editable Motive/Captury anatomical landmark correspondence map.
- `plot_bvh_c3d_angle_comparisons.py`: optional plotting helper for BVH q versus C3D angle channels.
- `environment_bvh_c3d_biobuddy.yml`: conda environment definition.
- `data/unknown.bvh`, `data/unknown.fbx`, `data/unknown.c3d`: expected local Captury input files. They are ignored by git because they are data files.

Generated outputs are written to `out_biobuddy_bvh_c3d/` and are intentionally ignored by git.

## Environment

Create or update the conda environment:

```bash
conda env create -f environment_bvh_c3d_biobuddy.yml
conda activate captury_biobuddy
```

If the environment already exists:

```bash
conda env update -n captury_biobuddy -f environment_bvh_c3d_biobuddy.yml --prune
conda activate captury_biobuddy
```

The environment installs BioBuddy from:

```text
biobuddy @ git+https://github.com/mickaelbegon/biobuddy.git@codex/add-model-editor-gui
```

This corresponds to the BioBuddy branch:

```text
https://github.com/mickaelbegon/biobuddy/tree/codex/add-model-editor-gui
```

## C3D Point Classification

The C3D contains both marker trajectories and angle-like point channels. The pipeline excludes detected angle channels from the marker cloud used for animation and local marker placement. In addition to labels/descriptions matching the angle regex, these Captury joint labels are treated as angles by default:

```text
RHip LHip RKne LKne RAnk LAnk RSho LSho RElb LElb RWri LWri
```

During animation, the marker overlay contains the C3D marker points after this angle filtering plus the generated joint centres (`BVHJC_*` or `FBXJC_*`). Angle point channels are never sent to pyorerun as markers.

## Run

To launch the graphical interface:

```bash
conda activate captury_biobuddy
python captury_biobuddy_gui.py
```

For quick P6 debugging, launch the GUI with the local P6 preset already loaded:

```bash
conda run -n captury_biobuddy python captury_biobuddy_gui.py --p6-debug
```

Run the lightweight regression tests:

```bash
python -m unittest discover -s tests -v
```

The interface is a graphical launcher around the existing command line scripts. The scientific processing remains in the CLI scripts; the GUI builds the equivalent command with `sys.executable`, copies it to the clipboard, launches it in a background `subprocess.Popen`, streams stdout/stderr into a log popup, and lets the running process be stopped.

The small `Commande` button in the bottom-left corner opens a compact command popup. It can target three workflows:

- `Analyse Captury/Motive`: runs `compare_p6_motive_captury.py`.
- `Pipeline BVH/FBX/C3D`: runs `bvh_c3d_biobuddy_pyorerun_compare.py`.
- `Comparaison générique`: runs `compare_capture_systems.py`.

The GUI tabs are organized for the Captury/Motive analysis:

- `Données`: choose the flattened `Captury/` + `Motive/` data root, output folder, static trial, model source and model-to-C3D axis conversion. The detected files are inventoried in a table, and the global trial menu in the top-right corner applies to every tab. The `Charger P6 debug` button fills a short `Static` run using `local_trials/2026-06-30_P6_flat`.
- `Occlusions`: analyze missing Motive marker trajectories.
- `Découpage`: estimate movement start/end and ground contacts from foot-marker kinematics, and open the selected trial in the lightweight 3D C3D viewer.
- `Dimensions`: compare model dimensions with an embedded graph and hierarchical metric/component selectors.
- `Centres`: compare model joint-centre positions after alignment.
- `Marqueurs`: compare reasonable Motive/Captury skin-marker correspondences.
- `Cinématiques`: compare available model q/angle channels and optionally run batch IK.
- `Visualisation`: launch the enriched C3D/Rerun visualization or run headless.
- `Avancé`: inspect the Python executable, script paths and compatibility options.

The metric tabs contain embedded Matplotlib graphs instead of PNG previews. Each graph panel has a hierarchical selector (`trial -> metric -> component`) so a metric can be plotted globally or narrowed to a specific marker, segment, joint, landmark or q component.

The model-centre workflow automatically handles the current P6 conventions by default: Captury BVH/FBX is treated as millimetres, Motive BVH/FBX as centimetres, and `--model-to-c3d-axis auto` currently resolves to the Y-up model -> Motive C3D Z-up conversion before writing `CAPJC_*` and `MOTJC_*` channels into enriched C3D copies. The bottom-left `Log` button opens the live process log when needed.

The C3D viewer is a lightweight PySide/QPainter widget. It uses orthographic projection, drag rotation, wheel zoom, double-click reset, a right-click view menu (`XY`, `YZ`, `ZX`, `Face`, `Dos`, `Côté`), a frame slider, playback, marker-table selection highlighting, a whole-body fit toggle and an RGB triad. Launch it directly with:

```bash
python c3d_trial_viewer.py local_trials/2026-06-30_P6_flat/Motive/P6_Static.c3d
```

For direct command line use:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d
```

To also launch pyorerun animations:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --animate
```

To display the BVH model, the FBX surface model and filtered C3D markers superposed in one scene:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --animate-superposed \
  --hide-extremities-in-rerun
```

To compute inverse kinematics from the C3D markers with biorbd nonlinear least squares, add:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-solver least_squares
```

To use biorbd's marker Kalman reconstruction instead:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-solver kalman
```

For a quick smoke test, limit inverse kinematics to the first few frames:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
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
- `bvh_c3d_marker_error_norm_boxplot.png`
- `fbx_c3d_marker_error_norm_boxplot.png`
- `bvh_fbx_c3d_marker_error_norm_overall_boxplot.png`
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

The FBX mesh is handled by the BioBuddy branch `codex/add-model-editor-gui`: the skinned visual mesh is split into per-segment `.ply` files for the FBX `bioMod`. The script also converts those generated `.ply` files to per-segment `.vtp` files before animation, because pyorerun accepts `.stl`/`.vtp` mesh files for rendering surfaces while biorbd keeps reading the `.ply` paths from the `bioMod`. Writing raw `mesh x y z` points into a `bioMod` only provides vertices and typically appears as a line/point cloud in the viewer.

The pyorerun display uses millimetre-scale marker radii by default (`--rerun-marker-radius 15`) and keeps the FBX mesh opaque so the scene is visible immediately in Rerun.

Use `--hide-hands-in-rerun`, `--hide-feet-in-rerun`, or `--hide-extremities-in-rerun` to hide hand/wrist/finger and/or foot/ankle/toe markers and meshes from pyorerun animations without changing the numerical outputs.

The script declares `Y` as the vertical axis in Rerun by default (`--rerun-up-axis y`). Use `--rerun-up-axis z`, `x`, or `none` if the viewer orientation should follow another convention.

## Motive vs Captury Comparison

The local Motive/Captury trial archive can be extracted into `local_trials/`, which is ignored by git:

```bash
mkdir -p local_trials
unzip -oq /Users/mickaelbegon/Downloads/data.zip -d local_trials
```

Run the comparison across all discovered trial pairs:

```bash
python compare_capture_systems.py \
  --data-root local_trials/data \
  --reference-system Motive \
  --test-system Captury \
  --landmark-map motive_captury_landmark_map.json \
  --out-dir out_capture_system_comparison
```

The comparison script is prepared for C3D, BVH and FBX on both systems. The current single-participant layout remains valid. It discovers either flat files:

```text
local_trials/data/Motive/P5_Marche_001.c3d
local_trials/data/Motive/P5_Marche_001.bvh
local_trials/data/Motive/P5_Marche_001.fbx
```

or one folder per trial:

```text
local_trials/data/Captury/P5_Marche_001/unknown.c3d
local_trials/data/Captury/P5_Marche_001/unknown.bvh
local_trials/data/Captury/P5_Marche_001/unknown.fbx
```

For population studies, use one directory per participant, with the same trial naming convention under each system:

```text
local_trials/data/P01/Motive/P01_Marche_001.c3d
local_trials/data/P01/Motive/P01_Marche_001.bvh
local_trials/data/P01/Motive/P01_Marche_001.fbx
local_trials/data/P01/Captury/P01_Marche_001/unknown.c3d
local_trials/data/P01/Captury/P01_Marche_001/unknown.bvh
local_trials/data/P01/Captury/P01_Marche_001/unknown.fbx
local_trials/data/P02/Motive/P02_Marche_001.c3d
local_trials/data/P02/Captury/P02_Marche_001/unknown.c3d
```

The GUI exposes participant and trial filters. From the command line, use repeated regex filters when needed:

```bash
python compare_capture_systems.py \
  --data-root local_trials/data \
  --participant-filter "P0[1-5]" \
  --trial-filter "Marche" \
  --landmark-map motive_captury_landmark_map.json \
  --out-dir out_capture_system_comparison
```

The current numerical comparison derives comparable anatomical landmarks from Motive markers and Captury `Q_*` points, time-normalizes each trial, applies a global rigid alignment by default, and writes raw/aligned landmark errors plus C3D and model-file inventories. Main outputs:

- `out_capture_system_comparison/all_landmark_metrics.csv`
- `out_capture_system_comparison/all_angle_metrics.csv`
- `out_capture_system_comparison/all_model_inventory.csv`
- `out_capture_system_comparison/population_landmark_summary.csv`
- `out_capture_system_comparison/population_angle_summary.csv`
- `out_capture_system_comparison/run_report.json`

For the current sample archive, Motive C3D files contain marker trajectories but no joint-angle POINT channels. Captury C3D files contain markerless `Q_*` points and joint-angle channels listed in `POINT:ANGLES`; therefore the script reports Captury angles in the inventory, but only computes angle agreement when both systems provide matching angle channels.

## Motive/Captury Kinematic Model-Centre Comparison

The original P6 folder can be flattened into a simpler local dataset with only `Captury/` and `Motive/` subfolders:

```bash
python prepare_kinematic_dataset.py \
  --source-root /Users/mickaelbegon/Downloads/2026-06-30_P6 \
  --output-root local_trials/2026-06-30_P6_flat
```

The flattened folder is ignored by git and has this structure:

```text
local_trials/2026-06-30_P6_flat/Captury/Static_P6.bvh
local_trials/2026-06-30_P6_flat/Captury/Static_P6.fbx
local_trials/2026-06-30_P6_flat/Captury/Static_P6.c3d
local_trials/2026-06-30_P6_flat/Motive/P6_Static_Skeleton 001.bvh
local_trials/2026-06-30_P6_flat/Motive/P6_Static.fbx
local_trials/2026-06-30_P6_flat/Motive/P6_Static.c3d
```

List the detected trials:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --list-trials
```

Run the example batch for `Static`, `LKnee` and `Marche_001`:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Static \
  --trial LKnee \
  --trial Marche_001 \
  --joint-filter "Hip|Knee|Ankle|Leg|Foot" \
  --static-trial Static \
  --model-source bvh \
  --no-mesh \
  --out-dir out_p6_motive_captury_comparison
```

The script builds BioBuddy/biorbd models for both systems from BVH by default. Use `--model-source fbx` to force FBX, or `--model-source auto` to prefer BVH and fall back to FBX. Captury BVH/FBX is treated as millimetres; Motive BVH/FBX is treated as centimetres unless overridden with `--captury-unit-scale-to-m` or `--motive-unit-scale-to-m`.

The model coordinates are converted from Y-up to the Motive C3D Z-up convention before writing C3D outputs:

```bash
--model-to-c3d-axis auto
```

Main outputs:

- `out_p6_motive_captury_comparison/<trial>/<trial>_motive_with_capjc_motjc.c3d`
- `out_p6_motive_captury_comparison/<trial>/joint_centre_metrics.csv`
- `out_p6_motive_captury_comparison/<trial>/kinematics_q_metrics.csv`
- `out_p6_motive_captury_comparison/<trial>/motive_marker_occlusions.csv`
- `out_p6_motive_captury_comparison/<trial>/trial_events_contacts.csv`
- `out_p6_motive_captury_comparison/<trial>/model_dimensions.csv`
- `out_p6_motive_captury_comparison/<trial>/skin_marker_correspondence_metrics.csv`
- `out_p6_motive_captury_comparison/all_joint_centre_metrics.csv`
- `out_p6_motive_captury_comparison/all_kinematics_q_metrics.csv`
- `out_p6_motive_captury_comparison/all_motive_marker_occlusions.csv`
- `out_p6_motive_captury_comparison/all_model_dimensions.csv`
- `out_p6_motive_captury_comparison/all_skin_marker_correspondence_metrics.csv`
- `out_p6_motive_captury_comparison/run_report.json`

The GUI reads the CSV outputs directly and renders graphs inside the relevant tabs: occlusions, trial cutting/contact signals, model dimensions, joint centres, skin markers and kinematics. The hierarchical graph menus expose metrics such as `median_error_mm`, `p95_error_mm`, `mae_x`, `mae_y`, `mae_z`, `mae_euclidean`, `rmse_euclidean`, `mae_rad`, `rmse_rad`, waveform correlation/CCC, Motive occlusion percentages and contact-detection signals. Use `Rafraîchir graphes` after an analysis to reload the CSV data.

The enriched Motive C3D copies contain generated model joint centres:

- `CAPJC_*`: Captury centres after static rigid alignment Captury -> Motive.
- `MOTJC_*`: Motive model centres in the Motive C3D frame.

Open/validate a visualization for one trial without launching the Rerun viewer:

```bash
PYORERUN_HEADLESS=1 python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Static \
  --visualize \
  --visualize-trial Static \
  --headless \
  --rerun-wait-seconds 0 \
  --out-dir out_p6_motive_captury_visual_check
```

For an interactive Rerun view, remove `PYORERUN_HEADLESS=1` and `--headless`. The current visualization displays the enriched C3D joint-centre channels. FBX meshes are generated when `--model-source fbx` and mesh extraction is enabled, but Motive FBX files may not contain usable geometry; this is reported under each trial's `run_report.json`.

Run Motive inverse kinematics in batch through the existing BioBuddy/biorbd pipeline:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial LKnee \
  --run-ik-batch \
  --ik-max-frames 50 \
  --out-dir out_p6_motive_captury_ik_check
```

Kinematic comparisons in `kinematics_q_metrics.csv` are intentionally conservative: they compare matching generalized-coordinate names from the generated BioBuddy models. Translation channels are useful for gross motion checks. Rotation channels are reported, but Captury and Motive BVH/FBX exports may use different local segment frames, Euler sequences or axis signs, so those angular differences should be interpreted as diagnostic rather than direct biomechanical agreement. Captury C3D angle channels are inventoried when present; the Motive C3D files inspected here do not expose matching C3D angle channels. Captury duplicate C3D labels are inventoried in `run_report.json`; current marker correspondences average duplicate labels until they are renamed more explicitly.

## Local Marker Test

For each C3D marker, the script uses biorbd segment rototranslations to express the marker in every segment's local frame. It assigns the marker to the segment where that local position varies least across frames, writes the local mean position into the corresponding `bioMod`, and reports stability statistics in the local marker CSV files.

The script then recomputes each local marker position in the global frame while the BVH or FBX model is animated. The Euclidean norm between this model marker and the measured C3D marker is saved in `*_c3d_marker_error_norm_mm.csv`, summarized in JSON, and displayed as per-marker and overall boxplots in millimetres. If several C3D channels have the same visible marker label, their channel indices are retained so every physical marker remains distinct. This measures residual fit on the same trial used to attach markers to segments; it is not an independent validation trial.

## Inverse Kinematics

With `--inverse-kinematics`, the script uses only the C3D marker channels, never the C3D angle channels. The solver can be `least_squares`, which calls `biorbd.InverseKinematics`, or `kalman`, which calls `biorbd.KalmanReconsMarkers`. The outputs contain reconstructed `q`, `qdot`, and `qddot`; no inverse dynamics or generalized forces are computed in this step.
