# Refactor Roadmap

This roadmap keeps the Captury/Motive comparison code maintainable while the
scientific workflow is still evolving. Each phase must start by adding or
strengthening focused tests, then refactor only the targeted ownership boundary,
then run a validation agent before commit.

## Validation Rule

For every phase:

1. Add or update tests that describe the current expected behavior.
2. Make the smallest refactor that improves ownership, naming or duplication.
3. Run the relevant unit tests, `py_compile` and formatting checks.
4. Ask an agent to review the phase for regressions, unclear CLI semantics and
   missing tests.
5. Document user-visible CLI or GUI changes in `README.md`.

## Phase 1 - Source-Explicit C3D Offset Diagnostic

Status: started.

Scope:

- `plot_c3d_initial_offset.py`
- `tests/test_plot_c3d_initial_offset.py`
- `README.md`

Purpose:

- Treat Motive and Captury independently in the diagnostic CLI.
- Make root-translation subtraction source-specific.
- Make point transforms source-specific.
- Preserve the old global `--subtract-root-offsets` flag only as hidden
  compatibility.

Validation:

- Unit tests cover independent source preparation and legacy flag behavior.
- The documented P6 command writes a diagnostic figure.

## Phase 2 - Shared Source Preparation Module

Status: planned.

Candidate scope:

- Extract source preparation helpers from `plot_c3d_initial_offset.py` into a
  small module, for example `c3d_source_preparation.py`.
- Reuse that module from plotting diagnostics and any future lightweight C3D
  inspection scripts.

Tests first:

- Move the existing source-preparation tests to the new module tests.
- Add one CLI integration test that verifies parsed Motive/Captury options map
  to the expected preparation configs.

## Phase 3 - Command Builders and GUI Option Semantics

Status: planned.

Candidate scope:

- `gui_commands.py`
- `captury_biobuddy_gui.py`

Purpose:

- Keep scientific choices in CLI commands, not in Tk callbacks.
- Centralize display labels versus CLI values for options such as root
  translation policy, model source, segment-frame rotations and axis conversion.

Tests first:

- Add command-builder tests for Captury-specific and Motive-specific options.
- Add regression tests for hidden/default options that should not appear in the
  wrong tab.

## Phase 4 - Metrics IO and Graph Contracts

Status: planned.

Candidate scope:

- `gui_graphs.py`
- `gui_run_report.py`
- `model_comparison_metrics.py`

Purpose:

- Keep fast `.npz` time-series readers separate from CSV summaries.
- Define small data contracts for joint centres, marker correspondences,
  segments and kinematics before GUI plotting.

Tests first:

- Add tests for plotting-data selection without creating GUI windows.
- Add fixture-level tests for each expected `.npz` schema.

## Phase 5 - BioBuddy Model and IK Integration

Status: planned.

Candidate scope:

- `create_biobuddy_c3d_model.py`
- `run_biobuddy_c3d_ik.py`
- `compare_p6_motive_captury.py`

Purpose:

- Make generated BioBuddy models a stable third source for dimensions, centres,
  segment rotations and kinematics.
- Separate model creation, static QLD reconstruction and batch IK orchestration.

Tests first:

- Add command-level tests that can run without BioBuddy internals by mocking the
  external model creation and IK calls.
- Add one minimal integration test when the BioBuddy API is available.
