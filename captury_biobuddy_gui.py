"""Graphical launcher for the Captury/BioBuddy BVH/FBX/C3D pipeline."""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_DIR / "bvh_c3d_biobuddy_pyorerun_compare.py"
MODEL_EDITOR_SCRIPT = PROJECT_DIR / "launch_biobuddy_model_editor.py"
COMPARISON_SCRIPT = PROJECT_DIR / "compare_capture_systems.py"
KINEMATIC_COMPARISON_SCRIPT = PROJECT_DIR / "compare_p6_motive_captury.py"

COMMAND_MODES = {
    "kinematic": "Analyse Captury/Motive",
    "pipeline": "Pipeline BVH/FBX/C3D",
    "comparison": "Comparaison générique",
}


class CapturyBioBuddyGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__(baseName="captury_biobuddy", className="CapturyBioBuddy")
        self.title("Captury BioBuddy")
        self.geometry("1180x780")
        self.minsize(980, 680)

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str | tuple[str, int]] = queue.Queue()
        self.figure_paths: list[Path] = []
        self.figure_photo: tk.PhotoImage | None = None
        self.analysis_buttons: list[ttk.Button] = []

        self.vars: dict[str, tk.Variable] = {}
        self._create_variables()
        self._configure_style()
        self._build_layout()
        self._bind_command_preview()
        self._update_command_preview()
        self.after(100, self._drain_output_queue)

    def _create_variables(self) -> None:
        defaults: dict[str, str | bool] = {
            "bvh": "data/unknown.bvh",
            "fbx": "data/unknown.fbx",
            "c3d": "data/unknown.c3d",
            "out_dir": "out_biobuddy_bvh_c3d",
            "bvh_unit_scale_to_m": "0.001",
            "fbx_unit_scale_to_m": "0.001",
            "c3d_angle_unit": "deg",
            "angle_label_regex": r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)",
            "extra_angle_labels": "",
            "comparison_map": "",
            "model_explorer_path": "out_biobuddy_bvh_c3d/model_from_fbx_biobuddy.bioMod",
            "no_biomod_joint_centre_markers": False,
            "no_root_offset_correction": False,
            "root_offset_mode": "auto",
            "no_fbx_mesh": False,
            "max_fbx_mesh_points": "0",
            "animate": False,
            "animate_superposed": False,
            "display_q_in_rerun": False,
            "rerun_marker_radius": "15",
            "rerun_wait_seconds": "2",
            "rerun_up_axis": "y",
            "hide_hands_in_rerun": False,
            "hide_feet_in_rerun": False,
            "hide_extremities_in_rerun": False,
            "headless": False,
            "inverse_kinematics": False,
            "inverse_kinematics_solver": "least_squares",
            "inverse_kinematics_method": "trf",
            "inverse_kinematics_max_frames": "0",
            "kalman_noise_factor": "1e-10",
            "kalman_error_factor": "1e-5",
            "inverse_dynamics": False,
            "inverse_dynamics_method": "",
            "inverse_dynamics_max_frames": "",
            "compare_data_root": "local_trials/data",
            "compare_reference_system": "Motive",
            "compare_test_system": "Captury",
            "compare_reference_c3d": "",
            "compare_reference_bvh": "",
            "compare_reference_fbx": "",
            "compare_test_c3d": "",
            "compare_test_bvh": "",
            "compare_test_fbx": "",
            "compare_trial_name": "",
            "compare_participant_filter": "",
            "compare_trial_filter": "",
            "compare_out_dir": "out_capture_system_comparison",
            "compare_landmark_map": "motive_captury_landmark_map.json",
            "compare_resample_points": "101",
            "compare_alignment": "global_rigid",
            "p6_data_root": "local_trials/2026-06-30_P6_flat",
            "p6_out_dir": "out_p6_motive_captury_comparison",
            "p6_trials": "",
            "p6_static_trial": "Static",
            "p6_joint_filter": "",
            "p6_no_figures": False,
            "p6_model_source": "bvh",
            "p6_model_to_c3d_axis": "y_up_to_z_up",
            "p6_no_mesh": False,
            "p6_max_mesh_points": "0",
            "p6_run_ik_batch": False,
            "p6_ik_max_frames": "0",
            "p6_visualize": False,
            "p6_visualize_trial": "",
            "p6_headless": False,
            "p6_rerun_wait_seconds": "1",
            "command_mode": COMMAND_MODES["kinematic"],
        }
        for name, value in defaults.items():
            var_cls = tk.BooleanVar if isinstance(value, bool) else tk.StringVar
            self.vars[name] = var_cls(value=value)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f6f7f8")
        style.configure("TLabelframe", background="#f6f7f8")
        style.configure("TLabelframe.Label", background="#f6f7f8", foreground="#1f2933")
        style.configure("TLabel", background="#f6f7f8", foreground="#1f2933")
        style.configure("TButton", padding=(10, 6))
        style.configure("Primary.TButton", padding=(12, 7))
        style.configure("Danger.TButton", padding=(12, 7))
        style.configure("Status.TLabel", foreground="#475569")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="Captury BioBuddy", font=("TkDefaultFont", 18, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Génération bioMod, comparaison C3D, visualisation Rerun et cinématique inverse",
            style="Status.TLabel",
        ).grid(row=1, column=0, sticky="w")

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        notebook = ttk.Notebook(left)
        self.main_notebook = notebook
        notebook.grid(row=0, column=0, sticky="nsew")

        self._build_loading_matching_tab(notebook)
        self._build_occlusions_tab(notebook)
        self._build_trial_cutting_tab(notebook)
        self._build_dimensions_tab(notebook)
        self._build_joint_centres_tab(notebook)
        self._build_skin_markers_tab(notebook)
        self._build_kinematics_compare_tab(notebook)
        self._build_visualization_tab(notebook)
        self._build_advanced_tab(notebook)

        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_actions(right)
        self._build_figure_panel(right)
        self._build_log_panel(right)

    def _tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(0, weight=1)
        notebook.add(frame, text=title)
        return frame

    def _analysis_action_row(self, parent: ttk.Widget, row: int) -> None:
        actions = ttk.Frame(parent)
        actions.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        for column in range(3):
            actions.columnconfigure(column, weight=1)
        run_button = ttk.Button(
            actions,
            text="Lancer analyse",
            style="Primary.TButton",
            command=self._run_p6_analysis,
        )
        run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.analysis_buttons.append(run_button)
        ttk.Button(actions, text="Copier commande", command=self._copy_p6_command).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(
            actions, text="Rafraîchir figures", command=self._refresh_figures
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

    def _build_loading_matching_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Données")
        data = ttk.LabelFrame(tab, text="Chargement et correspondance")
        data.grid(row=0, column=0, sticky="ew")
        data.columnconfigure(1, weight=1)
        self._path_row(data, 0, "Dossier racine", "p6_data_root", directory=True)
        self._path_row(data, 1, "Sortie", "p6_out_dir", directory=True)
        self._entry_row(data, 2, "Essais", "p6_trials")
        self._entry_row(data, 3, "Essai statique", "p6_static_trial")
        self._combo_row(
            data, 4, "Source modèle", "p6_model_source", ("bvh", "fbx", "auto")
        )
        self._combo_row(
            data,
            5,
            "Axes modèle -> C3D",
            "p6_model_to_c3d_axis",
            ("y_up_to_z_up", "identity"),
        )

        systems = ttk.LabelFrame(tab, text="Systèmes comparés")
        systems.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        systems.columnconfigure(1, weight=1)
        self._entry_row(systems, 0, "Référence", "compare_reference_system")
        self._entry_row(systems, 1, "Test", "compare_test_system")
        self._path_row(
            systems,
            2,
            "Carte repères",
            "compare_landmark_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._analysis_action_row(tab, 2)
        ttk.Button(
            tab,
            text="Charger P6 debug",
            command=self._load_p6_debug_preset,
        ).grid(row=3, column=0, sticky="ew", pady=(12, 0))

    def _build_occlusions_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Occlusions")
        panel = ttk.LabelFrame(tab, text="Marqueurs Motive")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Essais", "p6_trials")
        self._path_row(panel, 1, "Sortie", "p6_out_dir", directory=True)
        self._check(panel, 2, "Ne pas générer les figures PNG", "p6_no_figures")
        self._analysis_action_row(tab, 1)

    def _build_trial_cutting_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Découpage")
        panel = ttk.LabelFrame(tab, text="Début, fin et contacts au sol")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Essais", "p6_trials")
        self._entry_row(panel, 1, "Essai statique", "p6_static_trial")
        self._check(panel, 2, "Visualiser un essai enrichi", "p6_visualize")
        self._entry_row(panel, 3, "Essai visualisé", "p6_visualize_trial")
        self._analysis_action_row(tab, 1)

    def _build_dimensions_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Dimensions")
        panel = ttk.LabelFrame(tab, text="Dimensions des modèles")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._combo_row(
            panel, 0, "Source modèle", "p6_model_source", ("bvh", "fbx", "auto")
        )
        self._check(panel, 1, "Ne pas extraire les meshes FBX", "p6_no_mesh")
        self._entry_row(panel, 2, "Max points mesh", "p6_max_mesh_points")
        self._analysis_action_row(tab, 1)

    def _build_joint_centres_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Centres")
        panel = ttk.LabelFrame(tab, text="Centres articulaires")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Filtre centres", "p6_joint_filter")
        self._entry_row(panel, 1, "Essai statique", "p6_static_trial")
        self._combo_row(
            panel,
            2,
            "Alignement C3D",
            "compare_alignment",
            ("global_rigid", "per_frame_rigid", "none"),
        )
        self._analysis_action_row(tab, 1)

    def _build_skin_markers_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Marqueurs")
        panel = ttk.LabelFrame(tab, text="Marqueurs cutanés correspondants")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._path_row(
            panel,
            0,
            "Carte repères",
            "compare_landmark_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._entry_row(panel, 1, "Filtre essais", "p6_trials")
        self._analysis_action_row(tab, 1)

    def _build_kinematics_compare_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Cinématiques")
        panel = ttk.LabelFrame(tab, text="Angles articulaires et q")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Regex labels angles", "angle_label_regex")
        self._entry_row(panel, 1, "Labels angles extra", "extra_angle_labels")
        self._combo_row(panel, 2, "Unité angles C3D", "c3d_angle_unit", ("deg", "rad"))
        self._check(
            panel, 3, "Lancer l'IK du système de référence en batch", "p6_run_ik_batch"
        )
        self._entry_row(panel, 4, "Max frames IK batch", "p6_ik_max_frames")
        self._analysis_action_row(tab, 1)

    def _build_visualization_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Visualisation")
        panel = ttk.LabelFrame(tab, text="Rerun et figures")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._check(panel, 0, "Visualiser un essai enrichi", "p6_visualize")
        self._entry_row(panel, 1, "Essai visualisé", "p6_visualize_trial")
        self._check(panel, 2, "Headless", "p6_headless")
        self._entry_row(panel, 3, "Attente Rerun", "p6_rerun_wait_seconds")
        self._check(panel, 4, "Ne pas générer les figures PNG", "p6_no_figures")
        self._analysis_action_row(tab, 1)

    def _build_sources_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Sources")
        card = ttk.LabelFrame(tab, text="Fichiers")
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)

        self._path_row(
            card, 0, "BVH", "bvh", [("BVH", "*.bvh"), ("Tous les fichiers", "*")]
        )
        self._path_row(
            card, 1, "FBX", "fbx", [("FBX", "*.fbx"), ("Tous les fichiers", "*")]
        )
        self._path_row(
            card, 2, "C3D", "c3d", [("C3D", "*.c3d"), ("Tous les fichiers", "*")]
        )
        self._path_row(card, 3, "Sortie", "out_dir", directory=True)

        tips = ttk.LabelFrame(tab, text="Flux")
        tips.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            tips,
            text=(
                "BVH + C3D suffit pour générer le modèle BVH et les marqueurs locaux. "
                "Ajoute un FBX pour générer le modèle surfacique et l'affichage superposé."
            ),
            wraplength=760,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=8)

        kinematic = ttk.LabelFrame(tab, text="Données cinématiques multi-systèmes")
        kinematic.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        kinematic.columnconfigure(1, weight=1)
        self._path_row(kinematic, 0, "Dossier racine", "p6_data_root", directory=True)
        self._path_row(kinematic, 1, "Sortie", "p6_out_dir", directory=True)
        self._entry_row(kinematic, 2, "Essais", "p6_trials")
        self._entry_row(kinematic, 3, "Essai statique", "p6_static_trial")

    def _build_model_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Modèles")
        units = ttk.LabelFrame(tab, text="Unités et angles")
        units.grid(row=0, column=0, sticky="ew")
        units.columnconfigure(1, weight=1)
        self._entry_row(units, 0, "Échelle BVH vers m", "bvh_unit_scale_to_m")
        self._entry_row(units, 1, "Échelle FBX vers m", "fbx_unit_scale_to_m")
        self._combo_row(units, 2, "Unité angles C3D", "c3d_angle_unit", ("deg", "rad"))
        self._entry_row(units, 3, "Regex labels angles", "angle_label_regex")
        self._entry_row(units, 4, "Labels angles extra", "extra_angle_labels")
        self._path_row(
            units,
            5,
            "Mapping q/C3D",
            "comparison_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )

        generation = ttk.LabelFrame(tab, text="Génération")
        generation.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        generation.columnconfigure(1, weight=1)
        self._combo_row(
            generation,
            0,
            "Correction root offset",
            "root_offset_mode",
            ("auto", "subtract", "keep"),
        )
        self._entry_row(generation, 1, "Max points mesh FBX", "max_fbx_mesh_points")
        self._check(
            generation,
            2,
            "Ne pas ajouter les marqueurs de centres articulaires",
            "no_biomod_joint_centre_markers",
        )
        self._check(
            generation,
            3,
            "Ne pas corriger le root offset (compatibilité)",
            "no_root_offset_correction",
        )
        self._check(generation, 4, "Ne pas générer les meshes FBX", "no_fbx_mesh")

        chain_compare = ttk.LabelFrame(tab, text="Chaînes cinématiques comparées")
        chain_compare.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        chain_compare.columnconfigure(1, weight=1)
        self._combo_row(
            chain_compare,
            0,
            "Source modèle",
            "p6_model_source",
            ("bvh", "fbx", "auto"),
        )
        self._combo_row(
            chain_compare,
            1,
            "Axes modèle -> C3D",
            "p6_model_to_c3d_axis",
            ("y_up_to_z_up", "identity"),
        )
        self._check(chain_compare, 2, "Ne pas extraire les meshes FBX", "p6_no_mesh")
        self._entry_row(chain_compare, 3, "Max points mesh", "p6_max_mesh_points")
        ttk.Label(
            chain_compare,
            text=(
                "Utilise BVH/FBX pour construire les modèles BioBuddy/biorbd des deux systèmes. "
                "Le mode y_up_to_z_up place la hauteur modèle sur Z avant écriture dans le C3D cible."
            ),
            style="Status.TLabel",
            wraplength=760,
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

        explorer = ttk.LabelFrame(tab, text="Explorateur BioBuddy")
        explorer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        explorer.columnconfigure(1, weight=1)
        self._path_row(
            explorer,
            0,
            "Modèle",
            "model_explorer_path",
            [
                ("Modèles BioBuddy", "*.bioMod *.osim *.urdf *.bvh"),
                ("Tous les fichiers", "*"),
            ],
        )
        ttk.Button(
            explorer,
            text="BVH généré",
            command=lambda: self._set_generated_model_path("bvh"),
        ).grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 10))
        ttk.Button(
            explorer,
            text="FBX généré",
            command=lambda: self._set_generated_model_path("fbx"),
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 10))
        ttk.Button(
            explorer,
            text="Ouvrir dans BioBuddy",
            command=self._launch_biobuddy_model_explorer,
        ).grid(row=1, column=2, sticky="ew", padx=(4, 10), pady=(0, 10))

    def _build_rerun_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Rerun")
        launch = ttk.LabelFrame(tab, text="Visualisation")
        launch.grid(row=0, column=0, sticky="ew")
        launch.columnconfigure(1, weight=1)
        self._check(launch, 0, "Lancer une scène Rerun par modèle", "animate")
        self._check(
            launch,
            1,
            "Lancer la scène superposée BVH + FBX + C3D",
            "animate_superposed",
        )
        self._check(
            launch, 2, "Afficher les courbes q dans Rerun", "display_q_in_rerun"
        )
        self._check(launch, 3, "Mode headless", "headless")
        self._entry_row(launch, 4, "Rayon marqueurs", "rerun_marker_radius")
        self._entry_row(launch, 5, "Attente après envoi", "rerun_wait_seconds")
        self._combo_row(
            launch, 6, "Axe vertical", "rerun_up_axis", ("y", "z", "x", "none")
        )

        filters = ttk.LabelFrame(tab, text="Lisibilité")
        filters.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self._check(filters, 0, "Masquer mains/poignets/doigts", "hide_hands_in_rerun")
        self._check(filters, 1, "Masquer pieds/chevilles/orteils", "hide_feet_in_rerun")
        self._check(
            filters, 2, "Masquer toutes les extrémités", "hide_extremities_in_rerun"
        )

        kinematic = ttk.LabelFrame(
            tab, text="Visualisation des comparaisons cinématiques"
        )
        kinematic.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        kinematic.columnconfigure(1, weight=1)
        self._check(kinematic, 0, "Visualiser un essai enrichi", "p6_visualize")
        self._entry_row(kinematic, 1, "Essai visualisé", "p6_visualize_trial")
        self._check(kinematic, 2, "Headless", "p6_headless")
        self._entry_row(kinematic, 3, "Attente Rerun", "p6_rerun_wait_seconds")

    def _build_ik_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "IK")
        ik = ttk.LabelFrame(tab, text="Cinématique inverse")
        ik.grid(row=0, column=0, sticky="ew")
        ik.columnconfigure(1, weight=1)
        self._check(
            ik,
            0,
            "Lancer la cinématique inverse depuis les marqueurs C3D",
            "inverse_kinematics",
        )
        self._combo_row(
            ik, 1, "Solveur", "inverse_kinematics_solver", ("least_squares", "kalman")
        )
        self._combo_row(
            ik,
            2,
            "Méthode least-squares",
            "inverse_kinematics_method",
            ("trf", "lm", "only_lm"),
        )
        self._entry_row(ik, 3, "Nombre max de frames", "inverse_kinematics_max_frames")
        self._entry_row(ik, 4, "Kalman noise factor", "kalman_noise_factor")
        self._entry_row(ik, 5, "Kalman error factor", "kalman_error_factor")

        ttk.Label(
            ik,
            text="0 frame max signifie que toutes les frames du C3D sont utilisées.",
            style="Status.TLabel",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

        batch = ttk.LabelFrame(tab, text="Batch sur essais cinématiques")
        batch.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        batch.columnconfigure(1, weight=1)
        self._check(
            batch, 0, "Lancer l'IK du système de référence en batch", "p6_run_ik_batch"
        )
        self._entry_row(batch, 1, "Max frames IK batch", "p6_ik_max_frames")

    def _build_comparison_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Comparaison")
        batch = ttk.LabelFrame(tab, text="Mode dossier ou population")
        batch.grid(row=0, column=0, sticky="ew")
        batch.columnconfigure(1, weight=1)
        self._path_row(batch, 0, "Racine", "compare_data_root", directory=True)
        self._entry_row(batch, 1, "Système référence", "compare_reference_system")
        self._entry_row(batch, 2, "Système test", "compare_test_system")
        self._entry_row(batch, 3, "Filtre participants", "compare_participant_filter")
        self._entry_row(batch, 4, "Filtre essais", "compare_trial_filter")

        single = ttk.LabelFrame(tab, text="Mode paire C3D explicite")
        single.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        single.columnconfigure(1, weight=1)
        self._path_row(
            single,
            0,
            "Référence C3D",
            "compare_reference_c3d",
            [("C3D", "*.c3d"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            1,
            "Référence BVH",
            "compare_reference_bvh",
            [("BVH", "*.bvh"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            2,
            "Référence FBX",
            "compare_reference_fbx",
            [("FBX", "*.fbx"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            3,
            "Test C3D",
            "compare_test_c3d",
            [("C3D", "*.c3d"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            4,
            "Test BVH",
            "compare_test_bvh",
            [("BVH", "*.bvh"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            5,
            "Test FBX",
            "compare_test_fbx",
            [("FBX", "*.fbx"), ("Tous les fichiers", "*")],
        )
        self._entry_row(single, 6, "Nom essai", "compare_trial_name")

        options = ttk.LabelFrame(tab, text="Options")
        options.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        options.columnconfigure(1, weight=1)
        self._path_row(options, 0, "Sortie", "compare_out_dir", directory=True)
        self._path_row(
            options,
            1,
            "Carte repères",
            "compare_landmark_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._entry_row(options, 2, "Points normalisés", "compare_resample_points")
        self._combo_row(
            options,
            3,
            "Alignement",
            "compare_alignment",
            ("global_rigid", "per_frame_rigid", "none"),
        )

        centres = ttk.LabelFrame(tab, text="Centres articulaires et cinématiques")
        centres.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        centres.columnconfigure(1, weight=1)
        self._entry_row(centres, 0, "Filtre centres", "p6_joint_filter")
        self._check(centres, 1, "Ne pas générer les figures PNG", "p6_no_figures")
        ttk.Label(
            centres,
            text=(
                "Les centres issus des modèles sont ajoutés dans des copies du C3D de référence, "
                "puis comparés après alignement statique du système test vers le système de référence. "
                "Les rotations Euler sont reportées comme indicatives lorsque les conventions diffèrent."
            ),
            style="Status.TLabel",
            wraplength=760,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

        actions = ttk.Frame(tab)
        actions.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)
        self.compare_button = ttk.Button(
            actions,
            text="Lancer la comparaison",
            style="Primary.TButton",
            command=self._run_comparison,
        )
        self.compare_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            actions,
            text="Copier commande comparaison",
            command=self._copy_comparison_command,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        self.p6_button = ttk.Button(
            actions,
            text="Lancer analyse cinématique",
            style="Primary.TButton",
            command=self._run_p6_analysis,
        )
        self.p6_button.grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(
            actions, text="Copier commande cinématique", command=self._copy_p6_command
        ).grid(row=0, column=3, sticky="ew", padx=(6, 0))

    def _build_advanced_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Avancé")
        legacy = ttk.LabelFrame(tab, text="Compatibilité ancienne CLI")
        legacy.grid(row=0, column=0, sticky="ew")
        legacy.columnconfigure(1, weight=1)
        self._check(
            legacy, 0, "Utiliser --inverse-dynamics (déprécié)", "inverse_dynamics"
        )
        self._combo_row(
            legacy,
            1,
            "Méthode inverse dynamics",
            "inverse_dynamics_method",
            ("", "trf", "lm", "only_lm"),
        )
        self._entry_row(
            legacy, 2, "Max frames inverse dynamics", "inverse_dynamics_max_frames"
        )

        environment = ttk.LabelFrame(tab, text="Exécution")
        environment.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(environment, text=f"Python: {sys.executable}", wraplength=760).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 4)
        )
        ttk.Label(environment, text=f"Script: {PIPELINE_SCRIPT}", wraplength=760).grid(
            row=1, column=0, sticky="w", padx=10, pady=(0, 8)
        )

    def _build_actions(self, parent: ttk.Frame) -> None:
        actions = ttk.LabelFrame(parent, text="Commande")
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        actions.columnconfigure(0, weight=1)

        selector = ttk.Frame(actions)
        selector.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 0))
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="Commande").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            selector,
            textvariable=self.vars["command_mode"],
            values=tuple(COMMAND_MODES.values()),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.command_text = tk.Text(actions, height=7, wrap=tk.WORD, font=("Menlo", 11))
        self.command_text.grid(
            row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=8
        )

        self.run_button = ttk.Button(
            actions,
            text="Lancer",
            style="Primary.TButton",
            command=self._run_selected_command,
        )
        self.run_button.grid(row=2, column=0, sticky="ew", padx=(8, 4), pady=(0, 8))
        self.stop_button = ttk.Button(
            actions,
            text="Arrêter",
            style="Danger.TButton",
            command=self._stop_pipeline,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=2, column=1, sticky="ew", padx=4, pady=(0, 8))
        ttk.Button(actions, text="Copier", command=self._copy_command).grid(
            row=2, column=2, sticky="ew", padx=4, pady=(0, 8)
        )
        ttk.Button(actions, text="Ouvrir sortie", command=self._open_output_dir).grid(
            row=2, column=3, sticky="ew", padx=(4, 8), pady=(0, 8)
        )

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(actions, textvariable=self.status_var, style="Status.TLabel").grid(
            row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8)
        )

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Log")
        panel.grid(row=2, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            panel, wrap=tk.WORD, font=("Menlo", 11), state=tk.DISABLED
        )
        scrollbar = ttk.Scrollbar(
            panel, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_figure_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Figures")
        panel.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(1, weight=1)

        controls = ttk.Frame(panel)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        controls.columnconfigure(0, weight=1)
        ttk.Button(controls, text="Rafraîchir", command=self._refresh_figures).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(controls, text="Ouvrir", command=self._open_selected_figure).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        list_frame = ttk.Frame(panel)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.figure_listbox = tk.Listbox(list_frame, height=8, exportselection=False)
        figure_scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.figure_listbox.yview
        )
        self.figure_listbox.configure(yscrollcommand=figure_scrollbar.set)
        self.figure_listbox.grid(row=0, column=0, sticky="nsew")
        figure_scrollbar.grid(row=0, column=1, sticky="ns")
        self.figure_listbox.bind(
            "<<ListboxSelect>>", lambda _event: self._show_selected_figure()
        )

        preview = ttk.Frame(panel)
        preview.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.figure_preview = ttk.Label(preview, text="Aucune figure", anchor="center")
        self.figure_preview.grid(row=0, column=0, sticky="nsew")

    def _path_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
        directory: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(
            row=row, column=1, sticky="ew", padx=6, pady=6
        )
        ttk.Button(
            parent,
            text="Parcourir",
            command=lambda: self._browse_path(
                var_name, filetypes=filetypes, directory=directory
            ),
        ).grid(row=row, column=2, sticky="ew", padx=10, pady=6)

    def _entry_row(
        self, parent: ttk.Widget, row: int, label: str, var_name: str
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=10, pady=6
        )

    def _combo_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var_name: str,
        values: tuple[str, ...],
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Combobox(
            parent, textvariable=self.vars[var_name], values=values, state="readonly"
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=10, pady=6)

    def _check(self, parent: ttk.Widget, row: int, label: str, var_name: str) -> None:
        ttk.Checkbutton(parent, text=label, variable=self.vars[var_name]).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=10, pady=6
        )

    def _browse_path(
        self,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
        directory: bool = False,
    ) -> None:
        current = Path(str(self.vars[var_name].get()))
        initial_dir = current.parent if current.parent.exists() else PROJECT_DIR
        if directory:
            selected = filedialog.askdirectory(initialdir=initial_dir)
        else:
            selected = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=filetypes or [("Tous les fichiers", "*")],
            )
        if selected:
            self.vars[var_name].set(
                os.path.relpath(selected, PROJECT_DIR)
                if str(selected).startswith(str(PROJECT_DIR))
                else selected
            )

    def _bind_command_preview(self) -> None:
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self._update_command_preview())

    def _command_args(self) -> list[str]:
        args = [sys.executable, str(PIPELINE_SCRIPT)]
        self._append_path(args, "--bvh", "bvh", required=True)
        self._append_path(args, "--fbx", "fbx")
        self._append_path(args, "--c3d", "c3d", required=True)
        self._append_path(args, "--out-dir", "out_dir", required=True)

        self._append_value(args, "--bvh-unit-scale-to-m", "bvh_unit_scale_to_m")
        self._append_value(args, "--fbx-unit-scale-to-m", "fbx_unit_scale_to_m")
        self._append_value(args, "--c3d-angle-unit", "c3d_angle_unit")
        self._append_value(args, "--angle-label-regex", "angle_label_regex")
        for label in self._split_extra_labels():
            args.extend(["--extra-angle-label", label])
        self._append_path(args, "--comparison-map", "comparison_map")

        self._append_flag(
            args, "--no-biomod-joint-centre-markers", "no_biomod_joint_centre_markers"
        )
        self._append_flag(
            args, "--no-root-offset-correction", "no_root_offset_correction"
        )
        self._append_value(args, "--root-offset-mode", "root_offset_mode")
        self._append_flag(args, "--no-fbx-mesh", "no_fbx_mesh")
        self._append_value(args, "--max-fbx-mesh-points", "max_fbx_mesh_points")

        self._append_flag(args, "--animate", "animate")
        self._append_flag(args, "--animate-superposed", "animate_superposed")
        self._append_flag(args, "--display-q-in-rerun", "display_q_in_rerun")
        self._append_value(args, "--rerun-marker-radius", "rerun_marker_radius")
        self._append_value(args, "--rerun-wait-seconds", "rerun_wait_seconds")
        self._append_value(args, "--rerun-up-axis", "rerun_up_axis")
        self._append_flag(args, "--hide-hands-in-rerun", "hide_hands_in_rerun")
        self._append_flag(args, "--hide-feet-in-rerun", "hide_feet_in_rerun")
        self._append_flag(
            args, "--hide-extremities-in-rerun", "hide_extremities_in_rerun"
        )
        self._append_flag(args, "--headless", "headless")

        self._append_flag(args, "--inverse-kinematics", "inverse_kinematics")
        self._append_value(
            args, "--inverse-kinematics-solver", "inverse_kinematics_solver"
        )
        self._append_value(
            args, "--inverse-kinematics-method", "inverse_kinematics_method"
        )
        self._append_value(
            args, "--inverse-kinematics-max-frames", "inverse_kinematics_max_frames"
        )
        self._append_value(args, "--kalman-noise-factor", "kalman_noise_factor")
        self._append_value(args, "--kalman-error-factor", "kalman_error_factor")

        self._append_flag(args, "--inverse-dynamics", "inverse_dynamics")
        self._append_value(args, "--inverse-dynamics-method", "inverse_dynamics_method")
        self._append_value(
            args, "--inverse-dynamics-max-frames", "inverse_dynamics_max_frames"
        )
        return args

    def _append_path(
        self, args: list[str], option: str, var_name: str, required: bool = False
    ) -> None:
        value = str(self.vars[var_name].get()).strip()
        if value or required:
            args.extend([option, value])

    def _append_value(self, args: list[str], option: str, var_name: str) -> None:
        value = str(self.vars[var_name].get()).strip()
        if value:
            args.extend([option, value])

    def _append_flag(self, args: list[str], option: str, var_name: str) -> None:
        if bool(self.vars[var_name].get()):
            args.append(option)

    def _split_extra_labels(self) -> list[str]:
        raw = str(self.vars["extra_angle_labels"].get())
        return [
            part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()
        ]

    def _comparison_args(self) -> list[str]:
        args = [sys.executable, str(COMPARISON_SCRIPT)]
        reference_c3d = str(self.vars["compare_reference_c3d"].get()).strip()
        test_c3d = str(self.vars["compare_test_c3d"].get()).strip()
        self._append_value_to(args, "--reference-system", "compare_reference_system")
        self._append_value_to(args, "--test-system", "compare_test_system")
        if reference_c3d or test_c3d:
            args.extend(["--reference-c3d", reference_c3d, "--test-c3d", test_c3d])
            self._append_value_to(args, "--reference-bvh", "compare_reference_bvh")
            self._append_value_to(args, "--reference-fbx", "compare_reference_fbx")
            self._append_value_to(args, "--test-bvh", "compare_test_bvh")
            self._append_value_to(args, "--test-fbx", "compare_test_fbx")
            trial_name = str(self.vars["compare_trial_name"].get()).strip()
            if trial_name:
                args.extend(["--trial-name", trial_name])
        else:
            args.extend(
                ["--data-root", str(self.vars["compare_data_root"].get()).strip()]
            )
            for pattern in self._split_var_lines("compare_participant_filter"):
                args.extend(["--participant-filter", pattern])
            for pattern in self._split_var_lines("compare_trial_filter"):
                args.extend(["--trial-filter", pattern])
        self._append_value_to(args, "--out-dir", "compare_out_dir")
        self._append_value_to(args, "--landmark-map", "compare_landmark_map")
        self._append_value_to(args, "--resample-points", "compare_resample_points")
        self._append_value_to(args, "--alignment", "compare_alignment")
        return args

    def _p6_args(self) -> list[str]:
        args = [sys.executable, str(KINEMATIC_COMPARISON_SCRIPT)]
        self._append_value_to(args, "--data-root", "p6_data_root")
        self._append_value_to(args, "--out-dir", "p6_out_dir")
        for trial in self._split_var_lines("p6_trials"):
            args.extend(["--trial", trial])
        for pattern in self._split_var_lines("p6_joint_filter"):
            args.extend(["--joint-filter", pattern])
        self._append_value_to(args, "--static-trial", "p6_static_trial")
        self._append_value_to(args, "--model-source", "p6_model_source")
        self._append_value_to(args, "--model-to-c3d-axis", "p6_model_to_c3d_axis")
        self._append_flag(args, "--no-figures", "p6_no_figures")
        self._append_flag(args, "--no-mesh", "p6_no_mesh")
        self._append_value_to(args, "--max-mesh-points", "p6_max_mesh_points")
        self._append_flag(args, "--run-ik-batch", "p6_run_ik_batch")
        self._append_value_to(args, "--ik-max-frames", "p6_ik_max_frames")
        self._append_flag(args, "--visualize", "p6_visualize")
        self._append_value_to(args, "--visualize-trial", "p6_visualize_trial")
        self._append_flag(args, "--headless", "p6_headless")
        self._append_value_to(args, "--rerun-wait-seconds", "p6_rerun_wait_seconds")
        return args

    def _append_value_to(self, args: list[str], option: str, var_name: str) -> None:
        value = str(self.vars[var_name].get()).strip()
        if value:
            args.extend([option, value])

    def _split_var_lines(self, var_name: str) -> list[str]:
        raw = str(self.vars[var_name].get())
        return [
            part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()
        ]

    def _command_mode(self) -> str:
        value = str(self.vars["command_mode"].get())
        for key, label in COMMAND_MODES.items():
            if value == key or value == label:
                return key
        return "kinematic"

    def _current_args(self) -> list[str]:
        mode = self._command_mode()
        if mode == "pipeline":
            return self._command_args()
        if mode == "comparison":
            return self._comparison_args()
        return self._p6_args()

    def _current_output_dir(self) -> Path:
        mode = self._command_mode()
        if mode == "pipeline":
            value = str(self.vars["out_dir"].get()).strip()
        elif mode == "comparison":
            value = str(self.vars["compare_out_dir"].get()).strip()
        else:
            value = str(self.vars["p6_out_dir"].get()).strip()
        return self._resolve(value)

    def _validate_current_command(self) -> bool:
        mode = self._command_mode()
        if mode == "pipeline":
            return self._validate()
        if mode == "comparison":
            return self._validate_comparison()
        return self._validate_p6_analysis()

    def _update_command_preview(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._current_args())
        self.command_text.configure(state=tk.NORMAL)
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert(tk.END, command)
        self.command_text.configure(state=tk.DISABLED)

    def _validate(self) -> bool:
        required_paths = [("BVH", "bvh"), ("C3D", "c3d")]
        for label, var_name in required_paths:
            value = str(self.vars[var_name].get()).strip()
            if not value:
                messagebox.showerror(
                    "Champ manquant", f"Le fichier {label} est requis."
                )
                return False
            if not self._resolve(value).exists():
                messagebox.showerror("Fichier introuvable", f"{label}: {value}")
                return False
        optional_paths = [("FBX", "fbx"), ("Mapping q/C3D", "comparison_map")]
        for label, var_name in optional_paths:
            value = str(self.vars[var_name].get()).strip()
            if value and not self._resolve(value).exists():
                messagebox.showerror("Fichier introuvable", f"{label}: {value}")
                return False
        if (
            bool(self.vars["animate_superposed"].get())
            and not str(self.vars["fbx"].get()).strip()
        ):
            messagebox.showerror(
                "FBX requis", "La scène superposée Rerun nécessite un fichier FBX."
            )
            return False
        return True

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_DIR / path

    def _figure_dir(self) -> Path:
        return self._resolve(str(self.vars["p6_out_dir"].get()).strip()) / "figures"

    def _refresh_figures(self) -> None:
        figure_dir = self._figure_dir()
        self.figure_paths = (
            sorted(figure_dir.glob("**/*.png")) if figure_dir.exists() else []
        )
        self.figure_listbox.delete(0, tk.END)
        for path in self.figure_paths:
            try:
                label = path.relative_to(figure_dir).as_posix()
            except ValueError:
                label = path.name
            self.figure_listbox.insert(tk.END, label)
        if self.figure_paths:
            self.figure_listbox.selection_set(0)
            self._show_selected_figure()
            self.status_var.set(f"{len(self.figure_paths)} figure(s) trouvée(s)")
        else:
            self.figure_photo = None
            self.figure_preview.configure(image="", text="Aucune figure")
            self.status_var.set("Aucune figure trouvée")

    def _selected_figure_path(self) -> Path | None:
        selection = self.figure_listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.figure_paths):
            return None
        return self.figure_paths[index]

    def _show_selected_figure(self) -> None:
        path = self._selected_figure_path()
        if path is None:
            return
        try:
            photo = tk.PhotoImage(file=str(path))
        except tk.TclError as exc:
            self.figure_photo = None
            self.figure_preview.configure(
                image="", text=f"Impossible d'afficher:\n{exc}"
            )
            return
        target_width = max(240, self.figure_preview.winfo_width() or 360)
        target_height = max(180, self.figure_preview.winfo_height() or 260)
        factor = max(
            1,
            int(
                max(
                    photo.width() / max(1, target_width),
                    photo.height() / max(1, target_height),
                )
            ),
        )
        if factor > 1:
            photo = photo.subsample(factor, factor)
        self.figure_photo = photo
        self.figure_preview.configure(image=photo, text="")

    def _open_selected_figure(self) -> None:
        path = self._selected_figure_path()
        if path is None:
            messagebox.showinfo("Figures", "Sélectionne une figure à ouvrir.")
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _run_pipeline(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Le pipeline est déjà lancé.")
            return
        if not self._validate():
            return

        args = self._command_args()
        self._set_running(True)
        self._clear_log()
        self._append_log("$ " + " ".join(shlex.quote(part) for part in args) + "\n\n")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/captury_models_mplconfig")

        def worker() -> None:
            try:
                self.process = subprocess.Popen(
                    args,
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.output_queue.put(line)
                return_code = self.process.wait()
                self.output_queue.put(("__return_code__", return_code))
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                self.output_queue.put(f"\nErreur de lancement: {exc}\n")
                self.output_queue.put(("__return_code__", 1))

        threading.Thread(target=worker, daemon=True).start()

    def _validate_comparison(self) -> bool:
        reference_c3d = str(self.vars["compare_reference_c3d"].get()).strip()
        test_c3d = str(self.vars["compare_test_c3d"].get()).strip()
        if reference_c3d or test_c3d:
            if not reference_c3d or not test_c3d:
                messagebox.showerror(
                    "Paire incomplète",
                    "Référence C3D et Test C3D doivent être fournis ensemble.",
                )
                return False
            explicit_paths = [
                ("Référence C3D", reference_c3d),
                (
                    "Référence BVH",
                    str(self.vars["compare_reference_bvh"].get()).strip(),
                ),
                (
                    "Référence FBX",
                    str(self.vars["compare_reference_fbx"].get()).strip(),
                ),
                ("Test C3D", test_c3d),
                ("Test BVH", str(self.vars["compare_test_bvh"].get()).strip()),
                ("Test FBX", str(self.vars["compare_test_fbx"].get()).strip()),
            ]
            for label, value in explicit_paths:
                if not value:
                    continue
                if not self._resolve(value).exists():
                    messagebox.showerror("Fichier introuvable", f"{label}: {value}")
                    return False
        else:
            data_root = str(self.vars["compare_data_root"].get()).strip()
            if not data_root or not self._resolve(data_root).exists():
                messagebox.showerror("Dossier introuvable", f"Racine: {data_root}")
                return False
        landmark_map = str(self.vars["compare_landmark_map"].get()).strip()
        if landmark_map and not self._resolve(landmark_map).exists():
            messagebox.showerror(
                "Fichier introuvable", f"Carte repères: {landmark_map}"
            )
            return False
        return True

    def _validate_p6_analysis(self) -> bool:
        data_root = str(self.vars["p6_data_root"].get()).strip()
        if not data_root or not self._resolve(data_root).exists():
            messagebox.showerror(
                "Dossier introuvable", f"Dossier cinématique: {data_root}"
            )
            return False
        return True

    def _load_p6_debug_preset(self) -> None:
        self.vars["command_mode"].set(COMMAND_MODES["kinematic"])
        self.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        self.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        self.vars["p6_trials"].set("Static")
        self.vars["p6_static_trial"].set("Static")
        self.vars["p6_joint_filter"].set("Hip|Knee|Ankle|Leg|Foot")
        self.vars["p6_model_source"].set("bvh")
        self.vars["p6_model_to_c3d_axis"].set("y_up_to_z_up")
        self.vars["p6_no_mesh"].set(True)
        self.vars["p6_no_figures"].set(False)
        self.vars["p6_run_ik_batch"].set(False)
        self.vars["p6_ik_max_frames"].set("0")
        self.vars["p6_visualize"].set(False)
        self.vars["p6_visualize_trial"].set("Static")
        self.vars["p6_headless"].set(True)
        self.vars["p6_rerun_wait_seconds"].set("0")
        self.status_var.set("Preset P6 debug chargé")

    def _run_selected_command(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_current_command():
            return
        self._run_args(self._current_args())

    def _run_args(self, args: list[str]) -> None:
        self._set_running(True)
        self._clear_log()
        self._append_log("$ " + " ".join(shlex.quote(part) for part in args) + "\n\n")
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/captury_models_mplconfig")

        def worker() -> None:
            try:
                self.process = subprocess.Popen(
                    args,
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.output_queue.put(line)
                return_code = self.process.wait()
                self.output_queue.put(("__return_code__", return_code))
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                self.output_queue.put(f"\nErreur de lancement: {exc}\n")
                self.output_queue.put(("__return_code__", 1))

        threading.Thread(target=worker, daemon=True).start()

    def _run_comparison(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_comparison():
            return
        self._run_args(self._comparison_args())

    def _run_p6_analysis(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_p6_analysis():
            return
        self._run_args(self._p6_args())

    def _stop_pipeline(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        self.status_var.set("Arrêt demandé")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if hasattr(self, "compare_button"):
            self.compare_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if hasattr(self, "p6_button"):
            self.p6_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        for button in self.analysis_buttons:
            button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.status_var.set("Exécution en cours" if running else "Prêt")

    def _drain_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__return_code__":
                    return_code = item[1]
                    self.process = None
                    self._set_running(False)
                    self.status_var.set(
                        "Terminé" if return_code == 0 else f"Échec ({return_code})"
                    )
                    self._append_log(
                        f"\nProcessus terminé avec le code {return_code}.\n"
                    )
                    if return_code == 0:
                        self._refresh_figures()
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _copy_command(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._current_args())
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande copiée")

    def _copy_comparison_command(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._comparison_args())
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande comparaison copiée")

    def _copy_p6_command(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._p6_args())
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande cinématique copiée")

    def _generated_biomod_path(self, source: str) -> Path:
        filename = {
            "bvh": "model_from_bvh_biobuddy.bioMod",
            "fbx": "model_from_fbx_biobuddy.bioMod",
        }[source]
        return self._resolve(str(self.vars["out_dir"].get()).strip()) / filename

    def _set_generated_model_path(self, source: str) -> None:
        path = self._generated_biomod_path(source)
        self.vars["model_explorer_path"].set(
            os.path.relpath(path, PROJECT_DIR)
            if str(path).startswith(str(PROJECT_DIR))
            else str(path)
        )
        if not path.exists():
            self.status_var.set(
                f"Le modèle {source.upper()} généré n'existe pas encore"
            )

    def _launch_biobuddy_model_explorer(self) -> None:
        raw_path = str(self.vars["model_explorer_path"].get()).strip()
        if not raw_path:
            messagebox.showerror("Modèle manquant", "Sélectionne un modèle à explorer.")
            return
        model_path = self._resolve(raw_path)
        if not model_path.exists():
            messagebox.showerror("Modèle introuvable", str(model_path))
            return

        command = [sys.executable, str(MODEL_EDITOR_SCRIPT), str(model_path)]
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/captury_models_mplconfig")
        try:
            subprocess.Popen(command, cwd=PROJECT_DIR, env=env)
        except Exception as exc:
            messagebox.showerror(
                "BioBuddy", f"Impossible de lancer l'explorateur BioBuddy:\n{exc}"
            )
            return

        self._append_log("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        self.status_var.set("Explorateur BioBuddy lancé")

    def _open_output_dir(self) -> None:
        output_dir = self._current_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        elif os.name == "nt":
            os.startfile(output_dir)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Captury/BioBuddy GUI.")
    parser.add_argument(
        "--p6-debug",
        action="store_true",
        help="Load the local P6 debug preset on startup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = CapturyBioBuddyGui()
    if args.p6_debug:
        app._load_p6_debug_preset()
    app.mainloop()


if __name__ == "__main__":
    main()
