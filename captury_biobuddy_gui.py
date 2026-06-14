"""Graphical launcher for the Captury/BioBuddy BVH/FBX/C3D pipeline."""

from __future__ import annotations

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


class CapturyBioBuddyGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Captury BioBuddy")
        self.geometry("1180x780")
        self.minsize(980, 680)

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str | tuple[str, int]] = queue.Queue()

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
        ttk.Label(header, text="Captury BioBuddy", font=("TkDefaultFont", 18, "bold")).grid(row=0, column=0, sticky="w")
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
        notebook.grid(row=0, column=0, sticky="nsew")

        self._build_sources_tab(notebook)
        self._build_model_tab(notebook)
        self._build_rerun_tab(notebook)
        self._build_ik_tab(notebook)
        self._build_advanced_tab(notebook)

        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_actions(right)
        self._build_log_panel(right)

    def _tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(0, weight=1)
        notebook.add(frame, text=title)
        return frame

    def _build_sources_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Sources")
        card = ttk.LabelFrame(tab, text="Fichiers")
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)

        self._path_row(card, 0, "BVH", "bvh", [("BVH", "*.bvh"), ("Tous les fichiers", "*")])
        self._path_row(card, 1, "FBX", "fbx", [("FBX", "*.fbx"), ("Tous les fichiers", "*")])
        self._path_row(card, 2, "C3D", "c3d", [("C3D", "*.c3d"), ("Tous les fichiers", "*")])
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
        self._path_row(units, 5, "Mapping q/C3D", "comparison_map", [("JSON", "*.json"), ("Tous les fichiers", "*")])

        generation = ttk.LabelFrame(tab, text="Génération")
        generation.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        generation.columnconfigure(1, weight=1)
        self._combo_row(generation, 0, "Correction root offset", "root_offset_mode", ("auto", "subtract", "keep"))
        self._entry_row(generation, 1, "Max points mesh FBX", "max_fbx_mesh_points")
        self._check(generation, 2, "Ne pas ajouter les marqueurs de centres articulaires", "no_biomod_joint_centre_markers")
        self._check(generation, 3, "Ne pas corriger le root offset (compatibilité)", "no_root_offset_correction")
        self._check(generation, 4, "Ne pas générer les meshes FBX", "no_fbx_mesh")

        explorer = ttk.LabelFrame(tab, text="Explorateur BioBuddy")
        explorer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        explorer.columnconfigure(1, weight=1)
        self._path_row(
            explorer,
            0,
            "Modèle",
            "model_explorer_path",
            [("Modèles BioBuddy", "*.bioMod *.osim *.urdf *.bvh"), ("Tous les fichiers", "*")],
        )
        ttk.Button(explorer, text="BVH généré", command=lambda: self._set_generated_model_path("bvh")).grid(
            row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 10)
        )
        ttk.Button(explorer, text="FBX généré", command=lambda: self._set_generated_model_path("fbx")).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(0, 10)
        )
        ttk.Button(explorer, text="Ouvrir dans BioBuddy", command=self._launch_biobuddy_model_explorer).grid(
            row=1, column=2, sticky="ew", padx=(4, 10), pady=(0, 10)
        )

    def _build_rerun_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Rerun")
        launch = ttk.LabelFrame(tab, text="Visualisation")
        launch.grid(row=0, column=0, sticky="ew")
        launch.columnconfigure(1, weight=1)
        self._check(launch, 0, "Lancer une scène Rerun par modèle", "animate")
        self._check(launch, 1, "Lancer la scène superposée BVH + FBX + C3D", "animate_superposed")
        self._check(launch, 2, "Afficher les courbes q dans Rerun", "display_q_in_rerun")
        self._check(launch, 3, "Mode headless", "headless")
        self._entry_row(launch, 4, "Rayon marqueurs", "rerun_marker_radius")
        self._entry_row(launch, 5, "Attente après envoi", "rerun_wait_seconds")
        self._combo_row(launch, 6, "Axe vertical", "rerun_up_axis", ("y", "z", "x", "none"))

        filters = ttk.LabelFrame(tab, text="Lisibilité")
        filters.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self._check(filters, 0, "Masquer mains/poignets/doigts", "hide_hands_in_rerun")
        self._check(filters, 1, "Masquer pieds/chevilles/orteils", "hide_feet_in_rerun")
        self._check(filters, 2, "Masquer toutes les extrémités", "hide_extremities_in_rerun")

    def _build_ik_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "IK")
        ik = ttk.LabelFrame(tab, text="Cinématique inverse")
        ik.grid(row=0, column=0, sticky="ew")
        ik.columnconfigure(1, weight=1)
        self._check(ik, 0, "Lancer la cinématique inverse depuis les marqueurs C3D", "inverse_kinematics")
        self._combo_row(ik, 1, "Solveur", "inverse_kinematics_solver", ("least_squares", "kalman"))
        self._combo_row(ik, 2, "Méthode least-squares", "inverse_kinematics_method", ("trf", "lm", "only_lm"))
        self._entry_row(ik, 3, "Nombre max de frames", "inverse_kinematics_max_frames")
        self._entry_row(ik, 4, "Kalman noise factor", "kalman_noise_factor")
        self._entry_row(ik, 5, "Kalman error factor", "kalman_error_factor")

        ttk.Label(
            ik,
            text="0 frame max signifie que toutes les frames du C3D sont utilisées.",
            style="Status.TLabel",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

    def _build_advanced_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Avancé")
        legacy = ttk.LabelFrame(tab, text="Compatibilité ancienne CLI")
        legacy.grid(row=0, column=0, sticky="ew")
        legacy.columnconfigure(1, weight=1)
        self._check(legacy, 0, "Utiliser --inverse-dynamics (déprécié)", "inverse_dynamics")
        self._combo_row(legacy, 1, "Méthode inverse dynamics", "inverse_dynamics_method", ("", "trf", "lm", "only_lm"))
        self._entry_row(legacy, 2, "Max frames inverse dynamics", "inverse_dynamics_max_frames")

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

        self.command_text = tk.Text(actions, height=7, wrap=tk.WORD, font=("Menlo", 11))
        self.command_text.grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=8)

        self.run_button = ttk.Button(actions, text="Lancer", style="Primary.TButton", command=self._run_pipeline)
        self.run_button.grid(row=1, column=0, sticky="ew", padx=(8, 4), pady=(0, 8))
        self.stop_button = ttk.Button(actions, text="Arrêter", style="Danger.TButton", command=self._stop_pipeline, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 8))
        ttk.Button(actions, text="Copier", command=self._copy_command).grid(row=1, column=2, sticky="ew", padx=4, pady=(0, 8))
        ttk.Button(actions, text="Ouvrir sortie", command=self._open_output_dir).grid(
            row=1, column=3, sticky="ew", padx=(4, 8), pady=(0, 8)
        )

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(actions, textvariable=self.status_var, style="Status.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8)
        )

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Log")
        panel.grid(row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.log_text = tk.Text(panel, wrap=tk.WORD, font=("Menlo", 11), state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _path_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
        directory: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(
            parent,
            text="Parcourir",
            command=lambda: self._browse_path(var_name, filetypes=filetypes, directory=directory),
        ).grid(row=row, column=2, sticky="ew", padx=10, pady=6)

    def _entry_row(self, parent: ttk.Widget, row: int, label: str, var_name: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(row=row, column=1, columnspan=2, sticky="ew", padx=10, pady=6)

    def _combo_row(self, parent: ttk.Widget, row: int, label: str, var_name: str, values: tuple[str, ...]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        ttk.Combobox(parent, textvariable=self.vars[var_name], values=values, state="readonly").grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=10, pady=6
        )

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
            selected = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes or [("Tous les fichiers", "*")])
        if selected:
            self.vars[var_name].set(os.path.relpath(selected, PROJECT_DIR) if str(selected).startswith(str(PROJECT_DIR)) else selected)

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

        self._append_flag(args, "--no-biomod-joint-centre-markers", "no_biomod_joint_centre_markers")
        self._append_flag(args, "--no-root-offset-correction", "no_root_offset_correction")
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
        self._append_flag(args, "--hide-extremities-in-rerun", "hide_extremities_in_rerun")
        self._append_flag(args, "--headless", "headless")

        self._append_flag(args, "--inverse-kinematics", "inverse_kinematics")
        self._append_value(args, "--inverse-kinematics-solver", "inverse_kinematics_solver")
        self._append_value(args, "--inverse-kinematics-method", "inverse_kinematics_method")
        self._append_value(args, "--inverse-kinematics-max-frames", "inverse_kinematics_max_frames")
        self._append_value(args, "--kalman-noise-factor", "kalman_noise_factor")
        self._append_value(args, "--kalman-error-factor", "kalman_error_factor")

        self._append_flag(args, "--inverse-dynamics", "inverse_dynamics")
        self._append_value(args, "--inverse-dynamics-method", "inverse_dynamics_method")
        self._append_value(args, "--inverse-dynamics-max-frames", "inverse_dynamics_max_frames")
        return args

    def _append_path(self, args: list[str], option: str, var_name: str, required: bool = False) -> None:
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
        return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]

    def _update_command_preview(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._command_args())
        self.command_text.configure(state=tk.NORMAL)
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert(tk.END, command)
        self.command_text.configure(state=tk.DISABLED)

    def _validate(self) -> bool:
        required_paths = [("BVH", "bvh"), ("C3D", "c3d")]
        for label, var_name in required_paths:
            value = str(self.vars[var_name].get()).strip()
            if not value:
                messagebox.showerror("Champ manquant", f"Le fichier {label} est requis.")
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
        if bool(self.vars["animate_superposed"].get()) and not str(self.vars["fbx"].get()).strip():
            messagebox.showerror("FBX requis", "La scène superposée Rerun nécessite un fichier FBX.")
            return False
        return True

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_DIR / path

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

    def _stop_pipeline(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        self.status_var.set("Arrêt demandé")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
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
                    self.status_var.set("Terminé" if return_code == 0 else f"Échec ({return_code})")
                    self._append_log(f"\nProcessus terminé avec le code {return_code}.\n")
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
        command = self.command_text.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande copiée")

    def _generated_biomod_path(self, source: str) -> Path:
        filename = {
            "bvh": "model_from_bvh_biobuddy.bioMod",
            "fbx": "model_from_fbx_biobuddy.bioMod",
        }[source]
        return self._resolve(str(self.vars["out_dir"].get()).strip()) / filename

    def _set_generated_model_path(self, source: str) -> None:
        path = self._generated_biomod_path(source)
        self.vars["model_explorer_path"].set(
            os.path.relpath(path, PROJECT_DIR) if str(path).startswith(str(PROJECT_DIR)) else str(path)
        )
        if not path.exists():
            self.status_var.set(f"Le modèle {source.upper()} généré n'existe pas encore")

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
            messagebox.showerror("BioBuddy", f"Impossible de lancer l'explorateur BioBuddy:\n{exc}")
            return

        self._append_log("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        self.status_var.set("Explorateur BioBuddy lancé")

    def _open_output_dir(self) -> None:
        output_dir = self._resolve(str(self.vars["out_dir"].get()).strip())
        output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        elif os.name == "nt":
            os.startfile(output_dir)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])


def main() -> None:
    app = CapturyBioBuddyGui()
    app.mainloop()


if __name__ == "__main__":
    main()
