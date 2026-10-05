# Guide pour les agents de code

## Mission et démarrage

Ce dépôt compare les exports de capture de mouvement Captury et Motive (BVH/FBX, C3D) et les modèles BioBuddy/biorbd. Il fournit des outils CLI et GUI pour préparer les données, construire des modèles, mesurer centres/repères/cinématiques et produire des rapports traçables. Les résultats dépendent de conventions parfois propriétaires : un graphe ou une métrique disponible n'est pas une validation biomécanique.

Avant toute modification, lire ce fichier, `README.md` et la feuille de route liée au sujet (`docs/refactor_roadmap.md` ou `docs/scientific_kinematics_roadmap.md`). Inspecter `git status`, la branche et les changements préexistants. Ne pas écraser ni inclure dans un commit des changements qui ne sont pas liés à la tâche.

## Carte du code

- `bvh_c3d_biobuddy_pyorerun_compare.py` : pipeline historique BVH/FBX/C3D, modèle BioBuddy et visualisation.
- `compare_p6_motive_captury.py`, `compare_capture_systems.py` : découverte des essais, préparation et comparaison Captury/Motive.
- `captury_biobuddy_gui.py` : interface Tk; `gui_commands.py`, `gui_state.py`, `gui_graphs.py`, `gui_run_report.py`, `gui_trial_viewer.py`, `gui_isb_report.py`, `gui_marker_correspondence.py` portent des contrats GUI ciblés.
- `c3d_source_preparation.py`, `c3d_point_channels.py`, `mocap_labels.py`, `mocap_units.py`, `mocap_alignment.py`, `mocap_coordinate_frames.py`, `captury_frame_calibration.py` : préparation C3D, labels, unités, alignement et repères.
- `kinematic_conventions.py` + `kinematic_conventions.json`, `kinematic_rotations.py`, `joint_kinematics.py`, `isb_segment_audit.py`, `isb_compliance_report.py` : conventions et audits de rotations, repères et critères ISB.
- `model_comparison_metrics.py`, `spatial_calibration.py`, `temporal_synchronization.py` : métriques, recalage spatial et synchronisation temporelle.
- `create_biobuddy_c3d_model.py`, `run_biobuddy_c3d_ik.py`, `motive57_c3d_mapping.py` : création de modèle et cinématique inverse.
- `tests/` : tests unitaires/contrats; `data/`, `local_trials/` : entrées; `out_*/` : sorties générées, généralement ignorées par Git.

Le README décrit les CLI/GUI, options et formats de rapports plus en détail. Consulter la roadmap avant de déplacer les responsabilités entre modules.

## Invariants scientifiques à préserver

- Vérifier les unités par source et fichier. Les conventions P6 documentées traitent Captury BVH/FBX comme millimètres, Motive BVH/FBX comme centimètres, les coordonnées labo Captury comme `+Y` up et Motive comme `+Z` up. Les points C3D sont convertis depuis `POINT:UNITS`; ne pas propager une hypothèse d'un fichier à tous les autres.
- Dans l'audit cinématique canonique, les vecteurs sont colonnes et `R_lab_segment` a pour colonnes les axes du segment exprimés dans le laboratoire. La rotation relative proximal→distal est `R_proximal_distal = R_lab_proximal.T @ R_lab_distal`, après corrections de repères justifiées. Ne pas comparer des `q` de même nom comme s'ils avaient la même signification anatomique.
- Garder séparés les axes modèle/laboratoire et les conversions propres à chaque source. Captury C3D est converti vers le cadre commun `+Z` seulement pour les comparaisons inter-systèmes décrites dans le README; respecter les politiques de translation racine et d'alignement propres à chaque source.
- Exclure les canaux d'angles C3D du nuage de marqueurs; les traiter comme une source cinématique distincte tant que séquence, signes, axes et définition ne sont pas établis.
- Ne pas utiliser les mêmes centres pour calculer un recalage puis évaluer l'erreur de ces centres. Une calibration doit être indépendante et figée pour les essais évalués.
- `diagnostic_only`, `inconnu`, `partiel` ou `documente_non_evalue` ne signifient pas conformité. Ne jamais transformer une donnée manquante ou non justifiée en erreur nulle, substitution silencieuse ou comparaison finale. L'orientation Y-up/Z-up ou une séquence ZXY ne certifie pas la conformité ISB.
- Pour toute modification numérique/scientifique, vérifier explicitement unités, dimensions, repères, ordre/signe des transformations, handedness, séquence, tolérances et provenance. Comparer à une référence lorsque les résultats scientifiques changent. Suivre les gates de `docs/scientific_kinematics_roadmap.md`.

## Environnement et validation

Environnement de référence : Conda `captury_biobuddy`, Python 3.11, défini dans `environment_bvh_c3d_biobuddy.yml` (NumPy, SciPy, pandas, ezc3d, biorbd, pyorerun, PySide6, BioBuddy). Les tests du dépôt utilisent `unittest`, pas `pytest` par défaut :

```bash
conda activate captury_biobuddy
python -m unittest discover -s tests -v
```

Commencer par le test ciblé correspondant au module modifié, par exemple :

```bash
python -m unittest tests.test_mocap_units -v
python -m unittest tests.test_joint_kinematics -v
python -m unittest tests.test_gui_refactor_contracts -v
```

Vérifier que le nom de module existe dans `tests/` avant d'utiliser l'exemple; plusieurs fichiers peuvent être testés en les énumérant. Pour une modification Python, lancer au minimum le ou les tests pertinents et `python -m py_compile chemin/vers/module.py`; étendre aux tests d'intégration ou à toute la suite selon le risque. Une GUI nécessite un smoke test réel si le changement touche son lancement ou son interaction. Ne pas lancer une analyse complète sur des données utilisateur sans nécessité. Rapporter séparément contrôles réussis, non lancés ou bloqués.

## Git et méthode de travail

- Préserver les changements déjà présents. Avant édition et livraison, examiner `git status --short --branch` et le diff ciblé.
- Faire des changements petits et cohérents; ne pas inclure fichiers de données locales, sorties, artefacts ou modifications préexistantes sans rapport.
- Ne pas commit, push, fusionner ou publier sans demande explicite de l'utilisateur.
- Suivre `docs/refactor_roadmap.md` pour les refactorisations : tests de caractérisation d'abord, un périmètre à la fois, tests ciblés, `py_compile` et revue du diff. Pour les changements scientifiques, suivre en plus le protocole et les gates de la roadmap scientifique.
- Avant de conclure, relire le diff, contrôler l'absence de changements accidentels et exécuter `git diff --check`.
