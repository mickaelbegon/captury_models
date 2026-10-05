# Instructions pour Claude

Commence par lire [`AGENTS.md`](AGENTS.md) : il contient la carte du dépôt, les conventions scientifiques, l'environnement et les règles de validation applicables à tous les agents.

Ce dépôt traite de données biomécaniques Captury/Motive et BioBuddy/biorbd. Avant d'agir, examine `git status` et les modifications existantes, puis lis `README.md` et la roadmap pertinente. Préserve les changements locaux sans rapport avec la demande.

Repères de navigation rapide :

- Comparaisons P6 : `compare_p6_motive_captury.py`; comparaison générique : `compare_capture_systems.py`.
- Pipeline historique : `bvh_c3d_biobuddy_pyorerun_compare.py`; GUI : `captury_biobuddy_gui.py` et modules `gui_*.py`.
- Unités, labels, repères et données C3D : `mocap_*.py`, `c3d_*.py`, `captury_frame_calibration.py`.
- Conventions et audit cinématique : `kinematic_*.py`, `joint_kinematics.py`, `isb_*.py`.
- Tests : `tests/`; environnement : `environment_bvh_c3d_biobuddy.yml`.

Pour les tests ciblés, utiliser l'environnement Conda `captury_biobuddy` et `python -m unittest tests.test_<module> -v`; pour la suite, `python -m unittest discover -s tests -v`. Vérifie d'abord que le module de test choisi existe. Toute modification scientifique doit conserver unités, axes, sens des transformations et statuts de preuve; `diagnostic_only` n'autorise pas une conclusion d'accord biomécanique. Consulte les roadmaps avant toute refactorisation ou interprétation scientifique.

Ne commit/push que si l'utilisateur le demande. Avant livraison, inspecte le diff et lance `git diff --check`.
