"""Chemins et constantes globales du projet.

Tout ce qui est volumineux — venv, jeux de données, points de reprise —
vit hors du dépôt, sous ``CURVY_DATA_ROOT``. Le défaut convient à la machine
d'origine ; ailleurs, il faut le définir.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Racine des artefacts lourds. Surchargeable par la variable d'environnement
#: ``CURVY_DATA_ROOT`` pour rendre le projet portable.
DATA_ROOT = Path(os.environ.get("CURVY_DATA_ROOT", "~/curvy-data")).expanduser()

DATASET_DIR = DATA_ROOT / "data"
RUNS_DIR = DATA_ROOT / "runs"
CACHE_DIR = DATA_ROOT / "cache"

DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "article" / "figures"
LOGS_DIR = REPO_ROOT / "logs"


def ensure_dirs() -> None:
    """Crée les répertoires d'artefacts s'ils manquent."""
    for d in (DATASET_DIR, RUNS_DIR, CACHE_DIR, FIGURES_DIR, LOGS_DIR / "raw"):
        d.mkdir(parents=True, exist_ok=True)
