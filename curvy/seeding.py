"""Reproductibilité : une seule fonction pour tout semer.

Règle projet n°5 : toute expérience doit être rejouable. La graine est
enregistrée dans chaque fiche d'expérience et dans chaque checkpoint.

Arbitrage numpy (2026-08-19) : ruff (NPY002) réclame l'abandon de
``np.random.seed`` au profit d'un ``Generator`` explicite, et il a raison pour
*notre* code. Mais l'état global hérité reste le seul levier sur les
bibliothèques tierces qui l'utilisent en interne (scipy en particulier, qu'on
emploiera pour l'ajustement des constantes en Phase 5). On fait donc les deux :
on sème l'état global *et* on expose un ``Generator`` que le code de Curvy doit
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42

__all__ = ["DEFAULT_SEED", "make_rng", "seed_everything", "worker_seed_fn"]


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Générateur numpy moderne — à utiliser dans tout le code de Curvy."""
    return np.random.default_rng(seed)


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = False) -> np.random.Generator:
    """Sème python, numpy (état global hérité) et torch (CPU + tous les GPU).

    ``deterministic=True`` force les noyaux cuDNN déterministes : plus lent,
    réservé aux runs de vérification bit-à-bit.

    Retourne le ``Generator`` numpy à utiliser par l'appelant.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Volontaire : sème l'état global pour les bibliothèques tierces (scipy...).
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return make_rng(seed)


def worker_seed_fn(base_seed: int):
    """Fabrique un ``worker_init_fn`` pour DataLoader.

    Sans ça, chaque worker hérite du même état RNG et génère exactement les
    mêmes exemples — bug classique et silencieux des pipelines synthétiques.
    """

    def _init(worker_id: int) -> None:
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s % (2**32))  # noqa: NPY002
        torch.manual_seed(s)

    return _init
