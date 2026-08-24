"""Sélection du device de calcul.

Le projet doit tourner à l'identique sur trois cibles :
  - serveur Linux multi-GPU CUDA (machine de développement réelle) ;
  - MacBook Apple Silicon (backend MPS, cible annoncée dans la spec) ;
  - CPU (CI, machines sans accélérateur).

Piège rencontré le 2026-08-19 : par défaut le runtime CUDA numérote les GPU
par ordre de puissance décroissante (``CUDA_DEVICE_ORDER=FASTEST_FIRST``), pas
par ordre de bus PCI. Sur cette machine ``cuda:0`` désignait donc la RTX 3090
— entièrement occupée par un autre processus — et non la RTX 3060 libre.
``pick_device`` choisit sur la mémoire libre réelle, ce qui rend la question

Politique d'allocation : sur cette machine, la RTX 3090 est allouée
au projet et la RTX 3060 est réservée à d'autres travaux. La contrainte est
appliquée mécaniquement par la variable d'environnement ``CURVY_CUDA_ALLOW``
(liste d'UUID ou d'index CUDA), positionnée par le ``Makefile`` — et non par
la discipline de l'opérateur.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

__all__ = ["DeviceInfo", "pick_device", "describe_backends"]


@dataclass(frozen=True)
class DeviceInfo:
    """Device retenu et pourquoi."""

    device: torch.device
    name: str
    free_bytes: int | None
    reason: str

    def __str__(self) -> str:
        free = "n/a" if self.free_bytes is None else f"{self.free_bytes / 2**30:.1f} Gio libres"
        return f"{self.device} ({self.name}, {free}) — {self.reason}"


def _normalise_uuid(value: str) -> str:
    """Ramène un UUID de GPU à une forme comparable.

    Piège (2026-08-19) : ``nvidia-smi`` écrit ``GPU-1234abcd-...`` tandis que
    ``torch.cuda.get_device_properties(i).uuid`` écrit ``1234abcd-...``, sans
    le préfixe. Comparer les deux tels quels ne matche jamais — et l'allowlist
    se vidait donc silencieusement.
    """
    return value.strip().lower().removeprefix("gpu-")


def _allowlist() -> set[str] | None:
    """Ensemble des GPU autorisés, ou ``None`` si aucune restriction.

    Lu depuis ``CURVY_CUDA_ALLOW`` : liste séparée par des virgules d'index
    CUDA (``0``, ``1``) et/ou d'UUID, avec ou sans le préfixe ``GPU-``. Les
    UUID sont préférables : ils survivent aux renumérotations.
    """
    raw = os.environ.get("CURVY_CUDA_ALLOW", "").strip()
    if not raw:
        return None
    return {_normalise_uuid(tok) for tok in raw.split(",") if tok.strip()}


def _cuda_candidates() -> list[tuple[int, str, int]]:
    """(index, nom, octets libres) pour chaque GPU CUDA réellement interrogeable.

    Un GPU saturé échoue à l'initialisation de son contexte : on l'écarte
    silencieusement plutôt que de faire tomber tout le programme.

    Subtilité coûteuse (bug du 2026-08-19) : ``mem_get_info(i)`` bascule le
    device courant sur ``i`` *avant* d'échouer, et ne le restaure pas. Une
    sonde ratée laissait donc ``current_device()`` sur le GPU mort, et le
    premier ``torch.cuda.synchronize()`` sans argument explosait très loin de
    la cause. On restaure explicitement le device courant après chaque sonde.
    """
    allow = _allowlist()
    out: list[tuple[int, str, int]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        if (
            allow is not None
            and str(i) not in allow
            and _normalise_uuid(str(props.uuid)) not in allow
        ):
            continue  # GPU réservé à d'autres travaux : on ne le sonde même pas
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:  # contexte CUDA impossible à créer -> GPU inutilisable
            continue
        out.append((i, name, free))
    if out:
        # Ramène le device courant sur un GPU sain, quoi qu'aient fait les sondes.
        torch.cuda.set_device(out[0][0])
    return out


def pick_device(preference: str = "auto", min_free_gib: float = 2.0) -> DeviceInfo:
    """Retourne le device à utiliser.

    ``preference`` vaut ``auto``, ``cuda``, ``mps``, ``cpu`` ou un identifiant
    explicite du type ``cuda:1``. En mode ``auto`` on prend le GPU CUDA offrant
    le plus de mémoire libre, à défaut MPS, à défaut le CPU.
    """
    if preference not in ("auto", "cuda", "mps", "cpu"):
        dev = torch.device(preference)
        return DeviceInfo(dev, preference, None, "imposé explicitement par l'appelant")

    if preference in ("auto", "cuda") and torch.cuda.is_available():
        usable = [c for c in _cuda_candidates() if c[2] >= min_free_gib * 2**30]
        if usable:
            idx, name, free = max(usable, key=lambda c: c[2])
            torch.cuda.set_device(idx)  # sinon les appels sans device explicite visent cuda:0
            return DeviceInfo(
                torch.device(f"cuda:{idx}"),
                name,
                free,
                f"GPU CUDA le plus libre parmi {torch.cuda.device_count()} détecté(s)",
            )
        if preference == "cuda" or _allowlist() is not None:
            # Repli silencieux sur CPU interdit : une allowlist mal orthographiée
            # nous a déjà fait « choisir » le CPU sans le dire. Mieux vaut casser.
            raise RuntimeError(
                f"Aucun GPU autorisé n'a {min_free_gib} Gio libres. "
                f"CURVY_CUDA_ALLOW={os.environ.get('CURVY_CUDA_ALLOW', '(non défini)')} ; "
                f"GPU visibles et sondables : {_cuda_candidates()}. "
                f"Forcer le CPU avec preference='cpu' si c'est voulu."
            )

    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            torch.device("mps"), "Apple Silicon (MPS)", None, "backend MPS disponible"
        )

    if preference == "mps":
        raise RuntimeError("MPS demandé mais indisponible (machine non-Apple ou torch sans MPS).")

    return DeviceInfo(torch.device("cpu"), "CPU", None, "aucun accélérateur exploitable")


def describe_backends() -> dict[str, object]:
    """Instantané des backends, pour les fiches d'expérience."""
    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "(non défini -> FASTEST_FIRST)"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "(non défini)"),
        "CURVY_CUDA_ALLOW": os.environ.get("CURVY_CUDA_ALLOW", "(non défini -> tous les GPU)"),
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            {"index": i, "name": n, "free_gib": round(f / 2**30, 2)}
            for i, n, f in _cuda_candidates()
        ]
    return info
