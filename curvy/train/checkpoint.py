"""Sauvegarde et reprise.

Un entraînement doit survivre à une déconnexion SSH, à un OOM provoqué par un
processus voisin, et à une coupure. On sauvegarde donc l'état complet — modèle,
optimiseur, ordonnanceur, pas courant, graine — et pas seulement les poids.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

__all__ = ["latest_checkpoint", "load_checkpoint", "save_checkpoint"]


def save_checkpoint(
    path: Path, *, model, optimizer, scheduler, step: int, config: dict, best: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "best": best,
        },
        tmp,
    )
    tmp.replace(path)  # atomique : jamais de checkpoint tronqué
    (path.parent / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))


def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def latest_checkpoint(run_dir: Path) -> Path | None:
    p = run_dir / "last.pt"
    return p if p.exists() else None
