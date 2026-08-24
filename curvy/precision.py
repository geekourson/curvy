"""Précision numérique : bf16 et TF32.

Ampere (RTX 3090, sm_86) supporte nativement le bf16 et le TF32 :

- **bf16** a la même plage d'exposant que le fp32, donc pas d'``inf`` sur les
  logits et **aucun ``GradScaler``** — contrairement au fp16. C'est le bon
  défaut pour l'entraînement.
- **TF32** est une troncature de la mantisse appliquée aux matmuls et aux
  convolutions en fp32. On y gagne beaucoup de débit pour une perte de
  précision sans effet à notre échelle. PyTorch le désactive par défaut depuis
  la 1.12 : il faut l'activer explicitement.
"""

from __future__ import annotations

import torch

__all__ = ["bf16_supported", "configure_precision", "precision_report"]


def bf16_supported(device: torch.device) -> bool:
    """Le device sait-il faire du bf16 sans émulation ?"""
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "cpu":
        return hasattr(torch, "bfloat16")
    return False  # MPS : bf16 partiel selon les versions, on ne parie pas dessus


def configure_precision(tf32: bool = True) -> None:
    """Active TF32 pour les matmuls et cuDNN. Sans effet hors CUDA."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    # API récente, plus explicite que le booléen historique.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")


def precision_report(device: torch.device) -> dict[str, object]:
    """Instantané, pour l'en-tête des fiches d'expérience."""
    rep: dict[str, object] = {
        "device": str(device),
        "bf16_supported": bf16_supported(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        rep["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        rep["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    if device.type == "cuda":  # et non `cuda.is_available()` : le device peut être CPU
        props = torch.cuda.get_device_properties(device)
        rep["compute_capability"] = f"sm_{props.major}{props.minor}"
        rep["device_name"] = props.name
    return rep
