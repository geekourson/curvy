"""Configurations d'architecture.

Deux tailles, et l'ordre dans lequel on s'en sert compte : on valide la chaîne
complète avec ``SMALL`` (~5M) avant de lancer ``V1`` (~30M). Découvrir un bug
de dataloader après huit heures d'entraînement est une erreur évitable.
"""

from __future__ import annotations

from dataclasses import dataclass

from curvy.tokenizer.vocab import MAX_SEQ_LEN, VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    #: Encodage des coordonnées d'entrée. ``linear`` suit la spec ; ``fourier``
    #: ajoute des features sinusoïdales, connues pour aider les réseaux à lire
    #: des coordonnées brutes. À départager par ablation en Phase 4.
    point_encoding: str = "linear"
    n_fourier_bands: int = 8
    #: Encodage positionnel sur les points d'entrée. Désactivé = le nuage est
    #: traité comme un **ensemble**.
    point_positional: bool = False


#: ~5M de paramètres — validation du bout en bout en quelques minutes.
SMALL = ModelConfig(
    d_model=256, n_heads=4, n_encoder_layers=4, n_decoder_layers=4, dim_feedforward=512
)

#: ~30M de paramètres — cible v1. Les dimensions de la spec (d_model 512,
#: 6 + 6 couches) donnent 44M avec un FFN standard à 4×d_model ; on descend le
#: FFN à 2×d_model, ce qui ramène à ~31M sans toucher aux dimensions annoncées.
V1 = ModelConfig()

PRESETS = {"small": SMALL, "v1": V1}
