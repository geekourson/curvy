"""Architecture encodeur-décodeur.

- **Encodeur** : transformer sur l'ensemble des points. Chaque point ``(x, y)``
  est projeté en un vecteur, puis les points s'attendent mutuellement. Par
  défaut **aucun encodage positionnel** : un nuage est un ensemble, pas une
  séquence — et l'information d'ordre est déjà portée par la valeur de ``x``
  elle-même.
- **Décodeur** : autorégressif sur les tokens du squelette, cross-attention
  vers l'encodeur.

On s'appuie sur ``nn.TransformerEncoder``/``TransformerDecoder`` de PyTorch,
qui utilisent le SDPA fusionné en interne. Réimplémenter l'attention à la main
n'apporterait ici que des occasions de se tromper sur les masques.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from curvy.model.config import ModelConfig

__all__ = ["CurvyModel", "PointEmbedding", "count_parameters"]


class PointEmbedding(nn.Module):
    """``(x, y)`` -> vecteur de dimension ``d_model``.

    En mode ``fourier``, les coordonnées sont d'abord développées en
    sinus/cosinus à plusieurs fréquences. Un réseau lit très mal des
    coordonnées brutes ; ce développement est le remède standard. Il reste
    optionnel pour que l'ablation de la Phase 4 puisse le chiffrer.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mode = cfg.point_encoding
        if self.mode == "fourier":
            bands = 2.0 ** torch.arange(cfg.n_fourier_bands)
            self.register_buffer("bands", bands, persistent=False)
            in_dim = 2 + 4 * cfg.n_fourier_bands
        elif self.mode == "linear":
            in_dim = 2
        else:
            raise ValueError(f"point_encoding inconnu : {cfg.point_encoding!r}")
        self.proj = nn.Linear(in_dim, cfg.d_model)

    def forward(self, points: Tensor) -> Tensor:
        if self.mode == "fourier":
            angles = points.unsqueeze(-1) * self.bands * math.pi  # (B, N, 2, K)
            feats = torch.cat([points, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)
            return self.proj(feats)
        return self.proj(points)


class SinusoidalPositions(nn.Module):
    """Encodage positionnel classique, pour les tokens du décodeur."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class CurvyModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.point_embed = PointEmbedding(cfg)
        self.point_pos = SinusoidalPositions(cfg.d_model, 4096) if cfg.point_positional else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm : bien plus stable sans warmup agressif
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            cfg.n_encoder_layers,
            norm=nn.LayerNorm(cfg.d_model),
            # Le chemin « nested tensor » est incompatible avec norm_first ;
            # le laisser à True ne produit qu'un avertissement à chaque
            # construction de modèle.
            enable_nested_tensor=False,
        )

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.token_pos = SinusoidalPositions(cfg.d_model, cfg.max_seq_len)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, cfg.n_decoder_layers, norm=nn.LayerNorm(cfg.d_model)
        )
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        # Le partage des poids entre embedding et sortie est gratuit et régularise.
        self.head.weight = self.token_embed.weight

        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def encode_points(self, points: Tensor, point_mask: Tensor) -> Tensor:
        """``points`` (B, N, 2), ``point_mask`` (B, N) — True sur le remplissage."""
        h = self.point_embed(points)
        if self.point_pos is not None:
            h = self.point_pos(h)
        return self.encoder(h, src_key_padding_mask=point_mask)

    def decode(
        self, memory: Tensor, memory_mask: Tensor, tokens_in: Tensor, tokens_mask: Tensor
    ) -> Tensor:
        h = self.token_pos(self.token_embed(tokens_in))
        # Masque causal **booléen** et non flottant : mélanger un masque
        # d'attention float et un masque de padding bool déclenche un
        # avertissement de dépréciation et, à terme, un chemin non fusionné.
        causal = torch.ones(
            tokens_in.size(1), tokens_in.size(1), dtype=torch.bool, device=tokens_in.device
        ).triu(1)
        h = self.decoder(
            h,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tokens_mask,
            memory_key_padding_mask=memory_mask,
            tgt_is_causal=True,
        )
        return self.head(h)

    def forward(
        self, points: Tensor, point_mask: Tensor, tokens_in: Tensor, tokens_mask: Tensor
    ) -> Tensor:
        memory = self.encode_points(points, point_mask)
        return self.decode(memory, point_mask, tokens_in, tokens_mask)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Compte exact par bloc — la spec demande le chiffre, pas une estimation."""
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = name.split(".")[0]
        groups[block] = groups.get(block, 0) + p.numel()
    groups["TOTAL"] = sum(v for k, v in groups.items() if k != "TOTAL")
    return groups
