"""Décodage incrémental avec cache clé/valeur.

**Le problème, mesuré.** `nn.TransformerDecoder` n'a pas d'API incrémentale : au
pas ``t``, le décodage glouton ou en beam lui repasse le préfixe **entier**, et
tout est recalculé pour les ``t-1`` positions déjà traitées. Sur ``T = 40``
tokens, cela fait ``T(T+1)/2 = 820`` positions calculées au lieu de ``40``.

Profilage du 2026-08-20 : le décodage pèse **98,5 %** d'une requête à beam 8, et
dans ce décodage les passes du décodeur pèsent **99 %** (le masque d'arité en
Python, 1 %). C'est donc le seul poste qui vaille d'être optimisé.

**Ce que fait ce module.** Il rejoue à la main le corps d'une
``nn.TransformerDecoderLayer`` en ``norm_first``, en **réutilisant les
sous-modules du modèle entraîné** — `self_attn`, `multihead_attn`, `linear1/2`,
`norm1/2/3`. Aucun poids n'est copié ni remappé : c'est le même modèle, parcouru
autrement. Deux caches :

- les projections **clé/valeur de l'auto-attention**, allongées d'un token par
  pas ;
- les projections **clé/valeur de la cross-attention**, calculées **une seule
  fois** — la mémoire de l'encodeur ne change pas d'un pas à l'autre.

**Le risque est de se tromper sur les masques ou l'ordre des normalisations**,
et l'en-tête de `curvy.model.curvy` le dit : réimplémenter l'attention n'apporte
que des occasions de se tromper. La parade est un test d'équivalence stricte —
les logits du chemin incrémental doivent coïncider avec ceux du chemin complet,
sur des entrées aléatoires, à la tolérance flottante près. Sans ce test, ce
module ne devrait pas exister.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["CacheDecodeur", "pas_incremental"]


@dataclass
class CacheDecodeur:
    """État de décodage d'un batch. Une entrée par couche."""

    #: clés/valeurs de l'auto-attention, (B, têtes, t, dim_tête)
    self_k: list[Tensor] = field(default_factory=list)
    self_v: list[Tensor] = field(default_factory=list)
    #: clés/valeurs de la cross-attention, invariantes
    memoire_k: list[Tensor] = field(default_factory=list)
    memoire_v: list[Tensor] = field(default_factory=list)
    longueur: int = 0

    def reordonner(self, index: Tensor) -> None:
        """Réordonne le cache selon la provenance des faisceaux retenus.

        Indispensable en beam search : au pas suivant, le faisceau ``j`` peut
        descendre de n'importe quel faisceau du pas précédent. Sans ce
        réagencement, chaque faisceau hériterait du passé d'un autre — et rien
        ne le signalerait, les formes restant valides.
        """
        for liste in (self.self_k, self.self_v, self.memoire_k, self.memoire_v):
            for i, t in enumerate(liste):
                liste[i] = t.index_select(0, index)


def _projeter(module, entree: Tensor, quoi: str) -> Tensor:
    """q, k ou v d'une ``nn.MultiheadAttention`` à partir de ses poids groupés."""
    d = module.embed_dim
    dec = {"q": 0, "k": d, "v": 2 * d}[quoi]
    poids = module.in_proj_weight[dec : dec + d]
    biais = None if module.in_proj_bias is None else module.in_proj_bias[dec : dec + d]
    return F.linear(entree, poids, biais)


def _en_tetes(t: Tensor, n_tetes: int) -> Tensor:
    b, longueur, d = t.shape
    return t.view(b, longueur, n_tetes, d // n_tetes).transpose(1, 2)


def _fusionner(t: Tensor) -> Tensor:
    b, n_tetes, longueur, dim = t.transpose(1, 2).transpose(1, 2).shape
    return t.transpose(1, 2).reshape(b, longueur, n_tetes * dim)


@torch.no_grad()
def pas_incremental(
    model,
    memoire: Tensor,
    memoire_mask: Tensor,
    token: Tensor,
    cache: CacheDecodeur,
) -> Tensor:
    """Logits du prochain token, pour ``token`` (B, 1). Met le cache à jour.

    Ne renvoie **que** la dernière position : c'est tout ce dont un décodage
    autorégressif a besoin, et c'est précisément ce que le chemin complet
    recalcule inutilement pour tout le préfixe.
    """
    couches = model.decoder.layers
    premiere = cache.longueur == 0
    if premiere:
        cache.self_k = [None] * len(couches)
        cache.self_v = [None] * len(couches)
        cache.memoire_k = [None] * len(couches)
        cache.memoire_v = [None] * len(couches)

    # Position absolue du token courant. `SinusoidalPositions` ajoute
    # `pe[:len]` ; ici on ne traite qu'une position, celle d'indice
    # `cache.longueur`. Et le modèle ne met PAS l'embedding à l'échelle par
    # sqrt(d) — l'ajouter aurait produit des logits plausibles et faux.
    h = model.token_embed(token) + model.token_pos.pe[cache.longueur].view(1, 1, -1)

    pad_memoire = memoire_mask.unsqueeze(1).unsqueeze(2) if memoire_mask is not None else None

    for i, couche in enumerate(couches):
        n_tetes = couche.self_attn.num_heads

        # --- auto-attention (norm_first) ---
        x = couche.norm1(h)
        q = _en_tetes(_projeter(couche.self_attn, x, "q"), n_tetes)
        k = _en_tetes(_projeter(couche.self_attn, x, "k"), n_tetes)
        v = _en_tetes(_projeter(couche.self_attn, x, "v"), n_tetes)
        if cache.self_k[i] is not None:
            k = torch.cat([cache.self_k[i], k], dim=2)
            v = torch.cat([cache.self_v[i], v], dim=2)
        cache.self_k[i], cache.self_v[i] = k, v
        # Aucun masque causal : le cache ne contient que le passé.
        a = F.scaled_dot_product_attention(q, k, v)
        h = h + couche.self_attn.out_proj(_fusionner(a))

        # --- cross-attention, clés/valeurs calculées une seule fois ---
        x = couche.norm2(h)
        if cache.memoire_k[i] is None:
            cache.memoire_k[i] = _en_tetes(_projeter(couche.multihead_attn, memoire, "k"), n_tetes)
            cache.memoire_v[i] = _en_tetes(_projeter(couche.multihead_attn, memoire, "v"), n_tetes)
        q = _en_tetes(_projeter(couche.multihead_attn, x, "q"), n_tetes)
        a = F.scaled_dot_product_attention(
            q,
            cache.memoire_k[i],
            cache.memoire_v[i],
            attn_mask=~pad_memoire if pad_memoire is not None else None,
        )
        h = h + couche.multihead_attn.out_proj(_fusionner(a))

        # --- réseau à propagation avant ---
        x = couche.norm3(h)
        h = h + couche.linear2(couche.dropout(couche.activation(couche.linear1(x))))

    cache.longueur += 1
    if model.decoder.norm is not None:
        h = model.decoder.norm(h)
    return model.head(h)
