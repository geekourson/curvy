"""Décodage contraint par l'arité.

Le masque de ``curvy.tokenizer.vocab`` est appliqué aux logits à chaque pas :
toute séquence produite est donc syntaxiquement valide **par construction**
. Deux décodeurs, le même masque :

- ``greedy_decode`` — un candidat, utilisé par les métriques d'entraînement
  parce qu'il est rapide ;
- ``beam_search`` — ``k`` candidats, ce que le produit livre réellement
 . La différence n'est pas cosmétique : la baseline
  polynomiale a le droit d'essayer huit degrés et de garder le meilleur, alors
  qu'un décodage glouton n'a droit qu'à un essai. Comparer les deux, c'est
  comparer à armes inégales.
"""

from __future__ import annotations

import numpy as np
import torch

from curvy.data.expr import Node, from_prefix
from curvy.data.grammar import ID_TO_TOKEN
from curvy.infer.cache import CacheDecodeur, pas_incremental
from curvy.tokenizer.vocab import BOS_ID, EOS_ID, MAX_SEQ_LEN, DecodeState, legal_mask

__all__ = ["beam_search", "greedy_decode", "ids_to_node"]


def ids_to_node(ids: list[int]) -> Node | None:
    toks = [ID_TO_TOKEN[i] for i in ids]
    try:
        return from_prefix(toks)
    except ValueError:
        return None


@torch.no_grad()
def greedy_decode(
    model, points: torch.Tensor, point_mask: torch.Tensor, max_len: int = MAX_SEQ_LEN
) -> list[list[int]]:
    """Décodage glouton batché. Retourne les tokens d'arbre, sans `<bos>`/`<eos>`."""
    device = points.device
    b = points.size(0)
    memory = model.encode_points(points, point_mask)

    states = [DecodeState() for _ in range(b)]
    seqs: list[list[int]] = [[] for _ in range(b)]
    done = np.zeros(b, dtype=bool)
    tokens = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
    pad_mask = torch.zeros_like(tokens, dtype=torch.bool)

    for _ in range(max_len - 1):
        logits = model.decode(memory, point_mask, tokens, pad_mask)[:, -1]  # (B, V)
        masks = np.stack([legal_mask(s, max_len=max_len) for s in states])
        allowed = torch.from_numpy(masks).to(device)
        logits = logits.masked_fill(~allowed, float("-inf"))
        nxt = logits.argmax(dim=-1)

        for i in range(b):
            if done[i]:
                continue
            tid = int(nxt[i])
            if tid == EOS_ID or not masks[i].any():
                done[i] = True
                continue
            seqs[i].append(tid)
            states[i].advance(tid)
        if done.all():
            break
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
    return seqs


@torch.no_grad()
def beam_search(
    model,
    points: torch.Tensor,
    point_mask: torch.Tensor,
    beam: int = 8,
    max_len: int = MAX_SEQ_LEN,
    length_penalty: float = 0.0,
    cache: bool = True,
) -> list[list[tuple[list[int], float]]]:
    """Beam search batché sous masque d'arité.

    Retourne, pour chaque exemple, au plus ``beam`` candidats ``(tokens, score)``
    triés du meilleur au moins bon. Les tokens n'incluent ni ``<bos>`` ni
    ``<eos>``, comme ``greedy_decode``.

    ``length_penalty`` divise la log-vraisemblance par ``len ** alpha``. À 0 le
    score est la log-vraisemblance brute, qui favorise mécaniquement les
    squelettes courts — ce qui n'est pas neutre ici, la faiblesse mesurée du
    modèle étant précisément sur les squelettes profonds.

    ``cache`` active le décodage incrémental (``curvy.infer.cache``) : sans lui,
    chaque pas repasse le préfixe entier au décodeur et recalcule tout ce qui a
    déjà été calculé. Le mettre à ``False`` rejoue le chemin d'origine — c'est
    ce que fait le test d'équivalence.
    """
    device = points.device
    b = points.size(0)
    memory = model.encode_points(points, point_mask)
    mem_dim = memory.size(-1)

    # (B, K, ...) aplati en (B*K, ...) : chaque faisceau est une ligne du batch.
    mem_k = memory.unsqueeze(1).expand(b, beam, memory.size(1), mem_dim)
    mem_k = mem_k.reshape(b * beam, memory.size(1), mem_dim)
    pmask_k = point_mask.unsqueeze(1).expand(b, beam, point_mask.size(1))
    pmask_k = pmask_k.reshape(b * beam, point_mask.size(1))

    states = [DecodeState() for _ in range(b * beam)]
    seqs: list[list[int]] = [[] for _ in range(b * beam)]
    # -inf sur les faisceaux 1..K-1 au premier pas : sans ça, les K faisceaux
    # partent identiques et le top-K rend K fois le même candidat.
    scores = torch.full((b, beam), float("-inf"), device=device)
    scores[:, 0] = 0.0
    vivant = np.zeros((b, beam), dtype=bool)
    vivant[:, 0] = True

    fini: list[list[tuple[list[int], float]]] = [[] for _ in range(b)]
    tokens = torch.full((b * beam, 1), BOS_ID, dtype=torch.long, device=device)
    etat_cache = CacheDecodeur() if cache else None

    for pas in range(max_len - 1):
        if not vivant.any():
            break
        if etat_cache is not None:
            # Seul le dernier token entre : tout le préfixe est dans le cache.
            dernier = tokens[:, -1:] if pas else tokens
            logits = pas_incremental(model, mem_k, pmask_k, dernier, etat_cache)[:, -1]
        else:
            pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
            logits = model.decode(mem_k, pmask_k, tokens, pad_mask)[:, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)

        masks = np.stack([legal_mask(s, max_len=max_len) for s in states])
        allowed = torch.from_numpy(masks).to(device)
        logp = logp.masked_fill(~allowed, float("-inf"))
        # Un faisceau mort ou déjà retiré ne doit pas repeupler le top-K.
        logp = logp.masked_fill(
            ~torch.from_numpy(vivant.reshape(-1)).to(device).unsqueeze(1), float("-inf")
        )

        total = scores.reshape(-1, 1) + logp  # (B*K, V)
        total = total.reshape(b, beam * total.size(-1))
        k_eff = min(beam, total.size(-1))
        meilleurs, plats = torch.topk(total, k_eff, dim=-1)

        n_seqs, n_states, n_scores = [], [], np.zeros((b, beam), dtype=bool)
        # Pour chaque emplacement du pas suivant, l'indice de la ligne dont il
        # descend. Identité par défaut : un emplacement non repourvu garde son
        # cache, qui ne sera de toute façon plus lu.
        provenance = torch.arange(b * beam, dtype=torch.long)
        n_tokens = torch.full(
            (b * beam, tokens.size(1) + 1), BOS_ID, dtype=torch.long, device=device
        )
        nouveaux_scores = torch.full((b, beam), float("-inf"), device=device)

        for i in range(b):
            place = 0
            for rang in range(k_eff):
                sc = float(meilleurs[i, rang])
                if sc == float("-inf"):
                    continue
                plat = int(plats[i, rang])
                src, tid = divmod(plat, logp.size(-1))
                ligne = i * beam + src

                if tid == EOS_ID:
                    seq = list(seqs[ligne])
                    if seq:
                        pen = len(seq) ** length_penalty if length_penalty else 1.0
                        fini[i].append((seq, sc / pen))
                    continue
                if place >= beam:
                    continue
                cible = i * beam + place
                n_seqs.append((cible, [*seqs[ligne], tid]))
                etat = states[ligne].copy()
                etat.advance(tid)
                n_states.append((cible, etat))
                n_tokens[cible, : tokens.size(1)] = tokens[ligne]
                n_tokens[cible, tokens.size(1)] = tid
                provenance[cible] = ligne
                nouveaux_scores[i, place] = sc
                n_scores[i, place] = True
                place += 1

        if etat_cache is not None:
            # Le faisceau `cible` descend du faisceau `source` : son cache doit
            # suivre. Sans ce réagencement chaque faisceau hériterait du passé
            # d'un autre, sans que rien ne le signale — les formes restent
            # valides et les logits plausibles.
            etat_cache.reordonner(provenance.to(device))

        for cible, seq in n_seqs:
            seqs[cible] = seq
        for cible, etat in n_states:
            states[cible] = etat
        for i in range(b):
            for j in range(beam):
                if not n_scores[i, j]:
                    seqs[i * beam + j] = []
                    states[i * beam + j] = DecodeState()
        tokens, scores, vivant = n_tokens, nouveaux_scores, n_scores

    return [sorted(c, key=lambda t: -t[1])[:beam] for c in fini]
