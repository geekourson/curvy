"""Le décodage incrémental doit être indiscernable du décodage complet.

Ce module réimplémente à la main le corps d'une `nn.TransformerDecoderLayer`
pour y greffer un cache clé/valeur. L'en-tête de `curvy.model.curvy` prévient
que réimplémenter l'attention n'apporte que des occasions de se tromper sur les
masques — **ces tests sont la contrepartie de ce risque**. Sans eux, le module
ne devrait pas exister.

Une erreur d'ordre de normalisation ou de masque ne fait pas planter : elle
produit des logits plausibles et faux. Seule une comparaison numérique la voit.
"""

from __future__ import annotations

import pytest
import torch

from curvy.infer.cache import CacheDecodeur, pas_incremental
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel
from curvy.seeding import seed_everything
from curvy.tokenizer.vocab import BOS_ID


def _modele(preset="small"):
    seed_everything(0)
    m = CurvyModel(PRESETS[preset]).eval()
    return m


def _entree(b=3, n=40):
    torch.manual_seed(1)
    points = torch.randn(b, n, 2)
    masque = torch.zeros(b, n, dtype=torch.bool)
    return points, masque


@pytest.mark.parametrize("preset", ["small", "v1"])
def test_les_logits_coincident_avec_le_chemin_complet(preset):
    """Le test qui justifie le module."""
    m = _modele(preset)
    points, masque = _entree()
    with torch.no_grad():
        memoire = m.encode_points(points, masque)

    torch.manual_seed(2)
    tokens = torch.randint(3, 17, (3, 9))
    tokens = torch.cat([torch.full((3, 1), BOS_ID), tokens], dim=1)

    with torch.no_grad():
        complet = m.decode(memoire, masque, tokens, torch.zeros_like(tokens, dtype=torch.bool))

    cache = CacheDecodeur()
    incremental = []
    for t in range(tokens.size(1)):
        incremental.append(pas_incremental(m, memoire, masque, tokens[:, t : t + 1], cache))
    incremental = torch.cat(incremental, dim=1)

    ecart = (complet - incremental).abs().max().item()
    assert ecart < 2e-4, f"écart maximal {ecart:.2e} entre les deux chemins"


def test_le_cache_grandit_dun_token_par_pas():
    m = _modele()
    points, masque = _entree(b=2)
    with torch.no_grad():
        memoire = m.encode_points(points, masque)
    cache = CacheDecodeur()
    for attendu in (1, 2, 3):
        pas_incremental(m, memoire, masque, torch.full((2, 1), BOS_ID), cache)
        assert cache.longueur == attendu
        assert cache.self_k[0].size(2) == attendu


def test_les_cles_de_la_memoire_ne_sont_calculees_quune_fois():
    """La mémoire de l'encodeur ne change pas : la recalculer serait le gâchis
    que ce module existe pour supprimer."""
    m = _modele()
    points, masque = _entree(b=2)
    with torch.no_grad():
        memoire = m.encode_points(points, masque)
    cache = CacheDecodeur()
    pas_incremental(m, memoire, masque, torch.full((2, 1), BOS_ID), cache)
    premiere = cache.memoire_k[0]
    pas_incremental(m, memoire, masque, torch.full((2, 1), BOS_ID), cache)
    assert cache.memoire_k[0] is premiere, "les clés de mémoire ont été recalculées"


def test_le_reagencement_suit_la_provenance_des_faisceaux():
    """En beam search, le faisceau j du pas suivant peut descendre de n'importe
    quel faisceau du pas courant. Sans réagencement, chacun hériterait du passé
    d'un autre — et les formes resteraient valides, donc rien ne le dirait."""
    m = _modele()
    points, masque = _entree(b=3)
    with torch.no_grad():
        memoire = m.encode_points(points, masque)
    cache = CacheDecodeur()
    pas_incremental(m, memoire, masque, torch.tensor([[3], [9], [16]]), cache)
    avant = cache.self_k[0].clone()

    cache.reordonner(torch.tensor([2, 0, 1]))
    assert torch.equal(cache.self_k[0][0], avant[2])
    assert torch.equal(cache.self_k[0][1], avant[0])
    assert torch.equal(cache.self_k[0][2], avant[1])


def test_le_masque_de_remplissage_des_points_est_respecte():
    """Un nuage plus court que le batch a des positions de remplissage : les
    ignorer ferait attendre le décodeur sur du vide."""
    m = _modele()
    torch.manual_seed(3)
    points = torch.randn(2, 40, 2)
    masque = torch.zeros(2, 40, dtype=torch.bool)
    masque[1, 25:] = True  # le second nuage n'a que 25 points

    with torch.no_grad():
        memoire = m.encode_points(points, masque)
        tokens = torch.tensor([[BOS_ID, 5, 16], [BOS_ID, 5, 16]])
        complet = m.decode(memoire, masque, tokens, torch.zeros_like(tokens, dtype=torch.bool))

    cache = CacheDecodeur()
    sorties = [pas_incremental(m, memoire, masque, tokens[:, t : t + 1], cache) for t in range(3)]
    ecart = (complet - torch.cat(sorties, dim=1)).abs().max().item()
    assert ecart < 2e-4, f"écart {ecart:.2e} — le masque de remplissage est mal appliqué"


# --- le beam search doit rendre exactement la même chose avec et sans cache ---


@pytest.mark.parametrize("beam", [1, 4, 8])
def test_le_beam_search_donne_le_meme_resultat_avec_et_sans_cache(beam):
    """Le test qui protège le réagencement du cache.

    Une erreur de provenance ne casse rien de visible : les séquences restent
    syntaxiquement valides et les scores plausibles. Seule la comparaison avec
    le chemin d'origine la démasque.
    """
    from curvy.infer.decode import beam_search

    m = _modele()
    points, masque = _entree(b=3)
    sans = beam_search(m, points, masque, beam=beam, cache=False)
    avec = beam_search(m, points, masque, beam=beam, cache=True)

    assert len(sans) == len(avec)
    for a, b in zip(sans, avec, strict=True):
        assert [s for s, _ in a] == [s for s, _ in b], "séquences différentes"
        for (_, sa), (_, sb) in zip(a, b, strict=True):
            assert abs(sa - sb) < 1e-3, f"scores {sa} et {sb}"


def test_le_cache_donne_le_meme_resultat_que_le_glouton_a_beam_1():
    """Triple contrôle : glouton == beam 1 sans cache == beam 1 avec cache."""
    from curvy.infer.decode import beam_search, greedy_decode

    m = _modele()
    points, masque = _entree(b=3)
    glouton = greedy_decode(m, points, masque)
    assert [c[0][0] for c in beam_search(m, points, masque, beam=1, cache=False)] == glouton
    assert [c[0][0] for c in beam_search(m, points, masque, beam=1, cache=True)] == glouton
