"""Phase 5 — beam search, front de Pareto, sélection.

Le fil conducteur de ces tests : le beam ne doit jamais produire une séquence
que le masque d'arité interdirait, et la sélection ne doit jamais regarder les
points tenus à l'écart.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curvy.data.expr import from_prefix
from curvy.infer.decode import beam_search, greedy_decode, ids_to_node
from curvy.infer.pareto import Candidat, ajuster_candidats, front_de_pareto, selectionner
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel
from curvy.seeding import make_rng, seed_everything


def _modele():
    seed_everything(0)
    return CurvyModel(PRESETS["small"]).eval()


def _entree(b=3, n=40):
    return torch.randn(b, n, 2), torch.zeros(b, n, dtype=torch.bool)


def _p(s):
    return from_prefix(s.split())


def test_beam_de_taille_1_redonne_exactement_le_glouton():
    m = _modele()
    pts, pm = _entree()
    glouton = greedy_decode(m, pts, pm)
    beam = beam_search(m, pts, pm, beam=1)
    assert [c[0][0] for c in beam] == glouton


def test_tous_les_candidats_du_beam_sont_syntaxiquement_valides():
    m = _modele()
    pts, pm = _entree()
    for cands in beam_search(m, pts, pm, beam=6):
        assert cands, "le beam doit rendre au moins un candidat"
        for seq, _ in cands:
            assert ids_to_node(seq) is not None


def test_le_beam_rend_des_candidats_distincts_et_ordonnes():
    m = _modele()
    pts, pm = _entree()
    for cands in beam_search(m, pts, pm, beam=6):
        seqs = [tuple(s) for s, _ in cands]
        assert len(set(seqs)) == len(seqs), "deux fois la même formule n'est pas deux propositions"
        scores = [sc for _, sc in cands]
        assert scores == sorted(scores, reverse=True)


def test_le_beam_ne_depasse_jamais_la_taille_demandee():
    m = _modele()
    pts, pm = _entree()
    for k in (1, 2, 5):
        assert all(len(c) <= k for c in beam_search(m, pts, pm, beam=k))


def test_le_front_de_pareto_ecarte_les_candidats_domines():
    simple_mauvais = Candidat(_p("x"), [], 0.50, 2)
    simple_bon = Candidat(_p("sin x"), [], 0.90, 5)
    complexe_moins_bon = Candidat(_p("add mul C x C"), [1.0, 0.0], 0.80, 12)
    complexe_meilleur = Candidat(_p("add mul C sin x C"), [1.0, 0.0], 0.99, 15)

    front = front_de_pareto([simple_mauvais, simple_bon, complexe_moins_bon, complexe_meilleur])
    formules = [c.node for c in front]
    assert complexe_moins_bon.node not in formules, "dominé en simplicité ET en précision"
    assert simple_bon.node in formules and complexe_meilleur.node in formules
    assert [c.complexite for c in front] == sorted(c.complexite for c in front)


def test_le_front_ne_propose_pas_deux_fois_le_meme_squelette():
    a = Candidat(_p("sin x"), [], 0.90, 5)
    b = Candidat(_p("sin x"), [], 0.95, 5)
    front = front_de_pareto([a, b])
    assert len(front) == 1 and front[0].r2_fit == 0.95


def test_la_selection_prefere_le_plus_simple_a_precision_equivalente():
    """Le piège du degré 7 : à R² équivalent, ne pas prendre le plus tarabiscoté."""
    simple = Candidat(_p("add mul C x C"), [1.0, 0.0], 0.9950, 12)
    tarabiscote = Candidat(_p("add mul C sin mul C x C"), [1.0, 1.0, 0.0], 0.9952, 25)
    assert selectionner([simple, tarabiscote], tol=0.005).node == simple.node


def test_la_selection_prend_le_meilleur_quand_lecart_est_reel():
    mediocre = Candidat(_p("add mul C x C"), [1.0, 0.0], 0.70, 12)
    bon = Candidat(_p("add mul C sin x C"), [1.0, 0.0], 0.99, 18)
    assert selectionner([mediocre, bon], tol=0.005).node == bon.node


def test_la_selection_rend_none_sans_candidat():
    assert selectionner([]) is None


def test_lajustement_ecarte_les_candidats_qui_echouent():
    """`log x` est indéfini sur la moitié du domaine : il ne doit pas survivre."""
    rng = make_rng(3)
    x = np.linspace(-1, 1, 60)
    y = 2.0 * np.sin(x) + 0.5
    cands = ajuster_candidats([_p("add mul C sin x C"), _p("log x")], x, y, rng)
    formules = [c.node for c in cands]
    assert _p("add mul C sin x C") in formules
    assert _p("log x") not in formules


def test_lajustement_retrouve_la_vraie_formule_avec_un_r2_parfait():
    rng = make_rng(5)
    x = np.linspace(-1, 1, 80)
    y = 2.0 * np.sin(x) + 0.5
    cands = ajuster_candidats([_p("add mul C x C"), _p("add mul C sin x C")], x, y, rng)
    vrai = [c for c in cands if c.node == _p("add mul C sin x C")][0]
    assert vrai.r2_fit == pytest.approx(1.0, abs=1e-6)
    assert vrai.consts[0] == pytest.approx(2.0, abs=1e-6)


def test_la_tolerance_de_parcimonie_arbitre_entre_droite_et_sinus():
    """Pourquoi le défaut est ``tol = 0``, figé en test parce que c'est piégeux.

     Sur ``[-1, 1]``, ``sin(x) ≈ x`` : la droite atteint R² = 0,9978 contre 1,0
     pour le vrai sinus. Une tolérance de parcimonie rend donc la **droite** —
     défendable au vu des seules données observées, et destructeur en
     extrapolation où `x` et `sin(x)` divergent.

     J'avais posé ça comme un compromis à arbitrer : la tolérance devait acheter
     de la robustesse au bruit. **La mesure du 2026-08-20 dit qu'elle n'achète
     rien** — ``tol = 0`` gagne sur l'interpolation *et* sur l'extrapolation
    . Ce test garde donc le fait, qui reste vrai, sans la conclusion
     qu'on en tirait.
    """
    rng = make_rng(5)
    x = np.linspace(-1, 1, 80)
    y = 2.0 * np.sin(x) + 0.5
    cands = ajuster_candidats([_p("add mul C x C"), _p("add mul C sin x C")], x, y, rng)

    assert selectionner(cands, tol=0.005).node == _p("add mul C x C")
    assert selectionner(cands, tol=0.0).node == _p("add mul C sin x C")
