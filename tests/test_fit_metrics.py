"""Tests de l'ajustement des constantes et des métriques (Phases 4 et 5).

L'ajustement est la brique la plus susceptible d'être fausse *silencieusement* :
un R² médiocre ressemble à un modèle médiocre. D'où deux tests qui vérifient
qu'il retrouve exactement des constantes connues sur des données sans bruit.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvy.data.expr import evaluate, from_prefix
from curvy.infer.decode import ids_to_node
from curvy.infer.fit import fit_constants, r_squared
from curvy.seeding import make_rng
from curvy.tokenizer.vocab import encode


def _p(prefix: str):
    return from_prefix(prefix.split())


def test_r_squared_cas_limites():
    y = np.array([1.0, 2.0, 3.0])
    assert r_squared(y, y) == pytest.approx(1.0)
    assert r_squared(y, np.full(3, y.mean())) == pytest.approx(0.0)
    assert r_squared(y, np.array([np.nan, 0.0, 0.0])) == float("-inf")


def test_ajustement_exact_sans_constante_interne():
    """`C*sin(x) + C` : aucune constante interne, la solution est exacte."""
    rng = make_rng(0)
    node = _p("add mul C sin x C")
    x = np.linspace(-1, 1, 80)
    y = evaluate(node, x, [2.5, -0.75])
    res = fit_constants(node, x, y, rng)
    assert res.ok
    assert res.r2_fit == pytest.approx(1.0, abs=1e-9)
    assert res.consts[0] == pytest.approx(2.5, rel=1e-8)
    assert res.consts[1] == pytest.approx(-0.75, abs=1e-8)
    assert res.n_restarts_used == 1, "aucune itération ne devrait être nécessaire"


def test_ajustement_retrouve_une_constante_interne():
    """`C*sin(C*x) + C` : la projection variable ne laisse qu'un paramètre."""
    rng = make_rng(1)
    node = _p("add mul C sin mul C x C")
    x = np.linspace(-1, 1, 200)
    y = evaluate(node, x, [1.8, 4.2, 0.3])
    res = fit_constants(node, x, y, rng, n_restarts=10)
    assert res.ok and res.r2_fit > 0.999, res.r2_fit


def test_ajustement_tolere_un_arbre_hors_forme_canonique():
    """Régression : le décodeur produit des arbres sans enveloppe de racine
    tant que le modèle n'a rien appris. Supposer l'enveloppe faisait planter
    l'évaluation en plein entraînement."""
    rng = make_rng(2)
    node = _p("mul x C")  # ni `add` en racine, ni offset
    x = np.linspace(-1, 1, 50)
    y = 3.0 * x
    res = fit_constants(node, x, y, rng)
    assert res.ok and res.r2_fit > 0.999


def test_ajustement_sans_aucune_constante():
    rng = make_rng(3)
    node = _p("sin x")
    x = np.linspace(-1, 1, 50)
    res = fit_constants(node, x, np.sin(x), rng)
    assert res.ok and res.r2_fit == pytest.approx(1.0, abs=1e-12)


def test_ajustement_signale_lechec_sur_domaine_impossible():
    rng = make_rng(4)
    node = _p("log x")  # log(x) est nan sur la moitié du domaine
    x = np.linspace(-1, 1, 40)
    res = fit_constants(node, x, np.zeros(40), rng)
    assert not res.ok or res.r2_fit == float("-inf")


def test_decode_ids_vers_arbre_rejette_une_sequence_incomplete():
    from curvy.data.grammar import TOKEN_TO_ID

    assert ids_to_node([TOKEN_TO_ID["add"], TOKEN_TO_ID["x"]]) is None
    assert ids_to_node([TOKEN_TO_ID["x"]]) == ("x",)


def test_encode_puis_ids_to_node_est_un_aller_retour():
    node = _p("add mul C sin mul C x C")
    ids = encode(node, add_special=False)
    assert ids_to_node(ids) == node


def test_extrapolation_tient_a_lecart_le_bord_droit_pas_des_points_au_hasard():
    """Le mode extrapolation doit ajuster à gauche et juger à droite.

    Test par la conséquence observable : une droite ajustée sur la moitié
    gauche d'une parabole prédit très mal la moitié droite, alors qu'elle
    passe correctement entre des points tirés au hasard sur tout le domaine.
    """
    from curvy.train.metrics import _r2_holdout

    rng = make_rng(11)
    droite = _p("add mul C x C")  # C*x + C, incapable de courber
    x = np.linspace(-1, 1, 100)
    parabole = x**2

    r2_interp, _ = _r2_holdout(droite, x, parabole, parabole, rng, mode="interpolation")
    r2_extra, _ = _r2_holdout(droite, x, parabole, parabole, rng, mode="extrapolation")

    assert r2_extra < r2_interp, "extrapoler doit être plus dur qu'interpoler"
    assert r2_extra < 0.0, "une droite ajustée à gauche doit rater la droite d'une parabole"


def test_extrapolation_est_deterministe_contrairement_a_linterpolation():
    """Le bord droit ne dépend pas du tirage : deux appels donnent le même R²."""
    from curvy.train.metrics import _r2_holdout

    node = _p("add mul C sin x C")
    x = np.linspace(-1, 1, 60)
    y = 2.0 * np.sin(x) + 0.5

    a, _ = _r2_holdout(node, x, y, y, make_rng(1), mode="extrapolation")
    b, _ = _r2_holdout(node, x, y, y, make_rng(999), mode="extrapolation")
    assert a == pytest.approx(b, abs=1e-9)


def test_extrapolation_ignore_lordre_dans_lequel_les_points_arrivent():
    """Les points d'un canvas n'arrivent pas triés : le mode doit trier lui-même."""
    from curvy.train.metrics import _r2_holdout

    node = _p("add mul C x C")
    x = np.linspace(-1, 1, 80)
    y = 3.0 * x - 1.0
    perm = make_rng(7).permutation(len(x))

    trie, _ = _r2_holdout(node, x, y, y, make_rng(1), mode="extrapolation")
    melange, _ = _r2_holdout(node, x[perm], y[perm], y[perm], make_rng(1), mode="extrapolation")
    assert trie == pytest.approx(melange, abs=1e-9)
