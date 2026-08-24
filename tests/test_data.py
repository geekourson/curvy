"""Tests de la génération de données (Phase 1).

Deux invariants comptent plus que les autres :

- toute séquence produite est syntaxiquement valide **par construction**
  — si ce test casse, le masquage d'arité du beam search est faux ;
- la normalisation affine de y laisse le squelette inchangé — si ce
  test casse, le dataset entraîne le modèle sur des étiquettes fausses, et rien
  ne le signalera pendant l'entraînement.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvy.data.canonical import canonicalise, strip_absorbable_root, wrap_root
from curvy.data.expr import (
    count_constants,
    depth,
    evaluate,
    from_prefix,
    prefix_is_complete,
    prefix_validity,
    to_prefix,
)
from curvy.data.grammar import ARITY, MAX_CONSTANTS, VOCAB
from curvy.data.pointcloud import CloudConfig, normalise_y, sample_cloud
from curvy.data.sample import sample_skeleton
from curvy.data.weighting import DEFAULT_DEPTH_TARGET, describe_weights, stratified_weights
from curvy.seeding import make_rng


def _p(prefix: str):
    """`_p("mul C x")` -> l'arbre correspondant.

    Passer par une chaîne plutôt qu'une liste de littéraux garde les cas de
    test lisibles : une expression préfixe se relit d'un coup d'œil, une liste
    de quinze chaînes entre guillemets non.
    """
    return from_prefix(prefix.split())


def _skeletons(n: int = 300, seed: int = 3):
    rng = make_rng(seed)
    out = []
    while len(out) < n:
        sk = sample_skeleton(rng)
        if sk is not None:
            out.append(sk)
    return out


# --- grammaire et notation préfixe -------------------------------------------


def test_vocabulaire_taille_et_arites():
    assert len(VOCAB) == 18
    assert len(set(VOCAB)) == len(VOCAB)
    assert set(ARITY) == set(VOCAB) - {"<pad>", "<bos>", "<eos>"}


def test_roundtrip_prefixe():
    for sk in _skeletons(200):
        assert from_prefix(to_prefix(sk)) == sk


def test_toute_sequence_generee_est_valide():
    """la validité doit être garantie par construction, pas espérée."""
    for sk in _skeletons(300):
        assert prefix_is_complete(to_prefix(sk))


def test_prefixe_incomplet_ou_surnumeraire_est_detecte():
    assert prefix_validity(["add", "x"]) == 1  # il manque un sous-arbre
    assert prefix_validity(["x", "x"]) is None  # un token de trop
    assert prefix_validity(["zorglub"]) is None  # token inconnu
    with pytest.raises(ValueError):
        from_prefix(["add", "x"])


# --- canonicalisation ---------------------------------------------------------


def test_canonicalisation_idempotente():
    for sk in _skeletons(200):
        assert canonicalise(sk) == canonicalise(canonicalise(sk))


def test_regles_de_canonicalisation():
    def c(p: str) -> str:
        return " ".join(to_prefix(canonicalise(from_prefix(p.split()))))

    assert c("sin C") == "C"  # tout sous-arbre sans x est constant
    assert c("inv inv x") == "x"
    assert c("sqrt sq x") == "abs x"
    assert c("sub x x") == "C"
    assert c("mul C mul C x") == "mul C x"
    assert c("add mul C x mul C x") == "mul C x"  # C1*x + C2*x = C*x
    # Contre-épreuve : x + x vaut 2x, une fonction FIXE. La fusionner en C*x
    # ajouterait un paramètre libre et changerait la classe de fonctions.
    assert c("add x x") == "add x x"
    assert c("sin add x x") == "sin add x x"


def test_ordre_commutatif_stable():
    a = canonicalise(_p("mul add C x sin x"))
    b = canonicalise(_p("mul sin x add C x"))
    assert a == b


# --- squelettes ---------------------------------------------------------------


def test_enveloppe_de_racine_toujours_presente():
    """sans `C * (…) + C`, la normalisation de y n'est plus exacte."""
    for sk in _skeletons(200):
        assert sk[0] == "add"
        assert sk[2] == ("C",)
        assert sk[1][0] == "mul"
        assert ("C",) in sk[1][1:]


def test_budget_de_constantes_respecte():
    for sk in _skeletons(300):
        assert count_constants(sk) <= MAX_CONSTANTS


def test_strip_racine_retire_ce_que_l_enveloppe_absorbe():
    for src in ("mul C sin x", "add C sin x", "sub C sin x", "sub sin x C"):
        body = strip_absorbable_root(canonicalise(from_prefix(src.split())))
        assert " ".join(to_prefix(body)) == "sin x"
    assert wrap_root(("x",)) == ("add", ("mul", ("C",), ("x",)), ("C",))


# --- invariance affine (le test central de l') ------------------------


@pytest.mark.parametrize("alpha,beta", [(3.7, -2.1), (-0.4, 5.0), (1.0, 0.0)])
def test_normalisation_de_y_laisse_le_squelette_inchange(alpha, beta):
    """Pour toute affine sur y, il existe des constantes qui la reproduisent.

    C'est ce qui rend la normalisation « gratuite » : le modèle n'a aucune
    capacité à dépenser sur les changements d'échelle.
    """
    rng = make_rng(11)
    x = np.linspace(-1.0, 1.0, 64)
    for sk in _skeletons(60, seed=5):
        k = count_constants(sk)
        consts = list(rng.normal(size=k))
        y = evaluate(sk, x, consts)
        if not np.isfinite(y).all():
            continue
        # Les deux constantes de l'enveloppe sont la première (facteur) et la
        # dernière (offset) dans l'ordre préfixe.
        transformed = consts.copy()
        transformed[0] *= alpha
        transformed[-1] = alpha * consts[-1] + beta
        assert np.allclose(evaluate(sk, x, transformed), alpha * y + beta, rtol=1e-9, atol=1e-9)


def test_normalise_y_ramene_bien_dans_moins_un_un():
    y = np.array([3.0, 7.0, 5.0])
    n, scale, offset = normalise_y(y)
    assert n.min() == pytest.approx(-1.0)
    assert n.max() == pytest.approx(1.0)
    assert np.allclose(n * scale + offset, y)


def test_normalise_y_survit_a_une_courbe_plate():
    n, scale, offset = normalise_y(np.full(10, 4.0))
    assert np.all(n == 0.0) and scale == 1.0 and offset == 4.0


# --- nuages de points ---------------------------------------------------------


def test_nuage_respecte_les_bornes_et_les_tailles():
    rng = make_rng(17)
    cfg = CloudConfig()
    seen = 0
    for sk in _skeletons(120, seed=9):
        cloud, _ = sample_cloud(rng, sk, cfg)
        if cloud is None:
            continue
        seen += 1
        assert cloud.x.min() >= -1.0 - 1e-9 and cloud.x.max() <= 1.0 + 1e-9
        assert cloud.y.min() >= -1.0 - 1e-9 and cloud.y.max() <= 1.0 + 1e-9
        assert np.all(np.diff(cloud.x) >= -1e-12), "x doit être trié"
        assert np.isfinite(cloud.x).all() and np.isfinite(cloud.y).all()
        assert cloud.n_points <= cfg.n_points_max
    assert seen > 40, "trop de rejets, le générateur est cassé"


def test_desactiver_le_bruit_donne_un_nuage_exact():
    rng = make_rng(23)
    cfg = CloudConfig(
        use_white_noise=False,
        use_correlated_drift=False,
        use_x_jitter=False,
        use_quantisation=False,
    )
    for sk in _skeletons(40, seed=13):
        cloud, _ = sample_cloud(rng, sk, cfg)
        if cloud is None:
            continue
        clean = (evaluate(sk, cloud.x, cloud.consts) - cloud.y_offset) / cloud.y_scale
        assert np.allclose(clean, cloud.y, atol=1e-9)
        return
    pytest.fail("aucun nuage produit")


# --- pondération ---------------------------------------------------------------


def test_stratification_atteint_la_cible():
    depths = [3] * 1 + [6] * 5000 + [8] * 90000
    w = stratified_weights(depths, {3: 0.2, 6: 0.3, 8: 0.5})
    got = describe_weights(depths, w)
    assert got == {3: 20.0, 6: 30.0, 8: 50.0}


def test_stratification_redistribue_les_profondeurs_absentes():
    w = stratified_weights([7, 7, 8], DEFAULT_DEPTH_TARGET)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_profondeur_du_squelette_borne():
    for sk in _skeletons(200):
        assert depth(sk) <= 8  # 6 de corps + 2 d'enveloppe
