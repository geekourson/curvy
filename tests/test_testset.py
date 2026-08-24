"""Phase 6 — le jeu de test tient-il ses promesses ?

Ces tests vérifient des **affirmations écrites dans le catalogue**, plutôt que
le comportement du code : une formule annotée « la grammaire sait l'écrire »
doit vraiment tenir dans les budgets, et une courbe doit vraiment pouvoir
engendrer un nuage. Une annotation fausse fabriquerait un résultat faux sans
qu'aucun test classique ne bronche.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvy.data.canonical import canonicalise, strip_absorbable_root
from curvy.data.expr import count_constants, depth, from_prefix
from curvy.data.grammar import MAX_BODY_CONSTANTS, MAX_BODY_DEPTH
from curvy.data.pointcloud import CloudConfig, sample_cloud, sample_cloud_fn
from curvy.data.split import RESERVE_PAR_PROFONDEUR, partitionner, valeur_de_hachage
from curvy.data.testset import FORMULES_A_LA_MAIN
from curvy.seeding import make_rng

X = np.linspace(-1.0, 1.0, 400)


def _items(n_par_prof: dict[int, int]) -> list[dict]:
    out = []
    for profondeur, n in n_par_prof.items():
        for i in range(n):
            out.append({"prefix": f"add mul C p{profondeur}_{i} C", "depth": profondeur})
    return out


# --- la partition ------------------------------------------------------------


def test_la_partition_ne_laisse_aucun_chevauchement():
    items = _items({5: 300, 6: 400})
    part = partitionner(items, reserve={5: 40, 6: 100})
    entrainement = {it["prefix"] for it in part.entrainement}
    assert not (entrainement & part.prefixes_de_test)
    assert len(part.test) == 140
    assert len(part.entrainement) == 560


def test_la_partition_ne_depend_pas_de_lordre_du_fichier():
    """Une partition par indice de ligne se casserait à la première
    régénération du jeu de données, et en silence."""
    items = _items({5: 200, 6: 200})
    melange = list(reversed(items))
    a = partitionner(items, reserve={5: 20, 6: 20})
    b = partitionner(melange, reserve={5: 20, 6: 20})
    assert a.prefixes_de_test == b.prefixes_de_test


def test_changer_le_sel_change_le_jeu_de_test():
    items = _items({6: 500})
    a = partitionner(items, reserve={6: 50}, sel="curvy-test-v1")
    b = partitionner(items, reserve={6: 50}, sel="autre-sel")
    assert a.prefixes_de_test != b.prefixes_de_test


def test_le_hachage_est_stable_et_dans_lintervalle_unite():
    v = valeur_de_hachage("add mul C x C")
    assert 0.0 <= v < 1.0
    assert v == valeur_de_hachage("add mul C x C")


def test_la_partition_refuse_une_reserve_plus_grande_que_la_strate():
    """Mieux vaut une erreur franche qu'un jeu de test silencieusement tronqué."""
    with pytest.raises(ValueError, match="disponibles"):
        partitionner(_items({5: 10}), reserve={5: 40})


def test_aucune_reserve_avant_la_profondeur_5():
    """Il n'existe qu'un squelette de profondeur 3 : le réserver estropierait
    le modèle sur la forme la plus courante au lieu de le tester."""
    assert min(RESERVE_PAR_PROFONDEUR) == 5


# --- le catalogue hors distribution ------------------------------------------


@pytest.mark.parametrize("formule", FORMULES_A_LA_MAIN, ids=lambda f: f.nom)
def test_chaque_formule_engendre_une_courbe_utilisable(formule):
    with np.errstate(all="ignore"):
        y = np.asarray(formule.f(X), dtype=float)
    assert np.isfinite(y).all(), "une courbe non finie ne peut pas faire de nuage"
    assert float(y.max() - y.min()) > 1e-9, "une constante n'est pas une courbe"


@pytest.mark.parametrize(
    "formule", [f for f in FORMULES_A_LA_MAIN if f.prefixe], ids=lambda f: f.nom
)
def test_les_formules_annotees_dans_la_grammaire_tiennent_dans_les_budgets(formule):
    corps = strip_absorbable_root(canonicalise(from_prefix(formule.prefixe.split())))
    assert depth(corps) <= MAX_BODY_DEPTH
    assert count_constants(corps) <= MAX_BODY_CONSTANTS


def test_le_catalogue_couvre_les_deux_cotes_de_la_frontiere():
    dedans = [f for f in FORMULES_A_LA_MAIN if f.prefixe]
    dehors = [f for f in FORMULES_A_LA_MAIN if not f.prefixe]
    assert len(dedans) >= 5 and len(dehors) >= 5


def test_les_noms_du_catalogue_sont_uniques():
    noms = [f.nom for f in FORMULES_A_LA_MAIN]
    assert len(set(noms)) == len(noms)


# --- le pipeline de nuage ----------------------------------------------------


def test_le_nuage_dune_fonction_quelconque_suit_le_meme_chemin_que_celui_dun_squelette():
    """`sample_cloud` doit être exactement `sample_cloud_fn` sur la fonction du
    squelette : sinon on comparerait deux protocoles, pas deux familles de
    formules.

    Le seul écart légitime entre les deux entrées est **un tirage** :
    `sample_cloud` tire d'abord les constantes du squelette, puis délègue. Le
    test le reproduit explicitement plutôt que de le contourner — c'est cette
    relation-là qu'on veut figer, et elle casserait au premier tirage ajouté
    dans l'une des deux fonctions.
    """
    from curvy.data.expr import count_constants, evaluate
    from curvy.data.pointcloud import _sample_constants

    node = from_prefix(["add", "mul", "C", "sin", "mul", "C", "x", "C"])
    compares = 0
    for graine in range(30):
        a, raison_a = sample_cloud(make_rng(graine), node)

        rng = make_rng(graine)
        consts = _sample_constants(rng, count_constants(node), CloudConfig())
        b, raison_b = sample_cloud_fn(
            rng,
            lambda xs, c=consts: evaluate(node, xs, c),  # lié tôt : B023
            CloudConfig(),
            consts=consts,
        )

        assert raison_a == raison_b, "les deux entrées doivent rejeter pour la même raison"
        if a is None:
            continue
        assert consts == a.consts, "le tirage des constantes doit être le même"
        assert np.array_equal(a.x, b.x)
        assert np.array_equal(a.y, b.y)
        assert (a.y_scale, a.y_offset) == (b.y_scale, b.y_offset)
        compares += 1
    assert compares >= 5, f"trop peu de nuages comparés ({compares}) pour conclure"


@pytest.mark.parametrize("formule", FORMULES_A_LA_MAIN, ids=lambda f: f.nom)
def test_chaque_formule_produit_au_moins_un_nuage(formule):
    """Le filtre d'identifiabilité rejette beaucoup : on vérifie qu'aucune
    formule du catalogue n'est systématiquement rejetée, sans quoi elle serait
    silencieusement absente du jeu de test."""
    rng = make_rng(11)
    for _ in range(30):
        cloud, _ = sample_cloud_fn(rng, formule.f)
        if cloud is not None:
            assert cloud.n_points >= 20
            return
    pytest.fail(f"{formule.nom} : aucun nuage produit en 30 tentatives")
