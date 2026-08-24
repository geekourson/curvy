"""Le chemin tracé → formules (Phase 8).

Ces tests portent sur la partie qui décide de la justesse du résultat : la
normalisation doit être **exactement** celle de l'entraînement, sans quoi le
modèle reçoit une entrée d'une autre distribution et rien de ce qu'on a mesuré
ne s'applique.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvy.serve.pipeline import N_MAX, N_MIN_TRACE, est_univalue, preparer


def _trace(f, n=200, x0=50.0, x1=650.0, h=300.0):
    """Un tracé en pixels : x croissant, y vers le bas comme un canvas."""
    x = np.linspace(x0, x1, n)
    u = 2 * (x - x0) / (x1 - x0) - 1
    y = h - 100.0 * f(u)
    return np.stack([x, y], axis=1)


def test_la_normalisation_ramene_x_et_y_dans_lintervalle_unite():
    x, y, *_ = preparer(_trace(np.sin))
    assert x.min() == pytest.approx(-1.0) and x.max() == pytest.approx(1.0)
    assert y.min() == pytest.approx(-1.0) and y.max() == pytest.approx(1.0)


def test_laxe_y_du_canvas_est_retourne():
    """Un canvas compte vers le bas. Sans retournement, toutes les formules
    sortiraient à l'envers — et rien ne le signalerait."""
    montant = _trace(lambda u: u)  # y pixel DÉCROÎT quand la courbe monte
    x, y, *_ = preparer(montant)
    assert y[-1] > y[0], "une courbe qui monte à l'écran doit avoir un y croissant"


def test_laffine_rendue_permet_de_revenir_aux_pixels():
    """Le navigateur s'en sert pour superposer la courbe : elle doit être exacte."""
    trace = _trace(np.sin)
    x, y, echelle, decalage, _, _ = preparer(trace)
    reconstruit = -(y * echelle + decalage)
    attendu = np.interp(np.linspace(trace[0, 0], trace[-1, 0], len(x)), trace[:, 0], trace[:, 1])
    assert np.allclose(reconstruit, attendu, atol=1e-6)


def test_un_trace_trop_court_est_refuse_plutot_que_devine():
    assert preparer(np.zeros((N_MIN_TRACE - 1, 2))) is None


def test_un_trace_vertical_est_refuse():
    """Largeur nulle : aucune fonction y = f(x) ne le décrit."""
    y = np.linspace(0, 300, 50)
    assert preparer(np.stack([np.full(50, 42.0), y], axis=1)) is None


def test_les_points_de_meme_abscisse_sont_moyennes_pas_jetes():
    """Deux points à la même abscisse ne peuvent pas coexister dans y = f(x).

    L'exemple garde une courbure : une version antérieure de ce test moyennait
    vers une constante, ce que le garde-fou anti-dégénérescence refuse
    désormais — à raison.
    """
    base = np.linspace(0, 100, 30)
    x = np.repeat(base, 2)
    creux = 200.0 - 0.05 * (base - 50.0) ** 2
    y = np.empty(60)
    y[0::2] = creux - 40.0  # deux points encadrant la courbe...
    y[1::2] = creux + 40.0  # ...dont la moyenne la redonne exactement
    xn, yn, echelle, decalage, _, _ = preparer(np.stack([x, y], axis=1))

    assert len(xn) == 30, "une abscisse dupliquée doit donner un seul point"
    reconstruit = -(yn * echelle + decalage)
    assert np.allclose(reconstruit, creux, atol=1e-9)


def test_un_trace_trop_dense_est_sous_echantillonne_a_la_plage_dentrainement():
    x, *_ = preparer(_trace(np.sin, n=2000))
    assert len(x) == N_MAX


def test_une_courbe_qui_avance_est_univaluee():
    assert est_univalue(np.linspace(0, 100, 50))


def test_une_boucle_ne_lest_pas():
    t = np.linspace(0, 2 * np.pi, 200)
    assert not est_univalue(50 * np.cos(t))


def test_un_tremblement_de_la_main_reste_univalue():
    """Une vraie main recule de quelques pixels : ce n'est pas une boucle."""
    x = np.linspace(0, 600, 300) + np.array([0, -1.5, 1.0] * 100)
    assert est_univalue(x)


def test_un_trace_immobile_nest_pas_univalue():
    assert not est_univalue(np.full(50, 3.0))


def test_un_cercle_est_refuse_au_lieu_de_rendre_un_r2_parfait():
    """Le piège du 2026-08-20, figé en test.

    Un cercle trié par x avec ses doublons moyennés devient **exactement
    constant** : les moitiés haute et basse s'annulent. Comme
    ``r_squared(constante, constante)`` vaut 1,0, la démo affichait un R² parfait
    avec une formule vide de sens — et un modèle aux poids aléatoires obtenait
    le même 1,0000. Le chiffre annulait l'avertissement affiché juste au-dessus.
    """
    t = np.linspace(0.2, 2 * np.pi - 0.2, 200)
    cercle = np.stack([350 + 200 * np.cos(t), 250 + 150 * np.sin(t)], axis=1)
    assert not est_univalue(cercle[:, 0])
    assert preparer(cercle) is None, "un cercle aplati ne doit pas passer"


def test_une_droite_horizontale_est_refusee():
    """`y = constante` n'est pas dans la grammaire, et le générateur la rejette
    aussi (RejectReason.CONSTANT). Refuser est cohérent avec l'entraînement."""
    x = np.linspace(0, 600, 100)
    assert preparer(np.stack([x, np.full(100, 250.0)], axis=1)) is None


def test_une_courbe_a_peine_creusee_passe_quand_meme():
    """Le seuil ne doit pas écarter une courbe réelle mais peu marquée."""
    x = np.linspace(0, 600, 100)
    y = 250 - 0.5 * np.sin(np.linspace(-np.pi, np.pi, 100))
    assert preparer(np.stack([x, y], axis=1)) is not None


# --- constantes valorisées et import de données (2026-08-20) ---


def test_lordonnee_nest_pas_retournee_pour_des_donnees_importees():
    """Un canvas compte vers le bas, un fichier de mesures non. Retourner le
    signe de données importées inverserait toutes les formules."""
    trace = _trace(lambda u: u)
    canvas = preparer(trace, retourner_y=True)
    donnees = preparer(trace, retourner_y=False)
    assert canvas is not None and donnees is not None
    assert np.allclose(canvas[1], -donnees[1])


def test_les_affines_rendues_permettent_de_revenir_aux_unites_dorigine():
    """C'est ce qui rend la formule utilisable sur les données de l'utilisateur."""
    x = np.linspace(120.0, 480.0, 90)
    y = 3.5 * np.sin(0.02 * x) + 17.0
    _, y_norm, echelle, decalage, centre, demi = preparer(
        np.stack([x, y], axis=1), retourner_y=False
    )
    assert np.allclose(y_norm * echelle + decalage, y, atol=1e-9)
    assert centre == pytest.approx(300.0)
    assert demi == pytest.approx(180.0)


def test_la_formule_valorisee_porte_les_constantes():
    from curvy.data.expr import from_prefix
    from curvy.infer.rendu import formule_lisible

    node = from_prefix(["add", "mul", "C", "sin", "mul", "C", "x", "C"])
    rendu = formule_lisible(node, [2.31, 4.07, -0.5])
    assert "C" not in rendu
    assert "2.31" in rendu and "4.07" in rendu


def test_la_formule_valorisee_sexprime_dans_les_unites_dorigine():
    """Sans composition avec les affines, la formule serait fausse sur les
    abscisses de l'utilisateur — et rien ne le signalerait."""
    from curvy.data.expr import from_prefix
    from curvy.infer.rendu import Affines, formule_lisible

    node = from_prefix(["add", "mul", "C", "x", "C"])
    brut = formule_lisible(node, [2.0, 0.0])
    unites = formule_lisible(node, [2.0, 0.0], Affines(50.0, 25.0, 1.0, 0.0))
    assert brut != unites, "les affines doivent changer l'expression"
    assert "0.08" in unites, "2/25 = 0,08 : la pente est ramenée aux unités réelles"


def test_une_constante_minuscule_nest_pas_arrondie_a_zero():
    """`round(1.2e-5, 4)` vaut 0,0 et changerait la formule."""
    from curvy.data.expr import from_prefix
    from curvy.infer.rendu import formule_lisible

    node = from_prefix(["add", "mul", "C", "x", "C"])
    rendu = formule_lisible(node, [1.234e-5, 0.0])
    assert "e-5" in rendu or "0.00001" in rendu


def test_un_terme_negligeable_nest_pas_affiche():
    """`4.9*x^2 - 1.776e-15` doit se lire `4.9*x^2` : ce terme est du bruit
    d'arithmétique flottante, pas une constante."""
    from curvy.data.expr import from_prefix
    from curvy.infer.rendu import formule_lisible

    node = from_prefix(["add", "mul", "C", "sq", "x", "C"])
    rendu = formule_lisible(node, [4.9, -1.776e-15])
    assert "e-15" not in rendu


def test_le_seuil_de_negligeabilite_est_relatif():
    """Une donnée à l'échelle du micromètre a des constantes légitimement
    minuscules : les écraser serait pire que de les afficher."""
    from curvy.data.expr import from_prefix
    from curvy.infer.rendu import formule_lisible

    node = from_prefix(["add", "mul", "C", "x", "C"])
    rendu = formule_lisible(node, [2.0e-6, 5.0e-7])
    assert rendu.strip() != "0"
    assert "e-6" in rendu or "e-7" in rendu or "0.000" in rendu
