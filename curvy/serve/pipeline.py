"""Un tracé de canvas devient des formules.

Séparé du HTTP pour être testable sans serveur. Le chemin complet :

    points du canvas (pixels)
      → validité (assez de points ? univalué ?)
      → normalisation IDENTIQUE à celle de l'entraînement
      → sous-échantillonnage vers la plage vue à l'entraînement
      → beam search sous masque d'arité
      → ajustement des constantes sur les points observés
      → front de Pareto

**Le point qui compte, et qui n'est pas cosmétique :** la normalisation doit
être exactement celle du générateur (`normalise_y`), sinon le modèle
reçoit une entrée d'une autre distribution que celle sur laquelle il a été
entraîné, et tout ce qu'on a mesuré ne s'applique plus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from curvy.data.expr import to_infix
from curvy.data.pointcloud import normalise_y

#: Plage de tailles de nuage vue à l'entraînement (CloudConfig).
N_MIN, N_MAX = 20, 200

#: En dessous, le tracé n'est pas exploitable : on refuse plutôt que de rendre
#: une formule fabriquée sur trois points.
N_MIN_TRACE = 12

#: Étendue minimale de y après normalisation. Même seuil que le filtre
#: `DEGENERATE_NOISE` du générateur : une courbe plate n'est pas une courbe.
#:
#: Ce n'est pas un raffinement. Un cercle trié par x, doublons moyennés, devient
#: **exactement constant** — les moitiés haute et basse s'annulent. Et
#: `r_squared(constante, constante)` vaut **1,0** : la démo affichait donc
#: « R² 1,0000 » avec une formule dénuée de sens, juste à côté de
#: l'avertissement « ce tracé revient en arrière ». Le chiffre détruisait
#: l'avertissement. Vérifié aussi avec un modèle aux poids aléatoires, qui
#: obtenait le même 1,0000 (2026-08-20).
ETENDUE_MIN = 1e-9


#: Nombre de points de la courbe rendue au navigateur pour l'affichage.
N_APERCU = 600

#: Demi-largeur du domaine d'aperçu, en coordonnées normalisées. Les données
#: occupent `[-1, 1]` ; on rend la courbe sur `[-DOMAINE_APERCU, DOMAINE_APERCU]`
#: pour que le navigateur puisse dézoomer **sans aller-retour serveur**.
#:
#: C'est là qu'est l'intérêt : hors de la fenêtre observée, un polynôme rend
#: zéro sur 29 formules hors distribution sur 29 (mesuré le 2026-08-20), tandis
#: qu'une formule tient. La démo n'affichait jusqu'ici que l'intérieur du cadre,
#: c'est-à-dire précisément la zone où la baseline nous égale.
DOMAINE_APERCU = 3.0


@dataclass
class Formule:
    expression: str
    complexite: int
    r2: float
    constantes: list[float]
    prefixe: str
    #: L'expression avec ses constantes **valorisées**, et dans les unités
    #: d'origine quand elles ont un sens (import de données). Le squelette
    #: décrit une famille ; c'est cette ligne-ci que l'utilisateur veut lire.
    valorisee: str = ""
    #: La réponse principale retenue : meilleur R² d'ajustement,
    #: départagé par la simplicité. C'est elle qu'on affiche d'abord — le front
    #: est trié par complexité croissante, donc son premier élément est le plus
    #: simple ET le moins précis. L'afficher par défaut montrait le pire
    #: candidat en premier.
    principale: bool = False
    #: La courbe ajustée, échantillonnée sur `[-DOMAINE_APERCU, DOMAINE_APERCU]`
    #: en coordonnées normalisées — les données occupent le sous-intervalle
    #: `[-1, 1]`, le reste est de l'extrapolation.
    #: Renvoyée par le serveur plutôt que recalculée côté navigateur : la
    #: grammaire n'existe qu'ici, la réimplémenter en JavaScript créerait deux
    #: vérités qui divergeraient au premier opérateur ajouté.
    apercu: list[float] = field(default_factory=list)


@dataclass
class Reponse:
    ok: bool
    raison: str = ""
    #: Faux si le tracé revient en arrière : `y = f(x)` ne peut pas le décrire.
    #: C'est la limite de la v1, et la mesure qui décidera du mode paramétrique
    #:. On la rapporte, on ne la masque pas.
    univalue: bool = True
    n_points: int = 0
    formules: list[Formule] = field(default_factory=list)
    latence_ms: float = 0.0
    #: Affine appliquée à y, pour que le navigateur puisse revenir en pixels.
    y_scale: float = 1.0
    y_offset: float = 0.0
    #: Bornes en x du tracé d'origine, en pixels.
    x_min: float = 0.0
    x_max: float = 0.0
    #: Demi-largeur du domaine couvert par `apercu`, en unités normalisées.
    domaine_apercu: float = DOMAINE_APERCU


def est_univalue(x: np.ndarray, tolerance: float = 0.02) -> bool:
    """Le tracé avance-t-il toujours dans le même sens en x ?

    ``tolerance`` est exprimée en fraction de la largeur totale : une main qui
    tremble revient de quelques pixels en arrière sans que le tracé cesse d'être
    une fonction. Au-delà, c'est un vrai retour — une boucle, un cercle, un
    caractère.
    """
    if len(x) < 2:
        return True
    largeur = float(x.max() - x.min())
    if largeur < 1e-9:
        return False
    reculs = np.diff(x)
    recul_max = float(-reculs.min()) if reculs.min() < 0 else 0.0
    avance = float(reculs.sum())
    return recul_max <= tolerance * largeur and avance != 0.0


def preparer(
    points: np.ndarray, retourner_y: bool = True
) -> tuple[np.ndarray, np.ndarray, float, float, float, float] | None:
    """Points bruts → (x, y) normalisés comme à l'entraînement, plus les affines.

    ``retourner_y`` inverse l'ordonnée : vrai pour un canvas, dont l'axe croît
    vers le bas, faux pour des mesures importées, où le signe est celui des
    données.
    """
    if len(points) < N_MIN_TRACE:
        return None
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    if retourner_y:
        y = -y  # le canvas compte vers le bas

    ordre = np.argsort(x, kind="stable")
    x, y = x[ordre], y[ordre]

    # Deux points à la même abscisse ne peuvent pas coexister dans y = f(x) :
    # on garde leur moyenne plutôt que d'en jeter un au hasard.
    x_uniques, index = np.unique(np.round(x, 6), return_inverse=True)
    if len(x_uniques) < N_MIN_TRACE:
        return None
    y_moyens = np.bincount(index, weights=y) / np.bincount(index)

    largeur = x_uniques.max() - x_uniques.min()
    if largeur < 1e-9:
        return None
    x_centre = float((x_uniques.max() + x_uniques.min()) / 2.0)
    x_demi = float(largeur / 2.0)
    x_norm = (x_uniques - x_centre) / x_demi
    y_norm, echelle, decalage = normalise_y(y_moyens)
    if float(y_norm.max() - y_norm.min()) < ETENDUE_MIN:
        return None

    if len(x_norm) > N_MAX:
        # Sous-échantillonnage régulier : au-delà de 200 points le modèle n'a
        # jamais rien vu de tel, et l'attention coûte O(N²).
        idx = np.linspace(0, len(x_norm) - 1, N_MAX).round().astype(int)
        x_norm, y_norm = x_norm[idx], y_norm[idx]
    return x_norm, y_norm, echelle, decalage, x_centre, x_demi


def formules_depuis_trace(
    points: np.ndarray,
    modele,
    device,
    rng: np.random.Generator,
    beam: int = 8,
    max_formules: int = 5,
    retourner_y: bool = True,
    executor=None,
) -> Reponse:
    """Le chemin complet, du tracé aux formules classées."""
    import time

    import torch

    from curvy.data.dataset import collate
    from curvy.data.expr import evaluate, from_prefix, to_prefix
    from curvy.infer.decode import beam_search, ids_to_node
    from curvy.infer.pareto import ajuster_candidats, front_de_pareto, selectionner
    from curvy.infer.rendu import Affines, formule_lisible
    from curvy.tokenizer.vocab import encode

    t0 = time.perf_counter()
    brut_x = points[:, 0].astype(float)
    univalue = est_univalue(brut_x)

    prepare = preparer(points, retourner_y=retourner_y)
    if prepare is None:
        if not univalue:
            raison = (
                "ce tracé revient en arrière, et une fois ramené à une fonction y = f(x) "
                "il devient plat : il n'y a rien à décrire. Un cercle, une boucle ou un "
                "caractère demandent le mode paramétrique, prévu en v2."
            )
        elif len(points) < N_MIN_TRACE:
            raison = f"tracé trop court : {len(points)} points, {N_MIN_TRACE} minimum."
        else:
            raison = (
                "tracé dégénéré : soit vertical, soit horizontal. Une droite horizontale "
                "est y = constante, ce que la grammaire ne modélise pas — et une verticale "
                "n'est pas une fonction."
            )
        return Reponse(ok=False, raison=raison, univalue=univalue, n_points=len(points))
    x, y, echelle, decalage, x_centre, x_demi = prepare

    pts = np.stack([x, y], axis=1).astype(np.float32)
    faux_ids = encode(from_prefix(["add", "mul", "C", "x", "C"]))
    batch = collate([(pts, faux_ids)]).to(device)
    with torch.no_grad():
        candidats = beam_search(modele, batch.points, batch.point_mask, beam=beam)[0]

    noeuds = [ids_to_node(seq) for seq, _ in candidats]
    ajustes = ajuster_candidats(noeuds, x, y, rng, executor=executor)
    front_complet = front_de_pareto(ajustes)
    retenue = selectionner(ajustes)
    front = front_complet[:max_formules]
    # La réponse principale ne doit jamais tomber hors de la troncature : c'est
    # celle qu'on annonce.
    if retenue is not None and all(c is not retenue for c in front):
        front = front[: max_formules - 1] + [retenue]

    # Les affines ne s'expriment que pour des données importées : les pixels
    # d'un canvas ne sont pas une unité qui intéresse quelqu'un.
    affines = (
        None
        if retourner_y
        else Affines(x_centre=x_centre, x_demi=x_demi, y_echelle=echelle, y_decalage=decalage)
    )

    grille = np.linspace(-DOMAINE_APERCU, DOMAINE_APERCU, N_APERCU)
    formules = []
    for c in front:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            courbe = evaluate(c.node, grille, c.consts)
        # `null` en JSON pour les points non finis : le navigateur y coupera le
        # trait au lieu de dessiner une droite vers l'infini.
        apercu = [None if not np.isfinite(v) else round(float(v), 5) for v in courbe]
        formules.append(
            Formule(
                expression=to_infix(c.node),
                valorisee=formule_lisible(c.node, c.consts, affines),
                complexite=c.complexite,
                r2=round(c.r2_fit, 6),
                constantes=[round(v, 6) for v in c.consts],
                prefixe=" ".join(to_prefix(c.node)),
                principale=c is retenue,
                apercu=apercu,
            )
        )

    return Reponse(
        ok=True,
        univalue=univalue,
        n_points=len(x),
        formules=formules,
        latence_ms=round(1000 * (time.perf_counter() - t0), 1),
        y_scale=float(echelle),
        y_offset=float(decalage),
        x_min=float(brut_x.min()),
        x_max=float(brut_x.max()),
    )
