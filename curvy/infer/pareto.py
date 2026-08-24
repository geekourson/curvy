"""Des candidats du beam search aux 3-5 formules livrées (Phase 5).

Trois étapes, dans cet ordre :

1. **ajuster** les constantes de chaque candidat sur les points observés ;
2. **sélectionner** celui qu'on annonce comme réponse principale ;
3. **retenir le front de Pareto** — les candidats qu'aucun autre ne domine à la
   fois en simplicité et en précision. C'est ce que voit l'utilisateur.

Une règle gouverne tout le module : **la sélection ne regarde jamais les points
tenus à l'écart.** Un candidat est jugé sur son ajustement aux points observés,
comme à l'usage réel où il n'y a pas de vérité terrain. C'est exactement le
biais reproché à la baseline polynomiale le 2026-08-19 — choisir son degré
d'après le résultat final — et il serait malhonnête de se l'autoriser ici.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curvy.data.expr import Node, complexity
from curvy.infer.fit import estimer_bruit, fit_constants

__all__ = [
    "Candidat",
    "ajuster_candidats",
    "front_de_pareto",
    "selectionner",
    "selectionner_selon_bruit",
]


@dataclass
class Candidat:
    node: Node
    consts: list[float]
    r2_fit: float
    complexite: int
    score_modele: float = 0.0

    @property
    def n_consts(self) -> int:
        return len(self.consts)


def _ajuster_un(args):
    """Un ajustement isolé, picklable — cible du pool de processus."""
    node, x, y, graine = args
    from curvy.seeding import make_rng

    res = fit_constants(node, x, y, make_rng(graine))
    return (list(res.consts), float(res.r2_fit), bool(res.ok))


def ajuster_candidats(
    nodes: list[Node],
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    scores: list[float] | None = None,
    executor=None,
) -> list[Candidat]:
    """Ajuste chaque candidat sur ``(x, y)`` observés. Les échecs sont écartés.

    ``executor`` — un pool de processus optionnel. Les ajustements sont
    **indépendants** et dominent la latence : les paralléliser est le
    seul moyen d'élargir le beam sans quitter le budget d'une seconde. Mesuré à
    beam 24 : 1015 ms en série. Les threads n'aideraient pas — la fonction de
    résidu est du Python, donc tenue par le GIL.

    Sans pool, le comportement est strictement celui d'avant : c'est le chemin
    des tests et des scripts de mesure.
    """
    valides = [(i, n) for i, n in enumerate(nodes) if n is not None]
    if not valides:
        return []

    if executor is None:
        resultats = []
        for _, node in valides:
            r = fit_constants(node, x, y, rng)
            resultats.append((list(r.consts), float(r.r2_fit), bool(r.ok)))
    else:
        # Une graine par candidat, dérivée du rng appelant : le parallélisme ne
        # doit pas rendre le résultat dépendant de l'ordre d'arrivée.
        graines = [int(g) for g in rng.integers(0, 2**31 - 1, size=len(valides))]
        taches = [(n, x, y, g) for (_, n), g in zip(valides, graines, strict=True)]
        resultats = list(executor.map(_ajuster_un, taches))

    out: list[Candidat] = []
    for (i, node), (consts, r2, ok) in zip(valides, resultats, strict=True):
        if not ok or not np.isfinite(r2):
            continue
        out.append(
            Candidat(
                node=node,
                consts=consts,
                r2_fit=r2,
                complexite=complexity(node),
                score_modele=float(scores[i]) if scores is not None else 0.0,
            )
        )
    return out


def selectionner(cands: list[Candidat], tol: float = 0.0) -> Candidat | None:
    """La réponse principale : le meilleur ajustement, à parcimonie égale.

    Parmi les candidats dont le R² d'ajustement est à ``tol`` du meilleur, on
    retient **le plus simple**.

    ``tol = 0`` par défaut, et c'est une **mesure, pas une intuition**. J'avais
    prévu qu'une tolérance protégerait du sur-ajustement au bruit — choisir le
    maximum parmi 16 candidats, c'est en principe choisir celui qui épouse le
    mieux le bruit. Mesuré le 2026-08-20 sur 512 exemples, c'est faux : la
    tolérance ne fait que coûter, et sur les deux métriques à la fois.

    ===========  ==============  ==============
    tol          interpolation   extrapolation
    ===========  ==============  ==============
    0            **0,719**       **0,262**
    0,002        0,709           0,238
    0,005        0,691           0,229
    0,02         0,606           0,203
    ===========  ==============  ==============

    L'explication tient sans doute au masque d'arité : profondeur bornée,
    5 constantes au plus, vocabulaire de 18 tokens. Le beam n'a pas de quoi
    fabriquer un candidat assez tordu pour épouser le bruit — la grammaire fait
    déjà le travail de régularisation qu'on croyait devoir refaire ici.
    Le départage par complexité reste actif en cas d'égalité exacte.
    """
    if not cands:
        return None
    meilleur = max(c.r2_fit for c in cands)
    proches = [c for c in cands if c.r2_fit >= meilleur - tol]
    return min(proches, key=lambda c: (c.complexite, -c.r2_fit))


def selectionner_selon_bruit(
    cands: list[Candidat],
    x: np.ndarray,
    y: np.ndarray,
    marge: float = 1.0,
) -> Candidat | None:
    """Le plus simple des candidats dont le résidu est compatible avec le bruit.

    **L'hypothèse.** Prendre le maximum de R² d'ajustement revient, par
    construction, à retenir le candidat qui **épouse le mieux le bruit** : au-delà
    du niveau de bruit, tout R² supplémentaire est du sur-ajustement. La bonne
    règle serait donc « parmi ceux qui expliquent les données *jusqu'au bruit*,
    le plus simple ».

    Une tolérance **fixe** avait été essayée le 2026-08-20 et dégradait tout
    (0,691 contre 0,711 à 0,005). C'était attendu après coup : chaque tracé a son
    propre niveau de bruit, une tolérance uniforme est forcément trop large sur
    un nuage propre et trop étroite sur un nuage bruité. Ici le seuil est **tiré
    des données elles-mêmes**, par les pseudo-résidus de `estimer_bruit`.

    ``marge`` fixe la tolérance autour du bruit estimé : un candidat est accepté
    si son résidu quadratique moyen ne dépasse pas ``σ·(1 + marge)``.

    Retombe sur le maximum de R² si le bruit n'est pas estimable ou si aucun
    candidat n'atteint le niveau du bruit — auquel cas aucun ne « suffit », et
    prendre le meilleur reste le moins mauvais choix.
    """
    if not cands:
        return None
    sigma = estimer_bruit(x, y)
    variance = float(np.var(y))
    if sigma <= 0.0 or variance <= 1e-15:
        return selectionner(cands)

    # r2 = 1 − résidu²/variance : le seuil de bruit se convertit en seuil de R².
    seuil_r2 = 1.0 - (sigma * (1.0 + marge)) ** 2 / variance
    suffisants = [c for c in cands if c.r2_fit >= seuil_r2]
    if not suffisants:
        return selectionner(cands)
    return min(suffisants, key=lambda c: (c.complexite, -c.r2_fit))


def front_de_pareto(cands: list[Candidat]) -> list[Candidat]:
    """Candidats non dominés, du plus simple au plus précis.

    ``a`` domine ``b`` si ``a`` est au moins aussi simple **et** au moins aussi
    précis, avec un avantage strict quelque part. Les doublons de squelette sont
    écartés en amont : deux fois la même formule ne fait pas deux propositions.
    """
    vus: dict[tuple, Candidat] = {}
    for c in cands:
        cle = tuple(_aplatir(c.node))
        if cle not in vus or c.r2_fit > vus[cle].r2_fit:
            vus[cle] = c
    uniques = sorted(vus.values(), key=lambda c: (c.complexite, -c.r2_fit))

    front: list[Candidat] = []
    meilleur_r2 = -np.inf
    for c in uniques:
        if c.r2_fit > meilleur_r2:
            front.append(c)
            meilleur_r2 = c.r2_fit
    return front


def _aplatir(node: Node) -> list[str]:
    out = [node[0]]
    for enfant in node[1:]:
        out.extend(_aplatir(enfant))
    return out
