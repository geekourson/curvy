"""Partition entraînement / test des squelettes (Phase 6).

Un jeu de test n'a de valeur que si ses squelettes sont **exclus de
l'entraînement**. Jusqu'ici ce n'était le cas d'aucun run : le jeu de
validation était tiré du même fichier que le flux d'entraînement, et on a
mesuré qu'à 10,2 M tirages la probabilité qu'un de ses squelettes n'ait jamais
été vu vaut 5·10⁻⁸ en profondeur 8, et zéro en dessous. Tous les chiffres
publiés avant la Phase 6 mesurent donc de la **restitution**, pas de la
généralisation.

## Deux choix, et leurs raisons

**1. La réserve commence à la profondeur 5.** Il n'existe qu'*un* squelette de
profondeur 3 — `C*x+C`, la droite — et sept en profondeur 4. Les mettre au test
ne mesurerait pas une généralisation : ça retirerait de l'entraînement les
formes les plus courantes du produit. On tient donc à l'écart uniquement là où
la strate est peuplée (242 squelettes en profondeur 5, 182 231 en profondeur 8),
et **on l'écrit dans le rapport** plutôt que de laisser croire à une réserve
uniforme. Les formes peu profondes sont couvertes autrement, par le
sous-ensemble hors distribution écrit à la main.

**2. L'appartenance au test dépend du squelette, pas du fichier.** Elle est
tirée d'un hachage stable de la notation préfixe : le même squelette tombera
toujours du même côté, quel que soit l'ordre du fichier, sa taille, ou une
régénération future avec une autre graine. Une partition par indice de ligne
se casserait à la première régénération du jeu de données — silencieusement.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

__all__ = ["Partition", "RESERVE_PAR_PROFONDEUR", "SEL", "partitionner", "valeur_de_hachage"]

#: Sel du hachage. Le changer redistribue **toute** la partition : c'est
#: équivalent à changer de jeu de test, et ça invalide toute comparaison
#: antérieure. À ne faire que délibérément, en versionnant le nom.
SEL = "curvy-test-v1"

#: Nombre de squelettes tenus à l'écart, par profondeur. Rien avant la
#: profondeur 5 : voir l'en-tête du module.
#:
#: Les tailles sont dimensionnées sur la **précision par profondeur**, pas sur
#: le taux global. À 100 exemples, une strate se mesure à ±9,6 points près
#: (intervalle à 95 %) — de quoi lire une tendance, pas de quoi comparer deux
#: runs. À 400-950, on tombe à ±3 à ±5 points.
#:
#: La profondeur 5 est plafonnée par la grammaire elle-même : il n'existe que
#: **242 squelettes** de cette profondeur, en réserver 48 en retire déjà 20 %
#: de l'entraînement. Cette strate restera à ±14 points, quel que soit le
#: budget. C'est une limite de la grammaire, pas du jeu de test.
#:
#: Ajouter des nuages par squelette ne remplacerait pas des squelettes : six
#: nuages du même squelette ne valent pas six échantillons indépendants, c'est
#: la variance entre formules qui domine.
RESERVE_PAR_PROFONDEUR = {5: 48, 6: 400, 7: 600, 8: 950}


@dataclass
class Partition:
    entrainement: list[dict]
    test: list[dict]

    @property
    def prefixes_de_test(self) -> frozenset[str]:
        return frozenset(it["prefix"] for it in self.test)

    def rapport(self) -> dict:
        par_prof: dict[int, dict[str, int]] = defaultdict(lambda: {"entrainement": 0, "test": 0})
        for it in self.entrainement:
            par_prof[it["depth"]]["entrainement"] += 1
        for it in self.test:
            par_prof[it["depth"]]["test"] += 1
        return {
            "sel": SEL,
            "n_entrainement": len(self.entrainement),
            "n_test": len(self.test),
            "par_profondeur": {str(d): par_prof[d] for d in sorted(par_prof)},
            "profondeurs_sans_reserve": sorted(d for d in par_prof if par_prof[d]["test"] == 0),
        }


def valeur_de_hachage(prefix: str, sel: str = SEL) -> float:
    """Un réel de [0, 1) déterminé par le squelette seul, stable entre machines."""
    digest = hashlib.sha256(f"{sel}\x00{prefix}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def partitionner(
    items: list[dict],
    reserve: dict[int, int] | None = None,
    sel: str = SEL,
) -> Partition:
    """Sépare les squelettes en entraînement et test.

    Pour chaque profondeur, les ``n`` squelettes de plus petite valeur de
    hachage partent au test. Déterministe, indépendant de l'ordre du fichier.
    """
    reserve = RESERVE_PAR_PROFONDEUR if reserve is None else reserve
    par_prof: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        par_prof[it["depth"]].append(it)

    en_test: set[str] = set()
    for profondeur, n in reserve.items():
        candidats = par_prof.get(profondeur, [])
        if n > len(candidats):
            raise ValueError(
                f"profondeur {profondeur} : {n} squelettes demandés au test, "
                f"{len(candidats)} disponibles"
            )
        classes = sorted(candidats, key=lambda it: valeur_de_hachage(it["prefix"], sel))
        en_test.update(it["prefix"] for it in classes[:n])

    return Partition(
        entrainement=[it for it in items if it["prefix"] not in en_test],
        test=[it for it in items if it["prefix"] in en_test],
    )
