"""Jeu de test figé de la Phase 6 : ce que le modèle n'a jamais vu.

Trois sous-ensembles, mesurés séparément et jamais agrégés :

1. **tenu à l'écart** — mêmes grammaire et générateur que l'entraînement, mais
   squelettes explicitement exclus du flux (``curvy.data.split``). Répond à
   « le modèle généralise-t-il à des formules inédites de sa propre famille ? » ;
2. **hors distribution** — formules écrites à la main. Certaines sont dans la
   grammaire mais qu'un tirage aléatoire ne produirait jamais (une gaussienne,
   une sinusoïde à 30 rad, la fonction de Runge) ; d'autres en sortent
   franchement (une marche, un plancher, une fonction par morceaux). Répond à
   « que se passe-t-il quand on sort du bac à sable ? » ;
3. **réel** — tracés capturés au canvas. N'existe pas encore : l'outil de
   capture est en Phase 8.

Toutes traversent **le même** pipeline de nuage que l'entraînement — bruit
blanc, dérive corrélée, densité liée à la courbure, trous, quantification
 — via ``sample_cloud_fn``. Sans quoi on comparerait deux protocoles
au lieu de deux familles de formules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import erf, factorial

import numpy as np

__all__ = ["FORMULES_A_LA_MAIN", "FormuleAlaMain"]


@dataclass(frozen=True)
class FormuleAlaMain:
    nom: str
    f: Callable[[np.ndarray], np.ndarray]
    #: Notation préfixe si la grammaire v1 peut l'écrire, sinon ``None``.
    #: Vérifié par test contre les budgets réels, jamais affirmé à la main.
    prefixe: str | None
    commentaire: str


def _morceaux(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0.0, -(x**2), x**2)


def _dent_de_scie(x: np.ndarray) -> np.ndarray:
    return 3.0 * x - np.floor(3.0 * x)


def _puissance(x: np.ndarray) -> np.ndarray:
    # x^x n'est réel que sur x > 0 ; on décale pour rester défini partout.
    u = (x + 1.0) / 2.0 + 0.05
    return u**u


def _weierstrass(x: np.ndarray) -> np.ndarray:
    return sum(0.5**n * np.cos(3.0**n * np.pi * x) for n in range(6))


#: Le catalogue. `prefixe` renseigné = la grammaire sait l'écrire ; c'est le
#: modèle qu'on met en défaut, pas le vocabulaire. `prefixe = None` = la
#: grammaire ne peut pas, et l'échec attendu mesure une limite assumée.
FORMULES_A_LA_MAIN: tuple[FormuleAlaMain, ...] = (
    # --- exprimables, mais qu'un tirage ne produirait jamais ---
    FormuleAlaMain(
        "gaussienne",
        lambda x: np.exp(-8.0 * x**2),
        "exp mul C sq x",
        "la cloche : partout en science, jamais tirée par le générateur",
    ),
    FormuleAlaMain(
        "runge",
        lambda x: 1.0 / (1.0 + 25.0 * x**2),
        "inv add C mul C sq x",
        "le contre-exemple classique de l'interpolation polynomiale",
    ),
    FormuleAlaMain(
        "sinus_haute_frequence",
        lambda x: np.sin(30.0 * x),
        "sin mul C x",
        "structure triviale, constante inatteignable : mesuré le 2026-08-20, "
        "l'ajustement retrouve C jusqu'à 12 avec 6 essais et jusqu'à 20 avec 50, "
        "puis échoue. Le modèle peut proposer LE bon squelette et scorer 0.",
    ),
    FormuleAlaMain(
        "sigmoide_raide",
        lambda x: np.tanh(20.0 * x),
        "tanh mul C x",
        "quasi-marche, mais dérivable — la grammaire l'a",
    ),
    FormuleAlaMain(
        "oscillation_amortie",
        lambda x: np.exp(-3.0 * x) * np.sin(10.0 * x),
        "mul exp mul C x sin mul C x",
        "produit de deux motifs courants, rare en tirage",
    ),
    FormuleAlaMain(
        "sinus_de_inverse",
        lambda x: np.sin(1.0 / np.where(np.abs(x) < 0.05, 0.05, x)),
        "sin inv x",
        "pathologique près de 0 : oscille infiniment vite. L'appelable borne "
        "|x| à 0,05, il n'est donc pas exactement sin(1/x) — le squelette "
        "plafonne à R² 0,90 contre lui, et c'est voulu.",
    ),
    FormuleAlaMain(
        "logarithme_decale",
        lambda x: np.log(x + 1.5),
        "log add x C",
        "translation, pour sortir du domaine où log explose",
    ),
    FormuleAlaMain(
        "racine_de_valeur_absolue",
        lambda x: np.sqrt(np.abs(x)),
        "sqrt abs x",
        "pointe en 0, dérivée infinie",
    ),
    # --- pôles et asymptotes : aucun exemple jusqu'ici, et c'est un mode
    #     d'échec à part entière — la courbe part à l'infini dans le domaine ---
    FormuleAlaMain(
        "hyperbole_raide",
        lambda x: 1.0 / (x - 1.15),
        "inv add x C",
        "pôle JUSTE en dehors du domaine, donc pente très forte à droite. Un pôle "
        "*dans* le domaine a été essayé et retiré : le générateur le rejette "
        "systématiquement (filtre d'explosion à 1e6, aucun point non fini toléré). "
        "Ce n'est pas une lacune — un tracé au canvas est borné par l'écran, la "
        "démo ne recevra jamais de valeur infinie.",
    ),
    FormuleAlaMain(
        "sinus_cardinal",
        lambda x: np.sin(10.0 * x) / np.where(np.abs(x) < 1e-6, 1e-6, x),
        "mul sin mul C x inv x",
        "trou apparent en 0 alors que la fonction y est régulière",
    ),
    FormuleAlaMain(
        "chirp",
        lambda x: np.sin(25.0 * x**2),
        "sin mul C sq x",
        "fréquence variable : le motif change le long du tracé",
    ),
    FormuleAlaMain(
        "x_sinus_de_inverse",
        lambda x: x * np.sin(1.0 / np.where(np.abs(x) < 0.03, 0.03, x)),
        "mul x sin inv x",
        "oscillation dont l'amplitude s'annule — continue mais non dérivable en 0",
    ),
    FormuleAlaMain(
        "croissance_exponentielle",
        lambda x: np.exp(5.0 * x),
        "exp mul C x",
        "trois ordres de grandeur sur le domaine : la normalisation écrase la gauche",
    ),
    FormuleAlaMain(
        "pointe_etroite",
        lambda x: np.exp(-200.0 * x**2),
        "exp mul C sq x",
        "quasi nulle partout sauf sur 10 % du domaine : la densité par courbure est mise à l'épreuve",
    ),
    FormuleAlaMain(
        "coude",
        lambda x: np.abs(x - 0.2),
        "abs add x C",
        "dérivée discontinue en un point, valeur continue",
    ),
    # --- hors grammaire, modes d'échec distincts ------------------------------
    FormuleAlaMain(
        "arctangente",
        lambda x: np.arctan(8.0 * x),
        None,
        "saturation douce, très proche de tanh : le modèle devrait s'en tirer par substitution",
    ),
    FormuleAlaMain(
        "onde_triangulaire",
        lambda x: 2.0 * np.abs(2.0 * (1.5 * x - np.floor(1.5 * x + 0.5))) - 1.0,
        None,
        "périodique, continue, dérivée discontinue partout",
    ),
    FormuleAlaMain(
        "bessel_j0",
        lambda x: np.sum(
            [(-1.0) ** k / (factorial(k) ** 2) * (4.0 * x / 2.0) ** (2 * k) for k in range(12)],
            axis=0,
        ),
        None,
        "fonction spéciale oscillante amortie, absente du vocabulaire",
    ),
    FormuleAlaMain(
        "polynome_degre_7",
        lambda x: 0.5 * x**7 - 1.2 * x**5 + 0.9 * x**3 - 0.3 * x,
        None,
        "exactement ce que la baseline sait faire de mieux : le cas où elle doit gagner",
    ),
    FormuleAlaMain(
        "deux_echelles",
        lambda x: np.sin(2.0 * x) + 0.08 * np.sin(40.0 * x),
        None,
        "un motif lent et un rapide superposés : que rend le front de Pareto ?",
    ),
    FormuleAlaMain(
        "marche_douce_decalee",
        lambda x: 1.0 / (1.0 + np.exp(-25.0 * (x - 0.35))),
        None,
        "sigmoïde logistique décalée : la grammaire n'a ni exp(-u) ni décalage interne bon marché",
    ),
    # --- hors grammaire : l'échec est une limite assumée, pas un bug ---
    FormuleAlaMain(
        "marche",
        lambda x: np.sign(x),
        None,
        "discontinue : aucun opérateur du vocabulaire ne saute",
    ),
    FormuleAlaMain(
        "plancher",
        lambda x: np.floor(3.0 * x),
        None,
        "constante par morceaux, pas de `floor` au vocabulaire",
    ),
    FormuleAlaMain(
        "par_morceaux",
        _morceaux,
        None,
        "deux lois recollées en 0 : la grammaire n'a pas de conditionnelle",
    ),
    FormuleAlaMain(
        "dent_de_scie",
        _dent_de_scie,
        None,
        "périodique et discontinue, cumule les deux difficultés",
    ),
    FormuleAlaMain(
        "x_puissance_x",
        _puissance,
        None,
        "exposant variable : pas de puissance générale au vocabulaire",
    ),
    FormuleAlaMain(
        "weierstrass_tronquee",
        _weierstrass,
        None,
        "somme de six cosinus : profondeur très au-delà du budget",
    ),
    FormuleAlaMain(
        "erf",
        lambda x: np.vectorize(erf)(2.0 * x),
        None,
        "fonction spéciale, absente du vocabulaire (proche de tanh)",
    ),
    FormuleAlaMain(
        "cloche_asymetrique",
        lambda x: np.exp(-8.0 * (x - 0.3) ** 2) - 0.5 * np.exp(-20.0 * (x + 0.5) ** 2),
        None,
        "deux cloches décalées : profondeur et constantes au-delà du budget",
    ),
)
