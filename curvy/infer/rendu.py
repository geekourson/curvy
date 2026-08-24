"""Rendre une formule lisible, avec ses constantes et dans les bonnes unités.

Deux besoins que le squelette ne couvre pas :

1. **les valeurs.** `C * sin(C * x) + C` décrit une famille ; l'utilisateur veut
   `2.31*sin(4.07*x) - 0.5` ;
2. **les unités.** Les constantes sont ajustées en coordonnées **normalisées** —
   `x` ramené dans `[-1, 1]`, `y` centré-réduit. Rendre la formule
   telle quelle serait **faux** pour qui a fourni des mesures : elle ne
   s'appliquerait pas à ses abscisses. Il faut la composer avec les deux
   affines.

Pour un tracé au canvas la question ne se pose pas — les pixels ne sont pas une
unité qui intéresse quelqu'un. Elle se pose dès qu'on importe des données.

sympy sert ici et **seulement ici** : à mettre en forme une expression déjà
décidée. Il reste écarté de la canonicalisation, où il est sémantiquement faux
(cf. l'en-tête de `curvy.data.canonical` : il « simplifierait » `C*x + C*x` en
`2*C*x`, alors que les deux `C` sont indépendants).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curvy.data.expr import Node, const_name_iter, to_infix

__all__ = ["Affines", "formule_lisible"]


@dataclass(frozen=True)
class Affines:
    """Les transformations appliquées avant l'ajustement.

    ``x_norm = (x - x_centre) / x_demi`` et ``y = y_norm * y_echelle + y_decalage``.
    """

    x_centre: float = 0.0
    x_demi: float = 1.0
    y_echelle: float = 1.0
    y_decalage: float = 0.0

    @property
    def identite(self) -> bool:
        return (
            self.x_centre == 0.0
            and self.x_demi == 1.0
            and self.y_echelle == 1.0
            and self.y_decalage == 0.0
        )


def _arrondi_lisible(v: float, chiffres: int = 4) -> float:
    """Arrondit sans écraser les petites valeurs.

    ``round(1.2e-5, 4)`` vaut ``0.0``, ce qui transformerait une constante en
    zéro et changerait la formule. On arrondit donc à un nombre de **chiffres
    significatifs**, pas de décimales.
    """
    if v == 0.0 or not np.isfinite(v):
        return float(v)
    return float(f"%.{chiffres}g" % v)


def formule_lisible(
    node: Node,
    consts: list[float],
    affines: Affines | None = None,
    chiffres: int = 4,
) -> str:
    """L'expression avec ses constantes, dans les unités d'origine si fournies.

    Retombe sur un rendu textuel simple si sympy échoue — une mise en forme
    ratée ne doit jamais faire échouer une prédiction correcte.
    """
    noms = const_name_iter()
    infixe = to_infix(node, noms)
    valeurs = {f"c{i}": _arrondi_lisible(v, chiffres) for i, v in enumerate(consts)}

    try:
        import sympy

        x = sympy.Symbol("x")
        expr = sympy.sympify(infixe, locals={"x": x})
        expr = expr.subs({sympy.Symbol(k): sympy.Float(v) for k, v in valeurs.items()})

        if affines is not None and not affines.identite:
            # x du modèle = (x réel - centre) / demi, puis y réel = y*échelle + décalage
            expr = expr.subs(x, (x - sympy.Float(affines.x_centre)) / sympy.Float(affines.x_demi))
            expr = expr * sympy.Float(affines.y_echelle) + sympy.Float(affines.y_decalage)

        # Surtout PAS de `nsimplify` : il rationalise les flottants et rend
        # `231*sin(407*x/100)/100 - 1/2` là où on veut `2.31*sin(4.07*x) - 0.5`.
        if expr.count_ops() < 40:
            expr = sympy.expand(expr)

        # `evalf` n'arrondit que le résultat des opérations, pas les constantes
        # déjà présentes : la composition avec les affines laissait sortir
        # `exp(-0.0454533333333333*x)`. On arrondit chaque flottant de l'arbre.
        flottants = expr.atoms(sympy.Float)
        # Un terme négligeable devant les autres est du bruit d'arithmétique
        # flottante, pas une constante : `4.9*x^2 - 1.776e-15` doit se lire
        # `4.9*x^2`. Le critère est RELATIF — une donnée à l'échelle du
        # micromètre a des constantes légitimement minuscules.
        echelle = max((abs(float(f)) for f in flottants), default=0.0)
        seuil = 1e-10 * echelle
        expr = expr.xreplace(
            {
                f: sympy.Float(
                    0.0 if abs(float(f)) < seuil else _arrondi_lisible(float(f), chiffres)
                )
                for f in flottants
            }
        )
        rendu = str(expr.evalf(chiffres))
        return rendu.replace("**", "^")
    except Exception:
        # Repli : substitution textuelle, toujours juste même si moins jolie.
        rendu = infixe
        for nom, v in valeurs.items():
            rendu = rendu.replace(nom, repr(v))
        return rendu.replace("**", "^")
