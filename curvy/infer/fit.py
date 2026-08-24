"""Ajustement des constantes d'un squelette sur un nuage de points.

Implémente l'idée d'initialisation linéaire sous une
forme un peu plus forte : une **projection variable**.

Tout squelette a la forme ``a * body(x) + b`` (enveloppe de racine).
Pour un jeu de constantes internes donné, ``a`` et ``b`` sont solution exacte
d'une régression linéaire — inutile de les chercher par descente. On n'optimise
donc que les ``k - 2`` constantes internes, et le problème non linéaire perd
deux dimensions. Pour les squelettes les plus simples (`C*x + C`,
`C*sin(x) + C`), il ne reste **rien** à optimiser : la solution est exacte et
immédiate.

``scipy.optimize.least_squares`` sert de moteur ; c'est l'implémentation de
référence, celle contre laquelle un éventuel Levenberg-Marquardt
batché sur GPU devra être validé.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from curvy.data.expr import Node, count_constants, evaluate

__all__ = ["frequences_candidates", "FitResult", "fit_constants", "r_squared"]

_BIG = 1e6


@dataclass
class FitResult:
    consts: list[float]
    r2_fit: float
    n_restarts_used: int
    ok: bool


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² classique. Retourne ``-inf`` si la prédiction n'est pas finie."""
    if not np.isfinite(y_pred).all():
        return float("-inf")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-15:
        return 1.0 if np.allclose(y_pred, y_true) else float("-inf")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


_C: Node = ("C",)


def _unwrap_root(node: Node) -> Node | None:
    """``body`` si l'arbre a la forme ``add(mul(C, body), C)``, sinon ``None``.

    Tous les squelettes du dataset ont cette forme, mais **pas** ceux
    que produit le décodeur tant que le modèle n'a rien appris : il émet
    n'importe quel arbre syntaxiquement valide. Supposer l'enveloppe était un
    bug — trouvé au premier run de rodage, sur un `mul(x, C)` sans enveloppe.
    """
    if node[0] != "add" or node[2] != _C:
        return None
    scale = node[1]
    if scale[0] != "mul":
        return None
    if scale[1] == _C:
        return scale[2]
    if scale[2] == _C:
        return scale[1]
    return None


def _solve_affine(basis: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Meilleurs ``a, b`` tels que ``a*basis + b ≈ y``, en une seule opération."""
    design = np.stack([basis, np.ones_like(basis)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, a * basis + b


def frequences_candidates(x: np.ndarray, y: np.ndarray, n_max: int = 2) -> list[float]:
    """Pulsations dominantes du signal, lues par transformée de Fourier.

    **Pourquoi.** L'initialisation log-uniforme cherchait la fréquence au hasard
    dans ``[0,05 ; 20]`` — la plage de génération. Mesuré le
    2026-08-20 sur ``C·sin(C·x)+C`` avec le squelette **exact** : l'ajustement
    retrouve la fréquence jusqu'à 12 rad, puis décroche (R² 0,058 à 16 rad,
    0,002 à 30). Le paysage d'optimisation d'une fréquence est plein de minima
    locaux ; on ne l'atteint pas par tirage, on doit le viser.

    Or la fréquence est **lisible directement dans les points**. Une FFT sur
    ``y`` rééchantillonné coûte quelques microsecondes et donne le pic. Sur
    ``[-1, 1]``, un signal de ``k`` cycles sur le domaine correspond à une
    pulsation ``ω = π·k``.

    Retourne au plus ``n_max`` pulsations, la plus énergique d'abord. Liste vide
    si le signal n'a pas de pic exploitable — auquel cas on retombe sur le
    tirage aléatoire, qui reste correct en dessous de 12 rad.

    **Résolution.** Les raies d'une FFT sont espacées de ``2π/largeur``, soit
    π ≈ 3,14 rad sur ``[-1, 1]`` : une fréquence tombant entre deux raies serait
    lue à 1,6 rad près. Le pic est donc affiné par **interpolation parabolique**
    sur ses deux voisins — six lignes, et l'erreur tombe sous 0,2 rad. Ça compte
    aux fréquences élevées, où le bassin de convergence est étroit.
    """
    n = len(x)
    if n < 8:
        return []
    ordre = np.argsort(x)
    xs, ys = x[ordre], y[ordre]
    largeur = float(xs[-1] - xs[0])
    if not np.isfinite(largeur) or largeur < 1e-12:
        return []

    # La FFT exige un pas régulier ; les nuages ne le sont pas.
    grille = np.linspace(xs[0], xs[-1], max(64, n))
    yg = np.interp(grille, xs, ys)
    yg = yg - yg.mean()
    if not np.isfinite(yg).all() or float(np.abs(yg).max()) < 1e-12:
        return []

    spectre = np.abs(np.fft.rfft(yg * np.hanning(len(yg))))
    if len(spectre) < 3:
        return []
    spectre[0] = 0.0  # la composante continue est déjà absorbée par l'affine
    pics = np.argsort(spectre)[::-1][:n_max]

    out: list[float] = []
    for k in pics:
        if spectre[k] <= 0.0:
            continue
        out.append(2.0 * np.pi * _affiner_pic(spectre, int(k)) / largeur)
    return [w for w in out if 1e-3 < w < 1e4]


#: Opérateurs dont une constante interne est une pulsation.
_PERIODIQUES = frozenset({"sin", "cos"})


def _contient_periodique(node: Node) -> bool:
    if node[0] in _PERIODIQUES:
        return True
    return any(_contient_periodique(enfant) for enfant in node[1:])


def _affiner_pic(spectre: np.ndarray, k: int) -> float:
    """Position du pic entre les raies, par parabole sur ``k-1, k, k+1``.

    Sur trois points d'une parabole équidistants, le sommet est décalé de
    ``(g - d) / (2·(g - 2c + d))`` raie. Formule classique d'estimation
    spectrale ; on la borne à une demi-raie, au-delà c'est que le pic n'est pas
    parabolique et l'affinage n'a pas de sens.
    """
    # La raie 1 a pour voisine la composante continue, qu'on a mise à zéro :
    # la parabole s'appuierait sur un creux artificiel et tirerait l'estimation
    # vers le haut. Mesuré : 3,0 rad lu 3,64 avec, 3,14 sans.
    if k <= 1 or k >= len(spectre) - 1:
        return float(k)
    g, c, d = float(spectre[k - 1]), float(spectre[k]), float(spectre[k + 1])
    denom = g - 2.0 * c + d
    if abs(denom) < 1e-15:
        return float(k)
    delta = 0.5 * (g - d) / denom
    return float(k) + float(np.clip(delta, -0.5, 0.5))


def fit_constants(
    node: Node,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_restarts: int = 6,
    max_nfev: int = 120,
    spectral: bool = True,
) -> FitResult:
    """Meilleur jeu de constantes trouvé, avec plusieurs initialisations.

    ``spectral`` ajoute des points de départ tirés d'une FFT des données plutôt
    que du hasard — voir ``frequences_candidates``. Désactivable pour mesurer ce
    qu'il apporte.
    """
    k = count_constants(node)
    body = _unwrap_root(node)
    if body is None or count_constants(body) != k - 2:
        # Arbre hors forme canonique : pas de projection variable possible,
        # on optimise directement toutes les constantes.
        return _fit_generic(node, x, y, rng, k, n_restarts, max_nfev)

    n_inner = k - 2

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if n_inner == 0:
            basis = evaluate(body, x, [])
            if not np.isfinite(basis).all():
                return FitResult([], float("-inf"), 0, False)
            a, b, pred = _solve_affine(basis, y)
            return FitResult([a, b], r_squared(y, pred), 1, True)

        def residual(inner: np.ndarray) -> np.ndarray:
            basis = evaluate(body, x, list(inner))
            if not np.isfinite(basis).all():
                return np.full_like(y, _BIG)
            _, _, pred = _solve_affine(basis, y)
            return pred - y

        # Points de départ déterministes, essayés avant le tirage aléatoire :
        # le point neutre, puis la pulsation lue dans les données placée tour à
        # tour à chaque position (on ignore laquelle porte la fréquence).
        amorces: list[np.ndarray] = [np.ones(n_inner)]
        # Une pulsation n'a de sens que dans un opérateur périodique. Sur un
        # `inv(x + C)` ou un `exp(C·x)`, l'amorce spectrale est du bruit qui
        # coûte un essai — et l'essai aléatoire qu'elle remplacerait, lui,
        # trouvait. Mesuré : sans ce filtre, +49 % de temps d'ajustement pour un
        # gain concentré sur les seuls squelettes trigonométriques.
        if spectral and _contient_periodique(body):
            for omega in frequences_candidates(x, y):
                for i in range(n_inner):
                    depart = np.ones(n_inner)
                    depart[i] = omega
                    amorces.append(depart)

        # Les amorces spectrales s'AJOUTENT au budget aléatoire, elles ne le
        # remplacent pas. Première version : elles le consommaient, et
        # `hyperbole_raide` — un `inv(x + C)` sans aucune fréquence à lire —
        # est passé de 0,833 à 0,000, cinq tirages utiles ayant cédé la place à
        # trois amorces inutiles. La sortie anticipée sur R² > 0,9999 fait que
        # ce budget élargi ne coûte rien quand une amorce tombe juste.
        budget = len(amorces) + max(0, n_restarts - 1)

        best: tuple[float, list[float]] | None = None
        used = 0
        for attempt in range(budget):
            if attempt < len(amorces):
                x0 = amorces[attempt]
            else:
                # Repli log-uniforme sur la plage de génération.
                mag = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=n_inner))
                x0 = mag * rng.choice((-1.0, 1.0), size=n_inner)
            used += 1
            try:
                sol = least_squares(residual, x0, max_nfev=max_nfev, method="lm")
            except Exception:
                continue
            basis = evaluate(body, x, list(sol.x))
            if not np.isfinite(basis).all():
                continue
            a, b, pred = _solve_affine(basis, y)
            score = r_squared(y, pred)
            if best is None or score > best[0]:
                best = (score, [a, *sol.x.tolist(), b])
            if best[0] > 0.9999:  # inutile de continuer à chercher
                break

    if best is None:
        return FitResult([], float("-inf"), used, False)
    return FitResult(best[1], best[0], used, True)


def _fit_generic(
    node: Node,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    k: int,
    n_restarts: int,
    max_nfev: int,
) -> FitResult:
    """Ajustement sans projection variable, pour un arbre hors forme canonique."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if k == 0:
            pred = evaluate(node, x, [])
            return FitResult([], r_squared(y, pred), 1, np.isfinite(pred).all())

        def residual(c: np.ndarray) -> np.ndarray:
            pred = evaluate(node, x, list(c))
            if not np.isfinite(pred).all():
                return np.full_like(y, _BIG)
            return pred - y

        best: tuple[float, list[float]] | None = None
        used = 0
        for attempt in range(n_restarts):
            if attempt == 0:
                x0 = np.ones(k)
            else:
                mag = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=k))
                x0 = mag * rng.choice((-1.0, 1.0), size=k)
            used += 1
            try:
                sol = least_squares(residual, x0, max_nfev=max_nfev, method="lm")
            except Exception:
                continue
            pred = evaluate(node, x, list(sol.x))
            score = r_squared(y, pred)
            if best is None or score > best[0]:
                best = (score, sol.x.tolist())
            if best[0] > 0.9999:
                break

    if best is None:
        return FitResult([], float("-inf"), used, False)
    return FitResult(best[1], best[0], used, True)


def estimer_bruit(x: np.ndarray, y: np.ndarray) -> float:
    """Écart-type du bruit, estimé **sans connaître la fonction**.

    Pseudo-résidus de Gasser, Sroka et Jennen-Steinmetz (1986) : pour chaque
    point intérieur, on interpole ses deux voisins et on regarde de combien le
    point s'en écarte. Sur une fonction lisse, cet écart est dominé par le
    bruit ; la courbure n'y contribue qu'au second ordre.

    ``ε_i = a_i·y_{i-1} + b_i·y_{i+1} − y_i`` avec les poids de l'interpolation
    linéaire aux abscisses réelles, puis ``σ² = moyenne(c_i²·ε_i²)`` où ``c_i``
    normalise la variance. **La pondération par les espacements est
    indispensable ici** : les nuages du projet ne sont jamais à pas régulier
    (densité liée à la courbure, trous, jitter), et la différence
    seconde naïve prendrait l'irrégularité pour du bruit.

    Retourne ``0.0`` si l'estimation n'a pas de sens (trop peu de points,
    abscisses confondues).
    """
    n = len(x)
    if n < 5:
        return 0.0
    ordre = np.argsort(x)
    xs, ys = np.asarray(x, dtype=float)[ordre], np.asarray(y, dtype=float)[ordre]

    dx_g = xs[1:-1] - xs[:-2]
    dx_d = xs[2:] - xs[1:-1]
    ecart = dx_g + dx_d
    valide = ecart > 1e-12
    if not valide.any():
        return 0.0

    a = np.where(valide, dx_d / np.where(valide, ecart, 1.0), 0.0)
    b = np.where(valide, dx_g / np.where(valide, ecart, 1.0), 0.0)
    eps = a * ys[:-2] + b * ys[2:] - ys[1:-1]
    c2 = 1.0 / (a**2 + b**2 + 1.0)
    val = c2[valide] * eps[valide] ** 2
    if not val.size or not np.isfinite(val).all():
        return 0.0
    return float(np.sqrt(max(0.0, val.mean())))
