"""Du squelette au nuage de points, avec augmentation réaliste.

Le tirage des points suit la **courbure de la courbe** et non une loi uniforme :
un stylo ralentit dans les virages, donc les points s'y accumulent. C'est l'une
des trois composantes qui manquaient à la spec, et sans doute celle qui compte
le plus pour la robustesse au tracé à main levée.

Le bruit corrélé est appliqué **en espace d'indice** et non en espace ``x``.
Ce n'est pas une approximation par paresse : le tremblement de la main est
corrélé dans le *temps*, et le stylo avance le long de l'abscisse curviligne —
l'indice du point est donc plus proche du temps que ne l'est ``x``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from curvy.data.expr import Node, count_constants, evaluate

__all__ = [
    "sample_cloud_fn",
    "CloudConfig",
    "PointCloud",
    "RejectReason",
    "normalise_y",
    "sample_cloud",
]

#: Grille dense servant au filtrage des dégénérescences et au calcul de densité.
DENSE_N = 512


class RejectReason:
    """Causes de rejet, comptées séparément pour être publiables."""

    NON_FINITE = "non_fini"  # nan/inf sur le domaine (log, sqrt, inv hors domaine)
    EXPLOSION = "explosion"  # |y| > seuil
    CONSTANT = "constante"  # y quasi constant : pas d'information
    DEGENERATE_NOISE = "bruit_degenere"  # étendue de y nulle après bruit
    UNIDENTIFIABLE = "bruit_excessif"  # le bruit a effacé la formule
    SPIKE = "pic_isole"  # plate partout sauf une singularité qui fixe l'échelle
    ALL = (NON_FINITE, EXPLOSION, CONSTANT, DEGENERATE_NOISE, UNIDENTIFIABLE, SPIKE)


@dataclass(frozen=True)
class CloudConfig:
    n_points_min: int = 20
    n_points_max: int = 200
    #: |C| tiré log-uniformément : sur [-1, 1], une constante doit pouvoir
    #: engendrer aussi bien une pente douce qu'une oscillation rapide.
    const_log_range: tuple[float, float] = (0.05, 20.0)
    max_abs_y: float = 1e6
    #: Étendue relative minimale de y : en dessous, la courbe est une constante.
    min_y_range: float = 1e-6
    #: Fraction maximale de la grille dense autorisée à être non finie.
    max_nonfinite_frac: float = 0.0

    # --- augmentation (chaque composante désactivable pour l'ablation) ---
    use_curvature_density: bool = True
    use_white_noise: bool = True
    use_correlated_drift: bool = True
    use_x_jitter: bool = True
    use_gaps: bool = True
    use_quantisation: bool = True

    white_sigma_range: tuple[float, float] = (1e-3, 8e-2)
    drift_sigma_range: tuple[float, float] = (1e-3, 1.2e-1)
    drift_length_range: tuple[float, float] = (0.05, 0.35)  # en fraction du nuage
    #: R² minimal entre la courbe exacte et les points bruités, tous deux
    #: normalisés. En dessous, le bruit a effacé la formule : l'exemple n'est
    #: plus de l'augmentation, c'est du bruit d'étiquetage. Constaté sur la
    #: figure d'aperçu du jalon Phase 1, où plusieurs nuages ne décrivaient
    #: manifestement plus la formule qui les avait engendrés.
    min_identifiability_r2: float = 0.95
    #: Étendue minimale du décile central de la courbe **normalisée**. Le
    #: filtre « constante » regarde l'étendue totale et se laisse berner par
    #: une quasi-singularité : la courbe est plate partout, le pic fixe
    #: l'échelle, et après normalisation il ne reste qu'un trait horizontal
    #: avec une valeur aberrante. Constaté sur la figure d'aperçu du jalon.
    min_central_spread: float = 0.20
    p_gap: float = 0.25
    gap_width_range: tuple[float, float] = (0.08, 0.30)
    quantisation_levels: tuple[int, ...] = field(default=(128, 256, 512, 1024))
    p_quantisation: float = 0.4


@dataclass
class PointCloud:
    x: np.ndarray
    y: np.ndarray
    consts: list[float]
    y_scale: float  # y_normalisé = (y_brut - y_offset) / y_scale
    y_offset: float
    n_points: int


def _sample_constants(rng: np.random.Generator, k: int, cfg: CloudConfig) -> list[float]:
    lo, hi = cfg.const_log_range
    mag = np.exp(rng.uniform(np.log(lo), np.log(hi), size=k))
    sign = rng.choice((-1.0, 1.0), size=k)
    return list(mag * sign)


def _curvature_density(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Densité d'échantillonnage ∝ (1 + |dy/dx|)^α, mélangée à l'uniforme."""
    slope = np.abs(np.gradient(y))
    slope = slope / (slope.max() + 1e-12)
    alpha = rng.uniform(0.5, 2.5)
    mix = rng.uniform(0.2, 0.8)  # part d'uniforme, pour ne jamais tout concentrer
    dens = mix + (1.0 - mix) * (slope**alpha)
    return dens / dens.sum()


def _sample_x(
    rng: np.random.Generator, n: int, dense_x: np.ndarray, dense_y: np.ndarray, cfg: CloudConfig
) -> np.ndarray:
    """Positions en x, irrégulières, avec trous éventuels et bornes garanties.

    Les bornes -1 et +1 sont toujours présentes : à l'inférence, la
    normalisation envoie de toute façon le minimum et le maximum observés sur
    -1 et +1. Un nuage d'entraînement qui ne les atteindrait pas
    créerait un décalage train/test.
    """
    if cfg.use_curvature_density:
        p = _curvature_density(dense_y, rng)
    else:
        p = np.full(len(dense_x), 1.0 / len(dense_x))

    if cfg.use_gaps and rng.random() < cfg.p_gap:
        width = rng.uniform(*cfg.gap_width_range)
        start = rng.uniform(0.0, 1.0 - width)
        lo, hi = -1.0 + 2.0 * start, -1.0 + 2.0 * (start + width)
        p = np.where((dense_x > lo) & (dense_x < hi), 0.0, p)
        if p.sum() <= 0:
            p = np.full(len(dense_x), 1.0 / len(dense_x))
        p = p / p.sum()

    idx = rng.choice(len(dense_x), size=max(n - 2, 1), replace=True, p=p)
    x = np.concatenate([[-1.0, 1.0], dense_x[idx]])

    if cfg.use_x_jitter:
        step = 2.0 / len(dense_x)
        x = x + rng.normal(0.0, step * rng.uniform(0.3, 1.5), size=x.shape)
        x = np.clip(x, -1.0, 1.0)
        x[0], x[1] = -1.0, 1.0

    return np.sort(x)


def _correlated_drift(rng: np.random.Generator, n: int, cfg: CloudConfig) -> np.ndarray:
    """Bruit gaussien passé au filtre passe-bas : la dérive de la main."""
    length = max(2, int(rng.uniform(*cfg.drift_length_range) * n))
    raw = rng.normal(size=n + 4 * length)
    k = np.exp(-0.5 * (np.arange(-2 * length, 2 * length + 1) / length) ** 2)
    k /= k.sum()
    smooth = np.convolve(raw, k, mode="same")[2 * length : 2 * length + n]
    std = smooth.std()
    return smooth / std if std > 1e-12 else smooth


def normalise_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ramène y dans [-1, 1]. Retourne aussi l'affine pour l'inverser."""
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    if span < 1e-12:
        return np.zeros_like(y), 1.0, lo
    offset = (hi + lo) / 2.0
    scale = span / 2.0
    return (y - offset) / scale, scale, offset


def sample_cloud(
    rng: np.random.Generator, skeleton: Node, cfg: CloudConfig | None = None
) -> tuple[PointCloud | None, str | None]:
    """Un nuage de points, ou ``(None, raison_du_rejet)``."""
    cfg = cfg or CloudConfig()
    consts = _sample_constants(rng, count_constants(skeleton), cfg)

    def f(xs: np.ndarray) -> np.ndarray:
        return evaluate(skeleton, xs, consts)

    return sample_cloud_fn(rng, f, cfg, consts=consts)


def sample_cloud_fn(
    rng: np.random.Generator,
    f: Callable[[np.ndarray], np.ndarray],
    cfg: CloudConfig | None = None,
    consts: list[float] | None = None,
) -> tuple[PointCloud | None, str | None]:
    """Même chaîne, pour une fonction quelconque plutôt qu'un squelette.

    Sert au sous-ensemble **hors distribution** du jeu de test (Phase 6) : les
    formules écrites à la main n'ont pas toutes de représentation dans la
    grammaire, mais doivent traverser exactement le même bruit, la même
    densification par courbure et le même filtre d'identifiabilité. Sans quoi
    on comparerait deux protocoles au lieu de deux jeux de formules.
    """
    cfg = cfg or CloudConfig()
    consts = [] if consts is None else consts

    dense_x = np.linspace(-1.0, 1.0, DENSE_N)
    dense_y = f(dense_x)

    finite = np.isfinite(dense_y)
    if (~finite).mean() > cfg.max_nonfinite_frac:
        return None, RejectReason.NON_FINITE
    if np.abs(dense_y[finite]).max() > cfg.max_abs_y:
        return None, RejectReason.EXPLOSION
    span = float(dense_y[finite].max() - dense_y[finite].min())
    if span < cfg.min_y_range * max(1.0, float(np.abs(dense_y[finite]).max())):
        return None, RejectReason.CONSTANT

    dense_norm, _, _ = normalise_y(dense_y[finite])
    spread = float(np.percentile(dense_norm, 95) - np.percentile(dense_norm, 5))
    if spread < cfg.min_central_spread:
        return None, RejectReason.SPIKE

    n = int(rng.integers(cfg.n_points_min, cfg.n_points_max + 1))
    x = _sample_x(rng, n, dense_x, np.where(finite, dense_y, 0.0), cfg)
    y = f(x)
    ok = np.isfinite(y)
    if not ok.all():
        x, y = x[ok], y[ok]
        if len(x) < cfg.n_points_min:
            return None, RejectReason.NON_FINITE

    y_clean = y.copy()
    y_span = float(y.max() - y.min())
    if cfg.use_white_noise:
        sigma = np.exp(rng.uniform(*np.log(cfg.white_sigma_range)))
        y = y + rng.normal(0.0, sigma * max(y_span, 1e-9), size=y.shape)
    if cfg.use_correlated_drift:
        sigma = np.exp(rng.uniform(*np.log(cfg.drift_sigma_range)))
        y = y + sigma * max(y_span, 1e-9) * _correlated_drift(rng, len(y), cfg)

    y_norm, scale, offset = normalise_y(y)
    if float(y_norm.max() - y_norm.min()) < 1e-9:
        return None, RejectReason.DEGENERATE_NOISE

    # La formule doit rester identifiable à partir des points bruités.
    clean_norm = (y_clean - offset) / scale
    ss_res = float(np.sum((y_norm - clean_norm) ** 2))
    ss_tot = float(np.sum((y_norm - y_norm.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if r2 < cfg.min_identifiability_r2:
        return None, RejectReason.UNIDENTIFIABLE

    if cfg.use_quantisation and rng.random() < cfg.p_quantisation:
        levels = int(rng.choice(cfg.quantisation_levels))
        y_norm = np.round(y_norm * levels) / levels
        x = np.round((x + 1.0) / 2.0 * levels) / levels * 2.0 - 1.0

    return PointCloud(x, y_norm, consts, scale, offset, len(x)), None
