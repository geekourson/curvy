"""Pondération des squelettes à l'entraînement.

La déduplication inverse la distribution de profondeur : 71 % des squelettes
uniques sont à la profondeur maximale, alors que 35 % des *tirages* étaient de
profondeur minimale. Tirer uniformément dans l'ensemble unique reviendrait à
n'entraîner que sur des expressions complexes, et à échouer sur le cas d'usage
principal — une droite, une parabole, une sinusoïde.

La pondération est **stratifiée par profondeur**, avec une cible explicite.
L'alternative testée d'abord — un poids ``count ** tau`` — a été mesurée puis
abandonnée : entre ``tau = 0,75`` et ``tau = 1,25``, la part de la profondeur 3
passe de 5 % à 80 %. Un paramètre avec une falaise pareille n'est pas réglable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

__all__ = ["DEFAULT_DEPTH_TARGET", "stratified_weights", "describe_weights"]

#: Part de l'entraînement allouée à chaque profondeur de squelette.
#: Délibérément plus riche en formes simples que l'ensemble dédupliqué, moins
#: que la distribution brute de l'échantillonneur. À ablater en Phase 4.
DEFAULT_DEPTH_TARGET: dict[int, float] = {
    3: 0.05,  # C*x + C — l'affine, un seul squelette mais un cas très fréquent
    4: 0.08,
    5: 0.12,
    6: 0.20,
    7: 0.25,
    8: 0.30,
}


def stratified_weights(depths: Sequence[int], target: dict[int, float] | None = None) -> np.ndarray:
    """Poids de tirage par squelette réalisant la distribution cible.

    Dans chaque strate de profondeur, les squelettes sont équiprobables : la
    multiplicité ne sert plus qu'à décrire la strate, pas à départager ses
    membres. Les profondeurs absentes de la cible reçoivent un poids nul ;
    celles absentes des données voient leur part redistribuée.
    """
    target = target or DEFAULT_DEPTH_TARGET
    d = np.asarray(depths)
    members: dict[int, np.ndarray] = defaultdict(lambda: np.array([], dtype=int))
    for depth in np.unique(d):
        members[int(depth)] = np.flatnonzero(d == depth)

    present = {k: v for k, v in target.items() if len(members.get(k, ())) > 0}
    total = sum(present.values())
    if total <= 0:
        raise ValueError("aucune profondeur de la cible n'est présente dans les données")

    w = np.zeros(len(d), dtype=np.float64)
    for depth, share in present.items():
        idx = members[depth]
        w[idx] = (share / total) / len(idx)
    return w


def describe_weights(depths: Sequence[int], weights: np.ndarray) -> dict[int, float]:
    """Part effective de chaque profondeur, pour vérifier que la cible est tenue."""
    d = np.asarray(depths)
    return {int(k): round(100 * float(weights[d == k].sum()), 2) for k in sorted(np.unique(d))}
