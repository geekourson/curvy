"""Échantillonnage d'arbres d'expression.

Processus de branchement volontairement **sous-critique** : l'espérance du
nombre d'enfants vaut ``0,35 × 1 + 0,30 × 2 = 0,95 < 1``, donc les arbres
restent petits en moyenne et la coupure de profondeur ne sert que pour la
queue de distribution. Un processus sur-critique produirait une majorité
d'arbres butant sur la profondeur maximale, donc une distribution dégénérée.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curvy.data.canonical import canonicalise, strip_absorbable_root, wrap_root
from curvy.data.expr import Node, count_constants, depth, has_x
from curvy.data.grammar import BINARY, MAX_BODY_CONSTANTS, MAX_BODY_DEPTH, UNARY

__all__ = ["SamplerConfig", "sample_skeleton"]


@dataclass(frozen=True)
class SamplerConfig:
    max_body_depth: int = MAX_BODY_DEPTH
    max_body_constants: int = MAX_BODY_CONSTANTS
    p_leaf: float = 0.35
    p_unary: float = 0.35
    p_x_given_leaf: float = 0.75
    #: Poids par opérateur unaire ; ``sq``/``sin`` sont plus fréquents dans la
    #: nature que ``cube`` ou ``log``, et le dataset doit refléter cette réalité
    #: plutôt qu'une uniforme artificielle.
    unary_weights: tuple[float, ...] = (
        1.4,  # sin
        1.0,  # cos
        0.8,  # exp
        0.6,  # log
        0.7,  # sqrt
        0.6,  # abs
        1.0,  # tanh
        1.4,  # sq
        0.7,  # cube
        0.8,  # inv
    )


def _sample_tree(rng: np.random.Generator, remaining: int, cfg: SamplerConfig) -> Node:
    if remaining <= 1:
        return ("x",) if rng.random() < cfg.p_x_given_leaf else ("C",)
    u = rng.random()
    if u < cfg.p_leaf:
        return ("x",) if rng.random() < cfg.p_x_given_leaf else ("C",)
    if u < cfg.p_leaf + cfg.p_unary:
        w = np.asarray(cfg.unary_weights, dtype=np.float64)
        op = UNARY[rng.choice(len(UNARY), p=w / w.sum())]
        return (op, _sample_tree(rng, remaining - 1, cfg))
    op = BINARY[rng.integers(len(BINARY))]
    return (op, _sample_tree(rng, remaining - 1, cfg), _sample_tree(rng, remaining - 1, cfg))


def sample_skeleton(rng: np.random.Generator, cfg: SamplerConfig | None = None) -> Node | None:
    """Un squelette canonique enveloppé, ou ``None`` si le tirage est rejeté.

    Rejets possibles à ce stade : arbre sans ``x`` (donc constant), corps trop
    profond ou trop riche en constantes après canonicalisation.
    """
    cfg = cfg or SamplerConfig()
    body = _sample_tree(rng, cfg.max_body_depth, cfg)
    body = strip_absorbable_root(canonicalise(body))
    if not has_x(body):
        return None
    if depth(body) > cfg.max_body_depth:
        return None
    if count_constants(body) > cfg.max_body_constants:
        return None
    return wrap_root(body)
