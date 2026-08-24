"""Arbres d'expression, notation préfixe, évaluation.

Un nœud est un tuple ``(token, *enfants)`` : ``("add", ("mul", ("C",), ("x",)),
("C",))``. C'est immuable, hachable, et se compare directement — trois
propriétés dont la canonicalisation et la déduplication se servent beaucoup.

La notation préfixe rend toute séquence vérifiable par un simple
compteur d'arité, ce qui sert deux fois : valider les cibles à la génération, et
masquer les tokens impossibles pendant le beam search.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from curvy.data.grammar import (
    ARITY,
    COMPLEXITY_COST,
    INFIX_SYMBOL,
    UNARY_RENDER,
)

Node = tuple  # ("token", *enfants)

__all__ = [
    "Node",
    "complexity",
    "const_name_iter",
    "count_constants",
    "depth",
    "evaluate",
    "from_prefix",
    "has_x",
    "iter_nodes",
    "prefix_is_complete",
    "prefix_validity",
    "size",
    "to_infix",
    "to_prefix",
]


# --- structure ---------------------------------------------------------------


def to_prefix(node: Node) -> list[str]:
    out: list[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        out.append(cur[0])
        stack.extend(reversed(cur[1:]))
    return out


def from_prefix(tokens: Sequence[str]) -> Node:
    """Reconstruit l'arbre. Lève ``ValueError`` si la séquence est mal formée."""
    pos = 0

    def build() -> Node:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("séquence préfixe incomplète")
        tok = tokens[pos]
        pos += 1
        if tok not in ARITY:
            raise ValueError(f"token inconnu : {tok!r}")
        children = tuple(build() for _ in range(ARITY[tok]))
        return (tok, *children)

    root = build()
    if pos != len(tokens):
        raise ValueError(f"{len(tokens) - pos} token(s) en trop après l'arbre")
    return root


def prefix_validity(tokens: Sequence[str]) -> int | None:
    """Nombre de sous-arbres encore attendus, ou ``None`` si la séquence est morte.

    ``0`` signifie « arbre complet ». C'est exactement la quantité dont le beam
    search a besoin pour savoir quels tokens sont légaux à l'étape suivante.
    """
    remaining = 1
    for tok in tokens:
        if tok not in ARITY or remaining == 0:
            return None
        remaining += ARITY[tok] - 1
    return remaining


def prefix_is_complete(tokens: Sequence[str]) -> bool:
    return prefix_validity(tokens) == 0


def iter_nodes(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(cur[1:])


def size(node: Node) -> int:
    return sum(1 for _ in iter_nodes(node))


def depth(node: Node) -> int:
    return 1 + max((depth(c) for c in node[1:]), default=0)


def count_constants(node: Node) -> int:
    return sum(1 for n in iter_nodes(node) if n[0] == "C")


def has_x(node: Node) -> bool:
    return any(n[0] == "x" for n in iter_nodes(node))


def complexity(node: Node) -> int:
    """Coût pondéré, pour l'axe « simplicité » du front de Pareto."""
    return sum(COMPLEXITY_COST[n[0]] for n in iter_nodes(node))


# --- rendu -------------------------------------------------------------------


def to_infix(node: Node, const_names: Iterator[str] | None = None) -> str:
    """Rendu infixe parsable par sympy.

    ``const_names`` fournit un nom distinct par occurrence de ``C`` (``c0``,
    ``c1``, …) : indispensable, puisque deux ``C`` du même squelette sont deux
    paramètres indépendants et non la même valeur.
    """
    tok = node[0]
    if tok == "x":
        return "x"
    if tok == "C":
        return next(const_names) if const_names is not None else "C"
    if tok in INFIX_SYMBOL:
        left = to_infix(node[1], const_names)
        right = to_infix(node[2], const_names)
        return f"({left} {INFIX_SYMBOL[tok]} {right})"
    return UNARY_RENDER[tok].format(to_infix(node[1], const_names))


def const_name_iter(prefix: str = "c") -> Iterator[str]:
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1


# --- évaluation --------------------------------------------------------------

#: Seuil sous lequel un dénominateur est considéré comme une singularité.
INV_EPS = 1e-3


def evaluate(node: Node, x: np.ndarray, consts: Sequence[float]) -> np.ndarray:
    """Évalue l'arbre sur ``x``, en consommant ``consts`` dans l'ordre préfixe.

    Aucune protection numérique : les domaines interdits produisent ``nan`` ou
    ``inf``, et c'est le filtre de dégénérescence qui décide. Masquer
    les singularités ici reviendrait à apprendre au modèle des fonctions qui
    n'existent pas.
    """
    idx = 0

    def go(n: Node) -> np.ndarray:
        nonlocal idx
        tok = n[0]
        if tok == "x":
            return x
        if tok == "C":
            if idx >= len(consts):
                raise ValueError("pas assez de constantes fournies")
            idx += 1
            return np.full_like(x, consts[idx - 1])
        if tok == "add":
            return go(n[1]) + go(n[2])
        if tok == "sub":
            return go(n[1]) - go(n[2])
        if tok == "mul":
            return go(n[1]) * go(n[2])
        u = go(n[1])
        if tok == "sin":
            return np.sin(u)
        if tok == "cos":
            return np.cos(u)
        if tok == "tanh":
            return np.tanh(u)
        if tok == "exp":
            return np.exp(u)
        if tok == "sq":
            return u * u
        if tok == "cube":
            return u * u * u
        if tok == "abs":
            return np.abs(u)
        if tok == "log":
            return np.where(u > 0, np.log(np.where(u > 0, u, 1.0)), np.nan)
        if tok == "sqrt":
            return np.where(u >= 0, np.sqrt(np.where(u >= 0, u, 0.0)), np.nan)
        if tok == "inv":
            return np.where(
                np.abs(u) > INV_EPS, 1.0 / np.where(np.abs(u) > INV_EPS, u, 1.0), np.nan
            )
        raise ValueError(f"token non évaluable : {tok!r}")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = go(node)
    if idx != len(consts):
        raise ValueError(f"{len(consts) - idx} constante(s) non consommée(s)")
    return np.asarray(y, dtype=np.float64)
