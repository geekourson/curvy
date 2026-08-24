"""Canonicalisation des squelettes.

Objectif : que deux écritures d'une même famille de fonctions donnent le même
arbre, pour ne pas apprendre plusieurs fois la même chose.

Pourquoi pas sympy — la raison est importante et vaut d'être répétée : dans
notre algèbre, deux ``C`` du même squelette sont **deux paramètres
indépendants**. sympy, qui les voit comme un même symbole, « simplifierait »
``C*x + C*x`` en ``2*C*x``, ce qui est faux ici. Il faut donc nos propres
règles. Elles ont aussi l'avantage d'être des dizaines de fois plus rapides,
ce qui compte : la canonicalisation tourne sur des millions d'arbres.

Règles appliquées de bas en haut jusqu'au point fixe :

1. tout sous-arbre sans ``x`` se replie sur ``C`` (couvre ``unaire(C) → C`` et
   ``C op C → C``) ;
2. ``sub(u, u) → C`` (nul, donc constant) ;
3. involutions et annulations : ``inv(inv u) → u``, ``exp(log u) → u``,
   ``log(exp u) → u``, ``abs(abs u) → abs u``, ``abs(sq u) → sq u``,
   ``sq(abs u) → sq u``, ``sqrt(sq u) → abs u`` ;
4. ``sub(u, C) → add(u, C)`` — ``C`` est libre et de signe quelconque ;
5. dans une chaîne de ``mul`` (resp. ``add``), au plus un facteur (resp. terme)
   constant : ``mul(C, mul(C, u)) → mul(C, u)`` ;
6. termes semblables d'une somme fusionnés **si l'un au moins porte un facteur
   constant libre** : ``C*u + C*u → C*u`` et ``C*u + u → C*u``, puisque la
   somme de deux constantes libres est une constante libre. Attention à la
   symétrique fausse : ``u + u`` vaut ``2u``, qui est une fonction **fixe** et
   non une famille — le fusionner en ``C*u`` ajouterait un paramètre libre et
   changerait la classe de fonctions. ``sin(x + x)`` est ``sin(2x)``, ce n'est
   pas ``sin(C*x)`` ;
7. arguments des opérateurs commutatifs triés selon un ordre total.

L'enveloppe de racine ``C * (…) + C`` est posée en dernier, après
avoir retiré du sommet du corps tout ce qu'elle absorbe déjà.
"""

from __future__ import annotations

from curvy.data.expr import Node, has_x, size, to_prefix

__all__ = ["canonicalise", "canonical_key", "strip_absorbable_root", "wrap_root"]

C: Node = ("C",)

_INVOLUTIONS = {
    ("inv", "inv"): lambda inner: inner,
    ("exp", "log"): lambda inner: inner,
    ("log", "exp"): lambda inner: inner,
    ("abs", "abs"): lambda inner: ("abs", inner),
    ("abs", "sq"): lambda inner: ("sq", inner),
    ("sq", "abs"): lambda inner: ("sq", inner),
    ("sqrt", "sq"): lambda inner: ("abs", inner),
}


def _sort_key(node: Node) -> tuple:
    """Ordre total sur les sous-arbres : d'abord la taille, puis la séquence."""
    return (size(node), tuple(to_prefix(node)))


def _flatten(node: Node, op: str) -> list[Node]:
    """Aplatit une chaîne d'opérateurs associatifs identiques."""
    out: list[Node] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur[0] == op:
            stack.extend(cur[1:])
        else:
            out.append(cur)
    return out


def _rebuild(op: str, parts: list[Node]) -> Node:
    node = parts[0]
    for p in parts[1:]:
        node = (op, node, p)
    return node


def _strip_free_scale(term: Node) -> tuple[bool, Node]:
    """Sépare un éventuel facteur constant libre du reste du terme.

    ``mul(C, sin(x))`` -> ``(True, sin(x))`` ; ``sin(x)`` -> ``(False, sin(x))``.
    """
    if term[0] != "mul":
        return False, term
    factors = _flatten(term, "mul")
    if C not in factors:
        return False, term
    rest = [f for f in factors if f != C]
    if not rest:
        return True, C
    return True, _rebuild("mul", sorted(rest, key=_sort_key))


def _merge_like_terms(parts: list[Node]) -> list[Node]:
    """Fusionne les termes semblables d'une somme (règle 6)."""
    groups: dict[Node, list[bool]] = {}
    for term in parts:
        scaled, base = _strip_free_scale(term)
        groups.setdefault(base, []).append(scaled)
    out: list[Node] = []
    for base, scales in groups.items():
        if len(scales) == 1:
            out.append(("mul", C, base) if scales[0] else base)
        elif any(scales):
            # Au moins une constante libre dans le groupe : elle absorbe tout.
            out.append(("mul", C, base))
        else:
            # Aucune constante libre : `u + u` vaut `2u`, non fusionnable.
            out.extend([base] * len(scales))
    return out


def canonicalise(node: Node) -> Node:
    """Forme canonique d'un sous-arbre. Idempotente."""
    tok = node[0]
    if tok in ("x", "C"):
        return node

    children = tuple(canonicalise(c) for c in node[1:])
    node = (tok, *children)

    # 1. Tout sous-arbre sans x est une constante, quelle que soit sa forme.
    if not has_x(node):
        return C

    # 3. Involutions et annulations.
    if len(children) == 1:
        inner = children[0]
        rule = _INVOLUTIONS.get((tok, inner[0]))
        if rule is not None:
            return canonicalise(rule(inner[1]))

    if tok == "sub":
        left, right = children
        # 2. u - u = 0
        if left == right:
            return C
        # 4. u - C ≡ u + C (C est libre et de signe quelconque)
        if right == C:
            return canonicalise(("add", left, C))

    if tok in ("add", "mul"):
        parts = _flatten(node, tok)
        # 5. Au plus un élément constant dans la chaîne.
        consts = [p for p in parts if p == C]
        parts = [p for p in parts if p != C]
        if consts:
            parts.append(C)
        if tok == "add":
            # 6. Termes semblables (hors le terme constant isolé, déjà traité).
            merged = _merge_like_terms([p for p in parts if p != C])
            parts = merged + ([C] if C in parts else [])
        if len(parts) == 1:
            return parts[0]
        # 7. Ordre total sur les arguments commutatifs.
        parts.sort(key=_sort_key)
        return _rebuild(tok, parts)

    return node


def strip_absorbable_root(body: Node) -> Node:
    """Retire du sommet du corps ce que l'enveloppe de racine absorbe déjà.

    L'enveloppe ``C * (…) + C`` rend redondants, **au sommet du corps
    uniquement**, un facteur constant, un terme constant, et même un
    ``sub(C, u)`` — le signe étant absorbé par le ``C`` multiplicatif.
    Plus bas dans l'arbre ces formes sont significatives et sont conservées.
    """
    while True:
        tok = body[0]
        if tok in ("add", "mul") and C in body[1:]:
            body = body[1] if body[2] == C else body[2]
            continue
        if tok == "sub" and (body[1] == C or body[2] == C):
            body = body[2] if body[1] == C else body[1]
            continue
        return body


def wrap_root(body: Node) -> Node:
    """Enveloppe canonique ``C * body + C``."""
    return ("add", ("mul", C, body), C)


def canonical_key(node: Node) -> str:
    """Clé de déduplication exacte."""
    return " ".join(to_prefix(node))
