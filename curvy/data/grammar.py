"""Grammaire v1 des expressions.

Un squelette est un arbre d'expression où les constantes numériques sont
remplacées par un placeholder unique ``C``. Le modèle ne prédit que cette
structure ; les valeurs sont trouvées après coup par optimisation.

Le vocabulaire est délibérément minuscule — 18 tokens — ce qui garde les
séquences courtes et le décodeur petit.
"""

from __future__ import annotations

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL = (PAD, BOS, EOS)

BINARY = ("add", "sub", "mul")
UNARY = ("sin", "cos", "exp", "log", "sqrt", "abs", "tanh", "sq", "cube", "inv")
LEAVES = ("x", "C")

OPERATORS = BINARY + UNARY
VOCAB: tuple[str, ...] = SPECIAL + BINARY + UNARY + LEAVES

TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = dict(enumerate(VOCAB))

ARITY: dict[str, int] = (
    {tok: 2 for tok in BINARY} | {tok: 1 for tok in UNARY} | {tok: 0 for tok in LEAVES}
)

#: Profondeur maximale du *corps* de l'expression, hors enveloppe de racine.
MAX_BODY_DEPTH = 6
#: Constantes libres autorisées dans le corps. Les 2 de l'enveloppe s'ajoutent.
MAX_BODY_CONSTANTS = 5
MAX_CONSTANTS = MAX_BODY_CONSTANTS + 2

#: Coût de complexité par token, pour le front de Pareto (Phase 5).
#: Une composition transcendante coûte plus cher qu'une addition : entre deux
#: candidats de même R², on veut proposer le plus lisible.
COMPLEXITY_COST: dict[str, int] = {
    "x": 1,
    "C": 1,
    "add": 1,
    "sub": 1,
    "mul": 2,
    "sq": 2,
    "cube": 3,
    "abs": 2,
    "sqrt": 3,
    "inv": 3,
    "tanh": 4,
    "sin": 4,
    "cos": 4,
    "exp": 4,
    "log": 4,
}

#: Rendu infixe, pour l'affichage et le passage à sympy.
INFIX_SYMBOL = {"add": "+", "sub": "-", "mul": "*"}
UNARY_RENDER = {
    "sin": "sin({0})",
    "cos": "cos({0})",
    "exp": "exp({0})",
    "log": "log({0})",
    "sqrt": "sqrt({0})",
    "abs": "Abs({0})",
    "tanh": "tanh({0})",
    "sq": "({0})**2",
    "cube": "({0})**3",
    "inv": "1/({0})",
}

assert set(COMPLEXITY_COST) == set(OPERATORS) | set(LEAVES)
assert set(UNARY_RENDER) == set(UNARY)
