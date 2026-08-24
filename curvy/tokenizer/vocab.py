"""Encodage des squelettes en séquences d'identifiants, et masque d'arité.

Le vocabulaire tient en 18 tokens. Aucun littéral numérique n'y
figure : le modèle ne prédit jamais de valeur, seulement une structure
.

La pièce importante de ce module n'est pas ``encode``/``decode`` mais
``legal_mask``. En notation préfixe, un simple compteur d'arité dit
à chaque pas quels tokens peuvent encore mener à un arbre complet. En masquant
les logits des autres pendant le beam search, on obtient une garantie et non
une espérance : **le taux de sorties syntaxiquement invalides doit être
exactement 0 %**. Si la mesure de la Phase 5 dit autre chose, le bug est ici.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from curvy.data.expr import Node, from_prefix, to_prefix
from curvy.data.grammar import (
    ARITY,
    BOS,
    EOS,
    ID_TO_TOKEN,
    MAX_CONSTANTS,
    PAD,
    TOKEN_TO_ID,
    VOCAB,
)

PAD_ID, BOS_ID, EOS_ID = TOKEN_TO_ID[PAD], TOKEN_TO_ID[BOS], TOKEN_TO_ID[EOS]
VOCAB_SIZE = len(VOCAB)

#: Arité par identifiant ; -1 pour les tokens spéciaux, qui n'en ont pas.
ARITY_BY_ID = np.array([ARITY.get(tok, -1) for tok in VOCAB], dtype=np.int64)
IS_OPERAND = ARITY_BY_ID >= 0
CONST_ID = TOKEN_TO_ID["C"]

#: Longueur maximale d'une séquence, `<bos>` et `<eos>` compris.
#: Un arbre de profondeur 8 pourrait en théorie compter 255 nœuds, mais la
#: canonicalisation et le budget de constantes ramènent le maximum très en
#: deçà. **Mesuré** sur les 255 080 squelettes du dataset v1 : maximum 43,
#: médiane 17, p99 à 29 — aucune séquence au-delà de 48. On fixe 48 plutôt
#: qu'une puissance de deux confortable : le coût de l'attention du décodeur
#: est quadratique en cette longueur.
MAX_SEQ_LEN = 48

__all__ = [
    "BOS_ID",
    "CONST_ID",
    "EOS_ID",
    "MAX_SEQ_LEN",
    "PAD_ID",
    "VOCAB_SIZE",
    "DecodeState",
    "decode",
    "encode",
    "legal_mask",
]


def encode(skeleton: Node, add_special: bool = True) -> list[int]:
    ids = [TOKEN_TO_ID[t] for t in to_prefix(skeleton)]
    return [BOS_ID, *ids, EOS_ID] if add_special else ids


def decode(ids: Sequence[int]) -> Node:
    """Reconstruit l'arbre. Ignore ``<bos>``/``<pad>``, s'arrête à ``<eos>``."""
    toks: list[str] = []
    for i in ids:
        tok = ID_TO_TOKEN[int(i)]
        if tok == EOS:
            break
        if tok in (BOS, PAD):
            continue
        toks.append(tok)
    return from_prefix(toks)


class DecodeState:
    """Suit ce qu'une séquence préfixe partielle autorise encore.

    ``remaining`` est le nombre de sous-arbres encore attendus. Il vaut 1 au
    début, 0 quand l'arbre est complet, et jamais négatif tant qu'on respecte
    le masque.
    """

    __slots__ = ("remaining", "n_consts", "n_emitted")

    def __init__(self) -> None:
        self.remaining = 1
        self.n_consts = 0
        self.n_emitted = 0

    def copy(self) -> DecodeState:
        s = DecodeState()
        s.remaining, s.n_consts, s.n_emitted = self.remaining, self.n_consts, self.n_emitted
        return s

    def advance(self, token_id: int) -> DecodeState:
        tok = ID_TO_TOKEN[int(token_id)]
        if tok in (BOS, PAD, EOS):
            return self
        self.remaining += ARITY[tok] - 1
        self.n_consts += tok == "C"
        self.n_emitted += 1
        return self

    @property
    def complete(self) -> bool:
        return self.remaining == 0


def legal_mask(
    state: DecodeState, max_len: int = MAX_SEQ_LEN, max_consts: int = MAX_CONSTANTS
) -> np.ndarray:
    """Masque booléen des tokens légaux au pas suivant.

    Trois contraintes, toutes nécessaires :

    1. **complétude** — ``<eos>`` est interdit tant que l'arbre n'est pas
       complet, et *seul* ``<eos>`` est permis une fois qu'il l'est ;
    2. **budget de longueur** — un token n'est légal que s'il reste assez de
       place pour fermer tous les sous-arbres qu'il ouvre (au minimum une
       feuille par sous-arbre restant) ;
    3. **budget de constantes** — ``C`` disparaît une fois le quota atteint
      , sinon le problème d'ajustement de la Phase 5 devient
       arbitrairement dur.
    """
    mask = np.zeros(VOCAB_SIZE, dtype=bool)

    if state.complete:
        mask[EOS_ID] = True
        return mask

    # `budget` = tokens d'arbre encore émettables avant d'atteindre max_len,
    # en réservant une place pour <eos> (et une pour le <bos> déjà émis).
    budget = max_len - 2 - state.n_emitted
    if budget <= 0:
        return mask  # séquence morte : le beam search doit écarter ce faisceau

    for tid in np.flatnonzero(IS_OPERAND):
        new_remaining = state.remaining + ARITY_BY_ID[tid] - 1
        if new_remaining > budget - 1:
            continue  # impossible de refermer l'arbre dans le budget
        if tid == CONST_ID and state.n_consts >= max_consts:
            continue
        mask[tid] = True
    return mask
