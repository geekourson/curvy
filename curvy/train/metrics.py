"""Métriques d'entraînement et d'évaluation.

Trois familles, jamais agrégées entre elles :

- **token** : accuracy en teacher forcing. Courbe lisse, utile en continu,
  ne dit presque rien sur la qualité réelle ;
- **séquence** : le squelette décodé est-il exactement celui attendu. Diagnostic
  honnête mais pessimiste — plusieurs squelettes décrivent le même nuage ;
- **R²** : la courbe proposée retrouve-t-elle la fonction ? C'est la métrique
  principale. Les constantes sont ajustées sur 80 % des points **observés**
  (bruités), le R² est mesuré sur les 20 % restants contre la valeur **exacte**
  de la fonction génératrice.

Chaque évaluation rapporte l'**oracle** : la même procédure appliquée au vrai
squelette. Sans lui, un taux de 55 % est illisible.

Le R² est mesuré **deux fois**, avec deux façons de tenir des points à l'écart :

- **interpolation** — 20 % de points tirés au hasard. On juge le remplissage
  entre les points observés ;
- **extrapolation** — les 20 % d'abscisses les plus à droite. On juge la
  prédiction *au-delà* de ce qui a été vu.

Le second n'est pas un raffinement : c'est le seul endroit où une formule bat
franchement un polynôme ajusté. Mesuré le 2026-08-19 sur le vrai squelette,
0,785 en interpolation contre 0,428 en extrapolation ; un polynôme à degré
honnête tombe lui de 0,670 à **0,088**. Ne mesurer qu'en interpolation, c'est
se comparer là où on est le plus faible (cf. `docs/benchmarks/results.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from curvy.data.expr import Node, evaluate, to_prefix
from curvy.infer.decode import greedy_decode, ids_to_node
from curvy.infer.fit import fit_constants, r_squared
from curvy.tokenizer.vocab import PAD_ID

__all__ = ["EvalReport", "evaluate_model", "token_accuracy"]

R2_THRESHOLD = 0.99


def token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """(accuracy token, accuracy séquence en teacher forcing)."""
    pred = logits.argmax(dim=-1)
    valid = target != PAD_ID
    correct = (pred == target) & valid
    tok = float(correct.sum()) / max(1, int(valid.sum()))
    seq_ok = ((pred == target) | ~valid).all(dim=1)
    return tok, float(seq_ok.float().mean())


@dataclass
class EvalReport:
    n: int = 0
    token_acc: float = 0.0
    seq_acc_teacher: float = 0.0
    seq_acc_greedy: float = 0.0
    r2_rate: float = 0.0
    r2_median: float = 0.0
    r2_rate_oracle: float = 0.0
    r2_rate_extrap: float = 0.0
    r2_rate_extrap_oracle: float = 0.0
    invalid_rate: float = 0.0
    fit_failed_rate: float = 0.0
    per_depth: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_depth"}
        d["per_depth"] = {str(k): round(v, 4) for k, v in sorted(self.per_depth.items())}
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _r2_holdout(
    node: Node,
    x: np.ndarray,
    y_obs: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    holdout: float = 0.2,
    mode: str = "interpolation",
) -> tuple[float, bool]:
    """Ajuste sur 80 % des points observés, mesure sur 20 % contre la vérité.

    ``mode="interpolation"`` tire les points tenus à l'écart au hasard.
    ``mode="extrapolation"`` tient à l'écart les abscisses les plus à droite :
    on ajuste sur la partie gauche de la courbe et on doit prédire la suite.
    """
    n = len(x)
    n_hold = max(3, int(round(holdout * n)))
    if mode == "extrapolation":
        order = np.argsort(x)
        hold, keep = order[-n_hold:], order[:-n_hold]
    else:
        idx = rng.permutation(n)
        hold, keep = idx[:n_hold], idx[n_hold:]
    res = fit_constants(node, x[keep], y_obs[keep], rng)
    if not res.ok:
        return float("-inf"), True
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pred = evaluate(node, x[hold], res.consts)
    return r_squared(truth[hold], pred), False


@torch.no_grad()
def evaluate_model(model, examples, device, rng, batch_size: int = 128) -> EvalReport:
    """``examples`` : liste de ``ValidationExample`` (cf. curvy.data.dataset)."""
    from curvy.data.dataset import collate

    model.eval()
    rep = EvalReport(n=len(examples))
    tok_num = tok_den = 0
    seq_teacher = seq_greedy = 0
    invalid = fit_failed = 0
    r2s: list[float] = []
    r2s_oracle: list[float] = []
    r2s_extrap: list[float] = []
    r2s_extrap_oracle: list[float] = []
    by_depth: dict[int, list[bool]] = {}

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate([(ex.points, ex.ids) for ex in chunk]).to(device)

        logits = model(
            batch.points, batch.point_mask, batch.tokens[:, :-1], batch.token_mask[:, :-1]
        )
        target = batch.tokens[:, 1:]
        pred = logits.argmax(dim=-1)
        valid = target != PAD_ID
        tok_num += int(((pred == target) & valid).sum())
        tok_den += int(valid.sum())
        seq_teacher += int(((pred == target) | ~valid).all(dim=1).sum())

        decoded = greedy_decode(model, batch.points, batch.point_mask)
        for ex, ids in zip(chunk, decoded, strict=True):
            node = ids_to_node(ids)
            if node is None:
                invalid += 1
                r2s.append(float("-inf"))
                r2s_extrap.append(float("-inf"))
                by_depth.setdefault(ex.depth, []).append(False)
                continue
            if to_prefix(node) == to_prefix(ex.node):
                seq_greedy += 1
            r2, failed = _r2_holdout(node, ex.x, ex.y, ex.y_clean, rng)
            fit_failed += failed
            r2s.append(r2)
            by_depth.setdefault(ex.depth, []).append(r2 >= R2_THRESHOLD)
            r2_or, _ = _r2_holdout(ex.node, ex.x, ex.y, ex.y_clean, rng)
            r2s_oracle.append(r2_or)

            # Extrapolation : mêmes candidats, mais les points tenus à l'écart
            # sont les abscisses les plus à droite. C'est là que la structure
            # rapporte et qu'un polynôme s'effondre (cf. en-tête de module).
            r2_ex, _ = _r2_holdout(node, ex.x, ex.y, ex.y_clean, rng, mode="extrapolation")
            r2s_extrap.append(r2_ex)
            r2_ex_or, _ = _r2_holdout(ex.node, ex.x, ex.y, ex.y_clean, rng, mode="extrapolation")
            r2s_extrap_oracle.append(r2_ex_or)

    arr = np.array(r2s)
    rep.token_acc = tok_num / max(1, tok_den)
    rep.seq_acc_teacher = seq_teacher / rep.n
    rep.seq_acc_greedy = seq_greedy / rep.n
    rep.r2_rate = float((arr >= R2_THRESHOLD).mean())
    rep.r2_median = float(np.median(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else -1.0
    rep.r2_rate_oracle = float((np.array(r2s_oracle) >= R2_THRESHOLD).mean()) if r2s_oracle else 0.0
    rep.r2_rate_extrap = float((np.array(r2s_extrap) >= R2_THRESHOLD).mean()) if r2s_extrap else 0.0
    rep.r2_rate_extrap_oracle = (
        float((np.array(r2s_extrap_oracle) >= R2_THRESHOLD).mean()) if r2s_extrap_oracle else 0.0
    )
    rep.invalid_rate = invalid / rep.n
    rep.fit_failed_rate = fit_failed / rep.n
    rep.per_depth = {d: float(np.mean(v)) for d, v in by_depth.items()}
    model.train()
    return rep
