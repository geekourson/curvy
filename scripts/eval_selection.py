"""Quelle règle de sélection récupère les points perdus ? (2026-08-20)

Mesuré la veille : sur les 10,5 points qui séparent le produit de l'oracle,
**4,9 sont perdus à la sélection** — un bon candidat figurait parmi les huit et
n'a pas été retenu. Ce poste se corrige sans toucher au modèle.

Le script ajuste les candidats **une seule fois** par exemple, puis applique
toutes les règles au même jeu ajusté : leur comparaison est donc exacte, à
candidats et à ajustements identiques, et non pas approchée d'un run à l'autre.

Les règles en lice, et pourquoi :

- ``max_r2`` — l'actuelle. Maximum du R² d'ajustement ;
- ``validation_croisee`` — on impose la validation croisée au polynôme pour
  choisir son degré, et on s'autorise le maximum brut pour nos candidats. La
  même règle des deux côtés supprime l'asymétrie ;
- ``penalite_complexite`` — critère d'information : le R² payé au prix du
  nombre de constantes ;
- ``modele`` — la log-vraisemblance du beam, c'est-à-dire l'avis du réseau,
  jamais utilisé jusqu'ici ;
- ``modele_puis_r2`` — le réseau départage les candidats proches en R².

    .venv/bin/python scripts/eval_selection.py --run exp-005 --preset v1
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from curvy.config import DATASET_DIR, RUNS_DIR
from curvy.data.dataset import collate, make_validation_set
from curvy.data.expr import evaluate
from curvy.devices import pick_device
from curvy.infer.decode import beam_search, ids_to_node
from curvy.infer.fit import fit_constants, r_squared
from curvy.infer.pareto import Candidat, ajuster_candidats, selectionner_selon_bruit
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel
from curvy.seeding import make_rng

SEUIL = 0.99

#: Marges essayées autour du bruit estimé. Plusieurs valeurs plutôt qu'une :
#: un résultat négatif doit distinguer « l'idée est mauvaise » de « l'idée est
#: bonne mais mal réglée ».
MARGES_BRUIT = (0.0, 0.5, 1.0, 2.0)

NOMS_REGLES = [
    "max_r2",
    "penalite_complexite",
    "modele",
    "modele_puis_r2",
    "validation_croisee",
    *[f"bruit_marge_{m}" for m in MARGES_BRUIT],
]


def _cv_score(c: Candidat, x: np.ndarray, y: np.ndarray, rng, k: int = 4) -> float:
    """R² d'un candidat sur des points d'ajustement tenus à l'écart, en k plis.

    Ne regarde jamais les points d'évaluation : c'est la règle imposée au
    polynôme, appliquée à nous.
    """
    n = len(x)
    if n < 4 * k:
        return c.r2_fit
    idx = rng.permutation(n)
    plis = np.array_split(idx, k)
    scores = []
    for pli in plis:
        tr = np.setdiff1d(idx, pli)
        res = fit_constants(c.node, x[tr], y[tr], rng)
        if not res.ok:
            continue
        with np.errstate(all="ignore"):
            scores.append(r_squared(y[pli], evaluate(c.node, x[pli], res.consts)))
    return float(np.mean(scores)) if scores else float("-inf")


def regles(cands: list[Candidat], x, y, rng) -> dict[str, Candidat | None]:
    """Un candidat retenu par règle, tous jugés sur le même jeu ajusté."""
    if not cands:
        return {}
    out: dict[str, Candidat | None] = {}
    out["max_r2"] = min(
        (c for c in cands if c.r2_fit >= max(k.r2_fit for k in cands)),
        key=lambda c: c.complexite,
    )
    out["penalite_complexite"] = max(cands, key=lambda c: c.r2_fit - 0.002 * len(c.consts))
    out["modele"] = max(cands, key=lambda c: c.score_modele)
    meilleur = max(c.r2_fit for c in cands)
    proches = [c for c in cands if c.r2_fit >= meilleur - 0.01]
    out["modele_puis_r2"] = max(proches, key=lambda c: c.score_modele)
    cv = {id(c): _cv_score(c, x, y, rng) for c in cands}
    out["validation_croisee"] = max(cands, key=lambda c: cv[id(c)])
    for marge in MARGES_BRUIT:
        out[f"bruit_marge_{marge}"] = selectionner_selon_bruit(cands, x, y, marge=marge)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    choix = pick_device("auto")
    modele = CurvyModel(PRESETS[args.preset]).to(choix.device)
    etat = torch.load(RUNS_DIR / args.run / args.checkpoint, map_location="cpu", weights_only=False)
    modele.load_state_dict(etat["model"])
    modele.eval()
    print(f"{args.run}/{args.checkpoint} step {etat.get('step', '?')} — {choix}")

    val = make_validation_set(
        DATASET_DIR / "skeletons-v1.jsonl.gz", args.val_size, seed=args.val_seed
    )
    cands_par_ex = []
    for start in range(0, len(val), args.batch_size):
        chunk = val[start : start + args.batch_size]
        batch = collate([(ex.points, ex.ids) for ex in chunk]).to(choix.device)
        cands_par_ex.extend(beam_search(modele, batch.points, batch.point_mask, beam=args.beam))

    rng = make_rng(args.val_seed)
    reussites: dict[str, list[bool]] = defaultdict(list)
    for ex, cands in zip(val, cands_par_ex, strict=True):
        n = len(ex.x)
        nh = max(3, int(round(0.2 * n)))
        idx = rng.permutation(n)
        hold, keep = idx[:nh], idx[nh:]
        noeuds = [ids_to_node(s) for s, _ in cands]
        scores = [sc for _, sc in cands]
        ajustes = ajuster_candidats(noeuds, ex.x[keep], ex.y[keep], rng, scores)

        def note(c, ex=ex, hold=hold):
            if c is None:
                return False
            with np.errstate(all="ignore"):
                return r_squared(ex.y_clean[hold], evaluate(c.node, ex.x[hold], c.consts)) >= SEUIL

        choisis = regles(ajustes, ex.x[keep], ex.y[keep], rng)
        for nom, c in choisis.items():
            reussites[nom].append(note(c))
        if not choisis:
            for nom in NOMS_REGLES:
                reussites[nom].append(False)
        reussites["rappel@k"].append(any(note(c) for c in ajustes))

    base = np.mean(reussites["max_r2"])
    print(f"\n{'règle':>22} {'taux':>8} {'écart':>8}")
    for nom in [*NOMS_REGLES, "rappel@k"]:
        v = np.mean(reussites[nom])
        marque = (
            "   <- actuelle"
            if nom == "max_r2"
            else ("   <- plafond atteignable" if nom == "rappel@k" else "")
        )
        print(f"{nom:>22} {v:>8.4f} {100 * (v - base):>+7.1f}{marque}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
