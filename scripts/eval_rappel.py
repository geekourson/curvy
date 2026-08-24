"""De quoi l'écart au plafond est-il fait ? (question de Billy, 2026-08-20)

Le modèle rend 8 candidats et on en retient un. Quand le résultat échoue, deux
causes possibles, et elles appellent des remèdes opposés :

- **aucun des 8 candidats n'était bon** → le modèle ne sait pas proposer. C'est
  un problème de capacité ou de recherche, et plus de paramètres aideraient ;
- **un candidat était bon mais la sélection l'a écarté** → agrandir le modèle ne
  servirait à rien, il faut corriger le choix.

On mesure donc trois chiffres sur le même jeu :

- **retenu** — ce que le produit rend vraiment (sans regarder la
  réponse) ;
- **rappel@k** — au moins un des k candidats aurait réussi. C'est le plafond de
  la sélection, atteignable sans toucher au modèle ;
- **oracle** — le vrai squelette. Le plafond de la tâche.

    .venv/bin/python scripts/eval_rappel.py --run exp-005 --preset v1
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
from curvy.infer.pareto import ajuster_candidats, selectionner
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel
from curvy.seeding import make_rng

SEUIL = 0.99


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

    candidats = []
    for start in range(0, len(val), args.batch_size):
        chunk = val[start : start + args.batch_size]
        batch = collate([(ex.points, ex.ids) for ex in chunk]).to(choix.device)
        candidats.extend(beam_search(modele, batch.points, batch.point_mask, beam=args.beam))

    rng = make_rng(args.val_seed)
    par_prof: dict[int, dict[str, list[bool]]] = defaultdict(
        lambda: {"retenu": [], "rappel": [], "oracle": [], "exact": []}
    )
    for ex, cands in zip(val, candidats, strict=True):
        n = len(ex.x)
        n_hold = max(3, int(round(0.2 * n)))
        idx = rng.permutation(n)
        hold, keep = idx[:n_hold], idx[n_hold:]

        noeuds = [ids_to_node(s) for s, _ in cands]
        ajustes = ajuster_candidats(noeuds, ex.x[keep], ex.y[keep], rng)

        def score(c, ex=ex, hold=hold):  # liés tôt : B023
            with np.errstate(all="ignore"):
                return r_squared(ex.y_clean[hold], evaluate(c.node, ex.x[hold], c.consts))

        retenu = selectionner(ajustes)
        d = par_prof[ex.depth]
        d["retenu"].append(retenu is not None and score(retenu) >= SEUIL)
        d["rappel"].append(any(score(c) >= SEUIL for c in ajustes))
        res = fit_constants(ex.node, ex.x[keep], ex.y[keep], rng)
        ok_or = False
        if res.ok:
            with np.errstate(all="ignore"):
                ok_or = (
                    r_squared(ex.y_clean[hold], evaluate(ex.node, ex.x[hold], res.consts)) >= SEUIL
                )
        d["oracle"].append(ok_or)
        from curvy.data.expr import to_prefix

        vrai = to_prefix(ex.node)
        d["exact"].append(any(n_ is not None and to_prefix(n_) == vrai for n_ in noeuds))

    print(
        f"\n{'prof':>5} {'n':>5} {'retenu':>8} {'rappel@' + str(args.beam):>10} "
        f"{'oracle':>8} {'vrai squelette dans le beam':>28}"
    )
    tot = defaultdict(list)
    for d in sorted(par_prof):
        v = par_prof[d]
        for k in v:
            tot[k] += v[k]
        print(
            f"{d:>5} {len(v['retenu']):>5} {np.mean(v['retenu']):>8.3f} "
            f"{np.mean(v['rappel']):>10.3f} {np.mean(v['oracle']):>8.3f} {np.mean(v['exact']):>28.3f}"
        )
    print(
        f"{'tous':>5} {len(tot['retenu']):>5} {np.mean(tot['retenu']):>8.3f} "
        f"{np.mean(tot['rappel']):>10.3f} {np.mean(tot['oracle']):>8.3f} {np.mean(tot['exact']):>28.3f}"
    )

    retenu, rappel, oracle = np.mean(tot["retenu"]), np.mean(tot["rappel"]), np.mean(tot["oracle"])
    print(f"\nDécomposition de l'écart au plafond ({100 * (oracle - retenu):.1f} points) :")
    print(
        f"  perdu à la SÉLECTION      : {100 * (rappel - retenu):>5.1f} pts "
        f"— un bon candidat était là, on ne l'a pas pris"
    )
    print(
        f"  perdu à la PROPOSITION    : {100 * (oracle - rappel):>5.1f} pts "
        f"— aucun des {args.beam} candidats n'était bon"
    )
    print("\nLe premier poste se corrige sans toucher au modèle.")
    print("Le second est le seul que plus de paramètres pourrait réduire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
