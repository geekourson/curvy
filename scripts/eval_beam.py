"""Phase 5 — que rapporte le beam search, et à quel prix ?

Compare le décodage glouton (un candidat) au beam search (``k`` candidats
ajustés puis départagés) sur les deux protocoles : interpolation et
extrapolation. Le candidat annoncé est choisi **sans jamais regarder les points
tenus à l'écart** — même règle que la validation croisée imposée à la baseline
polynomiale.

    .venv/bin/python scripts/eval_beam.py --run exp-003 --preset v1 --beams 1,4,8,16
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from curvy.config import DATASET_DIR, RUNS_DIR
from curvy.data.dataset import collate, make_validation_set
from curvy.data.expr import evaluate
from curvy.devices import pick_device
from curvy.infer.decode import beam_search, ids_to_node
from curvy.infer.fit import r_squared
from curvy.infer.pareto import ajuster_candidats, front_de_pareto, selectionner
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel
from curvy.seeding import make_rng, seed_everything

TOLS = (0.0, 0.002, 0.005, 0.02)


def _split(x: np.ndarray, mode: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    n_hold = max(3, int(round(0.2 * n)))
    if mode == "extrapolation":
        order = np.argsort(x)
        return order[:-n_hold], order[-n_hold:]
    idx = rng.permutation(n)
    return idx[n_hold:], idx[:n_hold]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--beams", default="1,4,8,16")
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    seed_everything(args.val_seed)
    choix = pick_device("auto")
    device = choix.device
    model = CurvyModel(PRESETS[args.preset]).to(device)
    state = torch.load(
        RUNS_DIR / args.run / args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["model"])
    model.eval()
    print(f"{args.run}/{args.checkpoint} — step {state.get('step', '?')} — {choix}")

    val = make_validation_set(
        DATASET_DIR / "skeletons-v1.jsonl.gz", args.val_size, seed=args.val_seed
    )
    beams = [int(b) for b in args.beams.split(",")]

    print(
        f"\n{'beam':>5} {'mode':>15} "
        + " ".join(f"tol={t:<7}" for t in TOLS)
        + f" {'candidats':>10} {'s/exemple':>10}"
    )
    resultats = {}
    for k in beams:
        # Décodage : une seule passe par valeur de beam, réutilisée par les deux protocoles.
        t0 = time.perf_counter()
        tous_candidats = []
        for start in range(0, len(val), args.batch_size):
            chunk = val[start : start + args.batch_size]
            batch = collate([(ex.points, ex.ids) for ex in chunk]).to(device)
            tous_candidats.extend(beam_search(model, batch.points, batch.point_mask, beam=k))
        t_decode = time.perf_counter() - t0

        for mode in ("interpolation", "extrapolation"):
            rng = make_rng(args.val_seed)
            succes = {t: [] for t in TOLS}
            n_cands, t_fit = [], time.perf_counter()
            for ex, cands in zip(val, tous_candidats, strict=True):
                keep, hold = _split(ex.x, mode, rng)
                nodes = [ids_to_node(seq) for seq, _ in cands]
                scores = [sc for _, sc in cands]
                ajustes = ajuster_candidats(nodes, ex.x[keep], ex.y[keep], rng, scores)
                n_cands.append(len(ajustes))
                for tol in TOLS:
                    c = selectionner(ajustes, tol=tol)
                    if c is None:
                        succes[tol].append(False)
                        continue
                    with np.errstate(all="ignore"):
                        pred = evaluate(c.node, ex.x[hold], c.consts)
                    succes[tol].append(r_squared(ex.y_clean[hold], pred) >= 0.99)
            dt = (time.perf_counter() - t_fit + t_decode) / len(val)
            taux = " ".join(f"{np.mean(succes[t]):<11.4f}" for t in TOLS)
            print(f"{k:>5} {mode:>15} {taux} {np.mean(n_cands):>10.1f} {dt:>10.3f}")
            resultats[(k, mode)] = {t: float(np.mean(succes[t])) for t in TOLS}

    print("\nRappels mesurés le 2026-08-19 (docs/benchmarks/results.md) :")
    print("   polynome, degre par validation croisee : interpolation 0.670, extrapolation 0.088")
    print("   vrai squelette (oracle)                : interpolation 0.803, extrapolation 0.430")

    # Exemple de front de Pareto, pour montrer ce que le produit livrerait.
    ex = val[0]
    keep, _ = _split(ex.x, "interpolation", make_rng(args.val_seed))
    cands = ajuster_candidats(
        [ids_to_node(s) for s, _ in tous_candidats[0]], ex.x[keep], ex.y[keep], make_rng(1)
    )
    from curvy.data.expr import to_infix

    print(f"\nExemple de front de Pareto (exemple 0, beam {beams[-1]}) :")
    for c in front_de_pareto(cands):
        print(f"   complexite {c.complexite:>3}  R2_ajust {c.r2_fit:>8.4f}  {to_infix(c.node)}")
    print(f"   vraie formule                          : {to_infix(ex.node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
