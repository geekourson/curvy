"""Évalue un checkpoint déjà entraîné, sans reprendre l'entraînement.

Sert à mesurer un run terminé avec des métriques ajoutées **après** lui — par
exemple l'extrapolation, ajoutée le 2026-08-19 alors qu'exp-001 et exp-002
étaient déjà finis.

    .venv/bin/python scripts/eval_checkpoint.py --run exp-002 --preset v1
"""

from __future__ import annotations

import argparse
import json

import torch

from curvy.config import DATASET_DIR, RUNS_DIR
from curvy.data.dataset import make_validation_set
from curvy.devices import pick_device
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel, count_parameters
from curvy.seeding import make_rng, seed_everything
from curvy.train.metrics import evaluate_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    seed_everything(args.val_seed)
    path = RUNS_DIR / args.run / args.checkpoint
    choix = pick_device("auto")
    device = choix.device
    model = CurvyModel(PRESETS[args.preset]).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    step = state.get("step", "?")
    total = count_parameters(model)["TOTAL"]
    print(f"{args.run}/{args.checkpoint} — step {step}, {total:,} paramètres")
    print(f"device : {choix}")

    val = make_validation_set(
        DATASET_DIR / "skeletons-v1.jsonl.gz", args.val_size, seed=args.val_seed
    )
    rep = evaluate_model(model, val, device, make_rng(args.val_seed), batch_size=args.batch_size)
    d = rep.as_dict()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print()
    print("Lecture :")
    print(f"  interpolation : modèle {d['r2_rate']:.4f}  vs oracle {d['r2_rate_oracle']:.4f}")
    print(
        f"  extrapolation : modèle {d['r2_rate_extrap']:.4f}  vs oracle {d['r2_rate_extrap_oracle']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
