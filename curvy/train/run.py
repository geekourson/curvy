"""``python -m curvy.train.run`` — lance ou reprend un entraînement.

La configuration est assemblée **par recoupement** entre les arguments de la
ligne de commande et les champs de ``TrainConfig``, et non par une liste
recopiée à la main. La liste recopiée a déjà coûté un run : ``--exclure-test``
existait, était accepté sans erreur, et n'arrivait jamais jusqu'à la config —
l'entraînement voyait donc les squelettes du jeu de test (2026-08-20). Un
drapeau qui ne fait rien en silence est pire qu'un drapeau absent.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from curvy.train.config import TrainConfig
from curvy.train.loop import Trainer


def main(argv: list[str] | None = None) -> int:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default=cfg.run_name)
    ap.add_argument("--preset", default=cfg.preset)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--steps", type=int, default=cfg.steps)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps)
    ap.add_argument("--workers", type=int, default=cfg.workers)
    ap.add_argument("--eval-every", type=int, default=cfg.eval_every)
    ap.add_argument("--log-every", type=int, default=cfg.log_every)
    ap.add_argument("--val-size", type=int, default=cfg.val_size)
    ap.add_argument("--bucketing", action="store_true")
    ap.add_argument("--compile", dest="compile_model", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skeletons", type=Path, default=cfg.skeletons)
    ap.add_argument(
        "--exclure-test",
        dest="exclure_test",
        action="store_true",
        help="retire du flux les squelettes réservés au jeu de test",
    )
    args = ap.parse_args(argv)
    cfg = config_depuis_arguments(args)
    Trainer(cfg, resume=not args.no_resume).run()
    return 0


#: Arguments qui pilotent le lancement plutôt que la configuration du run.
HORS_CONFIG = frozenset({"no_resume"})


def config_depuis_arguments(args: argparse.Namespace) -> TrainConfig:
    """Assemble la ``TrainConfig`` par recoupement avec les champs du dataclass.

    Tout argument dont le nom correspond à un champ y est transporté. Un
    argument qui n'en vise aucun est une erreur franche : c'est le seul moyen de
    ne pas se retrouver avec un drapeau qui ne fait rien.
    """
    champs = {f.name for f in dataclasses.fields(TrainConfig)}
    fournis = {k: v for k, v in vars(args).items() if k not in HORS_CONFIG}
    orphelins = set(fournis) - champs
    if orphelins:
        raise ValueError(
            f"arguments sans champ correspondant dans TrainConfig : {sorted(orphelins)} — "
            "ajouter le champ, ou le déclarer dans HORS_CONFIG s'il pilote le lancement"
        )
    return TrainConfig(**fournis)


if __name__ == "__main__":
    raise SystemExit(main())
