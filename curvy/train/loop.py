"""Boucle d'entraînement.

Volontairement courte : la configuration est ailleurs (``config.py``), les
métriques ailleurs (``metrics.py``), la reprise ailleurs (``checkpoint.py``).
Ce fichier ne fait qu'orchestrer.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from curvy.data.dataset import BucketedBatches, CurvyStream, collate, make_validation_set
from curvy.devices import pick_device
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel, count_parameters
from curvy.precision import bf16_supported, configure_precision, precision_report
from curvy.seeding import make_rng, seed_everything
from curvy.tokenizer.vocab import PAD_ID
from curvy.train.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from curvy.train.config import TrainConfig
from curvy.train.metrics import evaluate_model, token_accuracy

__all__ = ["Trainer"]


def cosine_with_warmup(step: int, cfg: TrainConfig) -> float:
    """Facteur multiplicatif du learning rate."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: bool = True) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        configure_precision(tf32=True)
        self.device = pick_device("auto").device
        self.rng = make_rng(cfg.seed + 1)

        model_cfg = PRESETS[cfg.preset]
        self.model = CurvyModel(model_cfg).to(self.device)
        self.params = count_parameters(self.model)
        if cfg.compile_model:
            self.model = torch.compile(self.model)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: cosine_with_warmup(s, cfg)
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        self.use_bf16 = cfg.bf16 and bf16_supported(self.device) and self.device.type == "cuda"

        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = cfg.run_dir / "log.jsonl"
        self.step = 0
        self.best = -1.0

        ckpt = latest_checkpoint(cfg.run_dir) if resume else None
        if ckpt is not None:
            state = load_checkpoint(
                ckpt, model=self.model, optimizer=self.opt, scheduler=self.sched
            )
            self.step = state["step"]
            self.best = state.get("best", -1.0)
            self._log({"event": "reprise", "step": self.step, "checkpoint": str(ckpt)})

        self.val = make_validation_set(cfg.skeletons, cfg.val_size, seed=cfg.val_seed)
        self.loader = self._make_loader()
        self._log(
            {
                "event": "partition",
                "exclure_test": cfg.exclure_test,
                "n_squelettes_exclus": self._n_exclus,
            }
        )

    #: Un batch écarté de temps en temps n'est pas une avarie : le modèle
    #: continue d'apprendre normalement, et sauter 3 batches sur 11 000 ne se
    #: voit dans aucune métrique (mesuré sur exp-005, 2026-08-20). Ce qui doit
    #: arrêter un run, c'est un **taux**, pas un total — sinon un run sain finit
    #: par atteindre n'importe quel compteur cumulatif.
    #:
    #: Premier seuil posé à « 5 au total » : il aurait tué exp-005 au step
    #: 28 500 alors que son taux R² dépassait celui du run de contrôle.
    FENETRE_INCIDENTS = 1_000
    MAX_INCIDENTS_PAR_FENETRE = 20  # 2 % des steps de la fenêtre

    def _incident(self, genre: str, batch, valeur: float) -> None:
        """Consigne un batch non fini, le sauvegarde, et arrête si ça se répète.

        Le batch fautif est écrit sur disque : sans lui, la cause reste
        indevinable. Il pèse moins d'un mégaoctet, et c'est la seule occasion de
        l'attraper — le flux ne repasse jamais deux fois au même endroit.
        """
        recents = getattr(self, "_incidents_recents", None)
        if recents is None:
            recents = self._incidents_recents = []
        recents.append(self.step)
        # On ne garde que la fenêtre glissante.
        seuil_bas = self.step - self.FENETRE_INCIDENTS
        self._incidents_recents = recents = [s for s in recents if s > seuil_bas]
        self._n_incidents = getattr(self, "_n_incidents", 0) + 1
        chemin = self.cfg.run_dir / f"batch-non-fini-{self.step}.pt"
        try:
            torch.save(
                {
                    "points": batch.points.detach().cpu(),
                    "point_mask": batch.point_mask.detach().cpu(),
                    "tokens": batch.tokens.detach().cpu(),
                    "token_mask": batch.token_mask.detach().cpu(),
                },
                chemin,
            )
        except Exception as exc:  # ne jamais faire tomber le run sur la sauvegarde
            chemin = f"échec de sauvegarde : {exc}"

        pts = batch.points.detach().float()
        self._log(
            {
                "event": "incident",
                "genre": genre,
                "step": self.step,
                "valeur": valeur,
                "n_incidents": self._n_incidents,
                "n_dans_la_fenetre": len(recents),
                "batch_sauve": str(chemin),
                "points_min": round(float(pts.min()), 4),
                "points_max": round(float(pts.max()), 4),
                "points_finis": bool(torch.isfinite(pts).all()),
                "tokens_max": int(batch.tokens.max()),
            }
        )
        if len(recents) >= self.MAX_INCIDENTS_PAR_FENETRE:
            self._log(
                {
                    "event": "abandon",
                    "raison": (
                        f"{len(recents)} batches non finis en {self.FENETRE_INCIDENTS} steps"
                    ),
                    "step": self.step,
                }
            )
            raise RuntimeError(
                f"{len(recents)} batches non finis en {self.FENETRE_INCIDENTS} steps "
                f"— arrêt. Batches sauvés dans {self.cfg.run_dir}"
            )

    def _prefixes_de_test(self) -> frozenset[str] | None:
        """Les squelettes réservés au test, ou ``None`` si le run n'exclut rien.

        Calculé à partir du fichier de squelettes lui-même : la partition est
        une fonction du hachage de chaque squelette, il n'y a donc pas de
        fichier d'index à tenir synchronisé.
        """
        if not self.cfg.exclure_test:
            self._n_exclus = 0
            return None
        from curvy.data.generate import load_skeletons
        from curvy.data.split import partitionner

        prefixes = partitionner(load_skeletons(self.cfg.skeletons)).prefixes_de_test
        self._n_exclus = len(prefixes)
        return prefixes

    def _make_loader(self) -> DataLoader:
        cfg = self.cfg
        stream = CurvyStream(
            cfg.skeletons,
            seed=cfg.seed + 100 * self.step,
            depth_target=cfg.depth_target,
            exclure=self._prefixes_de_test(),
        )
        common = dict(
            num_workers=cfg.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.workers > 0,
            prefetch_factor=4 if cfg.workers > 0 else None,
        )
        if cfg.bucketing:
            return DataLoader(BucketedBatches(stream, cfg.batch_size), batch_size=None, **common)
        return DataLoader(stream, batch_size=cfg.batch_size, collate_fn=collate, **common)

    def _log(self, record: dict) -> None:
        record.setdefault("t", round(time.time(), 3))
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast(self.device.type, enabled=False)

    def run(self) -> None:
        cfg = self.cfg
        self._log(
            {
                "event": "demarrage",
                "config": cfg.to_dict(),
                "parametres": self.params["TOTAL"],
                "device": str(self.device),
                "precision": precision_report(self.device),
                "val_size": len(self.val),
            }
        )
        it = iter(self.loader)
        self.model.train()
        t0 = time.perf_counter()
        window: list[float] = []
        n_tokens = 0

        while self.step < cfg.steps:
            batch = next(it).to(self.device)
            tokens_in, target = batch.tokens[:, :-1], batch.tokens[:, 1:]
            with self._autocast():
                logits = self.model(
                    batch.points, batch.point_mask, tokens_in, batch.token_mask[:, :-1]
                )
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1))
            self.opt.zero_grad(set_to_none=True)

            # Un seul batch à loss non finie suffit à tuer le run : le clip de
            # gradient renvoie NaN, multiplie TOUS les gradients par NaN, et
            # l'optimiseur écrit NaN dans tous les poids. Plus rien ne revient
            # ensuite. exp-005 s'est entraîné 3 500 steps sur du NaN sans que
            # rien ne l'arrête (2026-08-20).
            if not torch.isfinite(loss):
                self.step += 1
                self._incident("loss_non_finie", batch, float(loss.detach()))
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            # La loss peut être finie et un gradient déborder quand même.
            if not torch.isfinite(grad_norm):
                self.step += 1
                self._incident("gradient_non_fini", batch, float(loss.detach()))
                continue

            self.opt.step()
            self.sched.step()
            self.step += 1

            window.append(loss.detach().item())
            n_tokens += int((target != PAD_ID).sum())

            if self.step % cfg.log_every == 0:
                dt = time.perf_counter() - t0
                tok_acc, seq_acc = token_accuracy(logits.detach().float(), target)
                self._log(
                    {
                        "event": "train",
                        "step": self.step,
                        "loss": round(sum(window) / len(window), 5),
                        "token_acc": round(tok_acc, 4),
                        "seq_acc_tf": round(seq_acc, 4),
                        "lr": round(self.sched.get_last_lr()[0], 8),
                        "grad_norm": round(float(grad_norm), 4),
                        "tokens_par_s": round(n_tokens / dt),
                        "s_par_step": round(dt / cfg.log_every, 4),
                        "vram_Mio": (
                            round(torch.cuda.max_memory_allocated(self.device) / 2**20)
                            if self.device.type == "cuda"
                            else None
                        ),
                    }
                )
                window, n_tokens, t0 = [], 0, time.perf_counter()

            if self.step % cfg.eval_every == 0 or self.step == cfg.steps:
                t_eval = time.perf_counter()
                rep = evaluate_model(self.model, self.val, self.device, self.rng)
                self._log(
                    {
                        "event": "eval",
                        "step": self.step,
                        "duree_s": round(time.perf_counter() - t_eval, 1),
                        **rep.as_dict(),
                    }
                )
                if rep.r2_rate > self.best:
                    self.best = rep.r2_rate
                    self._save("best.pt")
                t0 = time.perf_counter()

            if self.step % cfg.ckpt_every == 0:
                self._save("last.pt")

        self._save("last.pt")
        self._log({"event": "fin", "step": self.step, "meilleur_r2_rate": round(self.best, 4)})

    def _save(self, name: str) -> None:
        save_checkpoint(
            Path(self.cfg.run_dir) / name,
            model=self.model,
            optimizer=self.opt,
            scheduler=self.sched,
            step=self.step,
            config=self.cfg.to_dict(),
            best=self.best,
        )
