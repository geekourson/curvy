"""``python -m curvy.cli_smoke`` — validation du bout en bout avant d'entraîner.

Le but n'est pas d'apprendre quoi que ce soit mais de vérifier, en quelques
secondes, que la chaîne complète tient : dataset -> collate -> encodeur ->
décodeur -> loss -> backward. Et de **mesurer** la VRAM et le débit, plutôt que
de les estimer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from curvy.config import DATASET_DIR
from curvy.data.dataset import BucketedBatches, CurvyStream, collate
from curvy.devices import pick_device
from curvy.model.config import PRESETS
from curvy.model.curvy import CurvyModel, count_parameters
from curvy.precision import bf16_supported, configure_precision
from curvy.seeding import DEFAULT_SEED, seed_everything
from curvy.tokenizer.vocab import PAD_ID


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="steps exclus de la mesure — sinon on chronomètre surtout le "
        "remplissage initial du tampon de regroupement (10 s par worker)",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skeletons", type=Path, default=DATASET_DIR / "skeletons-v1.jsonl.gz")
    ap.add_argument(
        "--no-bucketing", action="store_true", help="désactive le regroupement par longueur"
    )
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    configure_precision(tf32=True)
    info = pick_device("auto")
    device = info.device
    cfg = PRESETS[args.preset]

    model = CurvyModel(cfg).to(device)
    params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    stream = CurvyStream(args.skeletons, seed=args.seed)
    if args.no_bucketing:
        loader = DataLoader(
            stream,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )
    else:
        loader = DataLoader(
            BucketedBatches(stream, args.batch_size),
            batch_size=None,  # le dataset produit déjà des batches
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )

    use_bf16 = bf16_supported(device)
    autocast = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else torch.autocast(device.type, enabled=False)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    it = iter(loader)
    losses: list[float] = []
    n_tokens = 0
    n_points = 0
    n_slots = 0
    t_data = 0.0
    t0 = time.perf_counter()

    for step in range(args.steps):
        td = time.perf_counter()
        batch = next(it).to(device)
        t_data += time.perf_counter() - td

        tokens_in = batch.tokens[:, :-1]
        target = batch.tokens[:, 1:]
        with autocast:
            logits = model(batch.points, batch.point_mask, tokens_in, batch.token_mask[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.detach().item())
        if step < args.warmup:
            continue
        n_tokens += int((target != PAD_ID).sum())
        n_points += int((~batch.point_mask).sum())
        n_slots += int(batch.point_mask.numel())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    report = {
        "preset": args.preset,
        "device": str(info),
        "parametres": params["TOTAL"],
        "bf16": use_bf16,
        "batch_size": args.batch_size,
        "bucketing": not args.no_bucketing,
        "remplissage_inutile": None,
        "steps": args.steps,
        "warmup": args.warmup,
        "duree_s": round(elapsed, 2),
        "s_par_step": round(elapsed / args.steps, 4),
        "part_attente_donnees": f"{100 * t_data / elapsed:.1f} %",
        "tokens_par_s": round(n_tokens / elapsed),
        "points_par_s": round(n_points / elapsed),
        "loss_initiale": round(losses[0], 4),
        "loss_finale": round(losses[-1], 4),
    }
    report["remplissage_inutile"] = f"{100 * (1 - n_points / n_slots):.1f} %"
    if device.type == "cuda":
        report["vram_pic_Mio"] = round(torch.cuda.max_memory_allocated(device) / 2**20)
        report["vram_reservee_Mio"] = round(torch.cuda.max_memory_reserved(device) / 2**20)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
