"""Génération de l'ensemble de squelettes (Phase 1).

Rappel du découpage : le coût symbolique — échantillonnage d'arbre,
canonicalisation, filtrage, déduplication — est payé **une fois par squelette
unique**, hors ligne. Les nuages de points sont tirés **en ligne** à
l'entraînement, si bien qu'un même squelette ne produit jamais deux fois le
même exemple. « 2M d'exemples » n'est donc pas la bonne unité de mesure ; les
chiffres qui comptent sont le nombre de squelettes uniques et le débit.

Déduplication à deux niveaux :

1. **exacte**, sur la forme canonique ;
2. **numérique**, sur une empreinte de la courbe évaluée avec un jeu de
   constantes fixé. Elle rattrape ce que la canonicalisation ne voit pas —
   ``mul(x, x)`` et ``sq(x)`` sont structurellement différents et
   numériquement identiques. C'est une heuristique : elle a des faux négatifs
   (deux membres différents d'une même famille ne collident pas), jamais de
   faux positifs à la tolérance choisie.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np

from curvy.config import DATASET_DIR, ensure_dirs
from curvy.data.expr import (
    Node,
    complexity,
    count_constants,
    depth,
    evaluate,
    to_prefix,
)
from curvy.data.pointcloud import DENSE_N, CloudConfig, normalise_y, sample_cloud
from curvy.data.sample import SamplerConfig, sample_skeleton
from curvy.seeding import DEFAULT_SEED, make_rng

#: Jeu de constantes fixe pour l'empreinte numérique. Fixe et non aléatoire :
#: deux squelettes ne sont comparables que s'ils sont évalués au même endroit.
FINGERPRINT_CONSTS = (1.7, -0.93, 2.31, 0.61, -1.42, 3.07, 0.84)
FINGERPRINT_DECIMALS = 4
#: Nombre de tirages de constantes accordés à un squelette avant de le juger
#: non viable. Sans ça, `inv(C + x)` serait jeté dès qu'un tirage malchanceux
#: place le pôle dans le domaine — alors que le squelette est bon.
VIABILITY_ATTEMPTS = 8


def fingerprint(skeleton: Node) -> str | None:
    """Empreinte de la courbe, ou ``None`` si elle n'est pas évaluable."""
    k = count_constants(skeleton)
    consts = list(FINGERPRINT_CONSTS[:k])
    if len(consts) < k:  # squelette plus riche que la table : pas d'empreinte
        return None
    x = np.linspace(-1.0, 1.0, DENSE_N)
    y = evaluate(skeleton, x, consts)
    if not np.isfinite(y).all():
        return None
    y_norm, _, _ = normalise_y(y)
    q = np.round(y_norm, FINGERPRINT_DECIMALS)
    return hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()


def _worker(task: tuple[int, int]) -> tuple[list[dict], Counter]:
    """Produit des squelettes candidats viables. Un flux RNG par worker."""
    seed, n_attempts = task
    rng = make_rng(seed)
    scfg, ccfg = SamplerConfig(), CloudConfig()
    stats: Counter = Counter()
    seen: dict[str, int] = {}
    out: list[dict] = []

    for _ in range(n_attempts):
        stats["tirages"] += 1
        sk = sample_skeleton(rng, scfg)
        if sk is None:
            stats["rejet_squelette"] += 1
            continue
        key = " ".join(to_prefix(sk))
        if key in seen:
            # La multiplicité n'est pas un déchet : c'est une mesure de la
            # probabilité a priori du squelette, dont on aura besoin pour
            # repondérer l'échantillonnage à l'entraînement.
            seen[key] += 1
            stats["doublon_local"] += 1
            continue
        seen[key] = 1

        # Viabilité : le squelette doit produire au moins un nuage valide.
        reason = None
        for _ in range(VIABILITY_ATTEMPTS):
            cloud, reason = sample_cloud(rng, sk, ccfg)
            if cloud is not None:
                break
        if reason is not None:
            stats[f"rejet_{reason}"] += 1
            continue

        stats["retenu"] += 1
        out.append(
            {
                "prefix": key,
                "depth": depth(sk),
                "n_consts": count_constants(sk),
                "complexity": complexity(sk),
                "fingerprint": fingerprint(sk),
                "count": 0,  # complété après la boucle
            }
        )
    for it in out:
        it["count"] = seen[it["prefix"]]
    return out, stats


def generate(n_attempts: int, seed: int, workers: int, out_path: Path) -> dict:
    ensure_dirs()
    per_worker = max(1, n_attempts // workers)
    tasks = [(seed + 1000 * i, per_worker) for i in range(workers)]

    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(_worker, tasks)
    elapsed = time.perf_counter() - t0

    stats: Counter = Counter()
    for _, s in results:
        stats.update(s)

    # Déduplication exacte, puis numérique (au profit du plus simple).
    by_key: dict[str, dict] = {}
    for items, _ in results:
        for it in items:
            prev = by_key.get(it["prefix"])
            if prev is None:
                by_key[it["prefix"]] = it
            else:
                prev["count"] += it["count"]
    stats["doublon_global"] = stats["retenu"] - len(by_key)

    by_fp: dict[str, dict] = {}
    kept: list[dict] = []
    for it in sorted(by_key.values(), key=lambda d: (d["complexity"], d["prefix"])):
        fp = it["fingerprint"]
        if fp is None:
            kept.append(it)
            continue
        if fp in by_fp:
            by_fp[fp]["count"] += it["count"]  # la multiplicité revient au survivant
            stats["doublon_numerique"] += 1
            continue
        by_fp[fp] = it
        kept.append(it)

    kept.sort(key=lambda d: (d["complexity"], d["prefix"]))
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for it in kept:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    summary = {
        "squelettes_uniques": len(kept),
        "tirages": stats["tirages"],
        "duree_s": round(elapsed, 2),
        "debit_tirages_par_s": round(stats["tirages"] / elapsed, 1),
        "workers": workers,
        "seed": seed,
        "fichier": str(out_path),
        "rejets": {k: v for k, v in sorted(stats.items()) if k.startswith(("rejet", "doublon"))},
        "distribution_profondeur": dict(sorted(Counter(d["depth"] for d in kept).items())),
        "distribution_constantes": dict(sorted(Counter(d["n_consts"] for d in kept).items())),
        "multiplicite_par_profondeur": {
            str(d): round(
                sum(k["count"] for k in kept if k["depth"] == d)
                / max(1, sum(1 for k in kept if k["depth"] == d)),
                1,
            )
            for d in sorted({k["depth"] for k in kept})
        },
    }
    return summary


def load_skeletons(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200_000, help="nombre de tirages d'arbres")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--out", type=Path, default=DATASET_DIR / "skeletons-v1.jsonl.gz")
    args = ap.parse_args(argv)

    summary = generate(args.n, args.seed, args.workers, args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    total_rejets = sum(summary["rejets"].values())
    print(f"\ntaux de rejet global : {100 * total_rejets / summary['tirages']:.1f} %")
    print(f"squelettes uniques   : {summary['squelettes_uniques']}")
    print(
        f"débit                : {summary['debit_tirages_par_s']:.0f} tirages/s "
        f"sur {summary['workers']} cœurs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
