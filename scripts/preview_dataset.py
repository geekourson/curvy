"""Aperçu visuel du dataset — jalon de la Phase 1.

Deux colonnes volontairement différentes :

- **gauche** : squelettes tirés uniformément dans l'ensemble dédupliqué ;
- **droite** : squelettes tirés avec la pondération par multiplicité
  (``tau = 0,5``).

La comparaison est le sujet de la figure : à gauche ce que produit un pipeline
naïf, à droite ce que le modèle doit réellement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy  # noqa: E402

from curvy.config import DATASET_DIR, FIGURES_DIR, ensure_dirs  # noqa: E402
from curvy.data.expr import const_name_iter, evaluate, from_prefix, to_infix  # noqa: E402
from curvy.data.generate import load_skeletons  # noqa: E402
from curvy.data.pointcloud import CloudConfig, sample_cloud  # noqa: E402
from curvy.data.weighting import describe_weights, stratified_weights  # noqa: E402
from curvy.seeding import DEFAULT_SEED, make_rng  # noqa: E402


def latex_of(prefix: str, consts: list[float]) -> str:
    """Formule lisible, constantes numériques substituées."""
    node = from_prefix(prefix.split())
    names = const_name_iter()
    infix = to_infix(node, names)
    subs = {f"c{i}": round(float(c), 2) for i, c in enumerate(consts)}
    try:
        expr = sympy.sympify(infix).subs(subs)
        return f"${sympy.latex(expr)}$"
    except Exception:
        return infix


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeletons", type=Path, default=DATASET_DIR / "skeletons-v1.jsonl.gz")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--out", type=Path, default=FIGURES_DIR / "phase1-echantillons.png")
    args = ap.parse_args(argv)

    ensure_dirs()
    rng = make_rng(args.seed)
    items = load_skeletons(args.skeletons)
    depths = [it["depth"] for it in items]
    weights = stratified_weights(depths)
    print("distribution de profondeur visée à l'entraînement :", describe_weights(depths, weights))

    picks = [
        ("tirage uniforme sur les uniques", rng.choice(len(items), size=5, replace=False)),
        ("tirage stratifié", rng.choice(len(items), size=5, replace=False, p=weights)),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(13, 15))
    fig.suptitle(
        "Curvy — Phase 1 : 10 exemples du dataset\n"
        "à gauche, tirage uniforme sur les squelettes uniques ; "
        "à droite, tirage stratifié par profondeur",
        fontsize=13,
    )
    dense_x = np.linspace(-1.0, 1.0, 400)

    for col, (label, idxs) in enumerate(picks):
        for row, i in enumerate(idxs):
            ax = axes[row, col]
            it = items[int(i)]
            node = from_prefix(it["prefix"].split())
            for _ in range(30):
                cloud, _ = sample_cloud(rng, node, CloudConfig())
                if cloud is not None:
                    break
            if cloud is None:
                ax.set_axis_off()
                continue
            clean = evaluate(node, dense_x, cloud.consts)
            clean_n = np.where(np.isfinite(clean), clean, np.nan)
            clean_n = (clean_n - cloud.y_offset) / cloud.y_scale
            ax.plot(dense_x, clean_n, lw=1.2, color="tab:orange", alpha=0.9, label="courbe exacte")
            ax.scatter(cloud.x, cloud.y, s=9, color="tab:blue", alpha=0.75, label="points bruités")
            ax.set_title(
                f"{latex_of(it['prefix'], cloud.consts)}\n"
                f"prof. {it['depth']} · {it['n_consts']} const. · "
                f"complexité {it['complexity']} · mult. {it['count']} · {cloud.n_points} pts",
                fontsize=8.5,
            )
            ax.set_ylim(-1.6, 1.6)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.legend(fontsize=7, loc="upper right")
            if row == 0:
                ax.text(
                    0.02,
                    1.28,
                    label,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    color="tab:red",
                )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    fig.savefig(args.out.with_suffix(".svg"))
    print(f"figure écrite : {args.out}")
    print(f"               {args.out.with_suffix('.svg')}")

    for label, idxs in picks:
        print(f"\n--- {label} ---")
        for i in idxs:
            it = items[int(i)]
            print(f"  prof {it['depth']} · mult {it['count']:>7} · {it['prefix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
