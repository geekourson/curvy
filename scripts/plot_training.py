"""Courbes d'entraînement depuis le ``log.jsonl`` d'un run.

Trace ce qui compte, et **l'oracle en trait tireté** sur le panneau du R² : sans
la référence de plafond, un taux de 45 % est illisible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curvy.config import FIGURES_DIR, RUNS_DIR, ensure_dirs  # noqa: E402


def load(path: Path) -> tuple[list[dict], list[dict]]:
    train, ev = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") == "train":
            train.append(rec)
        elif rec.get("event") == "eval":
            ev.append(rec)
    return train, ev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="exp-001")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    ensure_dirs()
    log = args.runs_dir / args.run / "log.jsonl"
    train, ev = load(log)
    if not train:
        print(f"aucune donnée dans {log}")
        return 1
    out = args.out or FIGURES_DIR / f"{args.run}-courbes.png"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Curvy — {args.run} — {len(train)} points de log", fontsize=13)

    ax = axes[0, 0]
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], lw=1.2)
    ax.set_title("loss (cross-entropy, teacher forcing)")
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        [r["step"] for r in train], [r["token_acc"] for r in train], lw=1.2, label="token (train)"
    )
    if ev:
        ax.plot(
            [r["step"] for r in ev],
            [r["token_acc"] for r in ev],
            lw=1.4,
            marker="o",
            ms=3,
            label="token (val)",
        )
        ax.plot(
            [r["step"] for r in ev],
            [r["seq_acc_greedy"] for r in ev],
            lw=1.4,
            marker="s",
            ms=3,
            label="séquence exacte (val, glouton)",
        )
    ax.set_title("accuracy")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if ev:
        ax.plot(
            [r["step"] for r in ev],
            [r["r2_rate"] for r in ev],
            lw=1.8,
            marker="o",
            ms=4,
            color="tab:green",
            label="Curvy : R² ≥ 0.99",
        )
        ax.plot(
            [r["step"] for r in ev],
            [r["r2_rate_oracle"] for r in ev],
            lw=1.4,
            ls="--",
            color="gray",
            label="oracle (vrai squelette)",
        )
        # Extrapolation : absente des runs antérieurs au 2026-08-19, on ne trace
        # la courbe que si le log la porte.
        if any("r2_rate_extrap" in r for r in ev):
            ax.plot(
                [r["step"] for r in ev if "r2_rate_extrap" in r],
                [r["r2_rate_extrap"] for r in ev if "r2_rate_extrap" in r],
                lw=1.8,
                marker="^",
                ms=4,
                color="tab:purple",
                label="Curvy : R² ≥ 0.99 en extrapolation",
            )
            ax.plot(
                [r["step"] for r in ev if "r2_rate_extrap_oracle" in r],
                [r["r2_rate_extrap_oracle"] for r in ev if "r2_rate_extrap_oracle" in r],
                lw=1.2,
                ls="--",
                color="tab:purple",
                alpha=0.5,
                label="oracle en extrapolation",
            )
        ax.axhline(0.5, color="tab:red", ls=":", lw=1.2, label="jalon Phase 4 (50 %)")
        # Baselines polynomiales mesurées le 2026-08-19 (docs/benchmarks/results.md).
        ax.axhline(0.670, color="tab:orange", ls="-.", lw=1.2, label="polynôme, interpolation")
        ax.axhline(0.088, color="tab:orange", ls=":", lw=1.2, label="polynôme, extrapolation")
    ax.set_title("MÉTRIQUE PRINCIPALE — taux de R² ≥ 0.99 (points tenus à l'écart)")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if ev and ev[-1].get("per_depth"):
        depths = sorted(ev[-1]["per_depth"], key=int)
        for d in depths:
            ax.plot(
                [r["step"] for r in ev],
                [r.get("per_depth", {}).get(d, float("nan")) for r in ev],
                lw=1.2,
                marker=".",
                label=f"profondeur {d}",
            )
    ax.set_title("R² ≥ 0.99 par profondeur de squelette")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".svg"))
    print(f"figure écrite : {out}")
    if ev:
        last = ev[-1]
        print(
            f"dernier eval (step {last['step']}) : R²≥0.99 = {100 * last['r2_rate']:.1f} % "
            f"(oracle {100 * last['r2_rate_oracle']:.1f} %), "
            f"séquence exacte = {100 * last['seq_acc_greedy']:.1f} %"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
