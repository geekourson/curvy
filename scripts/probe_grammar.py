"""Que sait dessiner la grammaire v1 ? Sonde d'expressivité.

Le modèle ne peut trouver que ce que sa grammaire contient. Ce script prend des
formes que tout le monde a en tête — cercle, cœur, carré — et vérifie, pour
chacune, si elle est **exprimable** dans la grammaire v1 : arbre valide,
profondeur de corps ≤ 6, constantes internes ≤ 5.

C'est une question sur la grammaire, pas sur le modèle : indépendante de
l'entraînement, vérifiable tout de suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from curvy.config import FIGURES_DIR, ensure_dirs  # noqa: E402
from curvy.data.canonical import canonicalise, strip_absorbable_root, wrap_root  # noqa: E402
from curvy.data.expr import (  # noqa: E402
    count_constants,
    depth,
    evaluate,
    from_prefix,
    to_prefix,
)
from curvy.data.grammar import MAX_BODY_CONSTANTS, MAX_BODY_DEPTH  # noqa: E402


def check(prefix: str) -> dict:
    """Le squelette tient-il dans les budgets de la grammaire v1 ?"""
    try:
        body = from_prefix(prefix.split())
    except ValueError as exc:
        return {"ok": False, "raison": f"non parsable : {exc}"}
    canon = strip_absorbable_root(canonicalise(body))
    d, k = depth(canon), count_constants(canon)
    ok = d <= MAX_BODY_DEPTH and k <= MAX_BODY_CONSTANTS
    raison = ""
    if d > MAX_BODY_DEPTH:
        raison += f"profondeur {d} > {MAX_BODY_DEPTH} ; "
    if k > MAX_BODY_CONSTANTS:
        raison += f"{k} constantes internes > {MAX_BODY_CONSTANTS} ; "
    return {
        "ok": ok,
        "profondeur_corps": d,
        "constantes_internes": k,
        "squelette": " ".join(to_prefix(wrap_root(canon))),
        "raison": raison.removesuffix(" ; "),
    }


#: (nom, x(t) en préfixe, constantes de x, y(t) en préfixe, constantes de y, commentaire)
PARAMETRIQUES = [
    (
        "Cercle",
        "mul C cos x",
        [1.0],
        "mul C sin x",
        [1.0],
        "x = C·cos t, y = C·sin t",
    ),
    (
        "Cœur (cardioïde)",
        "mul cos x sub C sin x",
        [1.0],
        "mul sin x sub C sin x",
        [1.0],
        "x = (C − sin t)·cos t, y = (C − sin t)·sin t",
    ),
    (
        "Cœur « classique »",
        "cube sin x",
        [],
        "sub sub sub mul C cos x mul C cos mul C x mul C cos mul C x mul C cos mul C x",
        [13 / 16, 5 / 16, 2.0, 2 / 16, 3.0, 1 / 16, 4.0],
        "x = sin³t, y = 13cos t − 5cos 2t − 2cos 3t − cos 4t",
    ),
    (
        "Lemniscate (∞)",
        "mul C cos x",
        [1.0],
        "mul C mul sin x cos x",
        [1.4],
        "x = C·cos t, y = C·sin t·cos t",
    ),
    (
        "Astroïde (carré à côtés creux)",
        "cube cos x",
        [],
        "cube sin x",
        [],
        "x = cos³t, y = sin³t",
    ),
    (
        # PIÈGE : sign(cos t), sign(sin t) ne prend que QUATRE valeurs — les
        # quatre coins. La « figure de carré » qu'on croit voir n'est que les
        # segments tracés entre ces quatre points. Conservé exprès : c'est
        # exactement le genre d'erreur qu'une figure fait commettre.
        "« Carré » par le signe (PIÈGE : 4 points)",
        "mul abs cos x inv cos x",
        [],
        "mul abs sin x inv sin x",
        [],
        "sign(cos t), sign(sin t) — 4 valeurs seulement, pas une courbe",
    ),
    (
        "Carré (vraie construction, norme max)",
        "mul cos x inv mul C add add abs cos x abs sin x abs sub abs cos x abs sin x",
        [0.5],
        "mul sin x inv mul C add add abs cos x abs sin x abs sub abs cos x abs sin x",
        [0.5],
        "x = cos t / max(|cos t|,|sin t|), avec max(a,b) = (a+b+|a−b|)/2",
    ),
]

#: Formes définies comme y = f(x), donc univaluées.
FONCTIONS = [
    ("Squircle d'ordre 4", "sqrt sqrt sub C sq sq x", [1.0], "y = (1 − x⁴)^(1/4)"),
    ("Squircle d'ordre 8", "sqrt sqrt sqrt sub C sq sq sq x", [1.0], "y = (1 − x⁸)^(1/8)"),
    ("Demi-cercle", "sqrt sub C sq x", [1.0], "y = √(1 − x²)"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=FIGURES_DIR / "grammaire-formes.png")
    args = ap.parse_args(argv)
    ensure_dirs()

    print("=" * 78)
    print("FORMES PARAMÉTRIQUES  x(t), y(t)  —  nécessitent le mode paramétrique (Phase 7)")
    print("=" * 78)
    results = []
    for name, px, cx, py, cy, note in PARAMETRIQUES:
        rx, ry = check(px), check(py)
        ok = rx["ok"] and ry["ok"]
        results.append((name, px, cx, py, cy, note, ok, rx, ry))
        flag = "OUI" if ok else "NON"
        print(f"\n[{flag}] {name}  —  {note}")
        for label, r in (("x(t)", rx), ("y(t)", ry)):
            det = f"profondeur {r.get('profondeur_corps')}, {r.get('constantes_internes')} const."
            print(f"      {label} : {det}" + (f"  -> {r['raison']}" if r.get("raison") else ""))

    print("\n" + "=" * 78)
    print("FORMES UNIVALUÉES  y = f(x)  —  exprimables dès la v1")
    print("=" * 78)
    for name, pf, _cf, note in FONCTIONS:
        r = check(pf)
        flag = "OUI" if r["ok"] else "NON"
        print(f"\n[{flag}] {name}  —  {note}")
        print(
            f"      profondeur {r.get('profondeur_corps')}, "
            f"{r.get('constantes_internes')} const."
            + (f"  -> {r['raison']}" if r.get("raison") else "")
        )

    # --- figure ---
    n = len(PARAMETRIQUES) + len(FONCTIONS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.2 * rows))
    axes = np.atleast_2d(axes).ravel()
    t = np.linspace(-np.pi, np.pi, 2000)

    for ax, (name, px, cx, py, cy, note, ok, _rx, _ry) in zip(axes, results, strict=False):
        with np.errstate(all="ignore"):
            xs = evaluate(from_prefix(px.split()), t, cx)
            ys = evaluate(from_prefix(py.split()), t, cy)
        good = np.isfinite(xs) & np.isfinite(ys)
        colour = "tab:green" if ok else "tab:red"
        ax.plot(xs[good], ys[good], lw=1.6, color=colour)
        ax.set_title(
            f"{'✓' if ok else '✗'} {name}\n{note}",
            fontsize=9,
            color="black" if ok else "tab:red",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    xg = np.linspace(-0.999, 0.999, 2000)
    for ax, (name, pf, cf, note) in zip(axes[len(results) :], FONCTIONS, strict=False):
        with np.errstate(all="ignore"):
            y = evaluate(from_prefix(pf.split()), xg, cf)
        r = check(pf)
        colour = "tab:green" if r["ok"] else "tab:red"
        ax.plot(xg, y, lw=1.6, color=colour)
        ax.plot(xg, -y, lw=1.6, color=colour, alpha=0.6)
        ax.set_title(
            f"{'✓' if r['ok'] else '✗'} {name}\n{note}",
            fontsize=9,
            color="black" if r["ok"] else "tab:red",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_axis_off()

    fig.suptitle(
        "Ce que la grammaire v1 sait dessiner (vert) et ne sait pas (rouge)\n"
        "profondeur de corps ≤ 6, constantes internes ≤ 5",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=140)
    fig.savefig(args.out.with_suffix(".svg"))
    print(f"\nfigure écrite : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
