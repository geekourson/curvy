"""Phase 6 — construit et **gèle** le jeu de test.

Règle non négociable : ce script s'exécute **avant** de regarder le moindre
résultat, et son produit ne bouge plus. Il écrit un fichier accompagné de son
empreinte SHA-256 ; toute mesure publiée doit citer cette empreinte, faute de
quoi rien ne prouve qu'on n'a pas rejoué la construction jusqu'à obtenir un
jeu flatteur.

    .venv/bin/python scripts/build_testset.py

Le jeu produit ne contient **aucun** squelette vu à l'entraînement, à condition
que l'entraînement ait tourné avec ``--exclure-test`` (). Les runs
exp-001 à exp-003 sont antérieurs : leurs chiffres sur ce jeu ne vaudraient
rien, et le script le rappelle en fin de sortie.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from curvy.config import DATASET_DIR, ensure_dirs
from curvy.data.dataset import make_validation_set
from curvy.data.expr import to_prefix
from curvy.data.generate import load_skeletons
from curvy.data.pointcloud import sample_cloud_fn
from curvy.data.split import SEL, partitionner
from curvy.data.testset import FORMULES_A_LA_MAIN
from curvy.seeding import make_rng

GRAINE = 20260820  # figée une fois pour toutes : la date du gel


def _empreinte(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeletons", type=Path, default=DATASET_DIR / "skeletons-v1.jsonl.gz")
    ap.add_argument("--out", type=Path, default=DATASET_DIR / "testset-v1.jsonl.gz")
    ap.add_argument("--n-hors-distribution", type=int, default=6, help="nuages par formule")
    args = ap.parse_args(argv)
    ensure_dirs()

    if args.out.exists():
        print(f"REFUS : {args.out} existe déjà — un jeu de test gelé ne se régénère pas.")
        print(f"        empreinte actuelle : {_empreinte(args.out)}")
        print("        Le supprimer à la main est une décision, pas une commande de routine.")
        return 1

    items = load_skeletons(args.skeletons)
    part = partitionner(items)
    print(json.dumps(part.rapport(), indent=2, ensure_ascii=False))

    lignes: list[dict] = []

    # --- 1. tenu à l'écart : même grammaire, squelettes jamais vus ------------
    reserve = make_validation_set(
        args.skeletons,
        n=len(part.test),
        seed=GRAINE,
        garder=part.prefixes_de_test,
        un_nuage_par_squelette=True,
    )
    for ex in reserve:
        lignes.append(
            {
                "sous_ensemble": "tenu_a_lecart",
                "nom": " ".join(to_prefix(ex.node)),
                "dans_la_grammaire": True,
                "profondeur": ex.depth,
                "x": [round(float(v), 6) for v in ex.x],
                "y": [round(float(v), 6) for v in ex.y],
                "y_exact": [round(float(v), 6) for v in ex.y_clean],
                "ids": ex.ids,
            }
        )
    print(f"\ntenu à l'écart : {len(reserve)} exemples sur {len(part.test)} squelettes réservés")

    # --- 2. hors distribution : formules écrites à la main --------------------
    rng = make_rng(GRAINE)
    par_formule: dict[str, int] = {}
    for formule in FORMULES_A_LA_MAIN:
        obtenus = 0
        for _ in range(200):
            if obtenus >= args.n_hors_distribution:
                break
            cloud, _ = sample_cloud_fn(rng, formule.f)
            if cloud is None:
                continue
            with np.errstate(all="ignore"):
                exact = (np.asarray(formule.f(cloud.x)) - cloud.y_offset) / cloud.y_scale
            if not np.isfinite(exact).all():
                continue
            lignes.append(
                {
                    "sous_ensemble": "hors_distribution",
                    "nom": formule.nom,
                    "dans_la_grammaire": formule.prefixe is not None,
                    "prefixe": formule.prefixe,
                    "commentaire": formule.commentaire,
                    "x": [round(float(v), 6) for v in cloud.x],
                    "y": [round(float(v), 6) for v in cloud.y],
                    "y_exact": [round(float(v), 6) for v in exact],
                }
            )
            obtenus += 1
        par_formule[formule.nom] = obtenus

    manquantes = {k: v for k, v in par_formule.items() if v < args.n_hors_distribution}
    print(f"hors distribution : {sum(par_formule.values())} exemples")
    if manquantes:
        # Une formule sous-représentée doit se voir, pas disparaître en silence.
        print(f"  formules incomplètes (filtre d'identifiabilité) : {manquantes}")

    # --- écriture et gel -----------------------------------------------------
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for ligne in lignes:
            fh.write(json.dumps(ligne, ensure_ascii=False, sort_keys=True) + "\n")

    manifeste = {
        "fichier": args.out.name,
        "sha256": _empreinte(args.out),
        "graine": GRAINE,
        "sel_de_partition": SEL,
        "n_total": len(lignes),
        "n_tenu_a_lecart": len(reserve),
        "n_hors_distribution": sum(par_formule.values()),
        "par_formule": par_formule,
        "partition": part.rapport(),
        "reel_canvas": "non construit — outil de capture en Phase 8",
    }
    chemin_manifeste = Path("docs/benchmarks/testset-v1.json")
    chemin_manifeste.write_text(
        json.dumps(manifeste, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nécrit   : {args.out}")
    print(f"sha256  : {manifeste['sha256']}")
    print(f"manifeste : {chemin_manifeste}")
    print(
        "\nRAPPEL : exp-001 à exp-003 ont vu ces squelettes à l'entraînement.\n"
        "         Les mesurer sur ce jeu ne dirait rien. Il faut un run lancé\n"
        "         avec la partition."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
