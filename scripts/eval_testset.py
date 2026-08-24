"""Évalue sur le jeu de test **figé** (Phase 6).

Trois concurrents, exactement le même protocole :

- **oracle** — le vrai squelette, constantes réajustées. Plafond de la tâche ;
- **polynôme** — degré 1 à 8 choisi par validation croisée sur les seuls points
  d'ajustement. La baseline à battre ;
- **Curvy** — beam search, candidat choisi sans regarder les points tenus à
  l'écart. Omis si ``--run`` n'est pas fourni : les deux premiers ne
  demandent aucun modèle et peuvent être mesurés avant qu'un run existe.

Résultats rendus **par sous-ensemble et par profondeur**, jamais agrégés en un
chiffre unique : « tenu à l'écart » et « hors distribution » ne répondent pas à
la même question.

    .venv/bin/python scripts/eval_testset.py                     # oracle + polynôme
    .venv/bin/python scripts/eval_testset.py --run exp-005 --preset v1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from curvy.config import DATASET_DIR, RUNS_DIR
from curvy.data.expr import evaluate, from_prefix
from curvy.infer.fit import fit_constants, r_squared
from curvy.seeding import make_rng

SEUIL = 0.99
DEGRES = range(1, 9)


def charger(path: Path) -> tuple[list[dict], str]:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ligne) for ligne in fh], h.hexdigest()


def _decoupe(x: np.ndarray, mode: str, rng: np.random.Generator):
    n = len(x)
    n_hold = max(3, int(round(0.2 * n)))
    if mode == "extrapolation":
        o = np.argsort(x)
        return o[:-n_hold], o[-n_hold:]
    idx = rng.permutation(n)
    return idx[n_hold:], idx[:n_hold]


def _degre_par_cv(x, y, k=5):
    n = len(x)
    idx = np.arange(n)
    plis = np.array_split(idx, k)
    best, best_mse = 1, np.inf
    for deg in DEGRES:
        err = []
        for pli in plis:
            tr = np.setdiff1d(idx, pli)
            if len(tr) <= deg + 1 or len(pli) == 0:
                continue
            with np.errstate(all="ignore"):
                try:
                    p = np.polyval(np.polyfit(x[tr], y[tr], deg), x[pli])
                except Exception:
                    continue
            if np.all(np.isfinite(p)):
                err.append(float(np.mean((p - y[pli]) ** 2)))
        if err and np.mean(err) < best_mse:
            best_mse, best = float(np.mean(err)), deg
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", type=Path, default=DATASET_DIR / "testset-v1.jsonl.gz")
    ap.add_argument("--run", default=None)
    ap.add_argument("--preset", default="v1")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args(argv)

    lignes, empreinte = charger(args.testset)
    print(f"jeu de test : {args.testset.name}")
    print(f"sha256      : {empreinte}")
    manifeste = Path("docs/benchmarks/testset-v1.json")
    if manifeste.exists():
        attendu = json.loads(manifeste.read_text())["sha256"]
        etat = "conforme au manifeste" if attendu == empreinte else "!! DIVERGENT DU MANIFESTE !!"
        print(f"              {etat}")
    print(f"exemples    : {len(lignes)}\n")

    modele = None
    if args.run:
        import torch

        from curvy.data.dataset import collate
        from curvy.devices import pick_device
        from curvy.infer.decode import beam_search, ids_to_node
        from curvy.infer.pareto import ajuster_candidats, selectionner
        from curvy.model.config import PRESETS
        from curvy.model.curvy import CurvyModel
        from curvy.tokenizer.vocab import encode

        choix = pick_device("auto")
        modele = CurvyModel(PRESETS[args.preset]).to(choix.device)
        etat = torch.load(
            RUNS_DIR / args.run / args.checkpoint, map_location="cpu", weights_only=False
        )
        modele.load_state_dict(etat["model"])
        modele.eval()
        cfg_run = json.loads((RUNS_DIR / args.run / "config.json").read_text())
        exclu = cfg_run.get("exclure_test", False)
        print(f"modèle      : {args.run}/{args.checkpoint}, step {etat.get('step', '?')} — {choix}")
        print(f"exclure_test: {exclu}" + ("" if exclu else "  <-- CHIFFRES SANS VALEUR"))
        print()

        # Décodage batché : un seul passage sur tout le jeu.
        candidats: list[list] = []
        for start in range(0, len(lignes), args.batch_size):
            chunk = lignes[start : start + args.batch_size]
            faux_ids = encode(from_prefix(["add", "mul", "C", "x", "C"]))
            batch = collate(
                [
                    (
                        np.stack([np.array(lg["x"]), np.array(lg["y"])], 1).astype(np.float32),
                        faux_ids,
                    )
                    for lg in chunk
                ]
            ).to(choix.device)
            candidats.extend(beam_search(modele, batch.points, batch.point_mask, beam=args.beam))

    resultats: dict = defaultdict(lambda: defaultdict(list))
    for i, ligne in enumerate(lignes):
        x = np.array(ligne["x"])
        y = np.array(ligne["y"])
        exact = np.array(ligne["y_exact"])
        sous = ligne["sous_ensemble"]
        cle = ligne["profondeur"] if sous == "tenu_a_lecart" else ligne["nom"]
        rng = make_rng(args.seed + i)

        for mode in ("interpolation", "extrapolation"):
            keep, hold = _decoupe(x, mode, rng)

            # --- polynôme ---
            deg = _degre_par_cv(x[keep], y[keep])
            with np.errstate(all="ignore"):
                try:
                    pred = np.polyval(np.polyfit(x[keep], y[keep], deg), x[hold])
                    ok_poly = r_squared(exact[hold], pred) >= SEUIL
                except Exception:
                    ok_poly = False
            resultats[(sous, mode, "polynome")][cle].append(ok_poly)

            # --- oracle : seulement si le vrai squelette existe ---
            prefixe = ligne.get("nom") if sous == "tenu_a_lecart" else ligne.get("prefixe")
            if sous == "hors_distribution" and prefixe:
                prefixe = f"add mul C {prefixe} C"
            if prefixe:
                node = from_prefix(prefixe.split())
                res = fit_constants(node, x[keep], y[keep], rng)
                ok_or = False
                if res.ok:
                    with np.errstate(all="ignore"):
                        ok_or = r_squared(exact[hold], evaluate(node, x[hold], res.consts)) >= SEUIL
                resultats[(sous, mode, "oracle")][cle].append(ok_or)

            # --- Curvy ---
            if modele is not None:
                noeuds = [ids_to_node(s) for s, _ in candidats[i]]
                ajustes = ajuster_candidats(noeuds, x[keep], y[keep], rng)
                c = selectionner(ajustes)
                ok_m = False
                if c is not None:
                    with np.errstate(all="ignore"):
                        ok_m = r_squared(exact[hold], evaluate(c.node, x[hold], c.consts)) >= SEUIL
                resultats[(sous, mode, "curvy")][cle].append(ok_m)

    concurrents = ["oracle", "polynome"] + (["curvy"] if modele is not None else [])
    for sous in ("tenu_a_lecart", "hors_distribution"):
        for mode in ("interpolation", "extrapolation"):
            dispo = [c for c in concurrents if (sous, mode, c) in resultats]
            if not dispo:
                continue
            print(f"=== {sous} — {mode} ===")
            cles = sorted({k for c in dispo for k in resultats[(sous, mode, c)]}, key=str)
            large = max(len(str(k)) for k in cles) + 2
            print(f"{'':{large}} " + " ".join(f"{c:>16}" for c in dispo))
            for cle in cles:
                vals = []
                for c in dispo:
                    v = resultats[(sous, mode, c)].get(cle, [])
                    # n par concurrent : l'oracle n'existe pas pour les formules
                    # hors grammaire, afficher un n commun ferait croire à une
                    # comparaison qui n'a pas lieu.
                    vals.append(f"{np.mean(v):>10.3f} ({len(v):>2})" if v else f"{'-':>16}")
                print(f"{str(cle):{large}} " + " ".join(vals))

            tot = []
            for c in dispo:
                v = [x for vs in resultats[(sous, mode, c)].values() for x in vs]
                tot.append(f"{np.mean(v):>10.3f} ({len(v):>2})" if v else f"{'-':>16}")
            print(f"{'TOTAL':{large}} " + " ".join(tot))

            # Comparaison à périmètre égal : restreinte aux clés où TOUS les
            # concurrents ont un chiffre. Sans ça, on compare l'oracle sur les
            # formules exprimables au polynôme sur toutes, ce qui ne veut rien dire.
            communes = {k for k in cles if all(resultats[(sous, mode, c)].get(k) for c in dispo)}
            if communes and len(communes) < len(cles):
                comm = []
                for c in dispo:
                    v = [x for k in communes for x in resultats[(sous, mode, c)][k]]
                    comm.append(f"{np.mean(v):>10.3f} ({len(v):>2})")
                print(f"{'  à périmètre égal':{large}} " + " ".join(comm))
                print(f"{'':{large}}   ({len(communes)} clés communes sur {len(cles)})")

            n_ref = sum(len(v) for v in resultats[(sous, mode, dispo[-1])].values())
            marge = 1.96 * (0.25 / max(n_ref, 1)) ** 0.5
            print(f"{'':{large}} (±{100 * marge:.1f} pt à 95 % sur n={n_ref})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
