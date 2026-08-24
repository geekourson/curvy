"""``python -m curvy.cli_gpu`` — contrôle avant vol de la VRAM.

À lancer avant tout entraînement (le ``Makefile`` en fait une dépendance de la
cible ``train``). Répond à trois questions :

1. Les GPU autorisés par ``CURVY_CUDA_ALLOW`` ont-ils assez de VRAM libre ?
2. Qui l'occupe, le cas échéant ?
3. Faut-il tuer ces processus ? — uniquement avec ``--kill``, et **jamais** sur
   un GPU absent de l'allowlist. Le cas le plus fréquent est un run Curvy
   zombie d'une session tmux précédente ; le second est un serveur d'inférence
   qu'on a le droit d'arrêter.

Cet outil n'importe volontairement pas ``torch`` : il doit pouvoir dire
« la carte est pleine » sans avoir besoin d'y créer un contexte CUDA — la 3090
saturée nous a déjà montré que ce n'était pas toujours possible.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

MIN_FREE_GIB_DEFAULT = 20.0


def _smi(query: str, entity: str = "gpu") -> list[list[str]]:
    """Interroge nvidia-smi et retourne les lignes découpées."""
    flag = "--query-gpu" if entity == "gpu" else "--query-compute-apps"
    out = subprocess.run(
        ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def _allowed() -> set[str] | None:
    raw = os.environ.get("CURVY_CUDA_ALLOW", "").strip()
    return {t.strip() for t in raw.split(",") if t.strip()} or None


def _is_allowed(index: str, uuid: str, allow: set[str] | None) -> bool:
    return allow is None or index in allow or uuid in allow


def _wait_gone(pid: int, timeout_s: float) -> bool:
    """Attend la disparition d'un PID, sans busy-wait agressif."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:  # existe mais ne nous appartient pas
            return False
        time.sleep(0.2)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-free-gib",
        type=float,
        default=float(os.environ.get("CURVY_MIN_FREE_GIB", MIN_FREE_GIB_DEFAULT)),
        help="VRAM libre exigée sur au moins un GPU autorisé",
    )
    ap.add_argument(
        "--kill",
        action="store_true",
        help="tue les processus occupant les GPU AUTORISÉS (SIGTERM puis SIGKILL)",
    )
    args = ap.parse_args(argv)

    allow = _allowed()
    gpus = _smi("index,name,uuid,memory.total,memory.free")
    apps = _smi("pid,gpu_uuid,used_gpu_memory,process_name", entity="apps")

    by_uuid: dict[str, list[list[str]]] = {}
    for pid, uuid, used, name in apps:
        by_uuid.setdefault(uuid, []).append([pid, used, name])

    print(f"allowlist : {sorted(allow) if allow else '(aucune — tous les GPU)'}")
    print(f"seuil     : {args.min_free_gib:.1f} Gio libres exigés\n")

    best_free = 0.0
    killed: list[str] = []

    for index, name, uuid, total_mib, free_mib in gpus:
        ok = _is_allowed(index, uuid, allow)
        free_gib = float(free_mib) / 1024
        tag = "AUTORISÉ" if ok else "réservé — ne pas toucher"
        print(
            f"[{tag}] cuda:{index} {name}  {free_gib:.1f}/{float(total_mib) / 1024:.1f} Gio libres"
        )
        for pid, used, pname in by_uuid.get(uuid, []):
            print(f"           occupé par PID {pid} — {used} MiB — {pname}")
            if args.kill and ok:
                print(f"           -> SIGTERM {pid}")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    print(f"           -> PID {pid} n'appartient pas à l'utilisateur, ignoré")
                    continue
                if not _wait_gone(int(pid), 20.0):
                    print(f"           -> toujours vivant, SIGKILL {pid}")
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        _wait_gone(int(pid), 10.0)
                    except ProcessLookupError:
                        pass
                killed.append(f"{pid} ({pname})")
            elif args.kill and not ok:
                print("           -> GPU hors allowlist : PROCESSUS ÉPARGNÉ")
        if ok:
            best_free = max(best_free, free_gib)

    if killed:
        print(f"\nprocessus arrêtés : {', '.join(killed)}")
        best_free = max(
            (
                float(free) / 1024
                for i, _n, u, _t, free in _smi("index,name,uuid,memory.total,memory.free")
                if _is_allowed(i, u, allow)
            ),
            default=0.0,
        )
        print(f"VRAM libre après nettoyage : {best_free:.1f} Gio")

    if best_free < args.min_free_gib:
        print(
            f"\nÉCHEC : {best_free:.1f} Gio libres < {args.min_free_gib:.1f} Gio exigés.\n"
            f"Relancer avec `--kill` pour libérer les GPU autorisés.",
            file=sys.stderr,
        )
        return 1
    print(f"\nOK : {best_free:.1f} Gio libres sur un GPU autorisé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
