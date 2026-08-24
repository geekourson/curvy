"""``python -m curvy.cli_env`` — carte d'identité de l'environnement de calcul.

Sert de livrable vérifiable pour la Phase 0 et d'en-tête pour toute fiche
d'expérience : on ne consigne jamais une métrique sans savoir sur quoi elle a
été mesurée.
"""

from __future__ import annotations

import json
import platform
import sys

from curvy.config import DATA_ROOT, REPO_ROOT
from curvy.devices import describe_backends, pick_device


def main() -> int:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": str(REPO_ROOT),
        "data_root": str(DATA_ROOT),
        **describe_backends(),
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print()
    print("device retenu :", pick_device("auto"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
