"""Serveur de la démo : une page, deux endpoints, zéro dépendance ajoutée.

`http.server` de la bibliothèque standard plutôt que FastAPI : le projet tient
sur cinq dépendances, et en ajouter trois pour un endpoint JSON et une page HTML
serait disproportionné. `make demo` doit marcher sans rien installer.

**Ce service n'écrit rien sur disque.** Il est destiné à être exposé en ligne
avec l'article, sans authentification ni limite de débit : toute écriture serait
une croissance non bornée offerte au premier robot venu — un tracé pesant 3,7
Kio, dix requêtes par seconde rempliraient un gigaoctet en huit heures. Un
utilisateur qui veut garder son tracé le **télécharge** depuis son navigateur.

**Politique GPU.** Le service tourne sur la 3060, jamais sur
la 3090 d'entraînement, et il **plafonne explicitement sa VRAM** : la carte est
partagée, on ne présume pas qu'elle est vide, et un OOM provoqué par le voisin
doit rendre une erreur, pas faire tomber le serveur.

    .venv/bin/python -m curvy.serve.app --run exp-005 --preset v1
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from curvy.config import REPO_ROOT, RUNS_DIR
from curvy.seeding import make_rng
from curvy.serve.debit import Limiteur
from curvy.serve.pipeline import formules_depuis_trace

#: UUID de la 3060 : carte de service. Jamais la 3090.
#: Carte réservée au service. À renseigner avec la vôtre (`nvidia-smi -L`).
GPU_SERVICE = "GPU-xxxxxxxx-CHANGEZ-MOI"

#: Fraction de la VRAM que le service s'autorise. La carte héberge d'autres
#: travaux ; on ne prend pas ce dont on n'a pas besoin.
FRACTION_VRAM = 0.35

WEB = REPO_ROOT / "web"


def _urls(host: str, port: int) -> list[str]:
    """Les adresses réellement tapables, pas celle passée en argument.

    `0.0.0.0` n'est pas une adresse qu'on met dans un navigateur : afficher
    l'adresse de liaison telle quelle envoie l'utilisateur dans le mur.
    """
    if host != "0.0.0.0":  # noqa: S104
        return [f"http://{host}:{port}/"]
    import socket

    urls = [f"http://127.0.0.1:{port}/"]
    try:
        # Une connexion UDP n'émet aucun paquet : elle sert seulement à demander
        # au noyau par quelle interface il sortirait. Adresse hors du réseau
        # local pour ne présumer d'aucun plan d'adressage.
        prise = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        prise.connect(("203.0.113.1", 1))  # RFC 5737, réservée à la documentation
        urls.append(f"http://{prise.getsockname()[0]}:{port}/")
        prise.close()
    except OSError:
        pass
    return urls


class Etat:
    """Modèle chargé une fois, partagé par les requêtes."""

    modele = None
    device = None
    rng = None
    beam = 8
    #: Pool d'ajustement. Les ajustements de constantes dominent la latence
    #: et sont indépendants : les paralléliser est ce qui permet
    #: d'élargir le beam sans quitter le budget d'une seconde.
    pool = None
    limiteur = None
    #: Origine autorisée pour les appels croisés. Vide = aucun en-tête CORS,
    #: donc seule une page servie par ce serveur peut l'appeler.
    origine = ""
    #: Adresses depuis lesquelles `X-Forwarded-For` est cru. Vide = aucune.
    proxys: frozenset[str] = frozenset()


def charger(run: str, preset: str, checkpoint: str, beam: int, limiteur=None) -> None:
    import os

    import torch

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CURVY_CUDA_ALLOW", GPU_SERVICE)

    from curvy.devices import pick_device
    from curvy.model.config import PRESETS
    from curvy.model.curvy import CurvyModel

    choix = pick_device("auto")
    Etat.device = choix.device
    if Etat.device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(FRACTION_VRAM, Etat.device)
    Etat.modele = CurvyModel(PRESETS[preset]).to(Etat.device)
    etat = torch.load(RUNS_DIR / run / checkpoint, map_location="cpu", weights_only=False)
    Etat.modele.load_state_dict(etat["model"])
    Etat.modele.eval()
    Etat.rng = make_rng(0)
    Etat.beam = beam
    Etat.limiteur = limiteur

    # `spawn` et non `fork` : le processus parent porte un contexte CUDA, qu'un
    # fork dupliquerait dans un état indéfini. Les enfants ne font que du calcul
    # numpy/scipy et n'ont pas besoin du GPU.
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    n_procs = max(2, min(8, (os.cpu_count() or 4) - 2))
    Etat.pool = ProcessPoolExecutor(
        max_workers=n_procs, mp_context=multiprocessing.get_context("spawn")
    )
    print(f"pool    : {n_procs} processus d'ajustement", flush=True)
    print(f"modèle  : {run}/{checkpoint}, step {etat.get('step', '?')}", flush=True)
    print(f"device  : {choix}", flush=True)
    print(f"VRAM    : plafonnée à {FRACTION_VRAM:.0%} de la carte", flush=True)


class Handler(BaseHTTPRequestHandler):
    #: Coût de chaque endpoint, en jetons. Une prédiction mobilise le GPU
    #: pendant ~779 ms ; un rééchantillonnage coûte 4 ms et ne touche pas la
    #: carte. Les facturer pareil reviendrait soit à brider l'exploration de la
    #: courbe, soit à laisser passer les requêtes coûteuses.
    COUTS = {"/api/formules": 1.0, "/api/courbe": 0.1}

    #: Coût d'un fichier statique. `do_GET` était la seule route non limitée :
    #: exposé derrière un tunnel public, un robot pouvait boucler dessus sans
    #: rien consommer. Le tarif est bas (une page en fait ~25 avant d'attendre)
    #: parce que la page est autonome — un affichage = une seule requête.
    COUT_STATIQUE = 0.2

    def _adresse(self) -> str:
        """L'adresse du client, en tenant compte d'un proxy de confiance.

        **Sans ceci, la limitation par adresse s'effondre derrière un reverse
        proxy ou un tunnel** : toutes les requêtes arrivent de `127.0.0.1`, donc
        le monde entier partage un seul seau — un visiteur suffirait à bloquer
        tous les autres.

        `X-Forwarded-For` est fourni par le client et **ne doit jamais être cru
        sans condition** : n'importe qui pourrait s'inventer une adresse par
        requête et contourner le quota. On ne l'honore donc que si la connexion
        vient d'un proxy explicitement déclaré, et on prend la **dernière**
        valeur de la chaîne — celle ajoutée par ce proxy, la seule qu'il n'a pas
        recopiée du client.
        """
        directe = self.client_address[0] if self.client_address else "inconnu"
        if directe not in Etat.proxys:
            return directe
        chaine = self.headers.get("X-Forwarded-For", "")
        maillons = [m.strip() for m in chaine.split(",") if m.strip()]
        return maillons[-1] if maillons else directe

    def _debit_ok(self, cout: float | None = None) -> bool:
        """Vrai si la requête passe. Répond 429 elle-même sinon."""
        if Etat.limiteur is None:
            return True
        if cout is None:
            cout = self.COUTS.get(self.path, 1.0)
        attente = Etat.limiteur.verifier(self._adresse(), time.monotonic(), cout)
        if attente <= 0.0:
            return True
        secondes = max(1, int(attente + 0.999))
        corps = json.dumps(
            {
                "erreur": "trop de requêtes — le service est partagé",
                "reessayer_dans_s": secondes,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(429)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Retry-After", str(secondes))
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)
        return False

    def _cors(self) -> None:
        """Autorise l'appel depuis une page hébergée ailleurs.

        Nécessaire si la page vit sur un site personnel et que le service
        tourne sur une autre machine. L'origine autorisée est un paramètre :
        `*` ouvre à tous, ce qui est acceptable pour un service public sans
        authentification ni écriture, mais doit rester un choix explicite.
        """
        if Etat.origine:
            self.send_header("Access-Control-Allow-Origin", Etat.origine)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Requête préalable du navigateur avant un appel croisé."""
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code: int, charge: dict) -> None:
        corps = json.dumps(charge, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def do_GET(self) -> None:  # noqa: N802
        if not self._debit_ok(self.COUT_STATIQUE):
            return
        chemin = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        fichier = (WEB / chemin).resolve()
        if not fichier.is_file() or WEB.resolve() not in fichier.parents:
            self._json(404, {"erreur": "introuvable"})
            return
        corps = fichier.read_bytes()
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css"}
        self.send_response(200)
        self.send_header("Content-Type", types.get(fichier.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def do_POST(self) -> None:  # noqa: N802
        # Le débit est vérifié AVANT de lire le corps : sinon un robot ferait
        # transiter des mégaoctets pour se voir refuser ensuite.
        if not self._debit_ok():
            return
        taille = int(self.headers.get("Content-Length", 0))
        if taille > 2_000_000:
            self._json(413, {"erreur": "tracé trop volumineux"})
            return
        try:
            charge = json.loads(self.rfile.read(taille) or b"{}")
            points = np.asarray(charge.get("points", []), dtype=float)
        except Exception as exc:
            self._json(400, {"erreur": f"corps illisible : {exc}"})
            return
        if self.path != "/api/courbe" and (points.ndim != 2 or points.shape[1] != 2):
            self._json(400, {"erreur": "attendu : une liste de couples [x, y]"})
            return

        if self.path == "/api/courbe":
            self._courbe(charge)
            return
        if self.path != "/api/formules":
            self._json(404, {"erreur": "endpoint inconnu"})
            return

        try:
            # `source: "donnees"` : l'ordonnée n'est pas retournée (un fichier de
            # mesures n'a pas l'axe inversé d'un canvas) et la formule est rendue
            # dans les unités fournies, pas dans les coordonnées normalisées.
            importe = charge.get("source") == "donnees"
            rep = formules_depuis_trace(
                points,
                Etat.modele,
                Etat.device,
                Etat.rng,
                beam=Etat.beam,
                retourner_y=not importe,
                executor=Etat.pool,
            )
        except Exception as exc:  # un OOM du voisin ne doit pas tuer le serveur
            self._json(503, {"erreur": f"{type(exc).__name__} : {exc}"})
            return

        self._json(
            200,
            {
                "ok": rep.ok,
                "raison": rep.raison,
                "univalue": rep.univalue,
                "n_points": rep.n_points,
                "latence_ms": rep.latence_ms,
                "y_scale": rep.y_scale,
                "y_offset": rep.y_offset,
                "x_min": rep.x_min,
                "x_max": rep.x_max,
                "domaine_apercu": rep.domaine_apercu,
                "formules": [
                    {
                        "expression": f.expression,
                        "valorisee": f.valorisee,
                        "complexite": f.complexite,
                        "r2": f.r2,
                        "constantes": f.constantes,
                        "prefixe": f.prefixe,
                        "principale": f.principale,
                        "apercu": f.apercu,
                    }
                    for f in rep.formules
                ],
            },
        )

    #: Bornes du rééchantillonnage à la demande. Le domaine est plafonné parce
    #: qu'une formule évaluée trop loin ne donne plus que des infinis, et que
    #: rien ne justifie de calculer ce que personne ne peut lire.
    DOMAINE_MAX = 1000.0
    N_COURBE_MAX = 2000

    def _courbe(self, charge: dict) -> None:
        """Rééchantillonne une formule déjà trouvée sur un domaine plus large.

        Le navigateur en a besoin dès qu'on dézoome au-delà de l'aperçu initial.
        Aucun modèle, aucun ajustement : juste une évaluation, donc quelques
        centaines de microsecondes.

        **C'est le serveur qui évalue, pas le navigateur.** Réimplémenter la
        grammaire en JavaScript créerait deux vérités qui divergeraient au
        premier opérateur ajouté — même raison que pour l'aperçu initial.
        """
        from curvy.data.expr import evaluate, from_prefix

        try:
            noeud = from_prefix(str(charge.get("prefixe", "")).split())
            consts = [float(v) for v in charge.get("constantes", [])]
            domaine = min(abs(float(charge.get("domaine", 3.0))), self.DOMAINE_MAX)
            n = min(int(charge.get("n", 600)), self.N_COURBE_MAX)
        except Exception as exc:
            self._json(400, {"erreur": f"requête de courbe illisible : {exc}"})
            return
        if domaine <= 0 or n < 2:
            self._json(400, {"erreur": "domaine ou nombre de points invalide"})
            return

        grille = np.linspace(-domaine, domaine, n)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            try:
                y = evaluate(noeud, grille, consts)
            except Exception as exc:
                self._json(400, {"erreur": f"évaluation impossible : {exc}"})
                return
        self._json(
            200,
            {
                "domaine": domaine,
                "apercu": [None if not np.isfinite(v) else round(float(v), 5) for v in y],
            },
        )

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="exp-005")
    ap.add_argument("--preset", default="v1")
    ap.add_argument("--checkpoint", default="best.pt")
    #: 48, rendu abordable par le cache clé/valeur (514 ms contre 820 sans).
    #: Progression mesurée du taux rendu : 0,693 à beam 8, 0,719 à 24, **0,742
    #: à 48**. Et surtout, le poste « aucun candidat n'était bon » tombe de 5,7
    #: à **0,6 point** : le rappel atteint 0,811 pour un oracle à 0,816. Le
    #: modèle propose donc déjà le plafond de la tâche ; tout ce qui manque est
    #: dans la sélection.
    ap.add_argument("--beam", type=int, default=48)
    #: 8001 et non 8000 : le port 8000 est le défaut de trop d'outils, et la
    #: collision se manifeste par un « Address already in use » au démarrage.
    ap.add_argument("--port", type=int, default=8001)
    #: Par défaut on n'écoute QUE en local. Exposer un serveur d'inférence sur
    #: le réseau est une décision, pas un réglage par défaut.
    ap.add_argument("--host", default="127.0.0.1")
    #: Débit par adresse : rafale puis régime permanent. Les défauts laissent
    #: passer cinq requêtes d'affilée — un humain qui dessine plusieurs courbes
    #: — puis une toutes les deux secondes.
    ap.add_argument("--rafale", type=float, default=5.0)
    ap.add_argument("--par-seconde", type=float, default=0.5)
    #: Débit global. Le GPU soutient environ 1,3 prédiction par seconde ; on
    #: plafonne un peu au-dessus pour que la file se vide, pas pour la remplir.
    ap.add_argument("--rafale-globale", type=float, default=20.0)
    ap.add_argument("--par-seconde-global", type=float, default=2.0)
    ap.add_argument("--sans-limite", action="store_true", help="désactive la limitation")
    ap.add_argument(
        "--origine",
        default="",
        help="origine autorisée pour les appels croisés, ex. https://exemple.fr ou *",
    )
    ap.add_argument(
        "--proxy-de-confiance",
        default="",
        help=(
            "adresses d'un reverse proxy ou d'un tunnel, séparées par des virgules, "
            "dont l'en-tête X-Forwarded-For sera cru — ex. 127.0.0.1. "
            "SANS CELA, la limitation par adresse ne distingue plus les visiteurs."
        ),
    )
    args = ap.parse_args(argv)

    limiteur = (
        None
        if args.sans_limite
        else Limiteur(
            par_adresse=(args.rafale, args.par_seconde),
            global_=(args.rafale_globale, args.par_seconde_global),
        )
    )
    Etat.origine = args.origine
    Etat.proxys = frozenset(a.strip() for a in args.proxy_de_confiance.split(",") if a.strip())
    charger(args.run, args.preset, args.checkpoint, args.beam, limiteur)
    serveur = ThreadingHTTPServer((args.host, args.port), Handler)
    if Etat.proxys:
        print(f"proxy   : X-Forwarded-For cru depuis {sorted(Etat.proxys)}", flush=True)
    elif args.host == "127.0.0.1":
        print(
            "proxy   : aucun. Si un reverse proxy est devant, ajouter\n"
            "          --proxy-de-confiance 127.0.0.1, sinon tous les visiteurs\n"
            "          partageront un seul quota.",
            flush=True,
        )
    if limiteur is None:
        print("débit   : AUCUNE LIMITE — à réserver à un usage local", flush=True)
    else:
        print(
            f"débit   : {args.rafale:.0f} en rafale puis {args.par_seconde:.2g}/s par adresse ; "
            f"{args.rafale_globale:.0f} puis {args.par_seconde_global:.2g}/s au total",
            flush=True,
        )
    print("", flush=True)
    for url in _urls(args.host, args.port):
        print(f"démo : {url}", flush=True)
    if args.host == "0.0.0.0":  # noqa: S104
        print(
            "\nOUVERT AU RÉSEAU. Le service n'écrit rien sur disque et le débit est\n"
            "  limité, mais il n'y a AUCUNE AUTHENTIFICATION : qui atteint ce port\n"
            "  consomme le GPU, dans la limite du quota.\n"
            "  Le pare-feu devrait n'autoriser que le sous-réseau voulu, plutôt que\n"
            "  toutes les interfaces de la machine :\n"
            f"    sudo ufw allow from <votre-sous-réseau>/24 to any port {args.port} proto tcp",
            flush=True,
        )
    # Un SIGTERM n'exécute aucun `finally` : sans ce gestionnaire, les
    # processus du pool survivent au serveur, rattachés à init. Constaté le
    # 2026-08-22 — huit orphelins à 534 Mio, soit 4,3 Gio, après un simple
    # `kill`. Sur un service qu'on redémarre, la fuite s'accumule.
    import signal

    def _arreter(signum, cadre):  # noqa: ARG001
        print(f"\narrêt (signal {signum})", flush=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _arreter)
    signal.signal(signal.SIGINT, _arreter)

    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("arrêt", flush=True)
    finally:
        if Etat.pool is not None:
            Etat.pool.shutdown(wait=True, cancel_futures=True)
            print("pool arrêté", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
