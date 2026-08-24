#!/usr/bin/env bash
# Lance UN entraînement dans tmux, log complet vers logs/raw/.
# Usage : ./scripts/launch_run.sh <run-name> <preset> <steps> [workers]
#
# Passe --exclure-test PAR DÉFAUT : les squelettes réservés au jeu
# de test sortent du flux. Sans ça, les chiffres du run sur le jeu de test ne
# valent rien — c'est arrivé une fois, le 2026-08-20, et le run a été avorté
# après huit secondes. Pour reproduire un run d'avant la Phase 6 :
# CURVY_SANS_EXCLUSION=1 ./scripts/launch_run.sh ...
#
# CURVY_REPRENDRE=1 reprend au dernier checkpoint au lieu de repartir de zéro.
# Le checkpoint porte le modèle, l'optimiseur, le scheduler et le compteur de
# steps : la reprise retrouve la position exacte dans le cosine, ce n'est pas
# un redémarrage déguisé. Sert quand on libère le GPU pour autre chose.
#
# Empreinte CPU volontairement limitée : la machine a 12 cœurs et les fait
# tourner pour autre chose. Un run qui les prend tous a déjà fait tomber le
# réseau de la machine (2026-08-19). Défaut 5 workers, `nice 10`, threads
# torch plafonnés — : 5 workers suffisent à alimenter le GPU.
# Le run survit à une déconnexion SSH ; `--resume` (défaut) reprendrait au
# dernier checkpoint après une coupure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN="${1:?usage: launch_run.sh <run-name> <preset> <steps>}"
PRESET="${2:?preset manquant}"
STEPS="${3:?steps manquant}"
WORKERS="${4:-5}"
# Le drapeau est écrit dans le config.json du run : c'est lui qui dira si les
# mesures sur le jeu de test veulent dire quelque chose.
EXCLUSION="--exclure-test"
[ -n "${CURVY_SANS_EXCLUSION:-}" ] && EXCLUSION=""
# Par défaut on repart de zéro : un run neuf ne doit jamais ramasser par
# accident le checkpoint d'un homonyme.
REPRISE="--no-resume"
[ -n "${CURVY_REPRENDRE:-}" ] && REPRISE=""

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CURVY_CUDA_ALLOW=GPU-1234abcd-0000-0000-0000-000000000000

# Plafonds CPU : le GPU est le client, pas le patron de cette machine.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SESSION="curvy-train"
STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG="logs/raw/${STAMP}-${RUN}-${PRESET}.log"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "session tmux '$SESSION' déjà active — attache-toi avec : tmux attach -t $SESSION"
  exit 1
}

# Refus net si la VRAM est prise : mieux vaut ne pas démarrer qu'OOM à 3 h de run.
.venv/bin/python -m curvy.cli_gpu

tmux new-session -d -s "$SESSION" -c "$ROOT" "
  set -o pipefail
  echo '=== ${RUN} : preset ${PRESET}, ${STEPS} steps, ${WORKERS} workers, nice 10 ==='
  echo '=== exclusion du jeu de test : ${EXCLUSION:-AUCUNE} ==='
  echo '=== reprise : ${REPRISE:-OUI, au dernier checkpoint} ==='
  nice -n 10 .venv/bin/python -m curvy.train.run \
      --run-name ${RUN} --preset ${PRESET} --steps ${STEPS} \
      --eval-every 1000 --log-every 100 --batch-size 512 \
      --workers ${WORKERS} ${EXCLUSION} ${REPRISE} \
      2>&1 | tee ${LOG}
  echo '=== terminé ==='
  sleep 3600
"
echo "session tmux '$SESSION' lancée pour ${RUN}."
echo "  suivre  : tmux attach -t $SESSION   (détacher : Ctrl-b d)"
echo "  log     : ${LOG}"
echo "  courbes : .venv/bin/python scripts/plot_training.py --run ${RUN}"
echo "  CPU     : ${WORKERS} workers + 1 process principal, nice 10, sur 12 cœurs"
echo "  test    : ${EXCLUSION:-AUCUNE EXCLUSION (les chiffres sur le jeu de test seront nuls)}"
echo "  reprise : ${REPRISE:-au dernier checkpoint}"
