#!/usr/bin/env bash
# Lance les entraînements dans tmux, logs complets vers logs/raw/.
# Les runs survivent à une déconnexion SSH ; `make train` les reprendrait au
# dernier checkpoint en cas de coupure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CURVY_CUDA_ALLOW=GPU-1234abcd-0000-0000-0000-000000000000

SESSION="curvy-train"
STAMP="$(date '+%Y%m%d-%H%M%S')"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "session tmux '$SESSION' déjà active — attache-toi avec : tmux attach -t $SESSION"
  exit 1
}

tmux new-session -d -s "$SESSION" -c "$ROOT" "
  set -o pipefail
  echo '=== Contrôle VRAM avant lancement ==='
  .venv/bin/python -m curvy.cli_gpu || exit 1

  echo '=== exp-001 : preset small, 20 000 steps ==='
  .venv/bin/python -m curvy.train.run \
      --run-name exp-001 --preset small --steps 20000 \
      --eval-every 1000 --log-every 100 --batch-size 512 --no-resume \
      2>&1 | tee logs/raw/${STAMP}-exp-001-small.log

  echo '=== exp-002 : preset v1, 20 000 steps ==='
  .venv/bin/python -m curvy.train.run \
      --run-name exp-002 --preset v1 --steps 20000 \
      --eval-every 1000 --log-every 100 --batch-size 512 --no-resume \
      2>&1 | tee logs/raw/${STAMP}-exp-002-v1.log

  echo '=== terminé ==='
  sleep 3600
"
echo "session tmux '$SESSION' lancée."
echo "  suivre  : tmux attach -t $SESSION   (détacher : Ctrl-b d)"
echo "  logs    : logs/raw/${STAMP}-exp-00*.log"
echo "  courbes : .venv/bin/python scripts/plot_training.py --run exp-001"
