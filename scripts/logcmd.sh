#!/usr/bin/env bash
# logcmd.sh — exécute une commande, affiche sa sortie ET l'archive dans logs/commands.md
# Usage: ./scripts/logcmd.sh "description courte" -- <commande...>
# Règle projet n°1 : aucun chiffre dans la doc qui ne vienne d'ici.
set -uo pipefail
DESC="$1"; shift
[ "${1:-}" = "--" ] && shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/commands.md"
TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"
OUT="$(mktemp)"
"$@" > "$OUT" 2>&1
RC=$?
LINES=$(wc -l < "$OUT")
{
  echo ""
  echo "## [$TS] $DESC"
  echo ""
  echo '```console'
  echo "\$ $*"
  if [ "$LINES" -gt 120 ]; then
    RAW="logs/raw/$(date '+%Y%m%d-%H%M%S')-$(echo "$DESC" | tr -cd '[:alnum:]' | cut -c1-30).log"
    cp "$OUT" "$ROOT/$RAW"
    head -60 "$OUT"
    echo "[... $((LINES-60)) lignes tronquées, sortie complète : $RAW ...]"
  else
    cat "$OUT"
  fi
  echo '```'
  echo ""
  echo "*code retour : $RC*"
} >> "$LOG"
cat "$OUT"; rm -f "$OUT"
exit $RC
