#!/usr/bin/env bash

set -euo pipefail

# Automatisches Commit + Push Skript
# Nutzung:
#   ./git-auto.sh "Meine Commit Nachricht"
#   ./git-auto.sh            # nutzt automatische Commit Nachricht

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Fehler: Dieses Verzeichnis ist kein Git-Repository."
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"

if [[ -z "${branch}" ]]; then
  echo "Fehler: Konnte aktuellen Branch nicht ermitteln."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  commit_message="$*"
else
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  commit_message="Auto-Commit ${timestamp}"
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Keine Änderungen gefunden. Nichts zu committen."
  exit 0
fi

echo "Staging aller Änderungen..."
git add -A

if [[ -z "$(git diff --cached --name-only)" ]]; then
  echo "Keine gestagten Änderungen gefunden. Abbruch."
  exit 0
fi

echo "Erstelle Commit: ${commit_message}"
git commit -m "${commit_message}"

echo "Push nach origin/${branch}..."
git push origin "${branch}"

echo "Fertig: Commit und Push erfolgreich."
