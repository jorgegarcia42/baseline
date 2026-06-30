#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python scripts/download_data.py
python main.py

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user restart baseline.service
fi
