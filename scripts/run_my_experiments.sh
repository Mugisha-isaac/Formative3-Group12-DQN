#!/usr/bin/env bash
# Run only exp11–exp20 (your infrastructure sweep). Skips exp1–exp10.
# Usage: ./run_my_experiments.sh
# Optional: MEMBER_NAME="Your Name" TOTAL_TIMESTEPS=500000 ./run_my_experiments.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/src"

export RUN_EXTENDED_EXPERIMENTS_ONLY=1
export MEMBER_NAME="${MEMBER_NAME:-Sid}"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

exec "$PY" train.py "$@"
