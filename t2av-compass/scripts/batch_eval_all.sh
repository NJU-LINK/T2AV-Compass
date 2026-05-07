#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "scripts/batch_eval_all.sh is kept for compatibility."
echo "Prefer running from the repository root: bash run_objective_batch.sh"

bash "${SCRIPT_DIR}/eval_all_metrics.sh" "$@"
