#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# (optional) use your venv if needed
# source .venv/bin/activate

echo "=== Starting overnight queue at $(date) ==="

# DON'T include baseline_cell2cell again if it's already running now.
# Add what you want next, e.g. telco methods:
python -m thesis_uq.gridsearch.baseline
python -m thesis_uq.gridsearch.mc_dropout
python -m thesis_uq.gridsearch.laplace
python -m thesis_uq.gridsearch.edl

# (optional) run eval scripts after gridsearch
python -m thesis_uq.eval.eval_best_baseline
python -m thesis_uq.eval.eval_best_mc_dropout
python -m thesis_uq.eval.eval_best_laplace
python -m thesis_uq.eval.eval_best_edl

echo "=== Done at $(date) ==="
