from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PY = REPO_ROOT / ".venv" / "bin" / "python"

EVAL_MODULES = [
    "thesis_uq.eval.eval_best_baseline",
    "thesis_uq.eval.eval_best_mc_dropout",
    "thesis_uq.eval.eval_best_laplace",
    "thesis_uq.eval.eval_best_edl",
]

def main():
    env = os.environ.copy()

    # choose what you want to evaluate
    env["DATASET"] = env.get("DATASET", "cell2cell")

    # IMPORTANT: for clean protocol, I'd recommend 5-14 (since 1-4 used for tuning)
    # If you truly want 4-14, set it here:
    env["TRAIN_SEEDS"] = env.get("TRAIN_SEEDS", "5-14")
    env["SPLIT_SEED"] = env.get("SPLIT_SEED", "42")
    env["DEVICE"] = env.get("DEVICE", "cpu")

    for mod in EVAL_MODULES:
        print(f"\n=== RUN {mod} ===")
        subprocess.check_call([str(PY), "-m", mod], cwd=str(REPO_ROOT), env=env)

if __name__ == "__main__":
    main()
