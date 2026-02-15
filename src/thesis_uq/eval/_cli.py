from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import os

@dataclass(frozen=True)
class EvalArgs:
    repo_root: Path
    dataset: str
    split_seed: int
    seed_lo: int
    seed_hi: int
    device: str

def get_repo_root() -> Path:
    # .../src/thesis_uq/eval/_cli.py -> repo root is 4 levels up
    return Path(__file__).resolve().parents[3]

def parse_eval_args(method: str) -> EvalArgs:
    """
    Standard CLI for all eval scripts.
    Works with CLI args OR env vars as defaults.
    """
    p = argparse.ArgumentParser(description=f"Eval best {method} model")
    p.add_argument("--dataset", default=os.getenv("DATASET", "cell2cell"))
    p.add_argument("--split-seed", type=int, default=int(os.getenv("SPLIT_SEED", "42")))
    p.add_argument("--seed-lo", type=int, default=int(os.getenv("SEED_LO", "5")))
    p.add_argument("--seed-hi", type=int, default=int(os.getenv("SEED_HI", "15")))
    p.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    ns = p.parse_args()

    return EvalArgs(
        repo_root=get_repo_root(),
        dataset=ns.dataset,
        split_seed=ns.split_seed,
        seed_lo=ns.seed_lo,
        seed_hi=ns.seed_hi,
        device=ns.device,
    )
