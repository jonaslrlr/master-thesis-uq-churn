from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

@dataclass
class RunMeta:
    dataset: str
    method: str
    seed: int
    tag: str

def save_uq_scores_npz(out_file: Path, *, y_valid, p_valid, u_valid, y_test, p_test, u_test):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, y_valid=y_valid, p_valid=p_valid, u_valid=u_valid,
                      y_test=y_test,   p_test=p_test,   u_test=u_test)

def save_metrics_json(out_file: Path, meta: RunMeta, metrics: dict, config: dict):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": asdict(meta), "metrics": metrics, "config": config}
    out_file.write_text(json.dumps(payload, indent=2))
