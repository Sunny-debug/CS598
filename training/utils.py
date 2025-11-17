from __future__ import annotations
import csv, random
from pathlib import Path
import numpy as np
import torch

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.n = 0
    @property
    def avg(self): return self.sum / max(1, self.n)
    def update(self, val, n=1): self.sum += float(val) * int(n); self.n += int(n)

def save_checkpoint(model: torch.nn.Module, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra: payload.update(extra)
    torch.save(payload, path)

class EarlyStopper:
    def __init__(self, patience=8, mode="max"):
        self.patience = patience; self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.bad_epochs = 0; self.should_stop = False
    def update(self, metric: float) -> bool:
        improved = (metric > self.best_metric) if self.mode == "max" else (metric < self.best_metric)
        if improved:
            self.best_metric = metric; self.bad_epochs = 0; return True
        self.bad_epochs += 1; self.should_stop = self.bad_epochs >= self.patience; return False

class CSVLogger:
    def __init__(self, path: Path, header: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path; self.header = header
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)   # <- fixed

    def write(self, row: list):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

def set_deterministic(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True