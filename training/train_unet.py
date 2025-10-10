from __future__ import annotations
import argparse, json, os, platform, random, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.datasets import InpaintSynthDataset
from training.datasets_paired import PairedMaskDataset
from models import UNNetSmall


def is_wsl() -> bool:
    try:
        with open("/proc/sys/kernel/osrelease") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False

def default_num_workers() -> int:
    if platform.system().lower().startswith("win") or is_wsl():
        return 0
    return max(1, os.cpu_count() // 2)

def set_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", help="Directory of clean images (on-the-fly synth)")
    p.add_argument("--paired_dir", help="Pre-generated paired dataset (imgs/masks)")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=-1)
    return p.parse_args()

def make_loader(args, device: torch.device):
    if args.paired_dir:
        ds = PairedMaskDataset(args.paired_dir)
    elif args.src_dir:
        ds = InpaintSynthDataset(args.src_dir, size=args.size)
    else:
        raise SystemExit("Need either --paired_dir or --src_dir")
    if len(ds) == 0: raise RuntimeError("Dataset is empty!")

    workers = default_num_workers() if args.workers == -1 else args.workers
    pin = device.type == "cuda"
    return DataLoader(ds, batch_size=args.bs, shuffle=True,
                      num_workers=workers, pin_memory=pin,
                      persistent_workers=(workers>0)), workers

def train_one_epoch(model, loader, optim, crit, scaler, device):
    model.train(); total=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(x); loss = crit(logits,y)
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        total += float(loss.item())
    return total/len(loader)

def main():
    args = parse_args(); set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    loader, workers = make_loader(args, device)

    model = UNNetSmall(in_ch=3, base=32).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    print(f"Training on {device} | samples={len(loader.dataset)} bs={args.bs} workers={workers}")

    history=[]
    for ep in range(1,args.epochs+1):
        loss=train_one_epoch(model, loader, optim, crit, scaler, device)
        history.append({"epoch":ep,"loss":loss})
        print(f"[epoch {ep:03d}/{args.epochs}] loss={loss:.4f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt=Path(args.out_dir)/"unet_small_paired.pth"
    torch.save(model.state_dict(), ckpt)
    with open(str(ckpt)+".meta.json","w") as f: json.dump(history,f,indent=2)
    print(f"Saved {ckpt}")

if __name__=="__main__":
    main()