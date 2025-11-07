from __future__ import annotations
import argparse, json, os, platform, random, time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# keep your existing imports/paths
from training.datasets_paired import PairedMaskDataset
from training.losses import BCEDiceLoss
from training.eval_metrics import batch_dice_iou
from training.utils import (
    AverageMeter, save_checkpoint, EarlyStopper, CSVLogger, set_deterministic
)
from models import UNNetSmall  # matches your codebase

# -------------------- environment helpers --------------------
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

# -------------------- args --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paired_dir", required=True,
                   help="Root containing imgs/ and masks/ subfolders")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--export_ts", action="store_true", default=True)
    return p.parse_args()

# -------------------- dataloaders --------------------
def make_loaders(args, device: torch.device):
    ds = PairedMaskDataset(
        root=args.paired_dir,
        size=args.size,
        augment=True  # enable training-time augments
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty!")

    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=g)

    # Validation dataset should not augment
    ds_val.dataset = ds_val.dataset.clone_with_augment(False)

    workers = default_num_workers() if args.workers == -1 else args.workers
    pin = device.type == "cuda"
    train_loader = DataLoader(
        ds_train, batch_size=args.bs, shuffle=True,
        num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0)
    )
    val_loader = DataLoader(
        ds_val, batch_size=max(1, args.bs // 2), shuffle=False,
        num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0)
    )
    return train_loader, val_loader, workers, n_train, n_val

# -------------------- train/validate --------------------
def train_one_epoch(model, loader, optim, criterion, scaler, device, clip_grad: float):
    model.train()
    loss_meter = AverageMeter()
    for x, y in tqdm(loader, leave=False, desc="train"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if clip_grad and clip_grad > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optim)
        scaler.update()

        loss_meter.update(loss.item(), x.size(0))
    return loss_meter.avg

@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    for x, y in tqdm(loader, leave=False, desc="val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), x.size(0))

        dice, iou = batch_dice_iou(logits, y, threshold=0.5)
        dice_meter.update(dice, x.size(0))
        iou_meter.update(iou, x.size(0))
    return loss_meter.avg, dice_meter.avg, iou_meter.avg

# -------------------- main --------------------
def main():
    args = parse_args()
    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader, workers, n_train, n_val = make_loaders(args, device)

    model = UNNetSmall(in_ch=3, base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = BCEDiceLoss(bce_weight=0.6, dice_weight=0.4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(Path("outputs") / "train_log.csv",
                       header=["epoch", "lr", "train_loss", "val_loss", "val_dice", "val_iou"])
    stopper = EarlyStopper(patience=args.patience, mode="max")  # maximize dice

    print(f"Device: {device} | workers={workers} | train={n_train} val={n_val}")
    best_path = out_dir / "unet_small_best.pth"
    last_path = out_dir / "unet_small_last.pth"

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, args.clip_grad)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        logger.write([epoch, lr_now, tr_loss, val_loss, val_dice, val_iou])

        print(f"[{epoch:03d}/{args.epochs}] "
              f"lr={lr_now:.2e} train={tr_loss:.4f} val={val_loss:.4f} "
              f"dice={val_dice:.4f} iou={val_iou:.4f}")

        history.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr_loss, "val_loss": val_loss,
            "val_dice": val_dice, "val_iou": val_iou
        })

        # save "last"
        save_checkpoint(model, last_path)

        # save "best" by dice
        if stopper.update(val_dice):
            save_checkpoint(model, best_path, extra={"epoch": epoch, "val_dice": val_dice})

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # write meta
    with open(str(best_path) + ".meta.json", "w") as f:
        json.dump({"history": history, "best_metric": stopper.best_metric}, f, indent=2)

    # optional TorchScript export for inference service
    if args.export_ts:
        ckpt = torch.load(best_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)    
        model.eval().cpu()
        example = torch.randn(1, 3, args.size, args.size)
        ts = torch.jit.trace(model, example)
        ts_path = out_dir / "unet_small_best.ts.pt"
        ts.save(str(ts_path))
        print(f"TorchScript exported to {ts_path}")

if __name__ == "__main__":
    main()