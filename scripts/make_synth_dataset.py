#!/usr/bin/env python3
from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import albumentations as A
import torch
from torchvision.utils import save_image

try:
    from tqdm import trange
except Exception:
    trange = None  # fallback to manual logging

# ---------------------------
# Helpers
# ---------------------------
def list_images(src_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in src_dir.rglob("*") if p.suffix.lower() in exts]

def load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.0
    return arr

def to_tensor_chw01(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()

def save_pair(x01: torch.Tensor, m01: torch.Tensor, out_img: Path, out_mask: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    save_image(x01.clamp(0, 1), str(out_img))
    save_image(m01.float().clamp(0, 1), str(out_mask))

# ---------------------------
# Mask generation
# ---------------------------
def random_irregular_mask(h: int, w: int, min_strokes=6, max_strokes=12) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    num = np.random.randint(min_strokes, max_strokes + 1)
    for _ in range(num):
        pts = []
        for _ in range(np.random.randint(4, 8)):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            pts.append([x, y])
        thickness = int(max(8, min(h, w) * np.random.uniform(0.01, 0.04)))
        cv2.polylines(mask, [np.array(pts, np.int32)], isClosed=False, color=1, thickness=thickness)
        cv2.line(
            mask,
            (np.random.randint(0, w), np.random.randint(0, h)),
            (np.random.randint(0, w), np.random.randint(0, h)),
            color=1,
            thickness=thickness,
        )
    return mask  # 0/1

# ---------------------------
# Edit modes
# ---------------------------
def edit_inpaint(img01: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    img255 = (img01 * 255).astype(np.uint8)
    mask255 = (mask01 * 255).astype(np.uint8)
    out255 = cv2.inpaint(img255, mask255, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return out255.astype(np.float32) / 255.0

def edit_blur(img01: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    k = int(np.random.choice([5, 7, 9, 11]))
    blurred = cv2.GaussianBlur(img01, (k, k), 0)
    m3 = np.repeat(mask01[..., None], 3, axis=2)
    return img01 * (1 - m3) + blurred * m3

def edit_noisefill(img01: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    noise = np.clip(np.random.normal(loc=img01.mean(), scale=0.05, size=img01.shape), 0, 1).astype(np.float32)
    m3 = np.repeat(mask01[..., None], 3, axis=2)
    return img01 * (1 - m3) + noise * m3

EDITORS = {
    "inpaint": edit_inpaint,
    "blur": edit_blur,
    "noise": edit_noisefill,
}

# ---------------------------
# Augs (resize, compress, jitter)
# ---------------------------
def build_augs(size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),
            A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.3),
        ]
    )

def apply_augs(aug: A.Compose, img01: np.ndarray, mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img255 = (img01 * 255).astype(np.uint8)
    mask255 = (mask01 * 255).astype(np.uint8)
    out = aug(image=img255, mask=mask255)
    img01_aug = out["image"].astype(np.float32) / 255.0
    mask01_aug = (out["mask"].astype(np.float32) / 255.0)
    mask01_aug = (mask01_aug > 0.5).astype(np.float32)
    return img01_aug, mask01_aug

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Make synthetic edited dataset (images + masks).")
    ap.add_argument("--src_dir", required=True, help="Folder with clean images")
    ap.add_argument("--out_dir", default="data/inpaint_synth", help="Output root folder")
    ap.add_argument("--num", type=int, default=1000, help="Total samples desired (pairs).")
    ap.add_argument("--size", type=int, default=256, help="Output square size")
    ap.add_argument("--modes", default="inpaint,blur,noise",
                    help="Comma-separated edit modes: inpaint, blur, noise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="Continue from existing files (does not overwrite).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out_imgs = out / "imgs"
    out_masks = out / "masks"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    paths = list_images(src)
    if not paths:
        raise SystemExit(f"No images found under: {src}")

    augs = build_augs(args.size)
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in EDITORS]
    if not modes:
        raise SystemExit("No valid modes provided. Use any of: inpaint, blur, noise")

    # --- RESUME LOGIC ---
    existing_imgs = sorted([p for p in out_imgs.glob("*.png")])
    existing_masks = sorted([p for p in out_masks.glob("*.png")])
    start_idx = 0
    if args.resume:
        # use the min count to stay consistent
        start_idx = min(len(existing_imgs), len(existing_masks))
        if start_idx >= args.num:
            print(f"Already have {start_idx} ≥ target {args.num}. Nothing to do.")
            return
        print(f"Resuming at index {start_idx} → target {args.num}")

    total_to_make = args.num
    # choose progress iterator
    iter_range = range(start_idx, total_to_make)
    if trange is not None:
        iter_range = trange(start_idx, total_to_make)

    print(f"Found {len(paths)} clean images. Generating up to {args.num} samples → {out}")
    for i in iter_range:
        p = paths[i % len(paths)]
        img01 = load_rgb(p)
        h, w = img01.shape[:2]

        mask01 = random_irregular_mask(h, w)
        mode = random.choice(modes)
        edited01 = EDITORS[mode](img01, mask01)

        edited01, mask01_aug = apply_augs(augs, edited01, mask01.astype(np.float32))

        x = to_tensor_chw01(edited01)           # (3,H,W)
        m = torch.from_numpy(mask01_aug)[None]  # (1,H,W)

        fname = f"{i:05d}.png"
        save_pair(x, m, out_imgs / fname, out_masks / fname)

        # non-tqdm periodic logging
        if trange is None and (i + 1) % 100 == 0:
            print(f"  [{i+1}/{args.num}]")

    print("Done. Sample files:")
    print(f"  {out_imgs / f'{start_idx:05d}.png'} (start)")
    print(f"  {out_imgs / f'{args.num-1:05d}.png'} (end)")

if __name__ == "__main__":
    main()