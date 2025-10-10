from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class PairedMaskDataset(Dataset):
    """
    Loads paired edited images and binary masks from:
      root/
        imgs/00000.png ...
        masks/00000.png ...
    Returns (image_tensor[3,H,W], mask_tensor[1,H,W]) in [0,1].
    """
    def __init__(self, root: str):
        root = Path(root)
        self.imgs_dir = root / "imgs"
        self.masks_dir = root / "masks"
        exts = {".png", ".jpg", ".jpeg"}
        self.imgs: List[Path] = sorted([p for p in self.imgs_dir.iterdir() if p.suffix.lower() in exts])
        if not self.imgs:
            raise FileNotFoundError(f"No images found under {self.imgs_dir}")
        # ensure mask exists
        self.pairs: List[Tuple[Path,Path]] = []
        for ip in self.imgs:
            mp = (self.masks_dir / ip.stem).with_suffix(".png")
            if not mp.exists():
                raise FileNotFoundError(f"Missing mask for {ip.name}: {mp}")
            self.pairs.append((ip, mp))

    def __len__(self): return len(self.pairs)

    def _load_img01(self, p: Path) -> np.ndarray:
        arr = np.asarray(Image.open(str(p)).convert("RGB"), dtype=np.float32) / 255.0
        return arr

    def _load_mask01(self, p: Path) -> np.ndarray:
        m = np.asarray(Image.open(str(p)).convert("L"), dtype=np.float32) / 255.0
        m = (m > 0.5).astype(np.float32)
        return m

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img01 = self._load_img01(ip)             # HWC
        mask01 = self._load_mask01(mp)           # HW
        x = torch.from_numpy(img01.transpose(2,0,1))  # CHW
        y = torch.from_numpy(mask01)[None]            # 1HW
        return x, y