from __future__ import annotations
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PairedMaskDataset(Dataset):
    """
    root/
      imgs/*.png|jpg
      masks/*.png|jpg (binary or 0..255)
    """
    def __init__(self, root: str | Path, size: int = 256, augment: bool = True):
        self.root = Path(root)
        self.img_paths = sorted((self.root / "imgs").glob("*"))
        self.mask_paths = sorted((self.root / "masks").glob("*"))
        assert len(self.img_paths) == len(self.mask_paths), "imgs/masks count mismatch"
        self.size = size
        self.augment = augment
        self._build_tfms()

    def _build_tfms(self):
        base = [A.Resize(self.size, self.size, interpolation=cv2.INTER_AREA)]
        if self.augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
                A.ColorJitter(p=0.4),
                A.ElasticTransform(alpha=20, sigma=5, alpha_affine=5, p=0.1),
            ]
        else:
            aug = []
        norm = [A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)), ToTensorV2()]
        self.tfms = A.Compose(base + aug + norm)

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx: int):
        ip = str(self.img_paths[idx]); mp = str(self.mask_paths[idx])

        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(mp)
        mask = (mask > 127).astype("float32")  # 0/1

        out = self.tfms(image=img, mask=mask)
        x = out["image"]                    # torch.FloatTensor CxHxW
        y = out["mask"]                     # torch.FloatTensor HxW (after ToTensorV2)

        if isinstance(y, torch.Tensor) and y.ndim == 2:
            y = y.unsqueeze(0)              # 1xHxW
        elif not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y).unsqueeze(0)

        return x.float(), y.float()

    def clone_with_augment(self, augment: bool):
        return PairedMaskDataset(self.root, self.size, augment)