import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class InpaintSynthDataset(Dataset):
    """
    Creates synthetic 'edited' samples by removing random regions and inpainting.
    Returns (image, mask) where mask=1 on edited pixels, else 0.
    Also yields clean negatives with zero mask.
    """
    def __init__(self, src_dir, size=256, num_clean_ratio=0.3):
        self.paths = [p for p in Path(src_dir).rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        if not self.paths:
            raise FileNotFoundError(f"No images under {src_dir}")
        self.size = size
        self.num_clean_ratio = num_clean_ratio
        self.aug = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
            A.GaussNoise(var_limit=(0.0, 25.0), p=0.2),
            A.ColorJitter(0.1,0.1,0.1,0.05, p=0.2),
            ToTensorV2(),
        ])

    def _random_irregular_mask(self, h, w):
        mask = np.zeros((h,w), np.uint8)
        num_strokes = np.random.randint(3, 8)
        for _ in range(num_strokes):
            l = np.random.randint(20, 60)
            p = (np.random.randint(0,w), np.random.randint(0,h))
            for _ in range(l):
                rr = np.random.randint(8, 20)
                cv2.circle(mask, p, rr, 255, -1)
                p = (np.clip(p[0]+np.random.randint(-10,10),0,w-1),
                     np.clip(p[1]+np.random.randint(-10,10),0,h-1))
        return mask

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        make_clean = np.random.rand() < self.num_clean_ratio
        if make_clean:
            mask = np.zeros((h,w), np.uint8)
            edited = img.copy()
        else:
            mask = self._random_irregular_mask(h, w)
            edited = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        aug = self.aug(image=edited)
        img_t = aug["image"].float()/255.0
        # recompute mask after resize/pad
        mask_resized = A.resize(mask, img_t.shape[1], img_t.shape[2], interpolation=cv2.INTER_NEAREST)
        mask_t = np.expand_dims(mask_resized, 0) / 255.0
        return img_t, mask_t.astype(np.float32)