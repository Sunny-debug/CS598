# TorchScript lightweight inference (for deployment)
from __future__ import annotations
import torch, numpy as np
from PIL import Image
from pathlib import Path

class UNetInfer:
    def __init__(self, model_path="checkpoints/unet_small_traced.pt", device="cuda", size=256):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.size = size

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB").resize((self.size, self.size)), dtype=np.float32)/255.0
        return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img)
        logits = self.model(x)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs