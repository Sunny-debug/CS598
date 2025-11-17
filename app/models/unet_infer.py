# app/models/unet_infer.py
import torch
import numpy as np
from PIL import Image

class UNetInfer:
    """
    Lightweight TorchScript UNet inference wrapper.
    """

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Always use autocast for speed on GPU
        self.amp_device = "cuda" if self.device == "cuda" else "cpu"

    @torch.no_grad()
    def predict(self, img: Image.Image) -> np.ndarray:
        """
        Args:
            img: RGB PIL image (any size)
        Returns:
            mask: np.ndarray (H,W) float32 in [0,1]
        """
        img = img.resize((256, 256))
        arr = np.asarray(img).astype("float32") / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))[None, ...].to(self.device)

        with torch.amp.autocast(self.amp_device, enabled=self.device == "cuda"):
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        return probs