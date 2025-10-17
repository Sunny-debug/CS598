from __future__ import annotations
import torch, hashlib, os

class UNetRuntime:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.weights_sha256 = self._sha256(model_path)

    def _sha256(self, p: str) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @torch.inference_mode()
    def predict_mask(self, tensor):  # Bx3xHxW float32 [0..1]
        y = self.model(tensor)       # expect Bx1xHxW
        return y.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy()