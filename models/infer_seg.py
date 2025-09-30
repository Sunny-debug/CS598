import base64, io
import numpy as np
from PIL import Image

def preprocess_pil(img: Image.Image, size=384):
    im = img.convert("RGB").resize((size,size))
    arr = (np.asarray(im).astype("float32")/255.0).transpose(2,0,1)  # CHW
    return arr

def mask_to_b64(mask01: np.ndarray):
    """mask01: (H,W) float [0..1] -> base64 PNG (grayscale)."""
    m8 = (np.clip(mask01,0,1)*255).astype("uint8")
    im = Image.fromarray(m8, mode="L")
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")