from __future__ import annotations
import base64, io
from PIL import Image
import numpy as np

def mask_to_base64_png(mask: "np.ndarray | Image.Image") -> str:
    """
    Accepts: HxW (0/255) uint8 or PIL Image (mode 'L' or 'RGBA'/'RGB')
    Returns: base64 string of a lossless PNG without data: prefix.
    """
    if mask.ndim == 2:
        img = Image.fromarray(mask, mode="L")
    elif mask.ndim == 3 and mask.shape[2] == 3:
        img = Image.fromarray(mask, mode="RGB")
    else:
        raise ValueError("mask must be HxW or HxWx3 uint8")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")