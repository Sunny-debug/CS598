from PIL import Image
from io import BytesIO

ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}

def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(BytesIO(data))
    img = img.convert("RGB")
    if img.format and img.format.upper() not in ALLOWED_FORMATS:
        # Still allow, but we’ve normalized to RGB; format is advisory.
        pass
    return img

def downscale_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    ratio = max_side / float(max(w, h))
    new_size = (int(w * ratio), int(h * ratio))
    return img.resize(new_size)