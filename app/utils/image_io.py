from __future__ import annotations

from io import BytesIO
from typing import Iterable, Set

from fastapi import UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError


async def validate_and_load_image(
    file: UploadFile,
    max_mb: int = 50,
    allowed_content_types: Iterable[str] = ("image/jpeg", "image/png"),
) -> Image.Image:
    """
    Validates size/content-type, loads as RGB PIL.Image, and lightly normalizes.
    """
    allowed: Set[str] = set(allowed_content_types)
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    size_mb = len(raw) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(status_code=413, detail=f"Image too large (> {max_mb} MB)")

    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Optional light downscale to keep inference predictable
    img = downscale_if_needed(img, max_side=1024)
    return img


def downscale_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    ratio = max_side / float(max(w, h))
    new_size = (int(w * ratio), int(h * ratio))
    return img.resize(new_size)