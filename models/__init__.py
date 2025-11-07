"""
Models package initializer.

This makes `models` importable directly via `from models import UNNetSmall`.
"""

# Try to import safely depending on your file content
try:
    from .unet import UNNetSmall  # your existing class
except ImportError:
    # fallback if the class is named UNet instead
    from .unet import UNet as UNNetSmall

__all__ = ["UNNetSmall"]