"""
Models package initializer.

This makes `models` a proper Python package so imports like
`from models.unet import UNNetSmall` or `UNetSmall` work.
"""

# ---- CHANGE HERE ----
# Import both spellings from unet.py
from .unet import UNNetSmall, UNetSmall

__all__ = ["UNNetSmall", "UNetSmall"]