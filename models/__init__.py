"""
Models package initializer.

This makes `models` a proper Python package so imports like
`from models.unet import UNNetSmall` work.
"""

# Optionally, you can re-export key classes here:
from .unet import UNNetSmall

__all__ = ["UNNetSmall"]