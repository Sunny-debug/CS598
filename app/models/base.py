from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image

class DeepfakeModel(ABC):
    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @abstractmethod
    def predict_proba(self, img: Image.Image) -> Dict[str, float]:
        """
        Returns class probabilities for {"real", "fake"} that sum to 1.0
        """
        ...