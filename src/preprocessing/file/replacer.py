from abc import ABC, abstractmethod
from src.maths import mean
import numpy as np

class AbstractReplacer(ABC):
    @abstractmethod
    def replace(original_without_missing: np.ndarray) -> float | None:
        pass

class NoneReplacer(AbstractReplacer):
    @staticmethod
    def replace(original_without_missing):
        return None

class MeanReplacer(AbstractReplacer):
    @staticmethod
    def replace(original_without_missing):
        return mean(original_without_missing)
