import numpy as np
from abc import ABC, abstractmethod
from .tensor import Tensor

class Loss(ABC):
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: Tensor) -> float:
        pass

class MSE(Loss):
    def __call__(self, y_true, y_pred): 
        return ((y_pred - y_true) ** 2).mean()

class BinaryCrossEntropy(Loss):
    def __call__(self, y_true, y_pred): 
        return (-(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())).mean()

class CategoricalCrossEntropy(Loss):
    def __call__(self, y_true, y_pred): 
        return (-(y_true * y_pred.log())).sum(axis=-1).mean()
