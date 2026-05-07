import numpy as np
from abc import ABC, abstractmethod
from .tensor import Tensor

class Normalization(ABC):
    @abstractmethod
    def normalize(self, input_data: Tensor) -> Tensor:
        pass

class RMSNorm(Normalization):
    def __init__(self, input_size, epsilon: float=1e-8):
        self.learnable = Tensor(np.ones((1, input_size)))
        self.epsilon = epsilon

    def normalize(self, input_data: Tensor) -> Tensor:
        ms = (input_data**2).mean(axis=-1, keepdims=True)
        rms = (ms + self.epsilon) ** 0.5
        return (input_data / rms) * self.learnable
    
