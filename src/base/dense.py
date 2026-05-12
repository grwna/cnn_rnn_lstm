from typing import Optional

import numpy as np
from src.base.activations import Activation, Linear

class Dense:
    def __init__( self,
        kernel: np.ndarray,
        bias: np.ndarray,
        activation: Optional[Activation] = None,
    ) -> None:
        self.weights = kernel
        self.bias = bias
        self.activation = activation or Linear() # default to linear

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = (x @ self.weights) + self.bias
        return self.activation(z)
