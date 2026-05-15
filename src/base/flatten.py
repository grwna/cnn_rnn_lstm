import numpy as np

class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
