import numpy as np


class Embedding:

    def __init__(self) -> None:
        self.W: np.ndarray = None  # set by load_weights()

    def load_weights(self, weights: dict) -> None:
        self.W = np.asarray(weights["weights"]["embeddings"])  # (vocab_size, embed_dim)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")
        return self.W[token_ids]