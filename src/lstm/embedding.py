import numpy as np


class Embedding:
    """Token-index -> dense vector lookup.

    Wraps a simple weight matrix W of shape (vocab_size, embed_dim).
    Forward pass is just an index operation — no arithmetic needed.

    Attributes
    ----------
    W : np.ndarray, shape (vocab_size, embed_dim)
        Embedding matrix loaded from a trained Keras Embedding layer.
    """

    def __init__(self) -> None:
        self.W: np.ndarray = None  # set by load_weights()

    def load_weights(self, weights: dict) -> None:
        """Load weights from weight_loaders.load_weights() result.

        Parameters
        ----------
        weights : dict
            Entry from load_weights() result, e.g.
            { "type": "Embedding", "weights": {"embeddings": np.ndarray}, ... }
        """
        self.W = np.asarray(weights["weights"]["embeddings"])  # (vocab_size, embed_dim)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embedding vectors for a batch of token ids.

        Parameters
        ----------
        token_ids : np.ndarray, shape (batch_size,) or (batch_size, seq_len)
            Integer token indices.

        Returns
        -------
        np.ndarray, shape (..., embed_dim)
            Embedding vectors.
        """
        if self.W is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")
        return self.W[token_ids]
