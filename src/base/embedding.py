import numpy as np

class Embedding:
    def __init__(self, embeddings: np.ndarray) -> None:
        # load bobot embedding matrix berukuran (vocab_size, embed_dim)
        self.embeddings = np.asarray(embeddings)
        self.vocab_size, self.embed_dim = self.embeddings.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        # menambahkan dimensi batch jika input hanya 1 sequence (1D)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # lookup token id ke matriks bobot
        return self.embeddings[x]