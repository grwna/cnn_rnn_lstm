from typing import Dict, Optional
import numpy as np

from src.base.activations import Softmax
from src.base.dense import Dense
from lstm.embedding import Embedding
from lstm.lstm_cell import LSTMCell


class LSTMDecoder:
    """Image-captioning decoder — pre-inject method.

    Architecture (from spec / Show and Tell, Vinyals et al. 2015):

        x₋₁  = Dense_proj(CNN_feature)          <- image injected as first input
        x_t   = Embedding(token_t),  t ≥ 0
        sequence = [x₋₁, x₀, x₁, ..., x_{N-1}]
        h₀ = 0,  c₀ = 0  (Keras default)

        At each step:  H, C = LSTMCell(x_t, H, C)
        output logits  = Dense_out(H)  ->  softmax  ->  prob over vocab

    Usage
    -----
    1. Build and train the Keras model (see notebook).
    2. Call load_weights(weights_dict) with the result of
       utils.weight_loaders.load_weights(keras_model, spec).
    3. Call forward() or generate_caption().
    """

    def __init__(self) -> None:
        self.embedding  = Embedding()
        self.dense_proj = Dense(kernel=None, bias=None, activation=None)  # CNN -> embed_dim
        self.lstm_cell  = LSTMCell()
        self.dense_out  = Dense(kernel=None, bias=None, activation=Softmax())  # hidden -> vocab

    # Weight loading 

    def load_weights(self, weights_dict: Dict[str, dict]) -> None:
        """Load all weights from a weight_loaders.load_weights() result dict.

        Parameters
        ----------
        weights_dict : dict
            Returned by utils.weight_loaders.load_weights(keras_model, spec).
            Keys are Keras layer names; values contain type + weights.

        Example spec passed to load_weights():
        [
            {"name": "embedding",        "type": "Embedding", "target": "embedding"},
            {"name": "dense_projection", "type": "Dense",     "target": "dense_proj"},
            {"name": "lstm",             "type": "LSTM",      "target": "lstm_cell"},
            {"name": "dense_output",     "type": "Dense",     "target": "dense_out"},
        ]
        """
        for layer_name, entry in weights_dict.items():
            target = entry.get("target")
            layer_type = entry["type"]

            if target == "embedding" or layer_type == "Embedding":
                self.embedding.load_weights(entry)

            elif target == "dense_proj":
                w = entry["weights"]
                self.dense_proj.weights = np.asarray(w["kernel"])
                self.dense_proj.bias    = np.asarray(w["bias"])

            elif target == "lstm_cell" or layer_type == "LSTM":
                self.lstm_cell.load_weights(entry)

            elif target == "dense_out":
                w = entry["weights"]
                self.dense_out.weights = np.asarray(w["kernel"])
                self.dense_out.bias    = np.asarray(w["bias"])

    # Forward pass (training / teacher-forcing eval) 

    def forward(
        self,
        cnn_features: np.ndarray,
        token_sequences: np.ndarray,
    ) -> np.ndarray:
        """Full forward pass over a batch (teacher-forcing style).

        Parameters
        ----------
        cnn_features : np.ndarray, shape (batch_size, cnn_feature_dim)
            Pre-extracted CNN feature vectors.
        token_sequences : np.ndarray, shape (batch_size, seq_len)
            Integer token sequences, e.g. [<start>, w1, ..., w_{N-1}].

        Returns
        -------
        all_logits : np.ndarray, shape (batch_size, seq_len + 1, vocab_size)
            Softmax probability distributions at each timestep,
            including the output at t=-1 (after injecting CNN feature).
        """
        batch_size  = cnn_features.shape[0]
        num_hiddens = self.lstm_cell.W_hi.shape[0]

        H = np.zeros((batch_size, num_hiddens))
        C = np.zeros((batch_size, num_hiddens))

        all_logits = []

        # t = -1 : inject CNN feature 
        x    = self.dense_proj.forward(cnn_features)   # (batch, embed_dim)
        H, C = self.lstm_cell.forward(x, H, C)
        all_logits.append(self.dense_out.forward(H))   # (batch, vocab_size)

        # t = 0 … seq_len-1 : caption tokens 
        for t in range(token_sequences.shape[1]):
            x    = self.embedding.forward(token_sequences[:, t])  # (batch, embed_dim)
            H, C = self.lstm_cell.forward(x, H, C)
            all_logits.append(self.dense_out.forward(H))

        return np.stack(all_logits, axis=1)  # (batch, seq_len+1, vocab_size)

    # Greedy decoding (inference) 

    def generate_caption(
        self,
        cnn_feature: np.ndarray,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        max_len: int = 20,
        start_token: str = "<start>",
        end_token: str = "<end>",
    ) -> str:
        """Greedy decoding — generate one caption for a single image.

        Parameters
        ----------
        cnn_feature : np.ndarray, shape (cnn_feature_dim,)
            Feature vector for ONE image (batch dim added internally).
        word2idx : dict  — vocabulary mapping word -> index
        idx2word : dict  — vocabulary mapping index -> word
        max_len  : int   — maximum number of words to generate
        start_token, end_token : str — special tokens used during training

        Returns
        -------
        str
            Generated caption (space-joined words, without special tokens).
        """
        cnn_feature = cnn_feature[np.newaxis, :]   # (1, cnn_feature_dim)
        num_hiddens = self.lstm_cell.W_hi.shape[0]

        H = np.zeros((1, num_hiddens))
        C = np.zeros((1, num_hiddens))

        # t = -1 : inject image
        x    = self.dense_proj.forward(cnn_feature)
        H, C = self.lstm_cell.forward(x, H, C)

        # Feed <start>, then greedily pick next word at each step
        token_id      = word2idx[start_token]
        caption_words = []

        for _ in range(max_len):
            x    = self.embedding.forward(np.array([token_id]))  # (1, embed_dim)
            H, C = self.lstm_cell.forward(x, H, C)
            probs = self.dense_out.forward(H)                    # (1, vocab_size)

            token_id  = int(np.argmax(probs[0]))
            next_word = idx2word[token_id]

            if next_word == end_token:
                break
            caption_words.append(next_word)

        return " ".join(caption_words)
