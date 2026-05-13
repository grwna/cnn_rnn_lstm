import numpy as np


# ------------ Activation helpers ------------

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    # numerically stable softmax; works on last axis
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ------------ Embedding layer ------------

class Embedding:
    """Token-index -> dense vector lookup.

    Weights are loaded from a trained Keras Embedding layer.
    """

    def load_keras_weights(self, keras_layer):
        # shape: (vocab_size, embed_dim)
        self.W = keras_layer.get_weights()[0]

    def forward(self, token_ids):
        """
        Parameters
        ----------
        token_ids : np.ndarray, shape (batch_size,) or (batch_size, seq_len)
            Integer token indices.

        Returns
        -------
        np.ndarray
            Embedded vectors, shape (..., embed_dim).
        """
        return self.W[token_ids]


# ------------ Dense layer ------------

class Dense:
    """Fully-connected layer: output = activation(x @ W + b).

    Used for:
    - CNN feature projection  (no activation, or linear)
    - Decoder output layer    (softmax activation)
    """

    def __init__(self, activation=None):
        """
        Parameters
        ----------
        activation : callable or None
            e.g. softmax, np.tanh.  None -> linear (identity).
        """
        self.activation = activation

    def load_keras_weights(self, keras_layer):
        weights = keras_layer.get_weights()
        self.W = weights[0]   # shape: (input_dim, output_dim)
        self.b = weights[1]   # shape: (output_dim,)

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray, shape (batch_size, input_dim)

        Returns
        -------
        np.ndarray, shape (batch_size, output_dim)
        """
        out = x @ self.W + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out


# ------------ LSTM cell ------------

class LSTMCell:
    """Single LSTM cell — one timestep forward pass.

    Keras stores LSTM weights in concatenated form:
        kernel            shape: (input_dim,   4 * num_hiddens)
        recurrent_kernel  shape: (num_hiddens,  4 * num_hiddens)
        bias              shape: (4 * num_hiddens,)

    Gate ordering inside those matrices follows Keras convention: i, f, c, o.
    (Note: some references use i, f, o, c — Keras uses i, f, c, o.)
    """

    def load_keras_weights(self, keras_layer):
        kernel, recurrent_kernel, bias = keras_layer.get_weights()
        h = kernel.shape[1] // 4   # num_hiddens

        # Split along the last axis in Keras gate order: i, f, c, o
        self.W_xi, self.W_xf, self.W_xc, self.W_xo = np.split(kernel,           4, axis=1)
        self.W_hi, self.W_hf, self.W_hc, self.W_ho = np.split(recurrent_kernel,  4, axis=1)
        self.b_i,  self.b_f,  self.b_c,  self.b_o  = np.split(bias,              4)

    def forward(self, x, H, C):
        """
        Parameters
        ----------
        x : np.ndarray, shape (batch_size, input_dim)
            Input at current timestep.
        H : np.ndarray, shape (batch_size, num_hiddens)
            Previous hidden state.
        C : np.ndarray, shape (batch_size, num_hiddens)
            Previous cell state.

        Returns
        -------
        H_new, C_new : np.ndarray, each shape (batch_size, num_hiddens)
        """
        I      = sigmoid(x @ self.W_xi + H @ self.W_hi + self.b_i)
        F      = sigmoid(x @ self.W_xf + H @ self.W_hf + self.b_f)
        O      = sigmoid(x @ self.W_xo + H @ self.W_ho + self.b_o)
        C_tilde = np.tanh(x @ self.W_xc + H @ self.W_hc + self.b_c)

        C_new = F * C + I * C_tilde
        H_new = O * np.tanh(C_new)
        return H_new, C_new


# ------------ Full LSTM decoder (pre-inject, image captioning) ------------

class LSTMDecoder:
    """Image-captioning decoder using the pre-inject method.

    Architecture (from spec):
        x₋₁  = Dense_proj(CNN_feature)          ← image injected here
        x_t   = Embedding(token_t),  t ≥ 0
        sequence input = [x₋₁, x₀, x₁, ..., x_{N-1}]
        h₀ = 0,  c₀ = 0
        At each step: H, C = LSTMCell(x_t, H, C)
        output logits = Dense_out(H)  -> softmax -> prob over vocab
    """

    def __init__(self):
        self.embedding   = Embedding()
        self.dense_proj  = Dense(activation=None)    # CNN feature -> embed_dim
        self.lstm_cell   = LSTMCell()
        self.dense_out   = Dense(activation=softmax) # hidden -> vocab_size

    def load_keras_weights(self, keras_model):
        """Load all weights from a trained Keras model.

        Adjust layer names/indices to match your actual Keras model structure.
        """
        self.embedding.load_keras_weights(keras_model.get_layer('embedding'))
        self.dense_proj.load_keras_weights(keras_model.get_layer('dense_projection'))
        self.lstm_cell.load_keras_weights(keras_model.get_layer('lstm'))
        self.dense_out.load_keras_weights(keras_model.get_layer('dense_output'))

    def forward(self, cnn_features, token_sequences):
        """Full forward pass over a batch of sequences.

        Parameters
        ----------
        cnn_features : np.ndarray, shape (batch_size, cnn_feature_dim)
            Pre-extracted CNN feature vectors.
        token_sequences : np.ndarray, shape (batch_size, seq_len)
            Integer token sequences (e.g. [<start>, w1, w2, ..., w_{N-1}]).

        Returns
        -------
        all_logits : np.ndarray, shape (batch_size, seq_len + 1, vocab_size)
            Softmax probabilities at each timestep (including t=-1 output).
        """
        batch_size = cnn_features.shape[0]
        num_hiddens = self.lstm_cell.W_hi.shape[0]

        # Initialise hidden and cell states to zeros
        H = np.zeros((batch_size, num_hiddens))
        C = np.zeros((batch_size, num_hiddens))

        all_logits = []

        # --------- Timestep t = -1: inject CNN feature ---------
        x_neg1 = self.dense_proj.forward(cnn_features)   # (batch, embed_dim)
        H, C   = self.lstm_cell.forward(x_neg1, H, C)
        logits = self.dense_out.forward(H)                # (batch, vocab_size)
        all_logits.append(logits)

        # --------- Timesteps t = 0 … seq_len-1: caption tokens ---------
        for t in range(token_sequences.shape[1]):
            x_t    = self.embedding.forward(token_sequences[:, t])  # (batch, embed_dim)
            H, C   = self.lstm_cell.forward(x_t, H, C)
            logits = self.dense_out.forward(H)
            all_logits.append(logits)

        # Stack -> (batch_size, seq_len + 1, vocab_size)
        return np.stack(all_logits, axis=1)

    def generate_caption(
            self, 
            cnn_feature, 
            word2idx, 
            idx2word,
            max_len=20, 
            start_token='<start>', 
            end_token='<end>'
        ):
        """Greedy decoding — generate one caption for a single image.

        Parameters
        ----------
        cnn_feature : np.ndarray, shape (cnn_feature_dim,)
            Feature vector for one image (will be expanded to batch_size=1).
        word2idx, idx2word : dict
            Vocabulary mappings.
        max_len : int
            Maximum caption length.
        start_token, end_token : str
            Special tokens used during training.

        Returns
        -------
        str
            Generated caption (words joined by spaces).
        """
        cnn_feature = cnn_feature[np.newaxis, :]   # (1, cnn_feature_dim)
        num_hiddens = self.lstm_cell.W_hi.shape[0]

        H = np.zeros((1, num_hiddens))
        C = np.zeros((1, num_hiddens))

        # Inject CNN feature at t = -1
        x = self.dense_proj.forward(cnn_feature)
        H, C = self.lstm_cell.forward(x, H, C)

        # Feed <start> token, then greedily pick next word
        token = np.array([[word2idx[start_token]]])
        caption_words = []

        for _ in range(max_len):
            x      = self.embedding.forward(token[:, 0])   # (1, embed_dim)
            H, C   = self.lstm_cell.forward(x, H, C)
            logits = self.dense_out.forward(H)              # (1, vocab_size)

            next_token_id = int(np.argmax(logits[0]))
            next_word     = idx2word[next_token_id]

            if next_word == end_token:
                break

            caption_words.append(next_word)
            token = np.array([[next_token_id]])

        return ' '.join(caption_words)
