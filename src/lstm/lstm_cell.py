import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    """Single LSTM cell — one timestep forward pass.

    Keras stores LSTM weights in concatenated form:
        kernel            shape: (input_dim,   4 * num_hiddens)
        recurrent_kernel  shape: (num_hiddens,  4 * num_hiddens)
        bias              shape: (4 * num_hiddens,)

    Gate ordering inside those matrices follows Keras convention: i, f, c, o.

    Attributes
    ----------
    W_xi, W_xf, W_xc, W_xo : np.ndarray  — input weights per gate
    W_hi, W_hf, W_hc, W_ho : np.ndarray  — recurrent weights per gate
    b_i, b_f, b_c, b_o     : np.ndarray  — biases per gate
    """

    def __init__(self) -> None:
        # Input weights (input_dim, num_hiddens)
        self.W_xi = self.W_xf = self.W_xc = self.W_xo = None
        # Recurrent weights (num_hiddens, num_hiddens)
        self.W_hi = self.W_hf = self.W_hc = self.W_ho = None
        # Biases (num_hiddens,)
        self.b_i = self.b_f = self.b_c = self.b_o = None

    def load_weights(self, weights: dict) -> None:
        """Load weights from weight_loaders.load_weights() result.

        Parameters
        ----------
        weights : dict
            Entry from load_weights() result, e.g.
            {
                "type": "LSTM",
                "weights": {
                    "kernel":            np.ndarray,  # (input_dim, 4*H)
                    "recurrent_kernel":  np.ndarray,  # (H, 4*H)
                    "bias":              np.ndarray,  # (4*H,)
                },
            }
        """
        kernel           = np.asarray(weights["weights"]["kernel"])
        recurrent_kernel = np.asarray(weights["weights"]["recurrent_kernel"])
        bias             = np.asarray(weights["weights"]["bias"])

        # Split along axis=1 in Keras gate order: i, f, c, o
        self.W_xi, self.W_xf, self.W_xc, self.W_xo = np.split(kernel,           4, axis=1)
        self.W_hi, self.W_hf, self.W_hc, self.W_ho = np.split(recurrent_kernel,  4, axis=1)
        self.b_i,  self.b_f,  self.b_c,  self.b_o  = np.split(bias,              4)

    def forward(
        self,
        x: np.ndarray,
        H: np.ndarray,
        C: np.ndarray,
    ) -> tuple:
        """One LSTM timestep.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, input_dim)
            Input at the current timestep.
        H : np.ndarray, shape (batch_size, num_hiddens)
            Hidden state from the previous timestep.
        C : np.ndarray, shape (batch_size, num_hiddens)
            Cell state from the previous timestep.

        Returns
        -------
        H_new : np.ndarray, shape (batch_size, num_hiddens)
        C_new : np.ndarray, shape (batch_size, num_hiddens)
        """
        if self.W_xi is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        I       = _sigmoid(x @ self.W_xi + H @ self.W_hi + self.b_i)  # input gate
        F       = _sigmoid(x @ self.W_xf + H @ self.W_hf + self.b_f)  # forget gate
        O       = _sigmoid(x @ self.W_xo + H @ self.W_ho + self.b_o)  # output gate
        C_tilde = np.tanh( x @ self.W_xc + H @ self.W_hc + self.b_c)  # candidate cell

        C_new = F * C + I * C_tilde
        H_new = O * np.tanh(C_new)
        return H_new, C_new
