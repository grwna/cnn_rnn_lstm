import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    def __init__(self) -> None:
        self.W_xi = self.W_xf = self.W_xc = self.W_xo = None # Input weights (input_dim, num_hiddens)
        self.W_hi = self.W_hf = self.W_hc = self.W_ho = None # Recurrent weights (num_hiddens, num_hiddens)
        self.b_i = self.b_f = self.b_c = self.b_o = None # Biases (num_hiddens,)

    def load_weights(self, weights: dict) -> None:
        kernel           = np.asarray(weights["weights"]["kernel"])
        recurrent_kernel = np.asarray(weights["weights"]["recurrent_kernel"])
        bias             = np.asarray(weights["weights"]["bias"])

        # Split along axis=1 in Keras gate order: i, f, c, o
        self.W_xi, self.W_xf, self.W_xc, self.W_xo = np.split(kernel, 4, axis=1)
        self.W_hi, self.W_hf, self.W_hc, self.W_ho = np.split(recurrent_kernel, 4, axis=1)
        self.b_i,  self.b_f,  self.b_c,  self.b_o  = np.split(bias, 4)

    def forward(
        self,
        x: np.ndarray,
        H: np.ndarray,
        C: np.ndarray,
    ) -> tuple:
        if self.W_xi is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        I       = _sigmoid(x @ self.W_xi + H @ self.W_hi + self.b_i)  # input gate
        F       = _sigmoid(x @ self.W_xf + H @ self.W_hf + self.b_f)  # forget gate
        O       = _sigmoid(x @ self.W_xo + H @ self.W_ho + self.b_o)  # output gate
        C_tilde = np.tanh( x @ self.W_xc + H @ self.W_hc + self.b_c)  # candidate cell

        C_new = F * C + I * C_tilde
        H_new = O * np.tanh(C_new)
        return H_new, C_new
