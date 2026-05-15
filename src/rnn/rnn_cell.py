import numpy as np

class SimpleRNNCell:
    def __init__(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray,) -> None:
        # W_xh: (input_dim, units), W_hh: (units, units), b_h: (units,)
        self.W_xh = np.asarray(kernel)
        self.W_hh = np.asarray(recurrent_kernel)
        self.b_h = np.asarray(bias)
        self.units = self.W_hh.shape[0]

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        # x_t: (batch_size, input_dim), h_prev: (batch_size, units)
        # menghitung hidden state baru dengan aktivasi tanh
        return np.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray = None,
        return_sequences: bool = True,
    ) -> tuple:
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        if h_prev is None:
            h_prev = np.zeros((batch_size, self.units), dtype=x.dtype)

        all_hidden = []
        h_t = h_prev

        # iterasi forward pass sepanjang sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.forward_step(x_t, h_t)
            all_hidden.append(h_t)

        h_final = h_t

        if return_sequences:
            outputs = np.stack(all_hidden, axis=1) # (batch, seq_len, units)
        else:
            outputs = h_final                      # (batch, units)

        return outputs, h_final