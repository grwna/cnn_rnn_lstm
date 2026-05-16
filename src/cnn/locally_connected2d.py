from typing import Optional, Tuple
import numpy as np
from src.base.activations import Activation, ReLU


class LocallyConnected2D:
	def __init__( self,
		kernel: np.ndarray,
		bias: np.ndarray,
		strides: Tuple[int, int] = (1, 1),
		activation: Optional[Activation] = None,
	) -> None:
		self.kernel = np.asarray(kernel)
		self.bias = np.asarray(bias)
		self.strides = strides
		self.activation = activation or ReLU()

	def _get_kernel_size(self, c_in: int) -> Tuple[int, int]:
		if self.kernel.ndim != 3:
			raise ValueError("Kernel must have shape (H_out*W_out, kH*kW*C_in, C_out)")

		flat_size = self.kernel.shape[1]
		if flat_size % c_in != 0:
			raise ValueError("Kernel size must be divisible by channels")

		k_hw = flat_size // c_in
		k_h = int(np.sqrt(k_hw))
		k_w = k_h
		if k_h * k_w != k_hw:
			raise ValueError("Kernel must be square")

		return k_h, k_w

	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim != 4:
			raise ValueError("Input must have shape (N, H, W, C_in)")

        # extract properties
		x = np.asarray(x)
		n, h_in, w_in, c_in = x.shape
		k_h, k_w = self._get_kernel_size(c_in)
		s_h, s_w = self.strides

		out_h = (h_in - k_h) // s_h + 1
		out_w = (w_in - k_w) // s_w + 1
		out_shape = out_h * out_w

        # check validity of bias and kernel
		if self.kernel.shape[0] != out_shape:
			raise ValueError("Kernel output positions must match output size")
		if self.bias.shape != (out_shape, self.kernel.shape[2]):
			raise ValueError("Bias must have shape (H_out*W_out, C_out)")
		if h_in < k_h or w_in < k_w:
			raise ValueError("Input dims must be >= kernel size")

		c_out = self.kernel.shape[2]

		windows = np.lib.stride_tricks.sliding_window_view(
			x,
			window_shape=(k_h, k_w),
			axis=(1, 2),
		)
		windows = windows[:, ::s_h, ::s_w, :, :, :]
		windows = windows[:, :out_h, :out_w, :, :, :]
		patches = np.moveaxis(windows, 3, -1)


		# alleviate OOM issues
		kernel_reshaped = self.kernel.reshape(out_h, out_w, k_h, k_w, c_in, c_out)

		# Perform einsum directly to prevent OOM
		outputs = np.einsum("nhwijc,hwijcf->nhwf", patches, kernel_reshaped, optimize=True)
		outputs = outputs + self.bias.reshape(1, out_h, out_w, c_out)

		return self.activation(outputs)
