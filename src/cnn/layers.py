from typing import Optional, Tuple
import numpy as np
from src.base.activations import Activation, ReLU


class Conv2D:
	def __init__( self,
		kernel: np.ndarray,
		bias: np.ndarray,
		strides: Tuple[int, int] = (1, 1),
		padding: str = "valid",     # valid: no padding, same: padding matches shape
		activation: Optional[Activation] = None,
	) -> None:
		self.kernel = np.asarray(kernel)
		self.bias = np.asarray(bias)
		self.strides = strides
		self.padding = padding.lower()
		self.activation = activation or ReLU()

	def _get_fmap_dims(self, h_in: int, w_in: int) -> Tuple[int, int, int, int]:
		k_h, k_w = self.kernel.shape[:2]
		s_h, s_w = self.strides

		if self.padding == "valid":
			if h_in < k_h or w_in < k_w:
				raise ValueError("Input dimensions must be larger than kernel for 'valid' padding")

			out_h = (h_in - k_h) // s_h + 1
			out_w = (w_in - k_w) // s_w + 1
			pad_top = pad_bottom = pad_left = pad_right = 0

			return out_h, out_w, pad_top, pad_bottom, pad_left, pad_right

		if self.padding == "same":
            # calculate padding to match shape
			out_h = int(np.ceil(h_in / s_h))
			out_w = int(np.ceil(w_in / s_w))

			pad_h = max((out_h - 1) * s_h + k_h - h_in, 0)
			pad_w = max((out_w - 1) * s_w + k_w - w_in, 0)

			pad_top = pad_h // 2
			pad_bottom = pad_h - pad_top
			pad_left = pad_w // 2
			pad_right = pad_w - pad_left

			return out_h, out_w, pad_top, pad_bottom, pad_left, pad_right

		raise ValueError("Paddings can only be:\n - valid\n - same")


	def _pad(self, x: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
		pad_top, pad_bottom, pad_left, pad_right = pads
		if pad_top == pad_bottom == pad_left == pad_right == 0:
			return x

		return np.pad( 
			x,
			((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
		)


	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim != 4:
			raise ValueError("Input must have shape (N, H, W, C_in)")
		if self.kernel.ndim != 4:
			raise ValueError("Kernel must have shape (kH, kW, C_in, C_out)")

		x = np.asarray(x)
		k_h, k_w, c_in, c_out = self.kernel.shape

		if x.shape[3] != c_in:
			raise ValueError("Input channels must match kernel channels")
		if self.bias.shape != (c_out,):
			raise ValueError("Bias must have shape (C_out,)")

        # get and apply input properties
		n, h_in, w_in, _ = x.shape
		out_h, out_w, pad_top, pad_bottom, pad_left, pad_right = self._get_fmap_dims(h_in, w_in)
		x_padded = self._pad(x, (pad_top, pad_bottom, pad_left, pad_right))
		s_h, s_w = self.strides

		output = np.zeros((n, out_h, out_w, c_out), dtype=x_padded.dtype) # init

		windows = np.lib.stride_tricks.sliding_window_view(
			x_padded,
			window_shape=(k_h, k_w),
			axis=(1, 2),
		)
		windows = windows[:, ::s_h, ::s_w, :, :, :]
		windows = windows[:, :out_h, :out_w, :, :, :]
		patches = np.moveaxis(windows, 3, -1)

		output = np.einsum("nhwklc,klcf->nhwf", patches, self.kernel, optimize=True)
		output = output + self.bias

		return self.activation(output)


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