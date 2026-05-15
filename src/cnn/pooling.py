from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Pooling(ABC):
	@abstractmethod
	def forward(self, x: np.ndarray) -> np.ndarray:
		raise NotImplementedError


class LocalPooling(Pooling, ABC):
	def __init__(self, pool_size: Tuple[int, int], strides: Tuple[int, int]) -> None:
		self.pool_size = pool_size
		self.strides = strides


class GlobalPooling(Pooling, ABC):
	pass


class MaxPooling2D(LocalPooling):
	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim != 4:
			raise ValueError("Input must have shape (N, H, W, C)")

		x = np.asarray(x)
		n, h_in, w_in, c = x.shape
		p_h, p_w = self.pool_size
		s_h, s_w = self.strides

		if h_in < p_h or w_in < p_w:
			raise ValueError("Input dims must be >= pool size")

		out_h = (h_in - p_h) // s_h + 1
		out_w = (w_in - p_w) // s_w + 1
		output = np.zeros((n, out_h, out_w, c), dtype=x.dtype)

		for i in range(out_h):
			h_start = i * s_h
			h_end = h_start + p_h
			for j in range(out_w):
				w_start = j * s_w
				w_end = w_start + p_w
				patch = x[:, h_start:h_end, w_start:w_end, :]
				output[:, i, j, :] = np.max(patch, axis=(1, 2))

		return output


class AveragePooling2D(LocalPooling):
	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim != 4:
			raise ValueError("Input must have shape (N, H, W, C)")

		x = np.asarray(x)
		n, h_in, w_in, c = x.shape
		p_h, p_w = self.pool_size
		s_h, s_w = self.strides

		if h_in < p_h or w_in < p_w:
			raise ValueError("Input dims must be >= pool size")

		out_h = (h_in - p_h) // s_h + 1
		out_w = (w_in - p_w) // s_w + 1
		output = np.zeros((n, out_h, out_w, c), dtype=x.dtype)

		for i in range(out_h):
			h_start = i * s_h
			h_end = h_start + p_h
			for j in range(out_w):
				w_start = j * s_w
				w_end = w_start + p_w
				patch = x[:, h_start:h_end, w_start:w_end, :]
				output[:, i, j, :] = np.mean(patch, axis=(1, 2))

		return output


class GlobalMaxPooling2D(GlobalPooling):
	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim != 4:
			raise ValueError("Input must have shape (N, H, W, C)")

		x = np.asarray(x)
		return np.max(x, axis=(1, 2))


class GlobalAveragePooling2D(GlobalPooling):
	def forward(self, x: np.ndarray) -> np.ndarray:
		if x.ndim != 4:
			raise ValueError("Input must have shape (N, H, W, C)")

		x = np.asarray(x)
		return np.mean(x, axis=(1, 2))
