from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
	@abstractmethod
	def forward(self, x: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def __call__(self, x: np.ndarray) -> np.ndarray:
		return self.forward(x)


class ReLU(Activation):
	def forward(self, x: np.ndarray) -> np.ndarray:
		return np.maximum(0.0, x)


class Softmax(Activation):
	def __init__(self, axis: int = -1) -> None:
		self.axis = axis

	def forward(self, x: np.ndarray) -> np.ndarray:
		shifted = x - np.max(x, axis=self.axis, keepdims=True)
		exp_x = np.exp(shifted)
		return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
