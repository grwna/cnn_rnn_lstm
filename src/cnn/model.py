from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import src.base.activations as act
from src.base.dense import Dense
from src.base.flatten import Flatten
from src.cnn.layers import Conv2D, LocallyConnected2D
import src.cnn.pooling as pool
from src.utils.weight_loaders import load_weights


LAYER_MAPPING = {
    "Conv2D": lambda w, m, a: Conv2D(w["kernel"], w["bias"], strides=m.get("strides", (1,1)), padding=m.get("padding", "valid"), activation=a),
    "LocallyConnected2D": lambda w, m, a: LocallyConnected2D(w["kernel"], w["bias"], strides=m.get("strides", (1,1)), padding=m.get("padding", "valid"), activation=a),
    "MaxPooling2D": lambda w, m, a: pool.MaxPooling2D(pool_size=m.get("pool_size", (2,2)), strides=m.get("strides", m.get("pool_size", (2,2)))),
    "AveragePooling2D": lambda w, m, a: pool.AveragePooling2D(pool_size=m.get("pool_size", (2,2)), strides=m.get("strides", m.get("pool_size", (2,2)))),
    "GlobalMaxPooling2D": lambda w, m, a: pool.GlobalMaxPooling2D(),
    "GlobalAveragePooling2D": lambda w, m, a: pool.GlobalAveragePooling2D(),
    "Flatten": lambda w, m, a: Flatten(),
    "Dense": lambda w, m, a: Dense(kernel=w["kernel"], bias=w["bias"], activation=a),
}

class CNNScratch:
	def __init__(self, keras_model: Optional[Any], spec: Iterable[Dict[str, Any]]) -> None:
		self.spec: List[Dict[str, Any]] = list(spec)
		self.layers: List[Any] = []

		if keras_model is not None:
			w_by_name = load_weights(keras_model, self.spec)
		else:
			w_by_name = self._extract_weights_from_spec(self.spec)

		for entry in self.spec:
			l_type = entry.get("type")
			l_name = entry.get("name")
			meta = entry.get("metadata", {})
			activation = self.init_activation(meta.get("activation"))
			
			if l_type not in LAYER_MAPPING:
				raise ValueError(f"Unsupported layer type: {l_type}")
			
			# build layer
			builder = LAYER_MAPPING.get(l_type)
			if l_type in {"Conv2D", "LocallyConnected2D", "Dense"}:
				w = w_by_name.get(l_name, {}).get("weights") 
			else: 
				w = None

			self.layers.append(builder(w, meta, activation))


	def forward(self, x: np.ndarray) -> np.ndarray:
		out = x
		for layer in self.layers:
			out = layer.forward(out)
		return out


	def predict(self, x: np.ndarray, batch_size: int = 32, verbose: bool = False) -> np.ndarray:
		if x.ndim < 2:
			raise ValueError("Input must include batch dimension")
		if batch_size <= 0:
			raise ValueError("batch_size must be positive")

		outputs: List[np.ndarray] = []
		total_samples = x.shape[0]
		total_batches = (total_samples + batch_size - 1) // batch_size

		for i, start_idx in enumerate(range(0, total_samples, batch_size)):
			batch = x[start_idx:start_idx + batch_size]
			outputs.append(self.forward(batch))
			
			if verbose:
				print(f"Batch {i + 1}/{total_batches} processed", end='\r' if i + 1 < total_batches else '\n')

		return np.concatenate(outputs, axis=0) if outputs else np.empty((0,))

	def count_params(self) -> int:
		total = 0
		for layer in self.layers:
			if isinstance(layer, Conv2D):
				total += int(np.prod(layer.kernel.shape) + np.prod(layer.bias.shape))
			elif isinstance(layer, LocallyConnected2D):
				total += int(np.prod(layer.kernel.shape) + np.prod(layer.bias.shape))
			elif isinstance(layer, Dense):
				total += int(np.prod(layer.weights.shape) + np.prod(layer.bias.shape))
		return total


	def init_activation(self, name: Optional[str]) -> act.Activation:
		if name is None:
			return act.ReLU()

		name_lower = str(name).lower()
		if name_lower == "linear":
			return act.Linear()
		if name_lower == "relu":
			return act.ReLU()
		if name_lower == "sigmoid":
			return act.Sigmoid()
		if name_lower == "softmax":
			return act.Softmax()

		raise ValueError(f"Unsupported activation: {name}")

	def _extract_weights_from_spec( self, spec: Iterable[Dict[str, Any]]
								    ) -> Dict[str, Dict[str, Any]]:
		w_by_name = {}
		for entry in spec:
			l_name = entry.get("name")
			w = entry.get("weights")
			if l_name and w:
				w_by_name[l_name] = {"weights": w}

		return w_by_name
