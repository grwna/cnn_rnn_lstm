from typing import Any, Dict, Iterable, List, Mapping
import numpy as np


_WEIGHT_KEYS: Mapping[str, List[str]] = {
    "Conv2D": ["kernel", "bias"],
    "Dense": ["kernel", "bias"],
    "LocallyConnected2D": ["kernel", "bias"],
    # RNN
    # LSTM
}


def load_weights(keras_model, spec: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Load weights based on a spec.

    Spec format:
    {
        "name": "original_model_layer_name",
        "type": "LayerType",
        "target": "attribute_name",  # optional
        "metadata": { ... }            # optional
    }

    Return format:
    {
        "keras_layer_name": {
            "type": "LayerType",
            "weights": {"kernel": np.ndarray, "bias": np.ndarray},
            "metadata": { ... },
            "target": "attribute_name"  # optional
        }
    }
    """

    if spec is None:
        raise ValueError("Invalid spec!")

    results: Dict[str, Dict[str, Any]] = {}

    for entry in spec:
        if "name" not in entry or "type" not in entry:
            raise ValueError("Spec must include 'name' and 'type'")

        layer_name = entry["name"]
        layer_type = entry["type"]
        weight_keys = _WEIGHT_KEYS.get(layer_type)
        if weight_keys is None:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # get layer form og model
        try:
            layer = keras_model.get_layer(layer_name)
        except Exception as exc:
            raise ValueError(f"Layer not found: {layer_name}") from exc

        # layer must contain weights
        if not layer.weights:
            raise ValueError(f"Layer has no weights: {layer_name}")

        # weights format have to match expected by type
        if len(layer.weights) != len(weight_keys):
            raise ValueError(
                f"Layer '{layer_name}' expected {len(weight_keys)} weights, "
                f"got {len(layer.weights)} instead."
            )

        weights = {
            key: np.asarray(weight.numpy()) for key, weight in zip(weight_keys, layer.weights)
        }

        item: Dict[str, Any] = {
            "type": layer_type,
            "weights": weights,
            "metadata": entry.get("metadata", {}),
        }
        if "target" in entry:
            item["target"] = entry["target"]

        results[layer_name] = item

    return results