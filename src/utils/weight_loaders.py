from typing import Any, Dict, Iterable, List, Mapping
import numpy as np


_WEIGHT_KEYS: Mapping[str, List[str]] = {
    "Conv2D":            ["kernel", "bias"],
    "Dense":             ["kernel", "bias"],
    "LocallyConnected2D":["kernel", "bias"],
    "Embedding":         ["embeddings"],
    "LSTM":              ["kernel", "recurrent_kernel", "bias"],
    "SimpleRNN":         ["kernel", "recurrent_kernel", "bias"],
}


def load_weights(keras_model, spec: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Load weights from a Keras model based on a spec list.

    Parameters
    ----------
    keras_model : Keras Model
        Trained Keras model to extract weights from.
    spec : list of dict
        Each entry describes one layer to extract:
        {
            "name":     "keras_layer_name",   # required
            "type":     "LayerType",          # required — must be a key in _WEIGHT_KEYS
            "target":   "attribute_name",     # optional — used by decoder.load_weights()
            "metadata": { ... }               # optional — any extra info you want to carry
        }

    Returns
    -------
    dict
        Keyed by Keras layer name:
        {
            "keras_layer_name": {
                "type":     "LayerType",
                "weights":  {"kernel": np.ndarray, "bias": np.ndarray, ...},
                "metadata": { ... },
                "target":   "attribute_name"   # only present if given in spec
            }
        }

    Example
    -------
    spec = [
        {"name": "embedding",        "type": "Embedding", "target": "embedding"},
        {"name": "dense_projection", "type": "Dense",     "target": "dense_proj"},
        {"name": "lstm",             "type": "LSTM",      "target": "lstm_cell"},
        {"name": "dense_output",     "type": "Dense",     "target": "dense_out"},
    ]
    weights = load_weights(keras_model, spec)
    decoder.load_weights(weights)
    """
    if spec is None:
        raise ValueError("spec cannot be None.")

    results: Dict[str, Dict[str, Any]] = {}

    for entry in spec:
        if "name" not in entry or "type" not in entry:
            raise ValueError(
                f"Each spec entry must have 'name' and 'type'. Got: {entry}"
            )

        layer_name  = entry["name"]
        layer_type  = entry["type"]
        weight_keys = _WEIGHT_KEYS.get(layer_type)

        if weight_keys is None:
            raise ValueError(
                f"Unknown layer type: '{layer_type}'. "
                f"Supported types: {list(_WEIGHT_KEYS.keys())}"
            )

        try:
            layer = keras_model.get_layer(layer_name)
        except Exception as exc:
            raise ValueError(f"Layer '{layer_name}' not found in model.") from exc

        if not layer.weights:
            raise ValueError(f"Layer '{layer_name}' has no trainable weights.")

        if len(layer.weights) != len(weight_keys):
            raise ValueError(
                f"Layer '{layer_name}' ({layer_type}): "
                f"expected {len(weight_keys)} weight tensors {weight_keys}, "
                f"but got {len(layer.weights)}."
            )

        weights = {
            key: np.asarray(w.numpy())
            for key, w in zip(weight_keys, layer.weights)
        }

        item: Dict[str, Any] = {
            "type":     layer_type,
            "weights":  weights,
            "metadata": entry.get("metadata", {}),
        }
        if "target" in entry:
            item["target"] = entry["target"]

        results[layer_name] = item

    return results
