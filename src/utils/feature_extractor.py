import os
from typing import Iterable, List, Tuple
import numpy as np
from .image_utils import load_batch


def extract_features(
	file_paths: Iterable[str], # input
	encoder_model,  # can use Keras Encoder or other similar encoder
	cache_path: str, # output
	batch_size: int = 32,
	target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
	if os.path.exists(cache_path):
		return np.load(cache_path)

	paths: List[str] = list(file_paths)
	if not paths:
		empty = np.empty((0,), dtype=np.float32)
		return empty

	features: List[np.ndarray] = []
    # process files in batches
	for start in range(0, len(paths), batch_size):
		batch_files = paths[start : start + batch_size]
		batch = load_batch(batch_files, target_size=target_size)
		batch_features = encoder_model.predict(batch) # assume teh model has this method
		features.append(batch_features)

	all_features = np.concatenate(features, axis=0)

    # output
	cache_dir = os.path.dirname(cache_path)
	if cache_dir:
		os.makedirs(cache_dir, exist_ok=True)

	np.save(cache_path, all_features)
	return all_features
