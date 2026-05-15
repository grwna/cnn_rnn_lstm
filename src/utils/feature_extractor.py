import os
from typing import Callable, Iterable, List, Optional, Tuple
import numpy as np
from .image_utils import load_batch


def extract_features(
    file_paths: Iterable[str],
    encoder_model,
    cache_path: str,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"Cache ditemukan: {cache_path} — skip ekstraksi.")
        return np.load(cache_path)

    paths: List[str] = list(file_paths)
    if not paths:
        return np.empty((0,), dtype=np.float32)

    features: List[np.ndarray] = []

    for start in range(0, len(paths), batch_size):
        batch_files = paths[start : start + batch_size]
        batch = load_batch(batch_files, target_size=target_size)  # [0, 1]

        # Apply preprocessing jika ada (misal konversi ke [-1, 1] untuk InceptionV3)
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)

        batch_features = encoder_model.predict(batch, verbose=0)
        features.append(batch_features)

        done = min(start + batch_size, len(paths))
        print(f"  [{done}/{len(paths)}] batch selesai", end="\r")

    print()

    all_features = np.concatenate(features, axis=0).astype(np.float32)

    # Output
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    np.save(cache_path, all_features)
    print(f"Tersimpan: {cache_path} — shape: {all_features.shape}")
    return all_features