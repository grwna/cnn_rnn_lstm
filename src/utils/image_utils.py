from typing import Iterable, List, Tuple
import numpy as np
from PIL import Image


# Load image, resize, and normalize to [0, 1].
# returns float32 array shapes (H, W, C)
def load_image(file_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
	with Image.open(file_path) as img:
		img = img.convert("RGB")
		img = img.resize(target_size, resample=Image.BILINEAR)
		arr = np.asarray(img, dtype=np.float32)

	return arr / 255.0


# Load images and stack into (N, H, W, C).
def load_batch(file_paths: Iterable[str], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
	images: List[np.ndarray] = [load_image(path, target_size=target_size) for path in file_paths]

	if not images:
		return np.empty((0, *target_size, 3), dtype=np.float32)

	return np.stack(images, axis=0)