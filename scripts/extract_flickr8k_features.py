import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from src.utils.feature_extractor import extract_features


# ---------- Konfigurasi encoder ----------

MODEL_CONFIGS = {
    "inceptionv3": {
        "builder":     keras.applications.InceptionV3,
        "preprocess":  inception_preprocess,
        "target_size": (299, 299),
        "feature_dim": 2048,
    },
    "vgg16": {
        "builder":     keras.applications.VGG16,
        "preprocess":  vgg16_preprocess,
        "target_size": (224, 224),
        "feature_dim": 512,
    },
}


# ---------- Build encoder ----------

def build_encoder(model_name: str) -> tuple:
    config = MODEL_CONFIGS[model_name]

    encoder = config["builder"](
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )
    encoder.trainable = False

    # Wrapper: konversi [0,1] -> [0,255] -> format model
    preprocess_fn = lambda batch: config["preprocess"]((batch * 255.0).astype(np.float32))

    print(f"Encoder: {model_name.upper()} — output dim: {config['feature_dim']}")
    return encoder, preprocess_fn, config["target_size"]


# ---------- Load split files ----------

def load_split_paths(split_file: str, images_dir: str) -> Tuple[List[str], List[str]]:
    with open(split_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    paths     = [os.path.join(images_dir, fname) for fname in filenames]
    image_ids = filenames

    # Validasi
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} gambar tidak ditemukan. Contoh: {missing[:3]}"
        )

    return paths, image_ids


# ---------- Main ----------

def main(images_dir: str, split_dir: str, output_dir: str, model_name: str, batch_size: int) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU tersedia: {gpus[0].name}")
    else:
        print("GPU tidak tersedia, menggunakan CPU.")

    os.makedirs(output_dir, exist_ok=True)

    encoder, preprocess_fn, target_size = build_encoder(model_name)

    splits = {
        "train": "Flickr_8k.trainImages.txt",
        "val":   "Flickr_8k.devImages.txt",
        "test":  "Flickr_8k.testImages.txt",
    }

    for split_name, split_filename in splits.items():
        print(f"\n── Split: {split_name} ──")
        split_path = os.path.join(split_dir, split_filename)

        if not os.path.exists(split_path):
            print(f"  File split tidak ditemukan: {split_path} — skip.")
            continue

        image_paths, image_ids = load_split_paths(split_path, images_dir)
        print(f"  Jumlah gambar: {len(image_paths)}")

        cache_path    = os.path.join(output_dir, f"{model_name}_{split_name}_features.npy")
        id_cache_path = os.path.join(output_dir, f"{model_name}_{split_name}_image_ids.npy")

        features = extract_features(
            file_paths=image_paths,
            encoder_model=encoder,
            cache_path=cache_path,
            batch_size=batch_size,
            target_size=target_size,
            preprocess_fn=preprocess_fn,
        )

        if not os.path.exists(id_cache_path):
            np.save(id_cache_path, np.array(image_ids))
            print(f"  Image IDs disimpan: {id_cache_path}")

        print(f"  Features shape : {features.shape}")

    print("\nEkstraksi fitur selesai!")
    print(f"Output tersimpan di: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi fitur CNN untuk Flickr8k")
    parser.add_argument("--images_dir", type=str, default="data/Images")
    parser.add_argument("--split_dir",  type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs/features")
    parser.add_argument(
        "--model", type=str, default="inceptionv3",
        choices=list(MODEL_CONFIGS.keys()),
        help="Encoder model yang digunakan (default: inceptionv3)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(
        images_dir=args.images_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )