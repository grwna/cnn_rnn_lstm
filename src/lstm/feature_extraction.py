"""
Bagian 1 — CNN Encoder + Feature Extraction (Flickr8k)
=======================================================
Script ini:
1. Load pretrained InceptionV3 (tanpa top layer, bobot ImageNet, frozen)
2. Ekstrak feature vector untuk semua gambar Flickr8k (train/val/test)
3. Simpan hasilnya ke .npy agar tidak perlu diekstraksi ulang

Struktur output yang dihasilkan:
    data/features/
    ├── train_features.npy   # shape: (6000, 2048)
    ├── val_features.npy     # shape: (1000, 2048)
    ├── test_features.npy    # shape: (1000, 2048)
    └── train_image_ids.npy  # urutan filename, penting untuk mapping ke caption
    └── val_image_ids.npy
    └── test_image_ids.npy

Cara pakai:
    python src/lstm/feature_extraction.py \
        --flickr8k_dir data/flickr8k/Images \
        --split_dir    data/flickr8k \
        --output_dir   data/features
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from src.utils.image_utils import load_batch


# ── Konstanta ─────────────────────────────────────────────────────────────────

# InceptionV3 butuh input 299x299, bukan 224x224
INCEPTION_INPUT_SIZE: Tuple[int, int] = (299, 299)
FEATURE_DIM = 2048   # dimensi output GlobalAveragePooling InceptionV3
BATCH_SIZE  = 32


# ── Build encoder ─────────────────────────────────────────────────────────────

def build_encoder() -> keras.Model:
    """Load InceptionV3 pretrained ImageNet, tanpa top layer, semua layer di-freeze.

    Returns
    -------
    keras.Model
        Model dengan output shape (batch, 2048) — hasil GlobalAveragePooling.
    """
    base = InceptionV3(
        include_top=False,       # buang classification head
        weights="imagenet",      # load bobot ImageNet
        pooling="avg",           # GlobalAveragePooling otomatis → output (batch, 2048)
    )
    base.trainable = False       # freeze semua layer
    print(f"Encoder: InceptionV3 — output dim: {base.output_shape[-1]}")
    return base


# ── Preprocessing wrapper ─────────────────────────────────────────────────────

def preprocess_batch(batch: np.ndarray) -> np.ndarray:
    """Konversi batch dari [0,1] (format image_utils) ke format InceptionV3 [-1, 1].

    image_utils.load_batch() menghasilkan float32 di [0, 1].
    InceptionV3 butuh range [-1, 1] via preprocess_input().

    Parameters
    ----------
    batch : np.ndarray, shape (N, H, W, 3), dtype float32, range [0, 1]

    Returns
    -------
    np.ndarray, shape (N, H, W, 3), dtype float32, range [-1, 1]
    """
    # Kembalikan ke [0, 255] dulu, lalu apply preprocess_input InceptionV3
    batch_255 = (batch * 255.0).astype(np.float32)
    return preprocess_input(batch_255)   # → [-1, 1]


# ── Ekstraksi fitur dengan caching ───────────────────────────────────────────

def extract_and_cache(
    image_paths: List[str],
    encoder: keras.Model,
    cache_path: str,
    id_cache_path: str,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ekstrak fitur dari list gambar, simpan ke .npy. Skip jika cache ada.

    Parameters
    ----------
    image_paths  : list of str — path lengkap ke tiap gambar
    encoder      : Keras model encoder (frozen InceptionV3)
    cache_path   : path output untuk feature matrix (.npy)
    id_cache_path: path output untuk array image_ids (.npy)
    batch_size   : jumlah gambar per batch

    Returns
    -------
    features  : np.ndarray, shape (N, 2048)
    image_ids : np.ndarray, shape (N,) — basename tiap gambar (tanpa path)
    """
    image_ids = np.array([os.path.basename(p) for p in image_paths])

    # Load cache jika sudah ada
    if os.path.exists(cache_path) and os.path.exists(id_cache_path):
        print(f"  Cache ditemukan: {cache_path} — skip ekstraksi.")
        features = np.load(cache_path)
        ids      = np.load(id_cache_path)
        return features, ids

    print(f"  Mengekstraksi {len(image_paths)} gambar → {cache_path}")
    all_features: List[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]

        # Load dan resize ke 299x299 (InceptionV3)
        batch = load_batch(batch_paths, target_size=INCEPTION_INPUT_SIZE)

        # Preprocessing [0,1] → [-1,1]
        batch = preprocess_batch(batch)

        # Forward pass encoder
        feats = encoder.predict(batch, verbose=0)   # (batch, 2048)
        all_features.append(feats)

        done = min(start + batch_size, len(image_paths))
        print(f"  [{done}/{len(image_paths)}] batch selesai", end="\r")

    print()  # newline setelah progress

    features = np.concatenate(all_features, axis=0).astype(np.float32)

    # Simpan ke disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, features)
    np.save(id_cache_path, image_ids)
    print(f"  Tersimpan: {cache_path} — shape: {features.shape}")

    return features, image_ids


# ── Load split files (Flickr8k) ───────────────────────────────────────────────

def load_split_paths(
    split_file: str,
    images_dir: str,
) -> List[str]:
    """Baca file split Flickr8k dan kembalikan list path gambar.

    Flickr8k menyediakan file teks berisi nama file gambar per split:
        Flickr_8k.trainImages.txt
        Flickr_8k.devImages.txt    (validation)
        Flickr_8k.testImages.txt

    Parameters
    ----------
    split_file  : path ke file .txt berisi daftar nama gambar
    images_dir  : path ke folder berisi semua gambar Flickr8k

    Returns
    -------
    list of str — path lengkap ke tiap gambar
    """
    with open(split_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    paths = [os.path.join(images_dir, fname) for fname in filenames]

    # Validasi — pastikan file ada
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} gambar tidak ditemukan. Contoh: {missing[:3]}"
        )

    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main(flickr8k_dir: str, split_dir: str, output_dir: str) -> None:
    """Jalankan ekstraksi fitur untuk semua split Flickr8k.

    Parameters
    ----------
    flickr8k_dir : path ke folder Images Flickr8k
    split_dir    : path ke folder yang berisi file split .txt
    output_dir   : path ke folder output untuk .npy
    """
    encoder = build_encoder()

    splits = {
        "train": "Flickr_8k.trainImages.txt",
        "val":   "Flickr_8k.devImages.txt",
        "test":  "Flickr_8k.testImages.txt",
    }

    for split_name, split_file in splits.items():
        print(f"\n── Split: {split_name} ──")
        split_path = os.path.join(split_dir, split_file)

        if not os.path.exists(split_path):
            print(f"  File split tidak ditemukan: {split_path} — skip.")
            continue

        image_paths = load_split_paths(split_path, flickr8k_dir)
        print(f"  Jumlah gambar: {len(image_paths)}")

        cache_path    = os.path.join(output_dir, f"{split_name}_features.npy")
        id_cache_path = os.path.join(output_dir, f"{split_name}_image_ids.npy")

        features, image_ids = extract_and_cache(
            image_paths,
            encoder,
            cache_path,
            id_cache_path,
        )
        print(f"  Features shape : {features.shape}")
        print(f"  Image IDs shape: {image_ids.shape}")

    print("\nEkstraksi fitur selesai!")
    print(f"Output tersimpan di: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi fitur CNN Flickr8k")
    parser.add_argument(
        "--flickr8k_dir",
        type=str,
        default="data/flickr8k/Images",
        help="Path ke folder Images Flickr8k",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="data/flickr8k",
        help="Path ke folder berisi file split .txt Flickr8k",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features",
        help="Path ke folder output untuk .npy",
    )
    args = parser.parse_args()
    main(args.flickr8k_dir, args.split_dir, args.output_dir)
