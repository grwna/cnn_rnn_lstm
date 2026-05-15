import os
import sys
import glob
import argparse
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from src.utils.feature_extractor import extract_features

MODEL_CONFIGS = {
    "inceptionv3": {
        "builder": keras.applications.InceptionV3,
        "preprocess": keras.applications.inception_v3.preprocess_input,
        "target_size": (299, 299),
        "feature_dim": 2048, 
    },
    "vgg16": {
        "builder": keras.applications.VGG16,
        "preprocess": keras.applications.vgg16.preprocess_input,
        "target_size": (224, 224),
        "feature_dim": 512, 
    },
}

def build_encoder(model_name: str, pooling: str = "avg"):
    config = MODEL_CONFIGS[model_name]
    encoder = config["builder"](
        weights="imagenet",
        include_top=False, 
        pooling=pooling, 
    )
    encoder.trainable = False
    return encoder, config["preprocess"], config["target_size"]

def get_image_paths(images_dir: str) -> list:
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(set(paths))

def create_image_id_mapping(image_paths: list) -> dict:
    return {os.path.basename(p): i for i, p in enumerate(image_paths)}

def main():
    parser = argparse.ArgumentParser(description="Extract CNN features from images")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="inceptionv3", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "features"))
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, f"{args.model}_features.npy")
    mapping_path = os.path.join(args.output_dir, f"{args.model}_image_mapping.npy")

    if os.path.exists(cache_path) and os.path.exists(mapping_path):
        return

    image_paths = get_image_paths(images_dir)
    if not image_paths:
        sys.exit(1)

    encoder, preprocess_fn, target_size = build_encoder(args.model)

    # menggunakan utils/extract_features
    features = extract_features(
        file_paths=image_paths,
        encoder_model=encoder,
        cache_path=cache_path,
        batch_size=args.batch_size,
        target_size=target_size,
    )

    mapping = create_image_id_mapping(image_paths)
    np.save(mapping_path, mapping)

if __name__ == "__main__":
    main()