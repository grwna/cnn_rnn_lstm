import os
import sys
import json
import argparse
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_training_data():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    feat_dir = os.path.join(PROJECT_ROOT, "features")

    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    padded = np.load(os.path.join(data_dir, "padded_captions.npy"))
    img_names = np.load(os.path.join(data_dir, "caption_image_names.npy"), allow_pickle=True)

    features = np.load(os.path.join(feat_dir, "inceptionv3_features.npy"))
    mapping = np.load(os.path.join(feat_dir, "inceptionv3_image_mapping.npy"), allow_pickle=True).item()

    # sejajarkan vektor fitur dengan matriks caption
    feat_indices = np.array([mapping[name] for name in img_names])
    caption_features = features[feat_indices]

    # teacher forcing: geser input dan target 1 timestep
    decoder_input = padded[:, :-1] 
    target = padded[:, 1:] 
    mask = (target != meta["pad_idx"]).astype(np.float32)

    return caption_features, decoder_input, target, mask, meta

def train_val_split(n, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    val_size = int(n * val_ratio)
    return indices[val_size:], indices[:val_size]

def build_decoder(vocab_size, embed_dim, feature_dim, seq_len, rnn_units, num_layers):
    feat_input = keras.Input(shape=(feature_dim,))
    cap_input = keras.Input(shape=(seq_len,), dtype="int32")

    # proyeksi fitur CNN menjadi input x_{-1}
    projected = layers.Dense(embed_dim)(feat_input)
    projected = layers.Reshape((1, embed_dim))(projected)

    embedded = layers.Embedding(vocab_size, embed_dim)(cap_input)

    # gabungkan
    x = layers.Concatenate(axis=1)([projected, embedded])

    for _ in range(num_layers):
        x = layers.SimpleRNN(rnn_units, return_sequences=True)(x)

    output = layers.Dense(vocab_size, activation="softmax")(x)

    return keras.Model(inputs=[feat_input, cap_input], outputs=output)

def prepare_targets(target, mask, start_idx):
    # menyisipkan <start> sebagai target wajib untuk t=0 (timestep CNN)
    batch_size = target.shape[0]
    start_col = np.full((batch_size, 1), start_idx, dtype=target.dtype)
    mask_col = np.ones((batch_size, 1), dtype=mask.dtype)

    full_target = np.concatenate([start_col, target], axis=1) 
    full_mask = np.concatenate([mask_col, mask], axis=1) 
    return full_target, full_mask

def train_single_config(config, caption_features, decoder_input, target, mask, meta, train_idx, val_idx, epochs, batch_size, save_dir):
    num_layers = config["layers"]
    rnn_units = config["units"]
    embed_dim = 256 
    config_name = f"rnn_L{num_layers}_H{rnn_units}"

    model = build_decoder(
        vocab_size=meta["vocab_size"],
        embed_dim=embed_dim,
        feature_dim=caption_features.shape[1],
        seq_len=decoder_input.shape[1],
        rnn_units=rnn_units,
        num_layers=num_layers,
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    full_target, full_mask = prepare_targets(target, mask, meta["start_idx"])

    train_feats, train_cap = caption_features[train_idx], decoder_input[train_idx]
    train_tgt, train_mask = full_target[train_idx], full_mask[train_idx]
    val_feats, val_cap = caption_features[val_idx], decoder_input[val_idx]
    val_tgt, val_mask = full_target[val_idx], full_mask[val_idx]

    start_time = time.time()
    history = model.fit(
        [train_feats, train_cap], train_tgt,
        sample_weight=train_mask,
        validation_data=([val_feats, val_cap], val_tgt, val_mask),
        epochs=epochs, batch_size=batch_size, verbose=1,
    )
    elapsed = time.time() - start_time

    # simpan bobot
    model_dir = os.path.join(save_dir, config_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{config_name}.keras")
    model.save(model_path)

    # simpan log
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_data.update({"training_time_s": elapsed, "config": config})
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(hist_data, f, indent=2)

    return {
        "config_name": config_name,
        "final_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "final_acc": history.history["accuracy"][-1],
        "final_val_acc": history.history["val_accuracy"][-1],
        "time_s": elapsed,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--units", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "saved_models", "rnn"))
    args = parser.parse_args()

    caption_features, decoder_input, target, mask, meta = load_training_data()
    train_idx, val_idx = train_val_split(len(caption_features))

    if args.layers is not None and args.units is not None:
        configs = [{"layers": args.layers, "units": args.units}]
    else:
        configs = [
            {"layers": 1, "units": 128}, {"layers": 1, "units": 512},
            {"layers": 2, "units": 128}, {"layers": 2, "units": 512},
            {"layers": 3, "units": 128}, {"layers": 3, "units": 512},
        ]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for config in configs:
        result = train_single_config(config, caption_features, decoder_input, target, mask, meta, train_idx, val_idx, args.epochs, args.batch_size, args.output_dir)
        all_results.append(result)

    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()