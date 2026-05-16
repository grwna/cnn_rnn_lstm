import os
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.extract_flickr8k_features import build_encoder
from src.base.embedding import Embedding
from src.lstm.lstm_cell import LSTMCell
from src.base.dense import Dense
from src.base.activations import Softmax

class LSTMScratch:
    def __init__(
        self,
        keras_model_path: str,
        metadata_path: str,
        encoder_name: str = "inceptionv3",
    ):
        print("Memuat metadata")
        with open(metadata_path, "r") as f:
            self.meta = json.load(f)

        # 1. CNN Encoder (Keras pretrained, frozen) 
        print(f"Memuat CNN encoder: {encoder_name}")
        self.encoder, self.preprocess_fn, self.target_size = build_encoder(encoder_name)

        # 2. Load bobot dari Keras decoder 
        print(f"Memuat bobot Keras dari: {keras_model_path}")
        keras_decoder = keras.models.load_model(keras_model_path, safe_mode=False)

        # Dense projection (CNN feature -> embed_dim)
        proj_layer = keras_decoder.get_layer("dense_projection")
        self.dense_proj = Dense(*proj_layer.get_weights())

        # Embedding layer
        emb_layer = keras_decoder.get_layer("embedding")
        self.embedding = Embedding(emb_layer.get_weights()[0])

        # LSTM cells (ambil bobot dari setiap layer LSTM di Keras)
        self.lstm_cells = []
        for layer in keras_decoder.layers:
            if isinstance(layer, keras.layers.LSTM):
                kernel, recurrent_kernel, bias = layer.get_weights()
                cell = LSTMCell()
                cell.load_weights({
                    "weights": {
                        "kernel":            kernel,
                        "recurrent_kernel":  recurrent_kernel,
                        "bias":              bias,
                    }
                })
                self.lstm_cells.append(cell)

        if not self.lstm_cells:
            raise ValueError("Model tidak memiliki layer LSTM yang valid.")

        print(f"Jumlah LSTM layer: {len(self.lstm_cells)}")

        # Dense output layer (hidden -> vocab_size, softmax)
        out_layer = keras_decoder.get_layer("dense_output")
        self.dense_out = Dense(*out_layer.get_weights(), activation=Softmax())

        print("Pipeline ready")

    def generate_caption(
        self,
        image_path: str,
        idx_to_word: dict,
        max_len: int = None,
    ) -> str:
        if max_len is None:
            max_len = self.meta["max_seq_len"]

        # A. Ekstraksi fitur CNN 
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.target_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
        img_array = self.preprocess_fn(img_array)       # preprocessing sesuai model (misal: scaling, mean-subtraction, dll)

        cnn_feature = self.encoder.predict(img_array, verbose=0)  # (1, 2048)

        # B. LSTM Decoder from scratch 

        # Proyeksi fitur CNN -> x_{-1}
        x_img = self.dense_proj.forward(cnn_feature)  # (1, embed_dim)

        # Inisialisasi hidden state dan cell state untuk setiap layer
        h_list = [np.zeros((1, cell.W_hi.shape[0])) for cell in self.lstm_cells]
        c_list = [np.zeros((1, cell.W_hi.shape[0])) for cell in self.lstm_cells]

        # Timestep t=-1: inject fitur CNN
        current_input = x_img
        for i, cell in enumerate(self.lstm_cells):
            h_list[i], c_list[i] = cell.forward(current_input, h_list[i], c_list[i])
            current_input = h_list[i]

        # Greedy decoding loop
        current_token = self.meta["start_idx"]
        predicted_tokens = []

        for _ in range(max_len):
            # Embedding 
            x_word = self.embedding.forward(np.array([current_token]))  # (1, 1, embed_dim)
            x_word = x_word[:, 0, :]  # (1, embed_dim)

            # Forward pass LSTM layer
            current_input = x_word
            for i, cell in enumerate(self.lstm_cells):
                h_list[i], c_list[i] = cell.forward(current_input, h_list[i], c_list[i])
                current_input = h_list[i]

            logits = self.dense_out.forward(current_input)  # Probabilitas vocab (1, vocab_size)

            next_token = int(np.argmax(logits, axis=-1)[0]) # next_token = probabilitas tertinggi

            if next_token == self.meta["end_idx"]:
                break

            predicted_tokens.append(next_token)
            current_token = next_token

        caption = " ".join([
            idx_to_word.get(str(idx), "<unk>") for idx in predicted_tokens
        ])
        return caption


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Image Captioning from Scratch")
    parser.add_argument("--image",    type=str, required=True,  help="Path ke gambar")
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(PROJECT_ROOT, "models", "lstm", "lstm_L1_H512.keras"),
        help="Path ke file .keras hasil training"
    )
    parser.add_argument(
        "--metadata", type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "vocab", "metadata.json"),
    )
    parser.add_argument(
        "--vocab", type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "vocab", "vocab.json"),
    )
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()

    with open(args.vocab, "r") as f:
        word_to_idx = json.load(f)
    idx_to_word = {str(v): k for k, v in word_to_idx.items()}

    captioner = LSTMScratch(args.model, args.metadata)
    caption   = captioner.generate_caption(args.image, idx_to_word, max_len=args.max_len)

    print("\n" + "=" * 50)
    print("Gambar :", args.image)
    print("Caption:", caption)
    print("=" * 50 + "\n")