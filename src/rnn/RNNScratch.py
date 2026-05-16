import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.extract_flickr8k_features import build_encoder
from src.rnn.embedding import Embedding
from src.rnn.rnn_cell import SimpleRNNCell
from src.base.dense import Dense
from src.base.activations import Softmax

class RNNScratch:
    def __init__(self, keras_model_path, metadata_path, encoder_name="inceptionv3"):
        with open(metadata_path, 'r') as f:
            self.meta = json.load(f)
            
        self.encoder, self.preprocess_fn, self.target_size = build_encoder(encoder_name)
        
        keras_decoder = keras.models.load_model(keras_model_path)
        
        proj_layer = keras_decoder.get_layer("dense_projection")
        self.dense_proj = Dense(*proj_layer.get_weights())
        
        emb_layer = keras_decoder.get_layer("embedding")
        self.embedding = Embedding(emb_layer.get_weights()[0])
        
        self.rnn_cells = []
        for layer in keras_decoder.layers:
            if isinstance(layer, keras.layers.SimpleRNN):
                cell = SimpleRNNCell(*layer.get_weights())
                self.rnn_cells.append(cell)
        
        out_layer = keras_decoder.get_layer("output")
        self.dense_out = Dense(*out_layer.get_weights(), activation=Softmax())
        
        if len(self.rnn_cells) > 0:
            self.hidden_units = self.rnn_cells[0].units
        else:
            raise ValueError("Model tidak memiliki layer SimpleRNN!")
        
    def generate_caption(self, image_path, idx_to_word, max_len=None):
        if max_len is None:
            max_len = self.meta["max_seq_len"]
            
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # (1, H, W, C)
        img_array = self.preprocess_fn(img_array)
        
        cnn_feature = self.encoder.predict(img_array, verbose=0) # Shape: (1, 2048)
    
        # proyeksi fitur CNN menjadi x_{-1}
        x_img = self.dense_proj.forward(cnn_feature) # Shape: (1, embed_dim)
        
        # inisialisasi hidden states dengan Zeros untuk setiap layer RNN
        h_t_list = [np.zeros((1, cell.units)) for cell in self.rnn_cells]
        
        # timestep t=-1
        current_input = x_img
        for i, cell in enumerate(self.rnn_cells):
            h_t_list[i] = cell.forward_step(current_input, h_t_list[i])
            current_input = h_t_list[i]
        
        current_token = self.meta["start_idx"]
        predicted_tokens = []
        
        for _ in range(max_len):
            # masukkan token ke Embedding
            x_word = self.embedding.forward(np.array([current_token]))
            x_word = x_word[0] # Hilangkan dimensi sequence -> (1, embed_dim)
            
            # hitung h_{t} baru untuk setiap layer secara berurutan
            current_input = x_word
            for i, cell in enumerate(self.rnn_cells):
                h_t_list[i] = cell.forward_step(current_input, h_t_list[i])
                current_input = h_t_list[i] # Output layer ini menjadi input layer berikutnya
            
            # hitung probabilitas kata berikutnya menggunakan output dari layer terakhir
            logits = self.dense_out.forward(current_input)
            
            # pilih kata dengan probabilitas tertinggi
            next_token = int(np.argmax(logits, axis=-1)[0])
            
            if next_token == self.meta["end_idx"]:
                break
                
            predicted_tokens.append(next_token)
            current_token = next_token
            
        caption = " ".join([idx_to_word.get(str(idx), "<unk>") for idx in predicted_tokens])
        return caption

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path ke gambar")
    parser.add_argument("--model", type=str, default=os.path.join(PROJECT_ROOT, "models", "rnn", "rnn_L3_H512", "rnn_L3_H512.keras"))
    parser.add_argument("--metadata", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "vocab", "metadata.json"))
    parser.add_argument("--vocab", type=str, default=os.path.join(PROJECT_ROOT,"outputs", "vocab", "vocab.json"))
    args = parser.parse_args()

    # load vocab untuk decode
    with open(args.vocab, 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = {str(idx): word for word, idx in word_to_idx.items()}

    captioner = RNNScratch(args.model, args.metadata)
    caption = captioner.generate_caption(args.image, idx_to_word)
    
    print("\n" + "="*50)
    print("Gambar:", args.image)
    print("Caption (RNN):", caption)
    print("="*50 + "\n")
