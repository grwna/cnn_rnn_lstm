import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.extract_flickr8k_features import build_encoder

class RNNKeras:
    def __init__(self, keras_model_path, metadata_path, encoder_name="inceptionv3"):
        with open(metadata_path, 'r') as f:
            self.meta = json.load(f)
            
        self.encoder, self.preprocess_fn, self.target_size = build_encoder(encoder_name)
        self.decoder = keras.models.load_model(keras_model_path)
        
    def generate_caption(self, image_path, idx_to_word, max_len=None):
        if max_len is None:
            max_len = self.meta["max_seq_len"]
            
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # (1, H, W, C)
        img_array = self.preprocess_fn(img_array)
        
        cnn_feature = self.encoder.predict(img_array, verbose=0) # Shape: (1, 2048)
        
        current_token = self.meta["start_idx"]
        predicted_tokens = []
        
        cap_input_len = self.meta["max_seq_len"] - 1
        cap_input = np.full((1, cap_input_len), self.meta["pad_idx"], dtype=np.int32)
        
        for t in range(max_len):
            if t < cap_input_len:
                cap_input[0, t] = current_token
                
            # Output Keras (1, max_seq_len + 1, vocab_size)
            logits = self.decoder.predict([cnn_feature, cap_input], verbose=0)
            
            # ambil probabilitas di timestep t
            # timestep 0 adalah output dari CNN feature (t=-1)
            # timestep t+1 adalah output untuk cap_input token ke-t
            step_logits = logits[0, t+1, :]
            
            next_token = int(np.argmax(step_logits))
            
            if next_token == self.meta["end_idx"]:
                break
                
            predicted_tokens.append(next_token)
            current_token = next_token
            
            if t == self.meta["max_seq_len"] - 1:
                break
            
        caption = " ".join([idx_to_word.get(str(idx), "<unk>") for idx in predicted_tokens])
        return caption

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path ke gambar")
    parser.add_argument("--model", type=str, default=os.path.join(PROJECT_ROOT, "models", "rnn", "rnn_L1_H128", "rnn_L1_H128.keras"))
    parser.add_argument("--metadata", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "vocab", "metadata.json"))
    parser.add_argument("--vocab", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "vocab", "vocab.json"))
    args = parser.parse_args()

    with open(args.vocab, 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = {str(idx): word for word, idx in word_to_idx.items()}

    captioner = RNNKeras(args.model, args.metadata)
    
    start_time = time.time()
    caption = captioner.generate_caption(args.image, idx_to_word)
    exec_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("Gambar:", args.image)
    print("Caption (Keras):", caption)
    print(f"Waktu eksekusi: {exec_time:.4f} detik")
    print("="*50 + "\n")
