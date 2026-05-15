import os
import sys
import argparse
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.caption_preprocessing import (
    load_captions, clean_all_captions, build_vocabulary,
    tokenize_all_captions, pad_sequences, save_vocab, save_caption_data,
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN,
)

def main():
    parser = argparse.ArgumentParser(description="Preprocess Flickr8k captions")
    parser.add_argument("--captions_path", type=str, default=os.path.join(PROJECT_ROOT, "Flickr8k", "captions.txt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()

    raw_captions = load_captions(args.captions_path)

    cleaned_captions = clean_all_captions(raw_captions)

    word_to_idx, idx_to_word = build_vocabulary(cleaned_captions, min_freq=args.min_freq)
    vocab_size = len(word_to_idx)

    tokenized = tokenize_all_captions(cleaned_captions, word_to_idx)

    all_lengths = [len(cap) for cap_list in tokenized.values() for cap in cap_list]
    max_len = args.max_len or int(np.max(all_lengths))

    all_image_names = []
    all_caption_seqs = []
    for image_name, cap_list in tokenized.items():
        for cap_ids in cap_list:
            all_image_names.append(image_name)
            all_caption_seqs.append(cap_ids)

    padded_captions = pad_sequences(
        all_caption_seqs, max_len=max_len, pad_value=word_to_idx[PAD_TOKEN]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_vocab(word_to_idx, os.path.join(args.output_dir, "vocab.json"))
    save_caption_data(tokenized, os.path.join(args.output_dir, "tokenized_captions.npy"))
    
    np.save(os.path.join(args.output_dir, "padded_captions.npy"), padded_captions)
    np.save(os.path.join(args.output_dir, "caption_image_names.npy"), np.array(all_image_names))

    metadata = {
        "vocab_size": vocab_size,
        "max_seq_len": max_len,
        "num_images": len(raw_captions),
        "min_freq": args.min_freq,
        "pad_idx": word_to_idx[PAD_TOKEN],
        "start_idx": word_to_idx[START_TOKEN],
        "end_idx": word_to_idx[END_TOKEN],
        "unk_idx": word_to_idx[UNK_TOKEN],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()