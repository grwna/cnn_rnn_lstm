import re
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# load from flickr8k dataset, captions.txt, format: image,caption
def load_captions(captions_path: str) -> Dict[str, List[str]]:
    captions: Dict[str, List[str]] = {}
    with open(captions_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",", 1)
        if len(parts) != 2:
            continue

        image_name, caption = map(str.strip, parts)
        if image_name not in captions:
            captions[image_name] = []
        captions[image_name].append(caption)

    return captions

def clean_caption(caption: str) -> str:
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption) 
    return re.sub(r"\s+", " ", caption).strip()

def clean_all_captions(captions: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cleaned = {}
    for image_name, caption_list in captions.items():
        cleaned[image_name] = []
        for cap in caption_list:
            cap_clean = clean_caption(cap)
            if cap_clean:
                cleaned[image_name].append(f"{START_TOKEN} {cap_clean} {END_TOKEN}")
    return cleaned

def build_vocabulary(captions: Dict[str, List[str]], min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    word_counts: Counter = Counter()
    for caption_list in captions.values():
        for caption in caption_list:
            word_counts.update(caption.split())

    for token in SPECIAL_TOKENS:
        word_counts.pop(token, None)

    vocab_words = sorted([w for w, c in word_counts.items() if c >= min_freq])

    word_to_idx: Dict[str, int] = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for word in vocab_words:
        word_to_idx[word] = len(word_to_idx)

    idx_to_word: Dict[int, str] = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def tokenize_caption(caption: str, word_to_idx: Dict[str, int]) -> List[int]:
    unk_idx = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(word, unk_idx) for word in caption.split()]

def tokenize_all_captions(captions: Dict[str, List[str]], word_to_idx: Dict[str, int]) -> Dict[str, List[List[int]]]:
    return {
        img: [tokenize_caption(cap, word_to_idx) for cap in cap_list]
        for img, cap_list in captions.items()
    }

def pad_sequences(sequences: List[List[int]], max_len: Optional[int] = None, pad_value: int = 0) -> np.ndarray:
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded

def save_vocab(word_to_idx: Dict[str, int], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f, ensure_ascii=False, indent=2)

def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(idx): word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def save_caption_data(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, data, allow_pickle=True)

def load_caption_data(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()

def decode_sequence(token_ids: List[int], idx_to_word: Dict[int, str]) -> str:
    words = []
    for idx in token_ids:
        word = idx_to_word.get(idx, UNK_TOKEN)
        if word == END_TOKEN:
            break
        if word not in (PAD_TOKEN, START_TOKEN):
            words.append(word)
    return " ".join(words)