# Tugas Besar 2 IF3270 — CNN, RNN, dan LSTM

Implementasi **Convolutional Neural Network (CNN)**, **Simple Recurrent Neural Network (RNN)**, dan **Long Short-Term Memory (LSTM)** dari scratch menggunakan NumPy, serta eksplorasi pipeline *image captioning* berbasis arsitektur encoder-decoder.

---

## Deskripsi Singkat

Proyek ini terdiri dari dua bagian utama:

1. **CNN** — Klasifikasi gambar menggunakan dataset Intel Image Classification (6 kategori). Implementasi forward propagation CNN dari scratch dengan variasi hyperparameter.

2. **RNN & LSTM** — *Image captioning* menggunakan dataset Flickr8k. Implementasi forward propagation SimpleRNN dan LSTM dari scratch, serta eksplorasi pipeline encoder-decoder yang menggabungkan CNN pretrained (InceptionV3) sebagai encoder dan RNN/LSTM sebagai decoder.

---

## Struktur Repository

```
.
├── data/
│   ├── Images/                          <- Gambar Flickr8k
│   ├── captions.txt                     <- Caption Flickr8k
│   ├── Flickr_8k.trainImages.txt        <- Split train
│   ├── Flickr_8k.devImages.txt          <- Split val
│   └── Flickr_8k.testImages.txt         <- Split test
│
├── models/
│   ├── lstm/                            <- Bobot model LSTM (.keras)
│   ├── cnn_keras/                       <- Bobot model CNN (.keras)
│   └── rnn/                             <- Bobot model RNN (.keras)
│
├── outputs/
│   ├── features/                        <- CNN feature vectors (.npy)
│   ├── vocab/                           <- Vocabulary dan metadata
│   ├── visualizations/                  <- Grafik dan visualisasi
│   ├── results/                         <- *.json
│   ├── cnn_training_results.pkl         <- training results
│   └── cnn_scratch_predictions.pkl      <- predictions
│
├── src/
│   ├── base/
│   │   ├── activations.py               <- ReLU, Softmax, Sigmoid
│   │   ├── dense.py                     <- Dense layer
│   │   ├── embedding.py                 <- Embedding layer
│   │   └── flatten.py                   <- Flatten layer
│   │
│   ├── cnn/
│   │   ├── layers.py                    <- Conv2D + lokalisasi layer
│   │   ├── locally_connected.py         <- layer tanpa bobot sharing, untuk eksperimen khusus
│   │   ├── pooling.py                   <- semua fungsi pooling dasar untuk CNN
│   │   ├── model.py                     <- tempat model CNN dibangun dan digunakan untuk inferensi/prediksi
│   │   └── conv2d.py                    <- Conv2D from scratch
│   │
│   ├── lstm/
│   │   ├── lstm_cell.py                 <- LSTM cell from scratch
│   │   ├── lstm_decoder.py              <- LSTM decoder pipeline
│   │   ├── LSTMKeras.py                 <- Inference menggunakan Keras
│   │   └── LSTMScratch.py               <- Inference from scratch
│   │
│   ├── rnn/
│   │   ├── scripts/
│   │   │   ├── preprocess_captions.py   <- Preprocess caption
│   │   │   └── train_rnn_decoder.py     <- SimpleRNN decoder pipeline
│   │   ├── rnn_cell.py                  <- SimpleRNN cell from scratch
│   │   ├── RNNKeras.py                  <- Inference menggunakan Keras
│   │   └── RNNScratch.py                <- Inference from scratch
│   │
│   ├── utils/
│   │   ├── caption_preprocessing.py     <- Tokenisasi dan preprocessing caption
│   │   ├── extract_flickr8k_features.py <- Script ekstraksi fitur Flickr8k
│   │   ├── feature_extractor.py         <- Utility ekstraksi fitur (generik)
│   │   ├── image_utils.py               <- Load dan preprocessing gambar
│   │   ├── generate_splits.py           <- Generate file split Flickr8k
│   │   └── weight_loaders.py            <- Load bobot dari Keras model
│   │
│   └── notebooks/
│       ├── lstm_training.ipynb          <- Training 6 variasi LSTM
│       ├── LSTM_eksperimen.ipynb        <- Evaluasi dan analisis LSTM
│       └── (CNN notebooks)
│
├── requirements.txt
└── README.md
```

---

## Setup dan Instalasi

### Prasyarat

- Python 3.12
- CUDA-compatible GPU (opsional, untuk mempercepat training)
- WSL2 (jika menggunakan Windows)

### Langkah Instalasi

**1. Clone repository:**
```bash
git clone https://github.com/<username>/CNN_RNN_LSTM.git
cd CNN_RNN_LSTM
```

**2. Buat virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/WSL
# atau
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Setup GPU (opsional, WSL2):**

Tambahkan ke `~/.zshrc` atau `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cublas/lib:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cusolver/lib:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cufft/lib:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cusparse/lib:\
/path/to/venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:\
/usr/lib/wsl/lib
```

---

## Cara Menjalankan

### Persiapan Dataset

**1. Download dataset Flickr8k** dari [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) dan taruh di `data/`:
```
data/
├── Images/
└── captions.txt
```

**2. Generate split files:**
```bash
python generate_splits.py \
    --images_dir data/Images \
    --output_dir data
```

**3. Ekstraksi fitur CNN:**
```bash
PYTHONPATH=. python scripts/extract_flickr8k_features.py \
    --images_dir data/Images \
    --split_dir  data \
    --model      inceptionv3 \
    --output_dir outputs/features
```

**4. Preprocessing caption:**
```bash
PYTHONPATH=. python src/rnn/scripts/preprocess_captions.py \
    --captions_path data/captions.txt \
    --output_dir    outputs/vocab
```

### Training Model

**LSTM:**
```bash
# Buka notebook di Jupyter atau VS Code
src/notebooks/lstm_training.ipynb
```

**RNN:**
```bash
PYTHONPATH=. python src/rnn/scripts/train_rnn_decoder.py \
    --epochs 15 \
    --batch_size 64
```

### Evaluasi dan Eksperimen

```bash
src/notebooks/evaluasi_analisis_lstm.ipynb    
src/notebooks/evaluasi_analisis_rnn.ipynb     
```

### Inference (Generate Caption untuk Satu Gambar)

**LSTM Keras:**
```bash
PYTHONPATH=. python src/lstm/LSTMKeras.py \
    --image   data/Images/contoh.jpg \
    --model   models/lstm/lstm_L3_H512.keras \
    --metadata outputs/vocab/metadata.json \
    --vocab    outputs/vocab/vocab.json
```

**LSTM Scratch:**
```bash
PYTHONPATH=. python src/lstm/LSTMScratch.py \
    --image   data/Images/contoh.jpg \
    --model   models/lstm/lstm_L3_H512.keras \
    --metadata outputs/vocab/metadata.json \
    --vocab    outputs/vocab/vocab.json
```

**RNN Keras:**
```bash
PYTHONPATH=. python src/rnn/RNNKeras.py \
    --image   data/Images/contoh.jpg \
    --model   models/rnn/rnn_L3_H128/rnn_L3_H128.keras \
    --metadata outputs/vocab/metadata.json \
    --vocab    outputs/vocab/vocab.json
```

**RNN Scratch:**
```bash
PYTHONPATH=. python src/rnn/RNNScratch.py \
    --image   data/Images/contoh.jpg \
    --model   models/rnn/rnn_L3_H128/rnn_L3_H128.keras \
    --metadata outputs/vocab/metadata.json \
    --vocab    outputs/vocab/vocab.json
```

---

## Pembagian Tugas

| Nama | NIM | Tugas |
|------|-----|-------|
| M. Rayhan Farrukh | 13523035 | CNN: implementasi forward prop, training, evaluasi |
| Muhammad Alfansya | 13523005 | RNN: implementasi forward prop, training, evaluasi |
| Hanif Kalyana Aditya | 13523041 | LSTM: implementasi forward prop, training, evaluasi|

---