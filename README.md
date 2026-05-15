# CNN_RNN_LSTM
Custom implementation of CNN, RNN, and LSTM from scratch

## Feature Extraction
```
source venv/bin/activate

PYTHONPATH=. python3 scripts/extract_flickr8k_features.py \
    --images_dir data/Images \
    --split_dir data \
    --model inceptionv3
```