# Twitter Sentiment Analysis — Two-Layer Neural Network from Scratch

A tweet sentiment classifier built **from scratch using only NumPy**, without any deep learning frameworks. The model is trained on NLTK's Twitter dataset and achieves high accuracy distinguishing positive from negative tweets.

## Project Structure

```
01-twitter-sentiment-nn/
├── twitter-sentiment.py # Main model: sigmoid, forward pass, BCE loss, backprop, training loop
├── utils.py             # Tweet preprocessing, frequency table, feature extraction
├── en_US_twitter.txt    # Raw Twitter corpus (used for reference / language model)
├── requirements.txt
└── README.md
```

## Key Concepts Implemented

| Component | Description |
|---|---|
| `sigmoid()` | Logistic activation function |
| `inference_layer()` | Single-layer forward pass |
| `inference_2layers()` | Two-layer neural network forward pass |
| `bce_forward()` | Binary Cross-Entropy loss |
| `gradients()` | Manual backpropagation |
| `update_params()` | Mini-batch SGD parameter update |
| `train_nn()` | Full training loop with loss logging |

## How It Works

1. **Preprocessing** — Tweets are tokenized, stopwords removed, and Porter-stemmed via `utils.py`
2. **Feature Extraction** — Each tweet becomes a 3D vector: `[bias, positive_word_freq, negative_word_freq]`
3. **Training** — A 2-layer neural network is trained with mini-batch SGD and BCE loss
4. **Inference** — Sigmoid output > 0.5 → Positive sentiment

## Setup

```bash
pip install -r requirements.txt
```

Download required NLTK data (handled automatically on first run):
```python
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
```

## Usage

```bash
# Run all unit tests, then train the model
python assignment2.py
```

The script will:
- Run unit tests for each component (`sigmoid_test`, `gradients_test`, etc.)
- Train on `twitter_data.pkl` (generated from NLTK twitter_samples)
- Save model weights to `model_params.npz` and `assignment2.pkl`
- Plot training loss curve
- Report test accuracy and BCE loss

### Custom Tweet Prediction

```python
from utils import extract_features
from assignment2 import inference_2layers
import numpy as np

# Load saved model
params = np.load('model_params.npz')
W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

tweet = "I absolutely love this! Best day ever."
x = extract_features(tweet, freqs).reshape(-1, 1)
score = inference_2layers(x, W1, W2, b1, b2)[0]
print("Positive" if score > 0.5 else "Negative", f"(score: {score:.4f})")
```

## Requirements

- Python 3.7+
- numpy
- nltk
- matplotlib
- pandas
