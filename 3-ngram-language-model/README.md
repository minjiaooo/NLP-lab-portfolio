# N-Gram Language Model & Next Word Prediction

An N-gram language model built **from scratch using NumPy**, trained on a Twitter corpus. Includes k-smoothing, next word prediction, and a bonus StyleGram classifier that identifies writing style across multiple authors.

## Project Structure

```
03-ngram-language-model/
├── ngram-language-model.py  # Core model: preprocessing, n-gram counting, probability estimation, prediction
├── utils.py                 # Count matrix and helper utilities
├── en_US_twitter.txt        # Twitter corpus (training data)
├── requirements.txt
└── README.md
```

## Key Concepts Implemented

| Component | Description |
|---|---|
| `read_and_tokenize_sentences()` | Reads corpus and tokenizes into word lists |
| `get_words_with_nplus_frequency()` | Builds closed vocabulary by frequency threshold |
| `replace_oov_words_by_unk()` | Replaces out-of-vocabulary words with `<unk>` |
| `count_n_grams()` | Counts all n-grams using a sliding window |
| `estimate_probabilities()` | Computes next-word probabilities with k-smoothing |
| `predict_next_word()` | Predicts the most likely next word for a partial sentence |
| `StyleGram` | Bonus: classifies writing style and predicts next word per author |

## How It Works

1. **Preprocessing** — Corpus is tokenized, rare words replaced with `<unk>`, split into train/test
2. **N-gram Counting** — Unigrams and bigrams counted using a sliding window with `<s>` / `<e>` tokens
3. **K-Smoothing** — Laplace-style smoothing prevents zero probabilities for unseen n-grams
4. **Prediction** — Given a partial sentence, the model returns the most probable next word
5. **StyleGram** (Extra Credit) — Trains one bigram model per author, classifies passage style by log-probability, then predicts next word using the winning model

## Setup

```bash
pip install -r requirements.txt
```

Download required NLTK data (handled automatically on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Usage

```bash
python ngram-language-model.py
```

The script will:
- Run all unit tests
- Train a bigram model on the Twitter corpus
- Predict the next word for an example sentence

### Example

```python
partial_sentence = "i love"
predicted_word = predict_next_word(partial_sentence, ngram_model)
# → "you"
```

## Requirements

- Python 3.7+
- numpy
- pandas
- nltk
