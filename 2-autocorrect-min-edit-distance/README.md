# Autocorrect & Minimum Edit Distance

A spelling correction system and edit distance calculator built **from scratch using only NumPy**, trained on a Shakespeare corpus as the language model.

## Project Structure

```
02-autocorrect-min-edit-distance/
├── autocorrect.py         # Core algorithms: data processing, autocorrect, min edit distance
├── utils.py               # Tweet preprocessing and feature extraction utilities
├── shakespeare-7k.txt     # Shakespeare corpus (~7k lines, used as language model)
├── shakespeare-edit.txt   # Full Shakespeare corpus (~121k lines)
├── requirements.txt
└── README.md
```

## Key Concepts Implemented

| Component | Description |
|---|---|
| `process_data()` | Reads corpus, tokenizes and computes word probability distribution |
| `edit_one_letter()` | Generates all words one edit away (delete, switch, replace, insert) |
| `edit_two_letters()` | Generates all words two edits away |
| `probable_substitutes()` | Returns top-N spelling correction candidates ranked by probability |
| `min_edit_distance()` | Dynamic programming algorithm to compute minimum edit distance |

## How It Works

1. **Language Model** — Word probabilities are computed from the Shakespeare corpus
2. **Candidate Generation** — For a misspelled word, all words within 2 edits are generated
3. **Ranking** — Candidates are filtered against the vocabulary and ranked by frequency
4. **Edit Distance** — A dynamic programming matrix computes the cost to transform any source string into a target string

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python autocorrect.py
```

The script will:
- Run unit tests for all components
- Launch an interactive autocorrect session (type any word to get suggestions)
- Launch an interactive min edit distance calculator

### Example

```
Enter any word: speling
[('spelling', 0.002), ('spieling', 0.0001), ...]
```

```
Enter any source word: hello
Enter any target word: help
hello and help are 3 apart
```

## Requirements

- Python 3.7+
- numpy
- pandas
- nltk
