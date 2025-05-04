 # Data Preprocessing for Toxic Comment Classification

This Section describes the preprocessing steps performed on the dataset for toxic comment classification.

## Overview

The preprocessing phase prepares both clean and poisoned datasets for training a neural network model. The process includes text cleaning, tokenization, and embedding preparation.

## Dependencies

- torchtext
- torch
- pandas
- sklearn
- sentencepiece
- nltk
- numpy

## Data Sources

- Clean dataset: `../data/balanced_dataset.csv`
- Poisoned dataset: `../poisoning/poisoned10p.csv` (Or any one you want)

## Preprocessing Steps

### 1. Data Loading and Initial Cleaning

- Load clean and poisoned datasets
- Remove any null values
- Rename columns for consistency
- Basic data validation

### 2. Text Preprocessing

- Expand contractions (e.g., "don't" → "do not")
- Remove special characters and punctuation
- Convert text to lowercase
- Remove stopwords
- Apply stemming and lemmatization

### 3. Tokenization and Vocabulary Building

- Use SentencePiece for tokenization
- Add special tokens:
  - `<sos>`: Start of sentence
  - `<eos>`: End of sentence
  - `<unk>`: Unknown token
  - `<pad>`: Padding token

### 4. Embedding Preparation

- Use GloVe embeddings (6B tokens, 300 dimensions)
- Map tokens to their corresponding embeddings
- Handle out-of-vocabulary words

## Configuration Parameters

- Maximum sentence length: 100 tokens
- Embedding dimension: 300
- Device: CUDA if available, else CPU
- Random seed: 1300

