# Akkadian Unicode → Transliteration Transformer

A research project exploring Transformer-based neural transliteration for **Akkadian cuneiform**.

This repository implements a character-level Transformer that learns to convert **Unicode cuneiform signs** into **scholarly Akkadian transliteration**, providing a foundation for computational research in ancient Mesopotamian languages.

---

## About Akkadian

Akkadian was the dominant language of ancient Mesopotamia for over two millennia and was written using **cuneiform**, one of the earliest writing systems in human history. Rather than using an alphabet, cuneiform is composed of wedge-shaped signs that may represent syllables, words, or semantic determinatives.

Today, hundreds of thousands of clay tablets survive, preserving literature, royal inscriptions, legal documents, mathematics, astronomy, and administrative records. However, computational resources for processing Akkadian remain limited compared to modern languages.

Before translation into English, Assyriologists first convert cuneiform signs into a standardized Latin-script representation known as **scholarly transliteration**. This project focuses on automating that step using deep learning. Transliteration is a crucial intermediate representation used throughout Akkadian research and digital humanities.

---

## Project Objective

The objective of this project is to train a Transformer model capable of learning the mapping:

```text
Unicode Cuneiform
        ↓
Transformer
        ↓
Scholarly Transliteration
```

Instead of relying on handcrafted linguistic rules, the model learns these mappings directly from aligned Unicode cuneiform and transliteration pairs.

---
## Example

```text
Input (Unicode)

𒈗𒌋𒆠

↓

Output

LUGAL URI
```

## Current Features

* Character-level Transformer encoder-decoder
* Unicode Cuneiform → Scholarly Transliteration
* Automatic preprocessing pipeline
* Character vocabulary generation
* Dynamic sequence padding
* Teacher forcing during training
* Greedy autoregressive decoding
* EOS-aware decoding
* Character Error Rate (CER) evaluation
* Word Error Rate (WER) evaluation
* Automatic checkpoint saving
* Resume interrupted training
* Apple Silicon (MPS) compatible
* Google Colab compatible

---

## Dataset

Current experiments use an aligned Akkadian corpus consisting of approximately:

| Statistic                      |         Value |
| ------------------------------ | ------------: |
| Original aligned pairs         |      ~190,000 |
| Filtered training pairs        |      ~152,000 |
| Unique Unicode cuneiform signs |           432 |
| Transliteration vocabulary     | 71 characters |

Filtering removes extremely long sequences to improve training stability and efficiency while preserving the majority of the corpus.

---


## Data Pipeline

The workflow consists of:

1. Loading aligned Unicode cuneiform and transliteration pairs
2. Cleaning and filtering long sequences
3. Character-level vocabulary generation
4. Train / Validation / Test split
5. Dynamic padding during batching
6. Transformer training
7. Validation using CER and WER
8. Checkpoint generation after every epoch

---

## Model Architecture

Current implementation:

* Transformer Encoder–Decoder
* Character Embeddings
* Sinusoidal Positional Encoding
* Multi-Head Self Attention
* Cross Attention
* Feed Forward Network
* Cross Entropy Loss
* Teacher Forcing
* Greedy Decoding

The architecture is intentionally lightweight so that experiments can be reproduced on consumer hardware such as an Apple Silicon MacBook.

---

## Evaluation

The model is evaluated using:

* Character Error Rate (CER)
* Word Error Rate (WER)
* Validation Loss
* EOS prediction rate
* Prediction length statistics
* Decoding diagnostics

These metrics provide insight into both transliteration accuracy and decoder stability.

---

## Repository Structure

.
├── checkpoints/        # Saved model checkpoints
├── data/
│   ├── raw/            # Original corpus
│   └── processed/      # Filtered datasets
├── notebooks/          # Exploration notebooks
├── scripts/            # Training scripts
├── src/                # Model components
└── README.md

---

## Technology Stack

* Python
* PyTorch
* Transformer Networks
* Pandas
* NumPy
* Apple Metal (MPS)
* Jupyter Notebook

---

## Current Status

## Development Progress

Current focus:

- Dataset preprocessing
- Character-level Transformer training
- Model evaluation using CER/WER
- Decoder optimization

Future updates, experiments, and results will be published in this repository as development continues.

Planned:

* Improved Transformer architectures
* Beam search decoding
* Hyperparameter optimization
* Unicode → Transliteration accuracy improvements
* Bidirectional transliteration experiments

---

## Research Motivation

Ancient languages remain underrepresented in modern Natural Language Processing. This project investigates whether compact Transformer architectures can effectively learn Unicode cuneiform transliteration while remaining reproducible on accessible hardware.


---

## Citation

If this repository contributes to your research or academic work, please cite the repository or include a link to it in your references.

Repository:
https://github.com/ERo2203/akkadian-transliteration

---

