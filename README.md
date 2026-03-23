# PhoenixRec

**PhoenixRec** is a lightweight, modular Transformer-based recommendation model designed for real-world recsys scenarios with asymmetric attention patterns.

It features:
- **Asymmetric attention mask**: History is causal (upper triangle masked), candidates see full history + self but are invisible to each other.
- **Multi-hash feature compression** for users/items/authors to handle massive sparse IDs efficiently.
- **Signed multi-hot action embeddings** for rich behavior modeling (like, click, share, etc.).
- **Product surface categorical embeddings** for easy multi-modal/contextual extension.
- Modern components: RMSNorm, Rotary Embeddings (RoPE), GQA support, and flexible Transformer backbone.

Perfect for research, prototyping, or as a baseline for production recommendation systems in 2026.

## Model Overview
Input → [User Embedding] ─┬─ [History Sequence] ─┬─ [Candidate Sequence]
│                      │
RecSysTransformer     (custom asymmetric mask)
│
Final RMSNorm
│
Extract Candidate Embeddings → Linear → Logits

## Features

- Dedicated recsys attention mask (history causal + candidate self-attention only)
- GQA (Grouped Query Attention) for efficiency
- RoPE positional embeddings
- Multi-hash + projection reduce for ultra-large vocabularies
- Signed embedding for multi-label user actions
- Separable Transformer backbone (easy to swap or extend)

## Project Structure
phoenixrec/
├── transformer.py       # RecSysTransformer backbone (RoPE, GQA, asymmetric mask)
├── phoenix_model.py     # Main model with hash reduce, action embedding, etc.
└── README.md
