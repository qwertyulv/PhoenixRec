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

## Quick Start

```python
# Install dependencies
pip install torch

# Example usage
from phoenixrec.transformer import RecSysTransformer, TransformerConfig
from phoenixrec.phoenix_model import PhoenixModel, PhoenixModelConfig

# Transformer backbone
transformer_config = TransformerConfig(
    model_dim=256,
    key_size=32,
    num_q_heads=8,
    num_kv_heads=2,
    num_layers=6
)
transformer = RecSysTransformer(transformer_config)

# Phoenix model
model_config = PhoenixModelConfig(
    emb_size=256,
    num_actions=10,  # e.g., like, click, add-to-cart, purchase...
    product_surface_vocab_size=2048,
    hash_config=HashConfig(num_user_hashes=4, num_item_hashes=4, num_author_hashes=4)
)
model = PhoenixModel(model_config, transformer)

Note: You need to provide pre-computed embeddings (RecsysEmbeddings) and batch data (RecsysBatch) from your feature pipeline.

phoenixrec/
├── transformer.py       # RecSysTransformer backbone (RoPE, GQA, asymmetric mask)
├── phoenix_model.py     # Main model with hash reduce, action embedding, etc.
└── README.md

