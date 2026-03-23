import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TransformerConfig:
    """Configuration for the RecSys Transformer model."""
    model_dim: int          # Embedding / hidden dimension
    key_size: int           # Dimension per attention head
    num_q_heads: int        # Number of query heads
    num_kv_heads: int       # Number of key/value heads (supports GQA)
    num_layers: int         # Number of transformer layers
    widening_factor: float = 4.0
    attn_output_multiplier: float = 1.0


def ffn_size(model_size: int, widening_factor: float) -> int:
    """Calculate intermediate FFN size and make it multiple of 8."""
    intermediate = int(widening_factor * model_size) * 2 // 3
    intermediate = intermediate + (8 - intermediate) % 8
    return intermediate


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fprop_dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.scale * norm).to(fprop_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the dimensions for rotary embeddings."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) module."""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        return cos, sin


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key."""
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-head attention with GQA support and rotary embeddings."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.model_dim, config.num_q_heads * config.key_size, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.num_kv_heads * config.key_size, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.num_kv_heads * config.key_size, bias=False)
        self.o_proj = nn.Linear(config.num_q_heads * config.key_size, config.model_dim, bias=False)
        self.rope = RotaryEmbedding(config.key_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.config.num_q_heads, self.config.key_size)
        k = self.k_proj(x).view(B, T, self.config.num_kv_heads, self.config.key_size)
        v = self.v_proj(x).view(B, T, self.config.num_kv_heads, self.config.key_size)

        cos, sin = self.rope(q)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Repeat KV for GQA
        n_rep = self.config.num_q_heads // self.config.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)

        # Attention computation
        logits = torch.einsum("bthd,bThd->bhht", q, k) * self.config.attn_output_multiplier
        logits = logits.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(logits.float(), dim=-1).to(x.dtype)

        out = torch.einsum("bhht,bthd->bthd", attn, v)
        return self.o_proj(out.reshape(B, T, -1))


class DenseBlock(nn.Module):
    """Feed-forward network with GELU approximation (SwiGLU style)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = ffn_size(config.model_dim, config.widening_factor)
        self.v_proj = nn.Linear(config.model_dim, hidden_dim, bias=False)
        self.w_proj = nn.Linear(config.model_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(F.gelu(self.w_proj(x)) * self.v_proj(x))


def make_recsys_attn_mask(seq_len: int, candidate_start_offset: int, device: torch.device) -> torch.Tensor:
    """
    Create attention mask for recommendation system:
    - History is causal
    - Candidates can see all history but not other candidates (except self)
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    mask[candidate_start_offset:, candidate_start_offset:] = 0
    for i in range(candidate_start_offset, seq_len):
        mask[i, i] = 1
    return mask.unsqueeze(0).unsqueeze(0)


class RecSysTransformer(nn.Module):
    """Transformer backbone optimized for recommendation systems."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "norm_1": RMSNorm(config.model_dim),
                "attn": MultiHeadAttention(config),
                "norm_2": RMSNorm(config.model_dim),
                "dense": DenseBlock(config),
                "norm_3": RMSNorm(config.model_dim)
            }))

    def forward(
        self,
        x: torch.Tensor,
        candidate_start_offset: int,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        attn_mask = make_recsys_attn_mask(T, candidate_start_offset, x.device)

        if padding_mask is not None:
            padding_mask = padding_mask.view(B, 1, 1, T)
            attn_mask = attn_mask.masked_fill(padding_mask == 0, 0)

        h = x
        for layer in self.layers:
            h_norm = layer["norm_1"](h)
            h = h + layer["attn"](h_norm, attn_mask)

            h_norm = layer["norm_2"](h)
            h = h + layer["dense"](h_norm)
            h = layer["norm_3"](h)

        return h
