import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class RecModelConfig:
    """ Configuration parameters for the Recommendation Model """
    vocab_size: int = 10000
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 500
    num_experts: int = 4
    top_k: int = 2
    num_categories: int = 100
    dropout: float = 0.1

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE) to provide 
    relative positional information to the Transformer.
    """
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        return x * cos + torch.roll(x, shifts=-1, dims=-1) * sin

class Expert(nn.Module):
    """ Standard Feed-Forward Network used as an individual expert within the MoE layer """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
    def forward(self, x):
        return self.net(x)

class EfficientMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer that routes tokens to the Top-K experts 
    based on a learned gating mechanism.
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(dim, dim * 2) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        if len(orig_shape) == 2: x = x.unsqueeze(1)
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        logits = self.gate(x_flat)
        weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_idx = torch.topk(weights, self.top_k, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        
        routed = torch.zeros_like(x_flat)
        flat_token_idx = torch.arange(x_flat.shape[0], device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        flat_expert_idx = topk_idx.reshape(-1)
        flat_weights = topk_weights.reshape(-1).unsqueeze(-1)
        
        for e_id in range(self.num_experts):
            mask = (flat_expert_idx == e_id)
            if mask.any():
                token_idx = flat_token_idx[mask]
                expert_out = self.experts[e_id](x_flat[token_idx])
                routed.index_add_(0, token_idx, expert_out * flat_weights[mask].to(expert_out.dtype))
        
        moe_out = routed.view(B, S, D)
        if len(orig_shape) == 2: moe_out = moe_out.squeeze(1)
        return moe_out, logits

class TransformerMoEBlock(nn.Module):
    """ A Transformer layer replacing the standard MLP with a Mixture-of-Experts (MoE) component """
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = EfficientMoELayer(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        moe_out, router_logits = self.moe(x)
        x = self.norm2(x + self.dropout(moe_out))
        return x, router_logits

class UltimateTwoTowerMoERecModel(nn.Module):
    """
    Target-Aware Two-Tower Model using MoE and Cross-Attention to allow 
    the user representation to dynamically adapt to candidate items.
    """
    def __init__(self, config: RecModelConfig):
        super().__init__()
        self.config = config
        D = config.embed_dim

        self.item_embedding = nn.Embedding(config.vocab_size, D)
        self.category_embedding = nn.Embedding(config.num_categories, D)
        self.rope = RotaryEmbedding(D, config.max_seq_len)

        self.user_blocks = nn.ModuleList([
            TransformerMoEBlock(D, config.num_heads, config.dropout) for _ in range(config.num_layers)
        ])
        self.user_norm = nn.LayerNorm(D)

        self.item_moe = EfficientMoELayer(D, config.num_experts, config.top_k)
        self.item_norm = nn.LayerNorm(D)

        self.cross_attn = nn.MultiheadAttention(D, config.num_heads, dropout=config.dropout, batch_first=True)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Linear(D, 1)
        )

    def load_balance_loss(self, router_logits_list):
        """ Calculates the auxiliary loss to ensure balanced expert utilization """
        loss = 0.0
        for logits in router_logits_list:
            probs = F.softmax(logits, dim=-1)
            avg_prob = probs.mean(dim=0)
            uniform = torch.ones_like(avg_prob) / self.config.num_experts
            loss += (avg_prob - uniform).pow(2).mean()
        return loss

    def user_tower(self, history_seq, padding_mask=None):
        """ Processes user history into a sequence of hidden states """
        x = self.item_embedding(history_seq)
        x = self.rope(x)
        all_routers = []
        for block in self.user_blocks:
            x, r = block(x, padding_mask)
            all_routers.append(r)
        return self.user_norm(x), all_routers

    def item_tower(self, candidate_ids, category_ids=None):
        """ Encodes candidate items and applies MoE for specialized item representation """
        item_emb = self.item_embedding(candidate_ids)
        if category_ids is not None:
            cat_emb = self.category_embedding(category_ids)
            item_emb = item_emb + cat_emb
        item_emb, item_routers = self.item_moe(item_emb)
        return self.item_norm(item_emb), item_routers

    def forward(self, history_seq, candidate_ids, category_ids=None, padding_mask=None):
        """
        Main forward pass:
        1. Extract history sequence features.
        2. Extract candidate item features.
        3. Use Cross-Attention (Candidate as Query) to build target-aware user vectors.
        4. Fuse features and predict ranking logits.
        """
        history_hidden, user_routers = self.user_tower(history_seq, padding_mask)
        item_emb, item_routers = self.item_tower(candidate_ids, category_ids)

        B, Num_Cand, D = item_emb.shape
        query = item_emb.view(B * Num_Cand, 1, D)
        key_value = history_hidden.unsqueeze(1).expand(B, Num_Cand, -1, -1).reshape(B * Num_Cand, -1, D)

        attn_out, _ = self.cross_attn(query, key_value, key_value)
        target_aware_user = attn_out.view(B, Num_Cand, D)

        fused = torch.cat([target_aware_user, item_emb], dim=-1)
        logits = self.fusion_mlp(fused).squeeze(-1)

        aux_loss = self.load_balance_loss(user_routers + [item_routers])

        return logits, aux_loss

if __name__ == "__main__":
    config = RecModelConfig()
    model = UltimateTwoTowerMoERecModel(config)

    B, Seq, Cand = 32, 20, 50
    history = torch.randint(0, config.vocab_size, (B, Seq))
    candidates = torch.randint(0, config.vocab_size, (B, Cand))
    categories = torch.randint(0, config.num_categories, (B, Cand))

    logits, aux_loss = model(history, candidates, categories)
    print(f"Logits shape: {logits.shape}")
    print(f"Auxiliary Loss: {aux_loss.item():.4f}")
