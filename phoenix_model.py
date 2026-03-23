import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Tuple


@dataclass
class HashConfig:
    """Configuration for number of hash buckets per entity."""
    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2


@dataclass
class RecsysEmbeddings:
    """Pre-computed embedding tables from external lookup."""
    user_embeddings: torch.Tensor
    history_post_embeddings: torch.Tensor
    candidate_post_embeddings: torch.Tensor
    history_author_embeddings: torch.Tensor
    candidate_author_embeddings: torch.Tensor


class RecsysModelOutput(NamedTuple):
    """Output structure of the recommendation model."""
    logits: torch.Tensor


@dataclass
class RecsysBatch:
    """Input batch for recommendation model."""
    user_hashes: torch.Tensor
    history_post_hashes: torch.Tensor
    history_author_hashes: torch.Tensor
    history_actions: torch.Tensor
    history_product_surface: torch.Tensor
    candidate_post_hashes: torch.Tensor
    candidate_author_hashes: torch.Tensor
    candidate_product_surface: torch.Tensor


def block_user_reduce(
    user_hashes: torch.Tensor,
    user_embeddings: torch.Tensor,
    num_user_hashes: int,
    emb_size: int,
    proj_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce user features using hash + linear projection."""
    B = user_embeddings.shape[0]
    D = emb_size
    user_embedding = user_embeddings.reshape(B, 1, num_user_hashes * D)
    user_embedding = torch.matmul(user_embedding.to(proj_weight.dtype), proj_weight.t())
    user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1)
    return user_embedding, user_padding_mask


def block_history_reduce(
    history_post_hashes: torch.Tensor,
    history_post_embeddings: torch.Tensor,
    history_author_embeddings: torch.Tensor,
    history_product_surface_embeddings: torch.Tensor,
    history_actions_embeddings: torch.Tensor,
    num_item_hashes: int,
    num_author_hashes: int,
    proj_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce history sequence features: post + author + action + surface."""
    B, S, _, D = history_post_embeddings.shape
    post_reshaped = history_post_embeddings.reshape(B, S, num_item_hashes * D)
    author_reshaped = history_author_embeddings.reshape(B, S, num_author_hashes * D)
    combined = torch.cat([
        post_reshaped,
        author_reshaped,
        history_actions_embeddings,
        history_product_surface_embeddings
    ], dim=-1)
    history_embedding = torch.matmul(combined.to(proj_weight.dtype), proj_weight.t())
    history_padding_mask = (history_post_hashes[:, :, 0] != 0)
    return history_embedding, history_padding_mask


def block_candidate_reduce(
    candidate_post_hashes: torch.Tensor,
    candidate_post_embeddings: torch.Tensor,
    candidate_author_embeddings: torch.Tensor,
    candidate_product_surface_embeddings: torch.Tensor,
    num_item_hashes: int,
    num_author_hashes: int,
    proj_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce candidate sequence features: post + author + surface."""
    B, C, _, D = candidate_post_embeddings.shape
    post_reshaped = candidate_post_embeddings.reshape(B, C, num_item_hashes * D)
    author_reshaped = candidate_author_embeddings.reshape(B, C, num_author_hashes * D)
    combined = torch.cat([
        post_reshaped,
        author_reshaped,
        candidate_product_surface_embeddings
    ], dim=-1)
    candidate_embedding = torch.matmul(combined.to(proj_weight.dtype), proj_weight.t())
    candidate_padding_mask = (candidate_post_hashes[:, :, 0] != 0)
    return candidate_embedding, candidate_padding_mask


@dataclass
class PhoenixModelConfig:
    """Configuration for the Phoenix recommendation model."""
    emb_size: int
    num_actions: int
    product_surface_vocab_size: int = 16
    fprop_dtype: torch.dtype = torch.bfloat16
    hash_config: HashConfig = field(default_factory=HashConfig)


class PhoenixModel(nn.Module):
    """
    Phoenix recommendation model with asymmetric attention for history and candidates.
    
    Features:
    - User + History + Candidates three-stage input
    - Multi-hash compression + projection
    - Signed multi-hot action embeddings
    - Product surface categorical embedding
    """

    def __init__(self, config: PhoenixModelConfig, transformer_module: nn.Module):
        super().__init__()
        self.config = config
        self.transformer = transformer_module

        D = config.emb_size
        h = config.hash_config

        self.user_proj = nn.Linear(h.num_user_hashes * D, D, bias=False)
        self.history_proj = nn.Linear(
            (h.num_item_hashes + h.num_author_hashes) * D + 2 * D, D, bias=False
        )
        self.candidate_proj = nn.Linear(
            (h.num_item_hashes + h.num_author_hashes) * D + D, D, bias=False
        )

        self.action_projection = nn.Parameter(torch.empty(config.num_actions, D))
        self.product_surface_emb = nn.Embedding(config.product_surface_vocab_size, D)
        self.unembeddings = nn.Parameter(torch.empty(config.num_actions, D))
        self.final_norm = RMSNorm(D)  # Assuming RMSNorm is defined in transformer.py or imported

        self._init_weights()

    def _init_weights(self):
        for m in [self.user_proj, self.history_proj, self.candidate_proj]:
            nn.init.xavier_uniform_(m.weight)
        for p in [self.action_projection, self.unembeddings]:
            nn.init.xavier_uniform_(p)
        nn.init.normal_(self.product_surface_emb.weight, mean=0.0, std=0.02)

    def _get_action_embeddings(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert multi-hot actions to signed embeddings."""
        actions_signed = (2.0 * actions.float() - 1.0)
        action_emb = torch.matmul(actions_signed, self.action_projection.t())
        valid_mask = (actions.sum(dim=-1, keepdim=True) > 0).to(action_emb.dtype)
        return action_emb * valid_mask

    def build_inputs(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Build input embeddings and masks for the transformer."""
        hist_ps_emb = self.product_surface_emb(batch.history_product_surface.long())
        cand_ps_emb = self.product_surface_emb(batch.candidate_product_surface.long())
        hist_act_emb = self._get_action_embeddings(batch.history_actions)

        user_emb_raw, user_mask = block_user_reduce(
            batch.user_hashes, recsys_embeddings.user_embeddings,
            self.config.hash_config.num_user_hashes, self.config.emb_size,
            self.user_proj.weight
        )
        user_emb = self.user_proj(user_emb_raw.squeeze(1)).unsqueeze(1)

        hist_emb_raw, hist_mask = block_history_reduce(
            batch.history_post_hashes, recsys_embeddings.history_post_embeddings,
            recsys_embeddings.history_author_embeddings, hist_ps_emb, hist_act_emb,
            self.config.hash_config.num_item_hashes, self.config.hash_config.num_author_hashes,
            self.history_proj.weight
        )
        hist_emb = self.history_proj(hist_emb_raw)

        cand_emb_raw, cand_mask = block_candidate_reduce(
            batch.candidate_post_hashes, recsys_embeddings.candidate_post_embeddings,
            recsys_embeddings.candidate_author_embeddings, cand_ps_emb,
            self.config.hash_config.num_item_hashes, self.config.hash_config.num_author_hashes,
            self.candidate_proj.weight
        )
        cand_emb = self.candidate_proj(cand_emb_raw)

        embeddings = torch.cat([user_emb, hist_emb, cand_emb], dim=1)
        padding_mask = torch.cat([user_mask, hist_mask, cand_mask], dim=1)
        candidate_start_offset = user_mask.shape[1] + hist_mask.shape[1]

        return embeddings.to(self.config.fprop_dtype), padding_mask, candidate_start_offset

    def forward(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings
    ) -> RecsysModelOutput:
        """Forward pass: output action logits for candidates."""
        embeddings, padding_mask, cand_offset = self.build_inputs(batch, recsys_embeddings)

        transformer_out = self.transformer(
            embeddings,
            candidate_start_offset=cand_offset,
            padding_mask=padding_mask
        )

        out_embeddings = self.final_norm(transformer_out)
        candidate_embeddings = out_embeddings[:, cand_offset:, :]

        logits = torch.matmul(
            candidate_embeddings.to(self.unembeddings.dtype),
            self.unembeddings.t()
        )

        return RecsysModelOutput(logits=logits.to(self.config.fprop_dtype))
