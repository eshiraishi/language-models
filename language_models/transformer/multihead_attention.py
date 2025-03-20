from typing import Self

import torch
import torch.nn.functional as F
from torch import nn


def attn_mask_like(size: int | tuple[int]) -> torch.Tensor:
    mask = torch.ones(size)
    mask = mask.triu(diagonal=1)
    mask = mask.bool()
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self: Self, embed_dim: int, n_heads: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads

        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def split_embeddings(
        self: Self,
        inputs: torch.Tensor,
        batches: int,
        tokens: int,
    ) -> torch.Tensor:
        split_inputs = inputs.view(batches, tokens, self.n_heads, self.head_dim)
        head_sorted_inputs = split_inputs.transpose(1, 2)
        return head_sorted_inputs

    def join_embeddings(
        self: Self,
        inputs: torch.Tensor,
        batches: int,
        tokens: int,
    ) -> torch.Tensor:
        token_sorted_inputs = inputs.transpose(1, 2)
        token_sorted_inputs = token_sorted_inputs.contiguous()
        joined_inputs = token_sorted_inputs.view(batches, tokens, self.embed_dim)
        return joined_inputs

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batches, tokens, _ = queries.size()

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = self.split_embeddings(queries, batches, tokens)
        keys = self.split_embeddings(keys, batches, tokens)
        values = self.split_embeddings(values, batches, tokens)

        keys = keys.transpose(2, 3)

        scores = queries @ keys / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=3)

        attn_outputs = weights @ values
        joined_outputs = self.join_embeddings(attn_outputs, batches, tokens)
        projected_outputs = self.output_projection(joined_outputs)

        return projected_outputs
