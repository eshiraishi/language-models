from typing import Self

import torch
from torch import nn

from language_models.transformer.model.attention import MultiheadAttention
from language_models.transformer.model.config import TransformerBlockConfig


class TransformerBlock(nn.Module):
    def __init__(self: Self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.config = config

        self.mha = MultiheadAttention(
            embed_dim=self.config.embed_dim,
            n_heads=self.config.n_heads,
        )
        self.mha_layernorm = nn.LayerNorm(normalized_shape=self.config.embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.embed_dim),
        )
        self.ff_layernorm = nn.LayerNorm(self.config.embed_dim)

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mha_res = values

        mha_outputs = self.mha(queries, keys, values, mask)
        mha_res_outputs = mha_outputs + mha_res
        norm_mha_outputs = self.mha_layernorm(mha_res_outputs)

        ff_res = norm_mha_outputs
        ff_outputs = self.ff(norm_mha_outputs)
        norm_ff_outputs = self.ff_layernorm(ff_outputs + ff_res)

        return norm_ff_outputs
