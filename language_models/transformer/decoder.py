from typing import Self

import torch
from torch import nn

from language_models.transformer.attention import (
    MultiheadAttention,
    attn_mask_like,
)
from language_models.transformer.block import TransformerBlock, TransformerBlockConfig
from language_models.transformer.config import DecoderConfig


class DecoderBlock(nn.Module):
    def __init__(self: Self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.mha = MultiheadAttention(
            embed_dim=self.config.embed_dim,
            n_heads=self.config.n_heads,
        )
        self.transformer_block = TransformerBlock(self.config)

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        batches, tokens, _ = keys.size()
        mask = attn_mask_like((batches, self.config.n_heads, tokens, tokens))

        outputs = self.mha(
            queries=queries,
            keys=keys,
            values=values,
            mask=mask,
        )

        outputs = self.transformer_block(
            queries=encoder_outputs,
            keys=encoder_outputs,
            values=outputs,
        )

        return outputs


class Decoder(nn.Module):
    def __init__(self: Self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config.block) for _ in range(self.config.n_blocks)
        )

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        block, *blocks = self.blocks

        outputs = block(
            queries=queries,
            keys=keys,
            values=values,
            encoder_outputs=encoder_outputs,
        )

        for block in blocks:
            outputs = block(
                queries=outputs,
                keys=outputs,
                values=outputs,
                encoder_outputs=encoder_outputs,
            )

        return outputs
