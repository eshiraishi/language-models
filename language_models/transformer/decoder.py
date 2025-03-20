from typing import Self

import torch
from torch import nn

from language_models.transformer.block import TransformerBlock, TransformerBlockConfig
from language_models.transformer.multihead_attention import (
    MultiheadAttention,
    attn_mask_like,
)


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
    def __init__(
        self: Self,
        n_blocks: int,
        config: TransformerBlockConfig,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.config = config
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config) for _ in range(self.n_blocks)
        )

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        block, *blocks = self.blocks
        block_outputs = block(
            queries=queries,
            keys=keys,
            values=values,
            encoder_outputs=encoder_outputs,
        )

        for block in blocks:
            block_outputs = block(
                queries=block_outputs,
                keys=block_outputs,
                values=block_outputs,
                encoder_outputs=encoder_outputs,
            )
        return block_outputs
