from typing import Self

import torch
from torch import nn

from language_models.transformer.block import TransformerBlock, TransformerBlockConfig


class Encoder(nn.Module):
    def __init__(self: Self, n_blocks: int, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.config = config
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.n_blocks)
        )

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        block, *blocks = self.blocks
        block_outputs = block(
            queries=queries,
            keys=keys,
            values=values,
        )

        for block in blocks:
            block_outputs = block(
                queries=block_outputs,
                keys=block_outputs,
                values=block_outputs,
            )

        return block_outputs
