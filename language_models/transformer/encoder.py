from typing import Self

import torch
from torch import nn

from language_models.transformer.block import TransformerBlock
from language_models.transformer.config import EncoderConfig


class Encoder(nn.Module):
    def __init__(self: Self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config.block) for _ in range(self.config.n_blocks)
        )

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        block, *blocks = self.blocks

        outputs = block(queries=queries, keys=keys, values=values)

        for block in blocks:
            outputs = block(queries=outputs, keys=outputs, values=outputs)

        return outputs
