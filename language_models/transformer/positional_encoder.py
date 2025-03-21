from typing import Self

import torch
from torch import nn

from language_models.transformer.config import PositionalEncoderConfig


class PositionalEncoder(nn.Module):
    def __init__(self: Self, config: PositionalEncoderConfig) -> None:
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batches, tokens, _ = inputs.size()

        indexes = torch.arange(self.config.embed_dim, dtype=torch.float)

        positions = torch.arange(tokens, dtype=torch.float)
        positions = positions.view(tokens, 1)

        i = torch.arange(self.config.embed_dim // 2)
        i = i.float()
        i = i.repeat_interleave(2)

        cos_indexes = indexes % 2
        cos_indexes = cos_indexes.bool()
        cos_indexes = cos_indexes.expand((tokens, self.config.embed_dim))

        sin_indexes = ~cos_indexes

        encodings = positions / (self.config.theta ** (2 * i / self.config.embed_dim))

        encodings[sin_indexes] = encodings[sin_indexes].sin()
        encodings[cos_indexes] = encodings[cos_indexes].cos()

        encodings = encodings.expand((batches, tokens, self.config.embed_dim))

        return inputs + encodings
