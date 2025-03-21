from typing import Self

import torch
from torch import nn

from language_models.transformer.config import InputProcessorConfig
from language_models.transformer.positional_encoder import PositionalEncoder


class InputProcessor(nn.Module):
    def __init__(self: Self, config: InputProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.embedder = nn.Embedding(
            num_embeddings=self.config.embedder.n_tokens,
            embedding_dim=self.config.embedder.embed_dim,
            padding_idx=self.config.pad_token_int,
        )
        self.positional_encoder = PositionalEncoder(self.config.positional_encoder)

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.embedder(inputs)
        outputs = self.positional_encoder(outputs)
        return outputs
