from typing import Self

import torch
from torch import nn

from language_models.transformer.positional_encoder import PositionalEncoder


class InputProcessor(nn.Module):
    def __init__(
        self: Self,
        embedding: nn.Embedding,
        positional_encoder: PositionalEncoder,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.positional_encoder = positional_encoder

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(inputs)
        encoded_embeddings = self.positional_encoder(embeddings)
        return encoded_embeddings
