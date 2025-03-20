from typing import Self

import torch
from torch import nn

from language_models.transformer.embedding import EmbeddingConfig


class OutputProcessor(nn.Module):
    def __init__(self: Self, config: EmbeddingConfig) -> None:
        super().__init__()

        self.config = config

        self.linear = nn.Linear(
            in_features=self.config.embed_dim,
            out_features=self.config.out_tokens,
        )

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        linear_outputs = self.linear(inputs)
        batches, tokens, _ = linear_outputs.size()

        probs = linear_outputs.softmax(dim=2)
        probs = probs.view(batches * tokens, self.config.out_tokens)

        predictions = probs.argmax(dim=1)
        predictions = predictions + 2
        predictions = predictions.view(batches, tokens)

        return predictions
