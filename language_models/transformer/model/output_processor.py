from typing import Self

import torch
from torch import nn

from language_models.transformer.model.config import EmbedderConfig


class OutputProcessor(nn.Module):
    def __init__(self: Self, config: EmbedderConfig) -> None:
        super().__init__()

        self.config = config

        self.linear = nn.Linear(
            in_features=self.config.in_features,
            out_features=self.config.out_features,
        )

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)

        batch_size, n_tokens, _ = outputs.size()

        outputs = outputs.softmax(dim=2)
        outputs = outputs.view(batch_size * n_tokens, self.config.out_features)
        ouputs = outputs.argmax(dim=1)
        ouputs = ouputs + 2
        ouputs = ouputs.view(batch_size, n_tokens)

        return ouputs
