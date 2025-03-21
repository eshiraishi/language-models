from typing import Self

import torch
from torch import nn

from language_models.transformer.model.config import TransformerConfig
from language_models.transformer.model.decoder import Decoder
from language_models.transformer.model.encoder import Encoder
from language_models.transformer.model.input_processor import InputProcessor
from language_models.transformer.model.output_processor import OutputProcessor


class Transformer(nn.Module):
    def __init__(self: Self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_processor = InputProcessor(self.config.input_processor)
        self.encoder = Encoder(self.config.encoder)
        self.decoder = Decoder(self.config.decoder)
        self.output_processor = OutputProcessor(self.config.output_processor)

    def forward(
        self: Self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        encoder_inputs = self.input_processor(encoder_inputs)
        decoder_inputs = self.input_processor(decoder_inputs)

        encoder_outputs = self.encoder(
            queries=encoder_inputs,
            keys=encoder_inputs,
            values=encoder_inputs,
        )

        decoder_outputs = self.decoder(
            queries=decoder_inputs,
            keys=decoder_inputs,
            values=decoder_inputs,
            encoder_outputs=encoder_outputs,
        )

        outputs = self.output_processor(decoder_outputs)

        return outputs
