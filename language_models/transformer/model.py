from typing import Self

import torch
from torch import nn

from language_models.transformer.block import TransformerBlockConfig
from language_models.transformer.decoder import Decoder
from language_models.transformer.embedding import EmbeddingConfig
from language_models.transformer.encoder import Encoder
from language_models.transformer.input_processor import InputProcessor
from language_models.transformer.output_processor import OutputProcessor
from language_models.transformer.positional_encoder import (
    PositionalEncoder,
    PositionalEncoderConfig,
)


# TODO: Create encoderconfig, decoderconfig and transformerconfig
class Transformer(nn.Module):
    def __init__(
        self: Self,
        encoder_config: TransformerBlockConfig,
        decoder_config: TransformerBlockConfig,
        embedding_config: EmbeddingConfig,
        positional_encoder_config: PositionalEncoderConfig,
        n_encoder_blocks: int = 6,
        n_decoder_blocks: int = 6,
    ) -> None:
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.embedding_config = embedding_config
        self.positional_encoder_config = positional_encoder_config
        self.n_encoder_blocks = n_encoder_blocks
        self.n_decoder_blocks = n_decoder_blocks

        self.embedding = nn.Embedding(
            num_embeddings=self.embedding_config.in_tokens,
            embedding_dim=self.embedding_config.embed_dim,
            padding_idx=self.embedding_config.pad_token_int,
        )

        self.positional_encoder = PositionalEncoder(self.positional_encoder_config)

        self.input_processor = InputProcessor(
            embedding=self.embedding,
            positional_encoder=self.positional_encoder,
        )

        self.encoder = Encoder(
            n_blocks=self.n_encoder_blocks,
            config=self.encoder_config,
        )

        self.decoder = Decoder(
            n_blocks=self.n_decoder_blocks,
            config=self.decoder_config,
        )

        self.output_processor = OutputProcessor(config=self.embedding_config)

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
