from dataclasses import dataclass
from typing import Self


@dataclass
class TransformerBlockConfig:
    embed_dim: int = 512
    n_heads: int = 8
    hidden_dim: int = 2048


@dataclass
class PositionalEncoderConfig:
    embed_dim: int
    theta: int = 10000


@dataclass
class TokenizerConfig:
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __post_init__(self):
        self.special_tokens = {
            self.pad_token,
            self.bos_token,
            self.eos_token,
        }


@dataclass
class DecoderConfig:
    block: TransformerBlockConfig
    n_blocks: int


@dataclass
class EncoderConfig:
    block: TransformerBlockConfig
    n_blocks: int


@dataclass
class EmbedderConfig:
    n_tokens: int
    embed_dim: int


@dataclass
class InputProcessorConfig:
    embedder: EmbedderConfig
    positional_encoder: PositionalEncoderConfig
    pad_token_int: int


@dataclass
class OutputProcessorConfig:
    in_features: int
    out_features: int


class TransformerConfig:
    def __init__(
        self: Self,
        n_tokens: int,
        pad_token_int: int = 0,
        n_blocks: int = 6,
        block: TransformerBlockConfig | None = None,
        input_processor: InputProcessorConfig | None = None,
        encoder: EncoderConfig | None = None,
        decoder: DecoderConfig | None = None,
        output_processor: OutputProcessorConfig | None = None,
    ):
        self.n_tokens = n_tokens
        self.pad_token_int = pad_token_int
        self.n_blocks = n_blocks

        self.block = block or TransformerBlockConfig()

        self.input_processor = input_processor or InputProcessorConfig(
            pad_token_int=self.pad_token_int,
            embedder=EmbedderConfig(
                n_tokens=self.n_tokens,
                embed_dim=self.block.embed_dim,
            ),
            positional_encoder=PositionalEncoderConfig(
                embed_dim=self.block.embed_dim,
            ),
        )

        self.encoder = encoder or EncoderConfig(
            block=self.block,
            n_blocks=self.n_blocks,
        )

        self.decoder = decoder or DecoderConfig(
            block=self.block,
            n_blocks=self.n_blocks,
        )

        self.output_processor = output_processor or OutputProcessorConfig(
            in_features=self.block.embed_dim,
            out_features=self.n_tokens - 2,
        )
