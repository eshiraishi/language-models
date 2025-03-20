import random
from dataclasses import dataclass
from string import printable
from typing import Generator, Literal, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PositionalEncoderConfig:
    embed_dim: int = 512
    theta: int = 10000


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


class Tokenizer:
    def __init__(
        self: Self,
        vocab: set[str],
        config: TokenizerConfig,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = config

        self.string_to_int = {
            self.config.pad_token: 0,
            self.config.bos_token: 1,
            self.config.eos_token: 2,
        }
        self.string_to_int.update(
            {char: (index + 3) for index, char in enumerate(self.vocab)}
        )

        self.pad_token_int = self.string_to_int[self.config.pad_token]
        self.bos_token_int = self.string_to_int[self.config.bos_token]
        self.eos_token_int = self.string_to_int[self.config.eos_token]

        self.vocab_size = len(self.string_to_int)

        self.int_to_string = {
            0: self.config.pad_token,
            1: self.config.bos_token,
            2: self.config.eos_token,
        }
        self.int_to_string.update(
            {(index + 3): char for index, char in enumerate(self.vocab)}
        )

    def split(self: Self, chars: str) -> Generator[str, None, None]:
        index = 0
        while index < len(chars):
            token = chars[index]
            for special_token in self.config.special_tokens:
                if chars.startswith(special_token, index):
                    token = special_token
                    break

            yield token
            index += len(token)

    def encode(self: Self, inputs: str) -> torch.Tensor:
        tokens = [
            self.string_to_int[token_string] for token_string in self.split(inputs)
        ]
        tokens = torch.tensor(tokens)
        return tokens

    def pad(
        self: Self,
        tokens: torch.Tensor,
        amount: int,
        fill_value: int,
        side: Literal["left", "right"],
    ) -> torch.Tensor:
        if amount == 0:
            return tokens

        padding = torch.full(
            size=(amount,),
            fill_value=fill_value,
        )
        padded_tensor = (tokens, padding) if side == "right" else (padding, tokens)
        padded_tensor = torch.cat(padded_tensor)

        return padded_tensor

    def batch_encode(
        self: Self,
        inputs: list[str],
        side: Literal["left", "right"] = "right",
        strategy: Literal["max", "fixed"] = "max",
        amount: int | None = None,
        truncate: bool = False,
    ) -> torch.Tensor:
        token_lists = [self.encode(text) for text in inputs]
        lengths = [len(tokens) for tokens in token_lists]

        max_length = max(lengths)
        max_length = max_length if amount is None else max(max_length, amount)

        padded_tokens = [
            self.pad(
                tokens=tokens,
                amount=max_length - length,
                fill_value=self.pad_token_int,
                side=side,
            )
            for tokens, length in zip(token_lists, lengths)
        ]
        padded_tokens = torch.stack(padded_tokens)

        if strategy == "fixed" and truncate:
            padded_tokens = padded_tokens[:, :amount]

        return padded_tokens

    def decode(self: Self, inputs: torch.Tensor) -> str:
        outputs = [self.int_to_string[token] for token in inputs.tolist()]
        outputs = "".join(outputs)
        return outputs

    def batch_decode(self: Self, inputs: torch.Tensor) -> list[str]:
        outputs = [self.decode(tokens) for tokens in inputs]
        return outputs

    def batch_shift(self: Self, inputs: torch.Tensor, output_tokens: str) -> str:
        outputs = torch.cat((inputs[:, 1:], output_tokens), dim=1)
        return outputs

    def add_special_tokens(self: Self, chars: str) -> str:
        return self.config.bos_token + chars + self.config.eos_token


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


def attn_mask_like(size: int | tuple[int]) -> torch.Tensor:
    mask = torch.ones(size)
    mask = mask.triu(diagonal=1)
    mask = mask.bool()
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self: Self, embed_dim: int, n_heads: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads

        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def split_embeddings(
        self: Self,
        inputs: torch.Tensor,
        batches: int,
        tokens: int,
    ) -> torch.Tensor:
        split_inputs = inputs.view(batches, tokens, self.n_heads, self.head_dim)
        head_sorted_inputs = split_inputs.transpose(1, 2)
        return head_sorted_inputs

    def join_embeddings(
        self: Self,
        inputs: torch.Tensor,
        batches: int,
        tokens: int,
    ) -> torch.Tensor:
        token_sorted_inputs = inputs.transpose(1, 2)
        token_sorted_inputs = token_sorted_inputs.contiguous()
        joined_inputs = token_sorted_inputs.view(batches, tokens, self.embed_dim)
        return joined_inputs

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batches, tokens, _ = queries.size()

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = self.split_embeddings(queries, batches, tokens)
        keys = self.split_embeddings(keys, batches, tokens)
        values = self.split_embeddings(values, batches, tokens)

        keys = keys.transpose(2, 3)

        scores = queries @ keys / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=3)

        attn_outputs = weights @ values
        joined_outputs = self.join_embeddings(attn_outputs, batches, tokens)
        projected_outputs = self.output_projection(joined_outputs)

        return projected_outputs


@dataclass
class TransformerBlockConfig:
    embed_dim: int = 512
    n_heads: int = 8
    hidden_dim: int = 2048


class TransformerBlock(nn.Module):
    def __init__(self: Self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.config = config

        self.mha = MultiheadAttention(
            embed_dim=self.config.embed_dim,
            n_heads=self.config.n_heads,
        )
        self.mha_layernorm = nn.LayerNorm(normalized_shape=self.config.embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.embed_dim),
        )
        self.ff_layernorm = nn.LayerNorm(self.config.embed_dim)

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mha_res = values

        mha_outputs = self.mha(queries, keys, values, mask)
        mha_res_outputs = mha_outputs + mha_res
        norm_mha_outputs = self.mha_layernorm(mha_res_outputs)

        ff_res = norm_mha_outputs
        ff_outputs = self.ff(norm_mha_outputs)
        norm_ff_outputs = self.ff_layernorm(ff_outputs + ff_res)

        return norm_ff_outputs


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


class DecoderBlock(nn.Module):
    def __init__(self: Self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.mha = MultiheadAttention(
            embed_dim=self.config.embed_dim,
            n_heads=self.config.n_heads,
        )
        self.transformer_block = TransformerBlock(self.config)

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        batches, tokens, _ = keys.size()
        mask = attn_mask_like((batches, self.config.n_heads, tokens, tokens))

        outputs = self.mha(
            queries=queries,
            keys=keys,
            values=values,
            mask=mask,
        )

        outputs = self.transformer_block(
            queries=encoder_outputs,
            keys=encoder_outputs,
            values=outputs,
        )

        return outputs


class Decoder(nn.Module):
    def __init__(
        self: Self,
        n_blocks: int,
        config: TransformerBlockConfig,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.config = config
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config) for _ in range(self.n_blocks)
        )

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        block, *blocks = self.blocks
        block_outputs = block(
            queries=queries,
            keys=keys,
            values=values,
            encoder_outputs=encoder_outputs,
        )

        for block in blocks:
            block_outputs = block(
                queries=block_outputs,
                keys=block_outputs,
                values=block_outputs,
                encoder_outputs=encoder_outputs,
            )
        return block_outputs


@dataclass
class EmbeddingConfig:
    pad_token_int: int
    in_tokens: int
    out_tokens: int | None = None
    embed_dim: int = 512

    def __post_init__(self: Self) -> None:
        self.out_tokens = self.out_tokens or (self.in_tokens - 2)


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


class TransformerExecutor(nn.Module):
    def __init__(self: Self, tokenizer: Tokenizer, transformer: Transformer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = transformer

    def get_output_tokens(
        self: Self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        token_lengths = mask != self.tokenizer.pad_token_int
        token_lengths = token_lengths.sum(dim=1)

        indexes = token_lengths - 1
        batch_size, _ = tokens.size()
        rows = torch.arange(batch_size)

        output_tokens = tokens[rows, indexes]
        output_tokens = output_tokens.unsqueeze(1)

        return output_tokens

    def get_first_output_tokens(self: Self, length: int) -> torch.Tensor:
        output_tokens = torch.full((length, 1), self.tokenizer.bos_token_int)
        return output_tokens

    def forward(
        self: Self,
        inputs: list[str],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        length = len(inputs)
        inputs = [self.tokenizer.add_special_tokens(chars) for chars in inputs]
        tokens = self.tokenizer.batch_encode(inputs)

        output_tokens = self.get_first_output_tokens(length)
        encoder_inputs = tokens
        decoder_inputs = self.tokenizer.batch_shift(encoder_inputs, output_tokens)
        new_tokens = [output for output in output_tokens]
        eos_indexes = [None for _ in inputs]
        index = 1

        while any(eos_index is None for eos_index in eos_indexes) and (
            max_new_tokens is None or index < max_new_tokens
        ):
            decoder_outputs = self.transformer(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
            )

            output_tokens = self.get_output_tokens(
                tokens=decoder_outputs,
                mask=tokens,
            )

            shifted_decoder_inputs = self.tokenizer.batch_shift(
                inputs=decoder_inputs,
                output_tokens=output_tokens,
            )

            encoder_inputs, decoder_inputs = decoder_inputs, shifted_decoder_inputs

            new_tokens = [
                new_tokens_tensor
                if eos_index is not None
                else torch.cat((new_tokens_tensor, output_token))
                for new_tokens_tensor, output_token, eos_index in zip(
                    new_tokens,
                    output_tokens,
                    eos_indexes,
                )
            ]

            eos_indexes = [
                eos_index
                if eos_index is not None
                else index
                if output_token == self.tokenizer.eos_token_int
                else None
                for eos_index, output_token in zip(eos_indexes, output_tokens)
            ]

            index += 1

        outputs = self.tokenizer.batch_decode(new_tokens)

        return outputs


def litstr(chars):
    return "".join(repr(char).replace("'", "") for char in chars)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=500)

    vocab = set(printable)

    tokenizer_config = TokenizerConfig()
    tokenizer = Tokenizer(vocab, tokenizer_config)

    block_config = TransformerBlockConfig()
    positional_encoder_config = PositionalEncoderConfig()
    embedding_config = EmbeddingConfig(
        pad_token_int=tokenizer.pad_token_int,
        in_tokens=tokenizer.vocab_size,
    )

    i = 0
    while True:
        torch.manual_seed(i)
        random.seed(i)
        np.random.seed(i)

        transformer = Transformer(
            encoder_config=block_config,
            decoder_config=block_config,
            embedding_config=embedding_config,
            positional_encoder_config=positional_encoder_config,
        )

        executor = TransformerExecutor(tokenizer, transformer)

        inputs = [
            "Why did the dinosaurs disappear?",
            "What is a large language model?",
            "Where is the closest hospital?",
            "Who lives in The White House?",
        ]

        executor(inputs, max_new_tokens=16)

        print()
        print(f"{i}:")
        lines = [
            f"{litstr(input_string):<35} | {litstr(output):<35}"
            for input_string, output in zip(
                inputs,
                executor(
                    inputs,
                    max_new_tokens=16,
                ),
            )
        ]

        print(*lines, sep="\n")
        i += 1
