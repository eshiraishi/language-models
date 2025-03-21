from typing import Generator, Literal, Self

import torch

from language_models.transformer.config import TokenizerConfig


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
