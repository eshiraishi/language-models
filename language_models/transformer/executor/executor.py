from typing import Self

import torch
from torch import nn

from language_models.transformer.model import Transformer
from language_models.transformer.tokenizer import Tokenizer


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
