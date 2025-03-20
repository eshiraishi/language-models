import random
from string import printable

import numpy as np
import torch

from language_models.transformer.block import TransformerBlockConfig
from language_models.transformer.embedding import EmbeddingConfig
from language_models.transformer.executor import TransformerExecutor
from language_models.transformer.model import Transformer
from language_models.transformer.positional_encoder import PositionalEncoderConfig
from language_models.transformer.tokenizer import Tokenizer, TokenizerConfig
from language_models.utils import litstr

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
