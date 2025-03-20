from dataclasses import dataclass
from typing import Self


@dataclass
class EmbeddingConfig:
    pad_token_int: int
    in_tokens: int
    out_tokens: int | None = None
    embed_dim: int = 512

    def __post_init__(self: Self) -> None:
        self.out_tokens = self.out_tokens or (self.in_tokens - 2)
