from dataclasses import dataclass


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
