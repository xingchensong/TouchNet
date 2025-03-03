from dataclasses import asdict, dataclass, field, fields


@dataclass
class TokenizerConfig:
    """Configuration object for tokenizer"""

    _argument_group_name = "tokenizer"

    tokenizer_model: str = field(
        default=None,
        metadata={
            "help": ("path of tokenizer."),
        },
    )
    tokenizer_type: str = field(
        default="HuggingFaceTokenizer",
        metadata={
            "help": ("type of tokenizer."),
            "choices": [
                "HuggingFaceTokenizer",
            ],
        },
    )
