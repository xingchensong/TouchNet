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
                "BestRQTokenizer",
            ],
        },
    )
    tokenizer_bestrq_vocab_size: int = field(
        default=8192,
        metadata={
            "help": ("vocab_size of best-rq random codebook."),
        },
    )
    tokenizer_bestrq_input_size: int = field(
        default=560,
        metadata={
            "help": ("input_size of best-rq random quantizer."),
        },
    )
    tokenizer_bestrq_emb_size: int = field(
        default=16,
        metadata={
            "help": ("output_size of best-rq random quantizer."),
        },
    )
    tokenizer_bestrq_init_method: str = field(
        default="default",
        metadata={
            "help": (
                "Initialization method of best-rq random quantizer and codebook.\n",
                "    default: xavier_uniform for quantizer and normal for codebook.\n",
            ),
            "choices": [
                "default",
            ],
        },
    )
