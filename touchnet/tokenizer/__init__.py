from dataclasses import asdict, dataclass, field, fields

from .tokenizer import BaseTokenizer, HuggingFaceTokenizer


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


# TODO(xcsong): padding vocab
# def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
#     """Pad vocab size so it is divisible by model parallel size and
#     still having GPU friendly size."""
#
#     after = orig_vocab_size
#     multiple = args.make_vocab_size_divisible_by * \
#         args.tensor_model_parallel_size
#     after = int(math.ceil(after / multiple) * multiple)
#     if args.rank == 0 and logging_enabled:
#         print(' > padded vocab (size: {}) with {} dummy tokens '
#               '(new size: {})'.format(
#                   orig_vocab_size, after - orig_vocab_size, after), flush=True)
#     return after


def build_tokenizer(args: TokenizerConfig, **kwargs):
    """Initialize tokenizer."""

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == "HuggingFaceTokenizer":
        tokenizer = HuggingFaceTokenizer(args.tokenizer_model, **kwargs)
    else:
        raise NotImplementedError(
            "{} tokenizer is not " "implemented.".format(args.tokenizer_type)
        )

    # TODO(xcsong): padding vocab
    # # Add vocab size (if not already set from a checkpoint).
    # if getattr(args, "padded_vocab_size", None) is None:
    #     args.padded_vocab_size = _vocab_size_with_padding(
    #         tokenizer.vocab_size, args
    #     )

    return tokenizer
