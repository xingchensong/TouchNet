# Copyright (c) 2023, NVIDIA CORPORATION (Megatron-LM teams). All rights reserved.
#               2025, WeNet Community. Xingchen Song(sxc19@tsinghua.org.cn)

"""Touch tokenizers."""

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Union

import numpy
import torch
import transformers

from touchnet.tokenizer import TokenizerConfig


class BaseTokenizer(ABC):
    """Abstract class for tokenizer

    Absent a config or class-specific tracking of which objects are uniquely identifying, we must
    include all key word arguments as unique identifiers

    Args:
        tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

        tokenizer_options (Dict[str, Any]): All tokenizer options
    """

    def __init__(self, *tokenizer_paths: str, **tokenizer_options: Any):

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["tokenizer_path"] = list(tokenizer_paths)
        for option in tokenizer_options:
            self.unique_identifiers[option] = str(tokenizer_options[option])

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)

        super().__init__()

    @abstractmethod
    def tokenize(self, inputs: Any) -> Union[numpy.ndarray, torch.Tensor]:
        """Convert text to embedding ids

        Args:
            inputs (Any): The text/audio/video to convert

        Returns:
            numpy.ndarray/torch.Tensor: The converted embedding ids
        """
        pass

    def detokenize(self, ids: Union[numpy.ndarray, torch.Tensor]) -> Any:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray/torch.Tensor): The ids to convert

        Returns:
            Any: The converted text/audio/video

        Raises:
            NotImplementedError: Non-abstract, optional method
        """
        raise NotImplementedError(
            "{} has no method 'detokenize'".format(type(self).__name__)
        )

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token"""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token"""
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """The vocabulary size"""
        pass

    @property
    def cls(self):
        """The CLS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'cls'".format(type(self).__name__)
        )

    @property
    def sep(self):
        """The SEP token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'sep'".format(type(self).__name__)
        )

    @property
    def pad(self):
        """The PAD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'pad'".format(type(self).__name__)
        )

    @property
    def eod(self):
        """The EOD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'eod'".format(type(self).__name__)
        )

    @property
    def bos(self):
        """The BOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'bos'".format(type(self).__name__)
        )

    @property
    def eos(self):
        """The EOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'eos'".format(type(self).__name__)
        )

    @property
    def mask(self):
        """The MASK token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError(
            "{} has no attribute 'mask'".format(type(self).__name__)
        )


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, config: TokenizerConfig, **kwargs):
        super().__init__(config.tokenizer_model, **kwargs)
        self.pretrained_model_name_or_path = config.tokenizer_model
        self.kwargs = kwargs
        self._tokenizer = None
        self._vocab = None
        self._inv_vocab = None

    def _build_hugging_face(self):
        if self._tokenizer is None:
            # TODO(bnorick): download tokenizer once to lustre and use force offline to make sure all tasks read it from there
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                **self.kwargs
            )
            self._vocab = self._tokenizer.get_vocab()
            self._inv_vocab = {
                token_id: token for token, token_id in self._vocab.items()
            }

    @property
    def vocab_size(self):
        self._build_hugging_face()
        return len(self._tokenizer)

    @property
    def vocab(self):
        self._build_hugging_face()
        """Dictionary from vocab text token to id token."""
        return self._vocab

    @property
    def inv_vocab(self):
        self._build_hugging_face()
        """Dictionary from vocab id token to text token."""
        return self._inv_vocab

    @property
    def decoder(self):
        self._build_hugging_face()
        return self._inv_vocab

    def tokenize(self, inputs, **kwargs):
        self._build_hugging_face()
        return self._tokenizer(inputs, **kwargs).input_ids

    def detokenize(self, token_ids, **kwargs):
        self._build_hugging_face()
        return self._tokenizer.decode(token_ids, **kwargs)

    @property
    def eos(self):
        self._build_hugging_face()
        return self._tokenizer.eos_token_id

    @property
    def bos(self):
        self._build_hugging_face()
        return self._tokenizer.bos_token_id

    @property
    def pad(self):
        self._build_hugging_face()
        return self._tokenizer.pad_token_id


class BestRQTokenizer(BaseTokenizer):
    def __init__(self, config: TokenizerConfig, **kwargs):
        super().__init__(f"BestRQ-{config.tokenizer_bestrq_init_method}-init", **kwargs)
        self.kwargs = kwargs
        self.config = config
        self._quantizer = None
        self._codebook = None

    def _build_quantizer_and_codebook(self):
        if self._quantizer is None:
            self._quantizer = torch.nn.parameter.Parameter(
                torch.empty(self.config.tokenizer_bestrq_input_size,
                            self.config.tokenizer_bestrq_emb_size),
                requires_grad=False,
            )
            self._codebook = torch.nn.parameter.Parameter(
                torch.empty(self.config.tokenizer_bestrq_emb_size,
                            self.config.tokenizer_bestrq_vocab_size),
                requires_grad=False,
            )
            if self.config.tokenizer_bestrq_init_method == "default":
                # default initialization follows best-rq https://arxiv.org/pdf/2202.01855
                torch.nn.init.xavier_uniform_(self._quantizer)
                torch.nn.init.normal_(self._codebook)
            else:
                raise NotImplementedError(
                    f"Initialization method {self.config.tokenizer_bestrq_init_method} is not implemented."
                )
            self._codebook /= self._codebook.norm(dim=0, p=2, keepdim=True) + 1e-8
            self._codebook_magnitude = torch.sum(self._codebook**2, 0, keepdim=True)
            self._id2emb = self._codebook.transpose(0, 1)

    @property
    def vocab_size(self):
        self._build_quantizer_and_codebook()
        return self._codebook.size(1)

    @property
    def vocab(self):
        """Dictionary from emb to id token."""
        self._build_quantizer_and_codebook()
        return None

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to emb token."""
        self._build_quantizer_and_codebook()
        return self._id2emb

    @property
    def decoder(self):
        self._build_quantizer_and_codebook()
        return self._id2emb

    def tokenize(self, inputs, **kwargs):
        self._build_quantizer_and_codebook()
        # get nearest embedding
        xs = torch.matmul(inputs, self._quantizer.to(inputs.device))
        xs = xs / (xs.norm(dim=-1, p=2, keepdim=True) + 1e-8)  # [T, D]
        distance = (
            # [T, D] --> [T, 1]
            torch.sum(xs**2, -1, keepdim=True) -
            # [T, D] @ [D, Vocab] --> [T, Vocab]
            2 * torch.matmul(xs, self._codebook.to(xs.device)) +
            # [1, Vocab]
            self._codebook_magnitude
        )
        codes = torch.argmin(distance, dim=-1)  # [T,]
        return codes

    def detokenize(self, token_ids, **kwargs):
        self._build_quantizer_and_codebook()
        return torch.index_select(self._id2emb, dim=0, index=token_ids)

    @property
    def eos(self):
        self._build_quantizer_and_codebook()
        return -1

    @property
    def bos(self):
        self._build_quantizer_and_codebook()
        return -1

    @property
    def pad(self):
        self._build_quantizer_and_codebook()
        return -1


def build_tokenizer(args: TokenizerConfig, **kwargs):
    """Initialize tokenizer."""

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == "HuggingFaceTokenizer":
        tokenizer = HuggingFaceTokenizer(args, **kwargs)
    elif args.tokenizer_type == "BestRQTokenizer":
        tokenizer = BestRQTokenizer(args, **kwargs)
    else:
        raise NotImplementedError(
            "{} tokenizer is not " "implemented.".format(args.tokenizer_type)
        )

    return tokenizer
