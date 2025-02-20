# Copyright (c) 2023, NVIDIA CORPORATION (Megatron-LM teams). All rights reserved.
#               2025, WeNet Community. Xingchen Song(sxc19@tsinghua.org.cn)

"""Touch tokenizers."""

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import numpy
import transformers


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
    def tokenize(self, text: str) -> numpy.ndarray:
        """Convert text to embedding ids

        Args:
            text (str): The text to convert

        Returns:
            numpy.ndarray: The converted embedding ids
        """
        pass

    def detokenize(self, ids: numpy.ndarray) -> str:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray): The ids to convert

        Returns:
            str: The converted text

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
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(pretrained_model_name_or_path, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
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

    def tokenize(self, text, **kwargs):
        self._build_hugging_face()
        return self._tokenizer(text, **kwargs).input_ids

    def detokenize(self, token_ids, **kwargs):
        self._build_hugging_face()
        return self._tokenizer.decode(token_ids, **kwargs)

    @property
    def eod(self):
        self._build_hugging_face()
        return self._tokenizer.eos_token_id
