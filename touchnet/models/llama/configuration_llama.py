# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaForASRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaForASR`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B with a linear projector.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        input_size (`int`, defaults to 4096):
            Dimension of the input representations (mel or fbank).
    """

    model_type = "llama4asr"
    sub_configs = {
        "llm_config": LlamaConfig,
    }

    def __init__(
        self,
        llm_config: Optional[LlamaConfig] = None,
        input_size: int = 4096,
        **kwargs,
    ):
        if llm_config is None:
            llm_config = LlamaConfig()
        elif isinstance(llm_config, dict):
            llm_config = LlamaConfig(**llm_config)
        else:
            raise RuntimeError("llm_config not provided.")

        self.llm_config = llm_config
        self.input_size = input_size

        super().__init__(**kwargs)


__all__ = ["LlamaForASRConfig"]
