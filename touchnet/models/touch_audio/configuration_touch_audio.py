# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class TouchAudioProjectorConfig(PretrainedConfig):
    model_type = "touch_audio_projector"

    def __init__(self, input_size: int = 4096, **kwargs):
        self.input_size = input_size
        super().__init__(**kwargs)


class TouchAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TouchAudioForCausalLM`].
    It is used to instantiate an Qwen2/Llama/... model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Qwen2 with a linear projector.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "touch_audio"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        input_seq_dropout=0.0,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            assert "model_type" in audio_config
            if audio_config["model_type"] == "touch_audio_projector":
                audio_config = TouchAudioProjectorConfig(**audio_config)
            else:
                assert "input_size" in audio_config
                audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = TouchAudioProjectorConfig()

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config
        self.input_seq_dropout = input_seq_dropout

        super().__init__(**kwargs)


__all__ = ["TouchAudioConfig"]
