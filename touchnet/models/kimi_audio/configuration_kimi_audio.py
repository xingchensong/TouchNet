# -*- coding: utf-8 -*-
# Copyright (c) 2025, The Moonshot AI Team
#                     Xingchen Song(sxc19@tsinghua.org.cn)

from transformers import Qwen2Config, WhisperConfig


# NOTE(xcsong): `WhisperVQConfig` for audio tokenizer
class WhisperVQConfig(WhisperConfig):
    def __init__(self,
                 pooling_kernel_size=4,
                 pooling_type="avg",
                 pooling_position=16,
                 quantize_vocab_size=16384,
                 quantize_position=16,
                 quantize_commit_coefficient=0.25,
                 quantize_loss_scale=10.0,
                 quantize_ema_decay=0.99,
                 quantize_restart_interval=100,
                 quantize_encoder_only=True,
                 quantize_causal_encoder=False,
                 quantize_causal_block_size=200,
                 skip_language_detection=True,
                 encoder_causal_attention=False,
                 encoder_causal_convolution=True,
                 **kwargs):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        super().__init__(**kwargs)


class KimiAudioConfig(Qwen2Config):
    model_type = "kimi_audio"
    sub_configs = {
        "speech_tokenizer_config": WhisperVQConfig,
        "speech_encoder_config": WhisperConfig,
    }

    def __init__(
        self,
        speech_tokenizer_config=None,
        speech_encoder_config=None,
        vocab_size=163840,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        kimia_mimo_layers: int = 6,
        kimia_mimo_audiodelaytokens: int = 5,
        kimia_mimo_transformer_from_layer_index: int = 21,
        kimia_audio_output_vocab: int = 16896,
        kimia_text_output_vocab: int = 152064,
        num_audio_special_tokens: int = 512,
        num_base_tokens: int = 151643,
        kimia_token_offset: int = 152064,
        use_whisper_feature: bool = True,
        kimia_adaptor_input_dim: int = 5120,
        kimia_media_begin: int = 151661,
        kimia_media_end: int = 151663,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **kwargs,
        )

        if speech_tokenizer_config is None:
            speech_tokenizer_config = {}
        self.speech_tokenizer_config = WhisperVQConfig(**speech_tokenizer_config)

        if speech_encoder_config is None:
            speech_encoder_config = {}
        self.speech_encoder_config = WhisperConfig(**speech_encoder_config)

        self.kimia_mimo_layers = kimia_mimo_layers
        self.kimia_mimo_audiodelaytokens = kimia_mimo_audiodelaytokens
        # vocab
        self.kimia_mimo_transformer_from_layer_index = (
            kimia_mimo_transformer_from_layer_index
        )
        self.kimia_audio_output_vocab = kimia_audio_output_vocab
        self.kimia_text_output_vocab = kimia_text_output_vocab
        self.num_audio_special_tokens = num_audio_special_tokens
        self.num_base_tokens = num_base_tokens
        self.kimia_token_offset = kimia_token_offset
        self.use_whisper_feature = use_whisper_feature
        self.kimia_adaptor_input_dim = kimia_adaptor_input_dim
        # special tokens
        self.kimia_media_begin = kimia_media_begin
        self.kimia_media_end = kimia_media_end
