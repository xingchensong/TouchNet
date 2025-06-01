# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from touchnet.models.kimi_audio.configuration_kimi_audio import KimiAudioConfig
from touchnet.models.kimi_audio.modeling_kimi_audio import \
    MoonshotKimiaForCausalLM


def post_init(model: MoonshotKimiaForCausalLM, init_device: torch.device):
    """Post-initialization function for MoonshotKimiaForCausalLM."""

    # NOTE(xcsong): Init rope and norm.weight

    # 0. speech tokenizer
    model.speech_tokenizer._freeze_parameters()

    # 1. whisper encoder
    std = (
        model.config.initializer_range
        if hasattr(model.config, "initializer_range")
        else model.config.speech_encoder_config.initializer_range
    )
    model.speech_encoder.conv1.weight.data.normal_(mean=0.0, std=std)
    model.speech_encoder.conv1.bias.data.zero_()
    model.speech_encoder.conv2.weight.data.normal_(mean=0.0, std=std)
    model.speech_encoder.conv2.bias.data.zero_()
    model.speech_encoder.layer_norm.reset_parameters()
    for layer in model.speech_encoder.layers:
        layer.self_attn_layer_norm.reset_parameters()
        layer.final_layer_norm.reset_parameters()
    model.model.vq_adaptor.layers[-1].reset_parameters()  # layernorm

    # 2. qwen backbone
    inv_freq, attention_scaling = model.model.rotary_emb.rope_init_fn(
        model.model.rotary_emb.config, device=init_device)
    model.model.rotary_emb.inv_freq = inv_freq
    model.model.rotary_emb.attention_scaling = attention_scaling
    model.model.rotary_emb.original_inv_freq = inv_freq
    assert isinstance(model.model.norm, Qwen2RMSNorm)
    torch.nn.init.ones_(model.model.norm.weight)
    assert isinstance(model.model.mimo_norm, Qwen2RMSNorm)
    torch.nn.init.ones_(model.model.mimo_norm.weight)
    for layer in model.model.layers:
        assert isinstance(layer.input_layernorm, Qwen2RMSNorm)
        assert isinstance(layer.post_attention_layernorm, Qwen2RMSNorm)
        torch.nn.init.ones_(layer.input_layernorm.weight)
        torch.nn.init.ones_(layer.post_attention_layernorm.weight)
    for layer in model.model.mimo_layers:
        assert isinstance(layer.input_layernorm, Qwen2RMSNorm)
        assert isinstance(layer.post_attention_layernorm, Qwen2RMSNorm)
        torch.nn.init.ones_(layer.input_layernorm.weight)
        torch.nn.init.ones_(layer.post_attention_layernorm.weight)

    # NOTE(xcsong): Do some NaN check
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/inf in model parameters `{name}`.")


def get_num_flop_per_token(num_params: int, model_config: KimiAudioConfig, seq_len: int) -> int:
    # NOTE(xcsong): We do not include flops from speech_encoder
    l, h, q, t = (
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size // model_config.num_attention_heads,
        seq_len,
    )
    l_mimo = model_config.kimia_mimo_layers
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * (l + l_mimo) * h * q * t

    return flop_per_token


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        base_model_prefix = getattr(model, "base_model_prefix", "model")
        submodel = getattr(model, f"{base_model_prefix}")
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in submodel.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params
