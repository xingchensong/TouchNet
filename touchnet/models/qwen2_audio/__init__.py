# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2_audio import (
    Qwen2AudioConfig, Qwen2AudioForConditionalGeneration)


def post_init(model: Qwen2AudioForConditionalGeneration, init_device: torch.device):
    """Post-initialization function for Qwen2AudioForConditionalGeneration."""

    # NOTE(xcsong): Init rope and norm.weight

    # 1. whisper encoder
    model.audio_tower.layer_norm.reset_parameters()
    for layer in model.audio_tower.layers:
        layer.self_attn_layer_norm.reset_parameters()
        layer.final_layer_norm.reset_parameters()

    # 2. qwen backbone
    inv_freq, attention_scaling = model.language_model.model.rotary_emb.rope_init_fn(
        model.language_model.model.rotary_emb.config, device=init_device)
    model.language_model.model.rotary_emb.inv_freq = inv_freq
    model.language_model.model.rotary_emb.attention_scaling = attention_scaling
    model.language_model.model.rotary_emb.original_inv_freq = inv_freq
    assert isinstance(model.language_model.model.norm, Qwen2RMSNorm)
    torch.nn.init.ones_(model.language_model.model.norm.weight)
    for layer in model.language_model.model.layers:
        assert isinstance(layer.input_layernorm, Qwen2RMSNorm)
        assert isinstance(layer.post_attention_layernorm, Qwen2RMSNorm)
        torch.nn.init.ones_(layer.input_layernorm.weight)
        torch.nn.init.ones_(layer.post_attention_layernorm.weight)

    # NOTE(xcsong): Do some NaN check
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/inf in model parameters `{name}`.")


def get_num_flop_per_token(num_params: int, model_config: Qwen2AudioConfig, seq_len: int) -> int:
    # NOTE(xcsong): We do not include flops from speech_encoder
    l, h, q, t = (
        model_config.text_config.num_hidden_layers,
        model_config.text_config.num_attention_heads,
        model_config.text_config.hidden_size // model_config.text_config.num_attention_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        base_model_prefix = getattr(model.language_model, "base_model_prefix", "model")
        submodel = getattr(model.language_model, f"{base_model_prefix}")
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in submodel.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params
