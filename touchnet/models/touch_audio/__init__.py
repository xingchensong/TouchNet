import torch
from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                       apply_liger_kernel_to_qwen2)

from touchnet.bin import TrainConfig
from touchnet.models.touch_audio.configuration_touch_audio import \
    TouchAudioConfig
from touchnet.models.touch_audio.modeling_touch_audio import \
    TouchAudioForCausalLM


def pre_init(args: TrainConfig):
    """Pre-initialization function for Qwen2AudioForConditionalGeneration."""
    if args.training_enable_liger_kernel:
        # 1. monkey patch the forward function to Qwen2Model
        apply_liger_kernel_to_qwen2()
        # 2. monkey patch the forward function to LlamaModel
        apply_liger_kernel_to_llama()


def post_init(model: TouchAudioForCausalLM, init_device: torch.device):
    """Post-initialization function for Qwen2AudioForConditionalGeneration."""

    # NOTE(xcsong): Init rope and norm.weight
    # 1. llama/qwen backbone
    inv_freq, attention_scaling = model.language_model.model.rotary_emb.rope_init_fn(
        model.language_model.model.rotary_emb.config, device=init_device)
    model.language_model.model.rotary_emb.inv_freq = inv_freq
    model.language_model.model.rotary_emb.attention_scaling = attention_scaling
    model.language_model.model.rotary_emb.original_inv_freq = inv_freq
    torch.nn.init.ones_(model.language_model.model.norm.weight)
    for layer in model.language_model.model.layers:
        torch.nn.init.ones_(layer.input_layernorm.weight)
        torch.nn.init.ones_(layer.post_attention_layernorm.weight)

    # NOTE(xcsong): Do some NaN check
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/inf in model parameters `{name}`.")


def get_num_flop_per_token(num_params: int, model_config: TouchAudioConfig, seq_len: int) -> int:
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
