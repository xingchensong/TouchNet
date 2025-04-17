from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from touchnet.data.dataloader import build_dataloader
from touchnet.loss.cross_entropy import cross_entropy_loss
from touchnet.models.llama.configuration_llama import LlamaForASRConfig
from touchnet.models.llama.modeling_llama import LlamaForASR
from touchnet.models.llama.parallelize_llama import parallelize_llama
from touchnet.models.llama.pipeline_llama import pipeline_llama
from touchnet.tokenizer.tokenizer import build_tokenizer
from touchnet.utils.metrics import accuracy, build_metrics_processor
from touchnet.utils.optimizer import build_lr_schedulers, build_optimizers
from touchnet.utils.train_spec import TrainSpec, register_train_spec


def post_init(model: LlamaForCausalLM, init_device: torch.device):
    """Post-initialization function for LlamaForCausalLM and LlamaForASR."""

    # NOTE(xcsong): Init rope and norm.weight
    inv_freq, attention_scaling = model.model.rotary_emb.rope_init_fn(
        model.model.rotary_emb.config, device=init_device)
    model.model.rotary_emb.inv_freq = inv_freq
    model.model.rotary_emb.attention_scaling = attention_scaling
    model.model.rotary_emb.original_inv_freq = inv_freq
    assert isinstance(model.model.norm, LlamaRMSNorm)
    torch.nn.init.ones_(model.model.norm.weight)
    for layer in model.model.layers:
        assert isinstance(layer.input_layernorm, LlamaRMSNorm)
        assert isinstance(layer.post_attention_layernorm, LlamaRMSNorm)
        torch.nn.init.ones_(layer.input_layernorm.weight)
        torch.nn.init.ones_(layer.post_attention_layernorm.weight)

    # NOTE(xcsong): Do some NaN check
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/inf in model parameters `{name}`.")


def get_num_flop_per_token(num_params: int, model_config: LlamaConfig, seq_len: int) -> int:
    l, h, q, t = (
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size // model_config.num_attention_heads,
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


register_train_spec(
    TrainSpec(
        name="llama",
        model_cls=LlamaForCausalLM,
        config_cls=LlamaConfig,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        loss_fn=cross_entropy_loss,
        acc_fn=accuracy,
        additional_post_init_fn=post_init,
        build_metrics_processor_fn=build_metrics_processor,
        get_num_flop_per_token_fn=get_num_flop_per_token,
    )
)

register_train_spec(
    TrainSpec(
        name="llama.asr",
        model_cls=LlamaForASR,
        config_cls=LlamaForASRConfig,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        loss_fn=cross_entropy_loss,
        acc_fn=accuracy,
        additional_post_init_fn=post_init,
        build_metrics_processor_fn=build_metrics_processor,
        get_num_flop_per_token_fn=get_num_flop_per_token,
    )
)

AutoConfig.register(LlamaForASRConfig.model_type, LlamaForASRConfig, True)
AutoModelForCausalLM.register(LlamaForASRConfig, LlamaForASR, True)
