# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

# Import the built-in models here so that the corresponding register_model_spec()
# will be called.
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.qwen2_audio import (
    Qwen2AudioConfig, Qwen2AudioForConditionalGeneration)

from touchnet.data.dataloader import build_dataloader
from touchnet.loss.cross_entropy import cross_entropy_loss
from touchnet.models import kimi_audio, llama, qwen2_audio, touch_audio
from touchnet.models.kimi_audio.configuration_kimi_audio import KimiAudioConfig
from touchnet.models.kimi_audio.modeling_kimi_audio import \
    MoonshotKimiaForCausalLM
from touchnet.models.kimi_audio.parallelize_kimi_audio import \
    parallelize_kimi_audio
from touchnet.models.llama.parallelize_llama import parallelize_llama
from touchnet.models.llama.pipeline_llama import pipeline_llama
from touchnet.models.qwen2_audio.parallelize_qwen2_audio import \
    parallelize_qwen2_audio
from touchnet.models.touch_audio.configuration_touch_audio import \
    TouchAudioConfig
from touchnet.models.touch_audio.modeling_touch_audio import \
    TouchAudioForCausalLM
from touchnet.models.touch_audio.parallelize_touch_audio import \
    parallelize_touch_audio
from touchnet.tokenizer.tokenizer import build_tokenizer
from touchnet.utils.metrics import accuracy, build_metrics_processor
from touchnet.utils.optimizer import build_lr_schedulers, build_optimizers
from touchnet.utils.train_spec import TrainSpec, register_train_spec

# NOTE(xcsong): step-1, register new models to touchnet
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
        additional_pre_init_fn=llama.pre_init,
        additional_post_init_fn=llama.post_init,
        build_metrics_processor_fn=build_metrics_processor,
        get_num_flop_per_token_fn=llama.get_num_flop_per_token,
        get_num_params_fn=llama.get_num_params,
    )
)

register_train_spec(
    TrainSpec(
        name="touch_audio",
        model_cls=TouchAudioForCausalLM,
        config_cls=TouchAudioConfig,
        parallelize_fn=parallelize_touch_audio,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        loss_fn=cross_entropy_loss,
        acc_fn=accuracy,
        additional_pre_init_fn=touch_audio.pre_init,
        additional_post_init_fn=touch_audio.post_init,
        build_metrics_processor_fn=build_metrics_processor,
        get_num_flop_per_token_fn=touch_audio.get_num_flop_per_token,
        get_num_params_fn=touch_audio.get_num_params,
    )
)

register_train_spec(  # TODO(xcsong): We only support FSDP2 for qwen2_audio, no tp/pp/cp
    TrainSpec(
        name="qwen2_audio",
        model_cls=Qwen2AudioForConditionalGeneration,
        config_cls=Qwen2AudioConfig,
        parallelize_fn=parallelize_qwen2_audio,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        loss_fn=cross_entropy_loss,
        acc_fn=accuracy,
        additional_pre_init_fn=qwen2_audio.pre_init,
        additional_post_init_fn=qwen2_audio.post_init,
        build_metrics_processor_fn=build_metrics_processor,
        get_num_flop_per_token_fn=qwen2_audio.get_num_flop_per_token,
        get_num_params_fn=qwen2_audio.get_num_params,
    )
)

register_train_spec(  # TODO(xcsong): We only support FSDP2 for kimi_audio, no tp/pp/cp
    TrainSpec(
        name="kimi_audio",
        model_cls=MoonshotKimiaForCausalLM,
        config_cls=KimiAudioConfig,
        parallelize_fn=parallelize_kimi_audio,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        loss_fn=cross_entropy_loss,
        acc_fn=accuracy,
        additional_pre_init_fn=None,
        additional_post_init_fn=kimi_audio.post_init,
        build_metrics_processor_fn=build_metrics_processor,
        get_num_flop_per_token_fn=kimi_audio.get_num_flop_per_token,
        get_num_params_fn=kimi_audio.get_num_params,
    )
)

# NOTE(xcsong): step-2, register new models to transformers
AutoConfig.register(TouchAudioConfig.model_type, TouchAudioConfig, exist_ok=False)
AutoModelForCausalLM.register(TouchAudioConfig, TouchAudioForCausalLM, exist_ok=False)
AutoConfig.register(KimiAudioConfig.model_type, KimiAudioConfig, exist_ok=False)
AutoModelForCausalLM.register(KimiAudioConfig, MoonshotKimiaForCausalLM, exist_ok=False)
