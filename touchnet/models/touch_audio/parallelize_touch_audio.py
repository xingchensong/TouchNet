# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (CPUOffloadPolicy, MixedPrecisionPolicy,
                                    fully_shard)
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)

from touchnet.bin import TrainConfig
from touchnet.models.helper_func import (apply_ac, apply_compile, apply_ddp,
                                         apply_fsdp)
from touchnet.models.touch_audio.modeling_touch_audio import \
    TouchAudioForCausalLM
from touchnet.utils.distributed import TORCH_DTYPE_MAP, ParallelDims
from touchnet.utils.logging import logger


def parallelize_touch_audio(
    model: TouchAudioForCausalLM,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: TrainConfig,
) -> torch.nn.Module:
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    # Apply tensor parallelism for maximum memory savings on lm_head
    # Current strategy: language_model full TP + audio_tower non-TP + projector for layout conversion
    if parallel_dims.tp_enabled:
        if (
            job_config.training_enable_async_tensor_parallel
            and not job_config.training_compile
        ):
            raise RuntimeError("Async TP requires --training_compile")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=False,  # TODO(xcsong): support fp8 training
            enable_async_tp=job_config.training_enable_async_tensor_parallel,
        )

    # TODO(xcsong): Pipeline Parallel not yet supported for qwen2_audio
    assert not parallel_dims.pp_enabled

    if job_config.training_activation_checkpoint_mode != "none":
        apply_ac(model.language_model, job_config, base_model_prefix="model")

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training_compile:
        apply_compile(model.language_model, base_model_prefix="model")

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model.language_model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training_enable_cpu_offload,
            reshard_after_forward_policy=job_config.training_fsdp_reshard_after_forward,
            base_model_prefix="model",
            shard_on_toplevel_model=False,
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_reduce],
        )
        fsdp_config = {"mesh": world_mesh[tuple(dp_mesh_dim_names)], "mp_policy": mp_policy}
        if job_config.training_enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()
        fully_shard(model, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled)

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if job_config.training_enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.training_compile,
            enable_compiled_autograd=job_config.training_enable_compiled_autograd,
        )

    return model


def apply_tp(
    model: TouchAudioForCausalLM,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    base_model_prefix = getattr(model.language_model, "base_model_prefix", "model")
    parallelize_module(
        model,
        tp_mesh,
        {
            f"language_model.{base_model_prefix}.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            f"language_model.{base_model_prefix}.norm": SequenceParallel(),
            "language_model.lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    if hasattr(model, 'projector'):
        parallelize_module(
            model,
            tp_mesh,
            {
                "projector": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
            },
        )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears
    if enable_float8:
        raise NotImplementedError("Float8 is not supported. TODO(xcsong): support it")
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    submodel = getattr(model.language_model, f"{base_model_prefix}")
    if isinstance(submodel.layers, torch.nn.ModuleDict):
        transformer_blocks = submodel.layers.values()
    else:
        transformer_blocks = submodel.layers
    for transformer_block in transformer_blocks:
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": prepare_module_input(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                },
            ),
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            "self_attn.o_proj": rowwise_parallel(output_layouts=Shard(1)),
            "post_attention_layernorm": SequenceParallel(),
            "mlp": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": colwise_parallel(),
            "mlp.down_proj": rowwise_parallel(output_layouts=Shard(1)),
            "mlp.up_proj": colwise_parallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import \
            enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )
