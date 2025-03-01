# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)

from touchnet.bin import TrainConfig
from touchnet.parallel_models.helper_func import (apply_ac, apply_compile,
                                                  apply_ddp, apply_fsdp)
from touchnet.utils.distributed import TORCH_DTYPE_MAP, ParallelDims
from touchnet.utils.logging import logger


def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: TrainConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

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

    if job_config.training_activation_checkpoint_mode != "none":
        apply_ac(model, job_config)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training_compile:
        apply_compile(model)

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training_enable_cpu_offload,
            reshard_after_forward_policy=job_config.training_fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

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


def apply_tp(
    model: nn.Module,
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
    base_model_prefix = getattr(model, "base_model_prefix", "model")
    parallelize_module(
        model,
        tp_mesh,
        {
            f"{base_model_prefix}.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            f"{base_model_prefix}.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
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
    model = getattr(model, f"{base_model_prefix}")
    for layer_id, transformer_block in model.layers.items():
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
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
