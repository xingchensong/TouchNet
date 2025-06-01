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
from transformers.models.qwen2_audio import Qwen2AudioForConditionalGeneration

from touchnet.bin import TrainConfig
from touchnet.models.helper_func import (apply_ac, apply_compile, apply_ddp,
                                         apply_fsdp)
from touchnet.utils.distributed import TORCH_DTYPE_MAP, ParallelDims
from touchnet.utils.logging import logger


def parallelize_qwen2_audio(
    model: Qwen2AudioForConditionalGeneration,
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

    # TODO(xcsong): We only support FSDP2 for qwen2_audio, no tp/pp/cp
    assert not parallel_dims.tp_enabled
    assert not parallel_dims.cp_enabled
    assert not parallel_dims.pp_enabled

    if job_config.training_activation_checkpoint_mode != "none":
        apply_ac(model.language_model, job_config, base_model_prefix="model")
        apply_ac(model, job_config, base_model_prefix="audio_tower")

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training_compile:
        apply_compile(model.language_model, base_model_prefix="model")
        apply_compile(model, base_model_prefix="audio_tower")

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
        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training_mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training_enable_cpu_offload,
            reshard_after_forward_policy=job_config.training_fsdp_reshard_after_forward,
            base_model_prefix="audio_tower",
            shard_on_toplevel_model=True,
        )

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
