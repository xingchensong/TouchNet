# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from typing import Callable, Generator, Iterable, List, Optional, Set, Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.pipelining.schedules import (PipelineScheduleMulti,
                                                    PipelineScheduleSingle,
                                                    _PipelineSchedule,
                                                    _PipelineScheduleRuntime,
                                                    get_schedule_class)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import DTensor

from touchnet.bin import TrainConfig
from touchnet.utils.logging import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq=1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        self.collect("Initial GC collection.")

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Peforming periodical GC collection.")

    @staticmethod
    def collect(reason: str):
        begin = time.monotonic()
        gc.collect(1)
        logger.info("[GC] %s %.2f seconds.", reason, time.monotonic() - begin)


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, cp, tp, pp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        """
            NOTE(xcsong): Assume we have world_size = 8 && tp = 4 && dp = 2:
                Calling mesh["tp"] on rank 0, 1, 2, 3 returns a 1D submesh of DeviceMesh:([0, 1, 2, 3]).
                Calling mesh["tp"] on rank 4, 5, 6, 7 returns a 1D submesh of  DeviceMesh:([4, 5, 6, 7]).
                Calling mesh["dp"] on rank 0, 4 returns a 1D submesh of  DeviceMesh:([0, 4]).
                Calling mesh["dp"] on rank 1, 5 returns a 1D submesh of  DeviceMesh:([1, 5]).
                Calling mesh["dp"] on rank 2, 6 returns a 1D submesh of  DeviceMesh:([2, 6]).
                Calling mesh["dp"] on rank 3, 7 returns a 1D submesh of  DeviceMesh:([3, 7]).
        """
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        logger.info(f"world_mesh: {mesh}")
        if mesh.mesh_dim_names:
            for name in mesh.mesh_dim_names:
                logger.info(f"[rank{dist.get_rank()}] world_mesh['{name}']: {mesh[name]}")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp


def _dist_reduce(x: torch.Tensor, reduceOp: str, mesh: DeviceMesh) -> float:
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()
    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_min(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.MIN.name, mesh=mesh)


def dist_mean(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh)


def dist_sum(x: torch.Tensor, mesh: DeviceMesh) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh)


def set_determinism(
    world_mesh: Optional[DeviceMesh],
    device: torch.device,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> None:
    """
    Set the same DTensor manual seed for all ranks within the same DTensor SPMD group, but different
    seeds across PP groups (if applicable).

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.
    """
    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
        # seeds for unique SPMD groups
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()

    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    if c10d.get_world_size() > 1 and "pp" in world_mesh.mesh_dim_names:
        pp_mesh = world_mesh["pp"]
        seed += pp_mesh.get_local_rank()
        seed %= 2**64

        logger.debug(
            f"PP rank {pp_mesh.get_local_rank()}, Global rank {c10d.get_rank()} using seed: {seed}"
        )
        spmd_mesh_dims = list(
            filter(lambda name: name != "pp", world_mesh.mesh_dim_names)
        )
        spmd_mesh = world_mesh[spmd_mesh_dims] if len(spmd_mesh_dims) else None
    else:
        spmd_mesh = world_mesh
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
    # IF PP is also used, this seed is unique per PP rank.
    if spmd_mesh and spmd_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, spmd_mesh)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import \
            set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def init_distributed(job_config: TrainConfig):
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
            )
        os.environ[env] = val

    def _get_distributed_backend(job_config: TrainConfig):
        backend = "nccl"
        if device_type in torch.distributed.Backend.default_device_backend_map.keys():
            backend = torch.distributed.Backend.default_device_backend_map.get(
                device_type
            )
        if job_config.training_enable_cpu_offload:
            backend = f"{device_type}:{backend},cpu:gloo"
        return backend

    TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
    TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(job_config.training_trace_buf_size))
    if job_config.training_trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = f"{job_config.training_trace_dump_folder}/comm_trace"
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/rank_")

    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(job_config),
        timeout=timedelta(seconds=job_config.training_init_timeout_seconds),
    )


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


# TODO: It's unclear if this API is general enough to be used by other models.
# If not, we should move it to a Transformer-specific directory.
def generate_split_points(
    job_config: TrainConfig, pp_dim: int, num_layers: int
) -> list[str]:
    """
    Generate a default split point based on the number of layers and
    pipeline parallel dimension.

    Args:
        job_config (TrainConfig): The job configuration.
        pp_dim (int): The pipeline parallel dimension.
        num_layers (int): The number of layers in the model.

    Returns:
        list[str]: A list of split point FQNs.
    """

    schedule_class = get_schedule_class(
        job_config.training_pipeline_parallel_schedule
    )
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified
        num_stages_per_rank = 2
    else:
        raise ValueError(
            f"Unsupported pipeline schedule: {job_config.training_pipeline_parallel_schedule}"
        )
    total_stages = pp_dim * num_stages_per_rank
    if total_stages > num_layers:
        raise ValueError("Total stages cannot be greater than the number of layers")

    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages

    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # Middle stages get an extra layer if there are any remaining
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))
    logger.info(
        f"No 'pipeline_parallel_split_points' provided so the generated splits are: {splits} "
        "This may be sub-optimal as the number of layers per stage may be unbalanced."
    )
    return splits


def build_pipeline_schedule(
    job_config: TrainConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        job_config (TrainConfig): The job configuration.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = job_config.training_pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(
            job_config.training_pipeline_parallel_schedule
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    n_microbatches = job_config.training_pipeline_parallel_microbatches
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.training_pipeline_parallel_degree * len(stages)
    if n_microbatches is None:
        n_microbatches = num_total_stages
    elif n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    # validate that the batch size is divisible by the number of microbatches otherwise we'll hang or error during training
    if job_config.training_batchsize % n_microbatches != 0:
        raise ValueError(
            f"Batch size {job_config.training_batchsize} must be divisible by number of microbatches {n_microbatches}. "
            "Update the config arguments for either dataset_batchsize or training_pipeline_parallel_microbatches."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Using pipeline schedule {job_config.training_pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        schedule._load_csv(pp_schedule_csv)

    return schedule


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> tuple[int]:
    """
        Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule.

        NOTE(xcsong):

        Assume we have pp_size=4 & num_stages=8 & schedule=ScheduleZBVZeroBubble(MultiStage) & num_layers=16
        Then we got loop style:
            pp_rank0 runs: stage0(layer0 + layer1) + stage4(layer8 + layer9)
            pp_rank1 runs: stage1(layer2 + layer3) + stage5(layer10 + layer11)
            pp_rank2 runs: stage2(layer4 + layer5) + stage6(layer12 + layer13)
            pp_rank3 runs: stage3(layer6 + layer7) + stage7(layer14 + layer15)
            Training batch pass through rank0~3 twice, this is so called `loop` style.
        and v style:
            pp_rank0 runs: stage0(layer0 + layer1) + stage7(layer14 + layer15)
            pp_rank1 runs: stage1(layer2 + layer3) + stage6(layer12 + layer13)
            pp_rank2 runs: stage2(layer4 + layer5) + stage5(layer10 + layer11)
            pp_rank3 runs: stage3(layer6 + layer7) + stage4(layer8 + layer9)
            Training batch pass through rank0~3 and rank3~0, vice versa.

        For pp_size=4 & num_stages=4 & schedule=Schedule1F1B(SingleStage) & num_layers=16
        We only have:
            pp_rank0 runs: stage0(layer0 + layer1 + layer2 + layer3)
            pp_rank1 runs: stage1(layer4 + layer5 + layer6 + layer7)
            pp_rank2 runs: stage2(layer8 + layer9 + layer10 + layer11)
            pp_rank3 runs: stage3(layer12 + layer13 + layer14 + layer15)
            Training batch pass through rank0~3 only once.
    """
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]
