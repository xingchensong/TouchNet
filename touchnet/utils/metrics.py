# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import time
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from touchnet.bin import TrainConfig
from touchnet.utils.distributed import ParallelDims, device_module, device_type
from touchnet.utils.logging import Color, logger
from touchnet.utils.optimizer import LRSchedulersContainer, OptimizersContainer


def accuracy(pred: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -100) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pred (Tensor): Prediction tensors (B, Lmax // cp, Vocab // tp) if pred.to_local()
                       else (B, Lmax // cp, Vocab) if pred.full_tensor()
        labels (LongTensor): Target label tensors (B, Lmax // cp).
        ignore_index (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    if isinstance(pred, DTensor):
        pred = pred.full_tensor()  # (B, T // cp, V)
    pred = pred.argmax(dim=-1)  # (B, T // cp, V) -> (B, T // cp)
    mask = labels != ignore_index
    numerator = torch.sum(
        pred.masked_select(mask) == labels.masked_select(mask))
    denominator = torch.sum(mask)
    if denominator > 0:
        return (numerator / denominator).detach()
    else:
        return torch.zeros_like(numerator).detach()


def flatten_config(config, parent_key=''):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


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


# hardcoded BF16 type peak flops for NVIDIA A100, H100, and H200 GPU
def get_peak_flops(device_name: str) -> int:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name or "A800" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name or "H800" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "3090" in device_name:
        return 71e12
    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass

    def add_hparams(self, hparam_dict: List[Dict]) -> None:
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict: Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.

        """
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: Optional[str] = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()

    def add_hparams(self, hparam_dict: List[Dict]) -> None:
        final_dict = {}
        for conf in hparam_dict:
            final_dict.update(flatten_config(conf))
        exp, ssi, sei = hparams(final_dict, {}, None)
        self.writer.file_writer.add_summary(exp, None)
        self.writer.file_writer.add_summary(ssi, None)
        self.writer.file_writer.add_summary(sei, None)


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(self, log_dir: str, tag: Optional[str] = None):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        self.wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            dir=log_dir,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


def ensure_pp_loss_visible(
    parallel_dims: ParallelDims, job_config: TrainConfig, color: Color
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.
    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if job_config.training_pipeline_parallel_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline parallel loss is not visible. "
            f"Add {color.yellow}rank {loss_visible_rank}{color.red} to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    parallel_dims: ParallelDims,
    job_config: TrainConfig,
) -> int:
    """
    Determines which rank should log metrics.
    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if job_config.training_pipeline_parallel_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


def _build_metric_logger(
    job_config: TrainConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
) -> BaseLogger:
    """
    Build an appropriate metric logger based on configuration.
    """
    # Log initial config state
    logger.debug(
        f"Building logger with config: wandb={job_config.training_enable_wandb}, "
        f"tensorboard={job_config.training_enable_tensorboard}"
    )

    # Check if any logging backend is enabled
    has_logging_enabled = (
        job_config.training_enable_tensorboard or job_config.training_enable_wandb
    )

    # Determine if this rank should log
    should_log = has_logging_enabled
    if job_config.training_tb_rank_0_only and should_log:
        metrics_rank = _get_metrics_rank(parallel_dims, job_config)
        should_log = torch.distributed.get_rank() == metrics_rank

    logger.debug(
        f"Logging decision: has_logging_enabled={has_logging_enabled}, should_log={should_log}"
    )

    if not should_log:
        logger.debug("Returning BaseLogger due to should_log=False")
        return BaseLogger()

    # Setup logging directory
    dump_dir = job_config.training_trace_dump_folder
    base_log_dir = os.path.join(
        dump_dir, job_config.training_save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
    )

    if not job_config.training_tb_rank_0_only:
        base_log_dir = os.path.join(
            base_log_dir, f"rank_{torch.distributed.get_rank()}"
        )

    # Create loggers in priority order
    if job_config.training_enable_wandb:
        logger.debug("Attempting to create WandB logger")
        try:
            return WandBLogger(base_log_dir, tag)
        except Exception as e:
            if "No module named 'wandb'" in str(e):
                logger.error(
                    "Failed to create WandB logger: No module named 'wandb'. Please install it using 'pip install wandb'."
                )
            else:
                logger.error(f"Failed to create WandB logger: {e}")

    if job_config.training_enable_tensorboard:
        logger.debug("Creating TensorBoard logger")
        return TensorBoardLogger(base_log_dir, tag)

    logger.debug("No loggers enabled, returning BaseLogger")
    return BaseLogger()


class MetricsProcessor:
    """Metrics processor to processes the metrics and log metrics.
    The current MetricsProcessor log some metrics to STDOUT and some metrics to
    TensorBoard or WandB.
    Args:
        job_config (JobConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        tag (Optional[str]): Tag to use for TensorBoard or WandB. Defaults to None.
    """

    logger: BaseLogger
    parallel_dims: ParallelDims
    job_config: TrainConfig
    device_memory_monitor: DeviceMemoryMonitor
    color: Color

    gpu_peak_flops: int
    ntokens_since_last_log: int
    data_loading_times: list[float]
    time_last_log: float

    num_flop_per_token: int
    optimizers: Optional[OptimizersContainer]
    lr_schedulers: Optional[LRSchedulersContainer]

    def __init__(
        self,
        job_config: TrainConfig,
        parallel_dims: ParallelDims,
        tag: Optional[str] = None,
    ):
        self.logger = _build_metric_logger(job_config, parallel_dims, tag)
        self.parallel_dims = parallel_dims
        self.job_config = job_config
        self.device_memory_monitor = build_device_memory_monitor()
        # used for colorful printing
        self.color = Color

        self.gpu_peak_flops = get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

        # These variables have to be set later as they depend on other components or model.
        self.num_flop_per_token = -1
        self.optimizers = None
        self.lr_schedulers = None

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.job_config.training_log_freq == 0

    def log(self, epoch: int, step: int, global_avg_loss_per_sample: float,
            global_avg_loss_per_token: float, global_max_loss_per_token: float,
            global_avg_grad_norm: float, global_max_grad_norm: float,
            global_avg_acc: float, global_min_acc: float):
        assert self.num_flop_per_token > 0, "num_flop_per_token must be set"

        time_delta = time.perf_counter() - self.time_last_log

        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        # model FLOPS utilization
        # For its definition and calculation, please refer to the PaLM paper:
        # https://arxiv.org/abs/2204.02311
        mfu = 100 * self.num_flop_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flop_per_token * tps / 1e12

        time_end_to_end = time_delta / self.job_config.training_log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss_per_sample": global_avg_loss_per_sample,
            "loss_metrics/global_avg_loss_per_token": global_avg_loss_per_token,
            "loss_metrics/global_max_loss_per_token": global_max_loss_per_token,
            "loss_metrics/global_avg_grad_norm": global_avg_grad_norm,
            "loss_metrics/global_max_grad_norm": global_max_grad_norm,
            "loss_metrics/global_avg_acc": global_avg_acc,
            "loss_metrics/global_min_acc": global_min_acc,
            "loss_metrics/learning_rate": lr,
            "throughput(tps)": tps,
            "tflops": tflops,
            "mfu(%)": mfu,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }
        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.red}epoch: {epoch:2} step: {step:2}  "
            f"{color.green}loss (per sample): {global_avg_loss_per_sample:7.4f}  "
            f"{color.green}loss (per token): {global_avg_loss_per_token:7.4f}  "
            f"{color.green}grad norm: {global_avg_grad_norm:5.2f}  "
            f"{color.green}acc: {global_avg_acc:5.2f}  "
            f"{color.green}lr: {lr:.4f}  "
            f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.cyan}tflops: {tflops:,.2f}  "
            f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    def log_dev(self, epoch: int, step: int, global_avg_loss_per_sample: float,
                global_avg_loss_per_token: float, global_max_loss_per_token: float,
                global_avg_acc: float, global_min_acc: float):
        metrics = {
            "loss_metrics/dev_global_avg_loss_per_sample": global_avg_loss_per_sample,
            "loss_metrics/dev_global_avg_loss_per_token": global_avg_loss_per_token,
            "loss_metrics/dev_global_max_loss_per_token": global_max_loss_per_token,
            "loss_metrics/dev_global_avg_acc": global_avg_acc,
            "loss_metrics/dev_global_min_acc": global_min_acc,
        }
        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.red}epoch: {epoch:2} dev-step: {step:2}  "
            f"{color.green}loss (per sample): {global_avg_loss_per_sample:7.4f}  "
            f"{color.green}loss (per token): {global_avg_loss_per_token:7.4f} "
            f"{color.green}acc (per token): {global_avg_acc:7.4f}{color.reset}"
        )

    def close(self):
        self.logger.close()


def build_metrics_processor(
    job_config: TrainConfig, parallel_dims: ParallelDims, tag: Optional[str] = None
) -> MetricsProcessor:
    """Create a metrics processor.
    Args:
        job_config (TrainConfig): Job configuration.
        parallel_dims (ParallelDims): Parallel dimensions.
        tag (Optional[str]): Tag to use for TensorBoard or WandB. Defaults to None.
    Returns:
        MetricsProcessor: A metrics processor.
    """
    return MetricsProcessor(job_config, parallel_dims, tag)
