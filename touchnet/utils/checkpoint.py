# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import os
import queue
import re
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._state_dict_utils import (_copy_state_dict,
                                                 _create_cpu_state_dict)
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     set_model_state_dict)
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader

from touchnet.bin import TrainConfig
from touchnet.utils.distributed import TORCH_DTYPE_MAP, GarbageCollection
from touchnet.utils.logging import init_logger, logger
from touchnet.utils.optimizer import LRSchedulersContainer, OptimizersContainer

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    def __init__(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def state_dict(self) -> Dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }


class Terminate:
    pass


class SaveDone:
    pass


@torch.no_grad()
def save_with_gc(state, checkpoint_id):
    dcp.save(state, checkpoint_id=checkpoint_id)
    GarbageCollection.collect("GC collection invoked by checkpointer.")


def checkpoint_mp(recv: mp.Queue, send: mp.Queue):
    """Process to save the checkpoint in the background.

    This is only used when async_checkpoint_with_pinned_memory is enabled.

    Args:
        recv (mp.Queue): The queue to receive the state_dict and Terminate signal.
        send (mp.Queue): The queue to send the SaveDone signal.
    """
    init_logger()
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Checkpoint background process is done.")
            send.put(SaveDone())
            logger.debug("Wait for the new state_dict.")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the checkpoint background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, checkpoint_id = obj
            save_with_gc(state, checkpoint_id=checkpoint_id)
            logger.info(
                "Finish saving the checkpoint in the background process in %.2f seconds.",
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the process group.")
        dist.destroy_process_group()


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is gauranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.


    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        job_config (TrainConfig): The job config used to configure the checkpointing.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        model_parts: List[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: Dict[str, Any],
        job_config: TrainConfig,
    ) -> None:
        self.enable_checkpoint = job_config.training_enable_ckpt

        async_mode = job_config.training_ckpt_async_mode.lower()
        self.enable_staging = (
            self.enable_checkpoint and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        )

        if not self.enable_checkpoint:
            return

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.staging_stream = torch.cuda.Stream() if self.enable_staging else None

        self.folder = os.path.join(job_config.training_trace_dump_folder, job_config.training_ckpt_folder)
        self.interval = job_config.training_ckpt_interval
        if async_mode == AsyncMode.ASYNC:
            self.pg = dist.new_group(backend="gloo")

        self.keep_latest_k = job_config.training_ckpt_keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.model_weights_only = job_config.training_ckpt_model_weights_only
        self.export_dtype = TORCH_DTYPE_MAP[job_config.training_ckpt_export_dtype]
        raw_args = job_config.training_ckpt_exclude_from_loading
        self.exclude_from_loading = [s.strip() for s in raw_args.split(",") if s.strip()]

        self.mp = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
            self.async_future = None
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
            ctx = mp.get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=checkpoint_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                ),
                daemon=True,
            )
            self.mp.start()
        else:
            raise ValueError(f"Unkown checkpoint async_mode {async_mode}")

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if self.enable_checkpoint:
            if self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if self.purge_thread and self.purge_thread.is_alive():
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

    @torch.no_grad()
    def save(self, curr_step: int, force: bool = False) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``force`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == job_config.lr_scheduler_steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            force (bool, optional): Whether to force save the checkpoint. Defaults to False.

        Returns:
            None
        """

        if not self._should_save(curr_step, force):
            return

        begin = time.monotonic()
        logger.info("Saving the checkpoint (or staging if async is enabled).")
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do
        # GC right after async_save -- the CPU memory is not able to be
        # freed until _async_wait()
        if force:
            self._save_last_step(curr_step)
        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self._async_with_pinned_memory(checkpoint_id)
        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self.async_future = dcp.async_save(
                self.states, checkpoint_id=checkpoint_id, process_group=self.pg
            )
            GarbageCollection.collect("GC collection invoked by checkpointer.")
        else:
            save_with_gc(self.states, checkpoint_id=checkpoint_id)
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled)"
            f"in {time.monotonic() - begin:.2f} seconds."
        )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """

        if not self.enable_checkpoint or not os.path.isdir(self.folder):
            return False

        if step == -1:
            step = self._find_load_step()
            if step == -1:
                return False

        checkpoint_id = self._create_checkpoint_id(step)
        if not os.path.isdir(checkpoint_id):
            return False

        logger.info(f"Loading the checkpoint at step {step}.")
        begin = time.monotonic()
        states = self._states_to_load(step)
        dcp.load(states, checkpoint_id=checkpoint_id)
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            if not self.staging_stream.query():
                begin = time.monotonic()
                self.staging_stream.synchronize()
                logger.info(
                    "Checkpointer waited staging %.2f seconds.",
                    time.monotonic() - begin,
                )
            self.staging = False

            if self.sending_to_checkpoint_mp:
                # Copy the sync staging result to another process.
                def sync_func():
                    self.mp_queue_send.put_nowait(
                        (self.cpu_offload_state_dict, self.staging_id)
                    )

                # This may be a faster way to do zero-overhead checkpointing staging
                # checkpointing but we need more thorough investigation before
                # swithing to this method.
                # self.my_thread = threading.Thread(target=func).start()
                begin = time.monotonic()
                sync_func()
                logger.info(
                    "Checkpointer sent staged state_dict to another process %.2f seconds",
                    time.monotonic() - begin,
                )
                self.sending_to_checkpoint_mp = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            metadata_probe = os.path.join(folder, filename, ".metadata")
            if match and os.path.isfile(metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _states_to_load(self, step: int) -> Dict[str, Any]:
        """Determines which states to load for the given step.

        When checkpointer determines which step of the checkpoint to load, this API is
        used to determine which states to load based on the step.

        Args:
            step (int): The step to load the checkpoint for.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model weights.
        states = {MODEL: self.states[MODEL]} if step == 0 else self.states
        states_to_load = {
            k: v for k, v in states.items() if k not in self.exclude_from_loading
        }
        for exclude_key in self.exclude_from_loading:
            if exclude_key not in states:
                raise ValueError(f"{exclude_key} not found in state_dict.")
        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving weights only at the end of the training. So
        # this won't affect preemption and training resume. We also only allow
        # dtype conversion when we are checkpoint model weights only and the
        # current dtype is not the same as the export dtype at the end of the training.

        if self.model_weights_only:
            # We update self.states to keep the model only.
            # After this update, self.states = {
            #      'tok_embeddings.weight':...,
            #      'layers.0.attention.wq.weight': ...
            # }.
            self.states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                self.states = {
                    k: v.to(self.export_dtype) for k, v in self.states.items()
                }
            logger.info(
                f"Saving a model weights only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")

        save_with_gc(self.states, checkpoint_id=self._create_checkpoint_id(curr_step))

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        # Force saving a checkpoint at step 1 to fail fast if checkpointer is not
        # compatible with the cluster.
        if curr_step == 1:
            return True

        if force:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            logger.debug(
                f"Waiting for the background process to finish, {time.monotonic()=}.:.2f"
            )
            if not self.mp.is_alive():
                raise RuntimeError("The checkpoint background process is dead.")
            _ = self.mp_queue_recv.get()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()

    def _async_with_pinned_memory(self, checkpoint_id: str) -> None:
        self._cpu_staging(checkpoint_id)
        self.sending_to_checkpoint_mp = True

    def _cpu_staging(self, checkpoint_id: Optional[str]) -> None:
        """Offload state_dict to CPU memory"""
        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_id = checkpoint_id

    def _purge_stale_checkpoints(self):
        if (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                path = os.path.join(self.folder, filename)
                discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
