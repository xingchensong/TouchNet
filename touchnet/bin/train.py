# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from io import BytesIO
from typing import Any, Dict, List

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.hf_argparser import HfArgumentParser

from touchnet.bin import TrainConfig
from touchnet.data import DataConfig
from touchnet.tokenizer import TokenizerConfig
from touchnet.utils.checkpoint import CheckpointManager
from touchnet.utils.distributed import (GarbageCollection, ParallelDims,
                                        clip_grad_norm_,
                                        create_context_parallel_ctx,
                                        device_module, device_type, dist_max,
                                        dist_mean, get_train_context,
                                        init_distributed, set_determinism,
                                        set_pg_timeouts)
from touchnet.utils.logging import Color, init_logger, logger
from touchnet.utils.metrics import (build_device_memory_monitor,
                                    build_metric_logger,
                                    get_num_flop_per_token, get_num_params,
                                    get_peak_flops)
from touchnet.utils.profiling import (maybe_enable_memory_snapshot,
                                      maybe_enable_profiling)
from touchnet.utils.train_spec import get_train_spec


@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(tokenizer_config: TokenizerConfig, data_config: DataConfig, job_config: TrainConfig):
    torch._dynamo.config.cache_size_limit = 1024
    # torch.set_float32_matmul_precision('high')
    logger.info(f"Starting job: {job_config.training_description}")

    if job_config.training_print_args:
        logger.info(f"Running with tokenizer args: {asdict(tokenizer_config)}")
        logger.info(f"                  data args: {asdict(data_config)}")
        logger.info(f"              training args: {asdict(job_config)}")

    # take control of garbage collection to avoid stragglers
    gc_handler = GarbageCollection(gc_freq=job_config.training_gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training_data_parallel_shard_degree,
        dp_replicate=job_config.training_data_parallel_replicate_degree,
        cp=job_config.training_context_parallel_degree,
        tp=job_config.training_tensor_parallel_degree,
        pp=job_config.training_pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training_enable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_world_size, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_world_size, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    set_determinism(
        world_mesh, device, job_config.training_seed, job_config.training_deterministic
    )
    train_spec = get_train_spec(job_config.training_model_name)

    # build dataloader
    tokenizer = train_spec.build_tokenizer_fn(tokenizer_config)
    dataloader = train_spec.build_dataloader_fn(
        tokenizer_config=tokenizer_config,
        data_config=data_config,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    # build model (using meta init)
    model_cls = train_spec.model_cls
    config_cls = train_spec.config_cls
    model_config = config_cls.from_pretrained(job_config.training_model_config_path)
    model_config.return_dict = False  # NOTE: for compatibility with pipeline parallel
    assert model_config.vocab_size == tokenizer.vocab_size
    assert model_config.bos_token_id == tokenizer.bos
    assert model_config.eos_token_id == tokenizer.eos

    logger.info(
        f"Building {train_spec.name} with {model_config}"
    )
    with torch.device("meta"):
        model = model_cls.from_config(model_config)
        # NOTE: defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))

    # log model size
    model_param_count = get_num_params(model, exclude_embedding=False)
    model_param_count_wo_emb = get_num_params(model, exclude_embedding=True)
    # TODO(xcsong): support encoder-decoder flops
    num_flop_per_token = get_num_flop_per_token(
        model_param_count_wo_emb,
        model_config,
        data_config.dataset_text_seqlen,
    )
    logger.info(
        f"{Color.red}size: {model_param_count:,} total parameters ({model_param_count/1000000000.0:4f} B){Color.reset}"
    )
    logger.info(
        f"{Color.red}size (wo emb): {model_param_count_wo_emb:,}" +
        f" total parameters (wo emb) ({model_param_count_wo_emb/1000000000.0:4f} B){Color.reset}"
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.training_create_seed_ckpt:
        init_device = "cpu"
    elif job_config.training_enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        job_config.training_batchsize = data_config.dataset_batchsize
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
                # TODO(xcsong): load weight from hf? currently only support random init
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
            # TODO(xcsong): load weight from hf? currently only support random init
        model.train()

        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.training_create_seed_ckpt:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            job_config.training_enable_ckpt
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.training_ckpt_load_step)
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting training_ckpt_interval to be a multiple of training_log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(dataloader)

    train_context = get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.training_enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    # train loop
    # TODO(xcsong): support gradient accumalation steps?
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {data_config.dataset_batchsize}, "
        f"global batch size {data_config.dataset_batchsize * dp_world_size}, "
        f"sequence length {data_config.dataset_text_seqlen}, "
        f"total steps {job_config.training_steps} "
        f"(warmup {job_config.training_warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training_steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            labels, position_ids = batch["labels"], batch["position_ids"]
            inputs_embeds = batch["inputs_embeds"]
            input_ids = batch["input_ids"]
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            """
            TODO[flame]: We need to carefully handle the position_ids for TP/CP
            Depending on the Models'PE, the position_ids might be different.

            e.g. for TP
                For RoPE, all ranks have the same position_ids. [FOR HF model]
                For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

            e.g. for CP, [optional_context_parallel_ctx should automatically distbute the position_ids]
                Each rank has the coresponding chunked position_ids. [FOR All model]

            """
            labels = labels.to(device_type)
            position_ids = position_ids.to(device_type)
            cp_buffers, cp_no_restore_buffers, cp_seq_dims = [labels, position_ids], [labels, position_ids], [1, 1]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.to(device_type)
                cp_buffers.append(inputs_embeds)
                cp_no_restore_buffers.append(inputs_embeds)
                cp_seq_dims.append(1)
            if input_ids is not None:
                input_ids = input_ids.to(device_type)
                cp_buffers.append(input_ids)
                cp_no_restore_buffers.append(input_ids)
                cp_seq_dims.append(1)
            optimizers.zero_grad()

            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=cp_buffers,
                    cp_seq_dims=cp_seq_dims,
                    cp_no_restore_buffers=set(cp_no_restore_buffers),
                    cp_rotate_method=job_config.training_context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # TODO(xcsong): we should distribute the position_ids as well with CP,
                #   currently we just disable CP if PP is enabled.
                assert not parallel_dims.cp_enabled, "CP is not supported with PP"
                # Pipeline Parallel forward / backward inside step() call
                with train_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        pp_schedule.step(
                            input_ids,
                            position_ids=position_ids,
                            target=targets,
                            losses=losses
                        )
                    else:
                        pp_schedule.step(
                            position_ids=position_ids,
                            target=targets,
                            losses=losses
                        )

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    pred = model(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                    )
                    # pred.logits.shape=(bs, seq_len, vocab_size)
                    loss = train_spec.loss_fn(pred, labels)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    loss.backward()

            # clip gradients
            clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training_max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            # TODO(xcsong): skip nan norm here?
            optimizers.step()
            lr_schedulers.step()

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.training_log_freq == 0
            ):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        dist_mean(loss, world_mesh["dp_cp"]),
                        dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops
                tflops = num_flop_per_token * tps / 1e12

                time_end_to_end = time_delta / job_config.training_log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
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
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{Color.red}step: {train_state.step:2}  "
                    f"{Color.green}loss: {global_avg_loss:7.4f}  "
                    f"{Color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{Color.blue}tps: {round(tps):,}  "
                    f"{Color.cyan}tflops: {tflops:,.2f}  "
                    f"{Color.magenta}mfu: {mfu:.2f}%{Color.reset}"
                )

                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training_steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.training_train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    parser = HfArgumentParser([TokenizerConfig, DataConfig, TrainConfig])
    (tok_conf, data_conf, train_conf) = parser.parse_args_into_dataclasses()
    main(tok_conf, data_conf, train_conf)
    torch.distributed.destroy_process_group()
