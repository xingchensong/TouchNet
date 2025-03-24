# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from dataclasses import asdict
from datetime import timedelta
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.hf_argparser import HfArgumentParser

from touchnet.bin import TrainConfig
from touchnet.data import DataConfig
from touchnet.data.dataloader import BaseDataLoader
from touchnet.tokenizer import TokenizerConfig
from touchnet.utils.checkpoint import CheckpointManager
from touchnet.utils.distributed import (GarbageCollection, ParallelDims,
                                        clip_grad_norm_,
                                        create_context_parallel_ctx,
                                        device_module, device_type, dist_max,
                                        dist_mean, dist_sum, get_train_context,
                                        init_distributed, set_determinism,
                                        set_pg_timeouts)
from touchnet.utils.logging import init_logger, logger
from touchnet.utils.metrics import (MetricsProcessor, ensure_pp_loss_visible,
                                    get_num_flop_per_token, get_num_params)
from touchnet.utils.optimizer import LRSchedulersContainer, OptimizersContainer
from touchnet.utils.profiling import (maybe_enable_memory_snapshot,
                                      maybe_enable_profiling)
from touchnet.utils.train_spec import TrainSpec, get_train_spec


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    job_config: TrainConfig
    tokenizer_config: TokenizerConfig
    data_config: DataConfig
    gc_handler: GarbageCollection

    parallel_dims: ParallelDims
    train_spec: TrainSpec
    world_mesh: DeviceMesh

    dataloader: BaseDataLoader
    dataloader_dev: BaseDataLoader
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    model_parts: list[torch.nn.Module]
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer

    pp_has_first_stage: bool
    pp_has_last_stage: bool

    device: torch.device

    # states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, tokenizer_config: TokenizerConfig, data_config: DataConfig, job_config: TrainConfig):
        self.job_config = job_config
        self.tokenizer_config = tokenizer_config
        self.data_config = data_config

        torch._dynamo.config.cache_size_limit = 4096
        logger.info(f"Starting job: {job_config.training_description}")

        if job_config.training_print_args:
            logger.info(f"Running with tokenizer args: {asdict(tokenizer_config)}")
            logger.info(f"                  data args: {asdict(data_config)}")
            logger.info(f"              training args: {asdict(job_config)}")

        # take control of garbage collection to avoid stragglers
        self.gc_handler = GarbageCollection(gc_freq=job_config.training_gc_freq)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])
        self.parallel_dims = ParallelDims(
            dp_shard=job_config.training_data_parallel_shard_degree,
            dp_replicate=job_config.training_data_parallel_replicate_degree,
            cp=job_config.training_context_parallel_degree,
            tp=job_config.training_tensor_parallel_degree,
            pp=job_config.training_pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=job_config.training_enable_loss_parallel,
        )
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)
        init_distributed(job_config)

        # build meshes
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            dp_world_size, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_world_size, dp_rank = 1, 0

        # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
        set_determinism(
            self.world_mesh, self.device,
            job_config.training_seed, job_config.training_deterministic
        )
        self.train_spec = get_train_spec(job_config.training_model_name)

        # build dataloader
        tokenizer = self.train_spec.build_tokenizer_fn(tokenizer_config)
        self.dataloader = self.train_spec.build_dataloader_fn(
            tokenizer_config=tokenizer_config,
            data_config=data_config,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            split='train',
        )
        self.dataloader_dev = self.train_spec.build_dataloader_fn(
            tokenizer_config=tokenizer_config,
            data_config=data_config,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            split='dev',
        )

        # build model (using meta init)
        model_cls = self.train_spec.model_cls
        config_cls = self.train_spec.config_cls
        model_config = config_cls.from_pretrained(job_config.training_model_config_path,
                                                  attn_implementation="flex_attention")
        model_config.return_dict = False  # NOTE: for compatibility with pipeline parallel
        self.use_flex_attention = model_config._attn_implementation == "flex_attention"
        if self.use_flex_attention:
            job_config.training_compile = False  # TODO(xcsong): support flex_attention with torch.compile
        assert model_config.vocab_size == tokenizer.vocab_size
        assert model_config.bos_token_id == tokenizer.bos
        assert model_config.eos_token_id == tokenizer.eos

        logger.info(
            f"Building {self.train_spec.name} with {model_config}"
        )
        logger.info(
            f"Attention: {model_config._attn_implementation}"
        )
        with torch.device("meta"):
            model = model_cls.from_config(model_config)
            # NOTE: defer weight initialization until after parallelisms are applied
            model.apply(lambda m: setattr(m, "_is_hf_initialized", False))

        # metrics logging
        self.metrics_processor = self.train_spec.build_metrics_processor_fn(
            job_config, self.parallel_dims)
        color = self.metrics_processor.color

        # log model size
        model_param_count = get_num_params(model, exclude_embedding=False)
        model_param_count_wo_emb = get_num_params(model, exclude_embedding=True)
        # TODO(xcsong): support encoder-decoder flops
        self.metrics_processor.num_flop_per_token = get_num_flop_per_token(
            model_param_count_wo_emb,
            model_config,
            data_config.dataset_text_seqlen,
        )
        logger.info(
            f"{color.red}size: {model_param_count:,} total parameters ({model_param_count/1000000000.0:4f} B){color.reset}"
        )
        logger.info(
            f"{color.red}size (wo emb): {model_param_count_wo_emb:,}" +
            f" total parameters (wo emb) ({model_param_count_wo_emb/1000000000.0:4f} B){color.reset}"
        )
        logger.info(model)

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.training_create_seed_ckpt:
            init_device = "cpu"
            assert (
                world_size == 1
            ), "Must create seed checkpoint using a single device, to disable sharding"
            assert (
                job_config.training_enable_ckpt
            ), "Must enable checkpointing when creating a seed checkpoint"

            model.to_empty(device=init_device)
            if job_config.training_model_pretrained_weight_dir:
                logger.info(f"Load pretrained weight from {job_config.training_model_pretrained_weight_dir}")
                # TODO(xcsong): tie_word_emb=False? (this is required for pp)
                with torch.no_grad():
                    model = model.from_pretrained(job_config.training_model_pretrained_weight_dir,
                                                  config=model_config)
            else:
                logger.info("Random initilize seed checkpoint.")
                with torch.no_grad():
                    model.post_init()
                    self.train_spec.additional_post_init_fn(model, init_device)
            folder = os.path.join(job_config.training_trace_dump_folder,
                                  job_config.training_ckpt_folder,
                                  "step-0")
            os.makedirs(folder, exist_ok=True)
            dcp.save({"model": model.state_dict()}, checkpoint_id=folder)
            self.checkpointer = None
            logger.info("Created seed checkpoint")
            return
        elif job_config.training_enable_cpu_offload:
            init_device = "cpu"
        else:
            init_device = device_type

        # apply parallelisms and initialization
        if self.parallel_dims.pp_enabled:
            assert not model_config.tie_word_embeddings, "TODO(xcsong): PP supports tied embeddings"
            job_config.training_batchsize = data_config.dataset_batchsize

            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                self.world_mesh,
                self.parallel_dims,
                job_config,
                self.device,
                model_config,
                self.train_spec.parallelize_fn,
                self.train_spec.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.post_init()
                    self.train_spec.additional_post_init_fn(m, init_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(self.parallel_dims, job_config, color)
        else:
            """
            NOTE(xcsong): apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            NOTE(xcsong): Before parallelizing the model, `model.lm_head` looks like:
                tensor(..., device='meta', size=(vocab_size, hidden), dtype=torch.bfloat16, requires_grad=True)
            """
            model = self.train_spec.parallelize_fn(
                model, self.world_mesh, self.parallel_dims, job_config
            )
            """
            NOTE(xcsong): Assume dp=2 cp=4, after parallelizing the model, `model.lm_head` looks like:
                DTensor(local_tensor=tensor(..., device='meta', size=(vocab_size // dp // cp, hidden),
                                            dtype=torch.bfloat16),
                        device_mesh=DeviceMesh('cuda', [0, 1, 2, 3, 4, 5, 6, 7],
                        mesh_dim_names=('dp_shard_cp',)),
                        placements=(Shard(dim=0),))
            NOTE(xcsong): Assume dp=2, cp=2, tp=2, after parallelizing the model, `model.lm_head` looks like:
                DTensor(local_tensor=tensor(..., device='meta', size=(vocab_size // dp // cp // tp, hidden),
                                            dtype=torch.bfloat16),
                        device_mesh=DeviceMesh('cuda', [[0, 1], [2, 3], [4, 5], [6, 7]],
                        mesh_dim_names=('dp_shard_cp', 'tp')),
                        placements=(_StridedShard(dim=0, sf=2), Shard(dim=0)))
            """
            model.to_empty(device=init_device)
            """
            NOTE(xcsong): to_empty() will allocate real memories on init_device for all meta-tensors,
                Without post_init(), the values of these tensors are undefined, and there may be NaNs.
                We also need to pay attention to the correct initialization of non-persistent buffers (such as rope).
            """
            with torch.no_grad():
                model.post_init()
                self.train_spec.additional_post_init_fn(model, init_device)
            model.train()

            self.model_parts = [model]

        logger.info(f"Peak FLOPS used for computing MFU: {self.metrics_processor.gpu_peak_flops:.3e}")
        device_mem_stats = self.metrics_processor.device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(self.model_parts, job_config)
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(self.optimizers, job_config)
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.lr_schedulers = self.lr_schedulers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            job_config=job_config,
        )

        self.checkpointer.load(step=job_config.training_ckpt_load_step)

        self.train_context = get_train_context(
            self.parallel_dims.loss_parallel_enabled,
            job_config.training_enable_compiled_autograd,
        )

        # train loop
        # TODO(xcsong): support gradient accumalation steps?
        logger.info(
            "Trainer initialized. "
            f"Training starts at step {self.step + 1}, "
            f"with local batch size {data_config.dataset_batchsize}, "
            f"global batch size {data_config.dataset_batchsize * dp_world_size}, "
            f"sequence length {data_config.dataset_text_seqlen}, "
            f"total steps {job_config.lr_scheduler_steps} "
            f"(warmup {job_config.lr_scheduler_warmup_steps})"
        )

    def next_batch(self, data_iterator: Iterable, perf: bool = True) -> Dict[str, Any]:
        if perf:
            data_load_start = time.perf_counter()
        batch = next(data_iterator)
        if not batch:
            return None
        labels, position_ids = batch["labels"], batch["position_ids"]
        inputs_embeds = batch["inputs_embeds"]
        input_ids = batch["input_ids"]
        sentence_ids = batch["sentence_ids"]
        sentence_lens = batch["sentence_lens"]
        num_sentence = batch["num_sentence"]
        if self.parallel_dims.dp_enabled:
            # NOTE(xcsong): we use global-batch, so we need to sum num_sentence
            num_sentence = torch.tensor(num_sentence, dtype=torch.int64).to(device_type)
            num_sentence = dist_sum(num_sentence, mesh=self.world_mesh["dp"])
        if perf:
            self.metrics_processor.ntokens_since_last_log += labels.numel()
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

        """
        NOTE(flame): We need to carefully handle the position_ids for TP/CP
        Depending on the Models'PE, the position_ids might be different.

        e.g. for TP
            For RoPE, all ranks have the same position_ids. [FOR HF model]
            For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

        e.g. for CP, [optional_context_parallel_ctx should automatically distbute the position_ids]
            Each rank has the coresponding chunked position_ids. [FOR All model]

        """
        labels = labels.to(device_type)
        position_ids = position_ids.to(device_type)
        sentence_ids = sentence_ids.to(device_type)
        sentence_lens = sentence_lens.to(device_type)

        cp_buffers = [labels, position_ids, sentence_ids, sentence_lens]
        cp_no_restore_buffers = [labels, position_ids, sentence_ids, sentence_lens]
        cp_seq_dims = [1, 1, 1, 1]
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

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        optional_context_parallel_ctx = (
            create_context_parallel_ctx(
                cp_mesh=self.world_mesh["cp"],
                cp_buffers=cp_buffers,
                cp_seq_dims=cp_seq_dims,
                cp_no_restore_buffers=set(cp_no_restore_buffers),
                cp_rotate_method=self.job_config.training_context_parallel_rotate_method,
            )
            if self.parallel_dims.cp_enabled
            else None
        )

        return {
            "input_ids": input_ids,          # [bs, seq_len]
            "inputs_embeds": inputs_embeds,  # [bs, seq_len, dim]
            "sentence_ids": sentence_ids,    # [bs, seq_len]
            "sentence_lens": sentence_lens,  # [bs, seq_len]
            "labels": labels,                # [bs, seq_len]
            "position_ids": position_ids,    # [bs, seq_len]
            "num_sentence": num_sentence,    # [1,]
            "optional_context_parallel_ctx": optional_context_parallel_ctx,
        }

    def train_step(self, data: Dict[str, Any]):
        self.optimizers.zero_grad()

        if self.parallel_dims.pp_enabled:
            # TODO(xcsong): we should distribute the position_ids as well with CP,
            #   currently we just disable CP if PP is enabled.
            assert not self.parallel_dims.cp_enabled, "CP is not supported with PP"
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(data["optional_context_parallel_ctx"]):
                targets, losses = (data["labels"], []) if self.pp_has_last_stage else (None, None)
                if self.pp_has_first_stage:
                    assert data["inputs_embeds"] is None, "TODO(xcsong): PP supports inputs_embeds"
                    self.pp_schedule.step(
                        data["input_ids"],
                        position_ids=data["position_ids"],
                        attention_mask=data["sentence_ids"] if self.use_flex_attention else None,
                        target=targets,
                        losses=losses
                    )
                else:
                    self.pp_schedule.step(
                        position_ids=data["position_ids"],
                        attention_mask=data["sentence_ids"] if self.use_flex_attention else None,
                        target=targets,
                        losses=losses
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            # TODO(xcsong): pack loss for pp
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(data["optional_context_parallel_ctx"]):
                assert len(self.model_parts) == 1
                pred = self.model_parts[0](
                    input_ids=data["input_ids"],
                    inputs_embeds=data["inputs_embeds"],
                    position_ids=data["position_ids"],
                    attention_mask=data["sentence_ids"] if self.use_flex_attention else None,
                )
                # logits.shape = pred[0].shape = (bs, seq_len // cp, vocab_size // tp)
                loss_per_sample, loss_per_token = self.train_spec.loss_fn(
                    pred, data["labels"], data["sentence_lens"], data["num_sentence"])  # (1,), (1,)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss_per_sample.backward()

        # clip gradients
        norm = clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training_max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if self.parallel_dims.pp_enabled else None,
        )

        # optimizer step
        self.checkpointer.maybe_wait_for_staging()
        if norm.isnan() or norm.isinf():
            logger.warning(
                f"Skipping optimizer step - detected invalid gradient norm: {norm:.4f}"
            )
            self.optimizers.zero_grad()
        else:
            self.optimizers.step()
        self.lr_schedulers.step()

        # log metrics
        if self.metrics_processor.should_log(self.step):
            if (
                self.parallel_dims.dp_replicate_enabled
                or self.parallel_dims.dp_shard_enabled
                or self.parallel_dims.cp_enabled
            ):
                loss_per_sample = loss_per_sample.detach()
                loss_per_token = loss_per_token.detach()
                global_avg_loss_per_sample = dist_sum(loss_per_sample, self.world_mesh["dp_cp"])
                global_avg_loss_per_token, global_max_loss_per_token = (
                    dist_mean(loss_per_token, self.world_mesh["dp_cp"]),
                    dist_max(loss_per_token, self.world_mesh["dp_cp"]),
                )
            else:
                raise NotImplementedError("TODO: support other parallelisms")

            self.metrics_processor.log(
                self.step, global_avg_loss_per_sample,
                global_avg_loss_per_token, global_max_loss_per_token,
                norm.mean().detach().item(), norm.max().detach().item(),
            )

    @record
    def train(self):
        with maybe_enable_profiling(
            self.job_config, global_step=self.step
        ) as torch_profiler, maybe_enable_memory_snapshot(
            self.job_config, global_step=self.step
        ) as memory_profiler:
            data_iterator = iter(self.dataloader)
            while self.step < self.job_config.lr_scheduler_steps:
                self.step += 1
                self.gc_handler.run(self.step)

                data = self.next_batch(data_iterator)
                if not data:
                    break
                self.train_step(data)

                self.checkpointer.save(
                    self.step, force=(self.step == self.job_config.lr_scheduler_steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                if self.checkpointer._should_save(self.step):
                    self.dev()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    set_pg_timeouts(
                        timeout=timedelta(seconds=self.job_config.training_train_timeout_seconds),
                        world_mesh=self.world_mesh,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Training completed")

    @torch.no_grad()
    def dev_step(self, data: Dict[str, Any]):
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("TODO(xcsong): support PP.")
        else:
            # Non-PP forward
            with self.train_context(data["optional_context_parallel_ctx"]):
                assert len(self.model_parts) == 1
                pred = self.model_parts[0](
                    input_ids=data["input_ids"],
                    inputs_embeds=data["inputs_embeds"],
                    position_ids=data["position_ids"],
                    attention_mask=data["sentence_ids"] if self.use_flex_attention else None,
                )
                # logits.shape = pred[0].shape = (bs, seq_len // cp, vocab_size // tp)
                loss_per_sample, loss_per_token = self.train_spec.loss_fn(
                    pred, data["labels"], data["sentence_lens"], data["num_sentence"])  # (1,), (1,)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss_per_sample = loss_per_sample.detach()
                loss_per_token = loss_per_token.detach()
                global_avg_loss_per_sample = dist_sum(loss_per_sample, self.world_mesh["dp_cp"])
                global_avg_loss_per_token, global_max_loss_per_token = (
                    dist_mean(loss_per_token, self.world_mesh["dp_cp"]),
                    dist_max(loss_per_token, self.world_mesh["dp_cp"]),
                )

        return (global_avg_loss_per_sample, global_avg_loss_per_token, global_max_loss_per_token)

    @record
    @torch.no_grad()
    def dev(self):
        for model in self.model_parts:
            model.eval()

        losses = []
        data_iterator = iter(self.dataloader_dev)

        data = self.next_batch(data_iterator, perf=False)
        while data:
            losses.append(self.dev_step(data))
            data = self.next_batch(data_iterator, perf=False)

        self.metrics_processor.log_dev(
            step=self.step,
            global_avg_loss_per_sample=sum([loss_tuple[0] for loss_tuple in losses]) / len(losses),
            global_avg_loss_per_token=sum([loss_tuple[1] for loss_tuple in losses]) / len(losses),
            global_max_loss_per_token=max([loss_tuple[2] for loss_tuple in losses]),
        )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        for model in self.model_parts:
            model.train()
        logger.info("Dev completed")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()


if __name__ == "__main__":
    init_logger()
    parser = HfArgumentParser([TokenizerConfig, DataConfig, TrainConfig])
    (tok_conf, data_conf, train_conf) = parser.parse_args_into_dataclasses()
    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(tok_conf, data_conf, train_conf)
        if not train_conf.training_create_seed_ckpt:
            trainer.train()
    finally:
        if trainer:
            trainer.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
