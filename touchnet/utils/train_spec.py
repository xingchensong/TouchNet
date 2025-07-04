# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Type

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining.schedules import _PipelineSchedule
from transformers import AutoConfig

from touchnet.bin import TrainConfig
from touchnet.data.dataloader import BaseDataLoader
from touchnet.tokenizer.tokenizer import BaseTokenizer
from touchnet.utils.distributed import ParallelDims
from touchnet.utils.metrics import MetricsProcessor
from touchnet.utils.optimizer import LRSchedulersContainer, OptimizersContainer


@dataclass
class TrainSpec:
    name: str
    model_cls: Type[nn.Module]
    config_cls: Type[AutoConfig]
    parallelize_fn: Callable[[nn.Module, DeviceMesh, ParallelDims, TrainConfig], None]
    pipelining_fn: Callable[
        ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
    ]
    build_optimizers_fn: Callable[[list[nn.Module], TrainConfig], OptimizersContainer]
    build_lr_schedulers_fn: Callable[[OptimizersContainer, TrainConfig], LRSchedulersContainer]
    build_dataloader_fn: Callable[..., BaseDataLoader]
    build_tokenizer_fn: Callable[..., BaseTokenizer]
    loss_fn: Callable[..., torch.Tensor]
    acc_fn: Callable[..., torch.Tensor]
    additional_pre_init_fn: Callable[[TrainConfig], None]
    additional_post_init_fn: Callable[[nn.Module, torch.device], None]
    get_num_flop_per_token_fn: Callable[[int, AutoConfig, int], int]
    get_num_params_fn: Callable[..., int]
    build_metrics_processor_fn: Callable[..., MetricsProcessor]


_train_specs = {}


def register_train_spec(train_spec: TrainSpec) -> None:
    global _train_specs
    if train_spec.name in _train_specs:
        raise ValueError(f"Model {train_spec.name} is already registered.")

    _train_specs[train_spec.name] = train_spec


def get_train_spec(name: str) -> TrainSpec:
    global _train_specs
    if name not in _train_specs:
        raise ValueError(f"Model {name} is not registered.")
    return _train_specs[name]


def apply_to_train_specs(func: Callable[[TrainSpec], TrainSpec]) -> None:
    global _train_specs
    for name, train_spec in _train_specs.items():
        _train_specs[name] = func(train_spec)
