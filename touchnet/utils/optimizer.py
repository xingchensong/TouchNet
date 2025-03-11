# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import math
from typing import Any, Callable, Dict, Generic, List, TypeVar, Union

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_optimizer_state_dict,
                                                     set_optimizer_state_dict)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from touchnet.bin import TrainConfig
from touchnet.utils.logging import logger

__all__ = [
    "OptimizersContainer",
    "LRSchedulersContainer",
    "build_optimizers",
    "build_lr_schedulers",
]


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: List[T]
    model_parts: List[nn.Module]

    def __init__(
        self,
        model_parts: List[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: Dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers: List[T] = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Optimizer:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(
            self.optimizers
        ), "Must pass one optimizer per model part or per param if using OptimizersInBackwardContainer"

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


def build_optimizers(
    model_parts: List[nn.Module], job_config: TrainConfig
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (TrainConfig): Job config containing the optimizer name and parameters.
    """
    name = job_config.optimizer_name
    lr = job_config.optimizer_lr

    optim_implementation = job_config.optimizer_impl
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": foreach,
    }
    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]
    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)


class LRSchedulersContainer(Stateful):
    """Container for multiple learning rate schedulers.

    This class is used to wrap multiple LRSchedulers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.lr_scheduler.LRScheduler``. The design concept is the same as
    ``OptimizersContainer``. This class currently only supports ``LambdaLR``.

    **Note**
    Users who want to customize the lr_scheduler behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same
    signature as ``torch.optim.lr_scheduler.LRScheduler`` class: ``step()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes all the lr schedulers are the same. There is no easy way to support
    resharding for multiple different LRSchedulers because LRScheduler.state_dict() is not
    resharding friendly. Therefore, the limitation is used to allow TorchTitan to support
    lr scheduler resharding.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
    """

    schedulers: List[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer, lr_lambda: Callable) -> None:
        assert (
            len(optimizers) > 0
        ), "Must have at least one optimizer to create LRScheduler"

        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> LRScheduler:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        # While there may be multiple schedulers, we only save the first one because
        # the state_dict is the same for all. See the limitations section in the
        # docstring.
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``lr_scheduler_steps`` and ``lr_scheduler_warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(copy.deepcopy(state_dict))


def build_lr_schedulers(
    optimizers: OptimizersContainer, job_config: TrainConfig
) -> LRSchedulersContainer:
    """Create a LRSchedulerContainer for the given optimizers and job config.

    This function creates a ``LRSchedulersContainer`` for the given optimizers.
    ``job_config`` should define the correct lr scheduler parameters.

    **Note**
    Users who want to customize the lr scheduler behavior can create their own
    ``LRSchedulersContainer`` subclass and ``build_lr_scheduler``. Passing the
    customized ``build_lr_schedulers`` to ``TrainSpec`` will create the customized
    ``LRSchedulersContainer``.


    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the
            lr_schedulers.
    """
    training_steps = job_config.lr_scheduler_steps
    warmup_steps = int(job_config.lr_scheduler_warmup_steps)
    lr_decay_ratio = job_config.lr_scheduler_decay_ratio
    lr_decay_type = job_config.lr_scheduler_decay_type
    lr_min = job_config.lr_scheduler_lr_min

    def linear_warmup_stable_decay(
        current_step: int,
        warmup_steps: int,
        lr_decay_ratio: Union[float, None],
        lr_decay_type: str,
        lr_min: float,
    ):
        """
        Computes linear warmup followed by stable learning rate for a while,
        then some type of decay.

        Per LambdaLR requirement, this is accomplished by returning
        a multiplicative factor `curr_adjustment` ranging from 1 to 0
        to adjust the learning rate to create the desired schedule.

        We offer three types of learning rate decay schedules:
        1. `linear`: decays linearly from 1 to 0 over the decay period.
        2. `sqrt`: decays as 1 minus the square root of the decay progress.
        3. `cosine`: follows a cosine curve, decaying according to the values of the half-period of the cosine function.

        If `lr_min` is specified, the decay range is scaled from 1 to `lr_min`
        to ensure the learning rate does not drop below this minimum value.
        """
        if lr_decay_ratio is None:
            warmup_stable_steps = warmup_steps
        else:
            warmup_stable_steps = training_steps * (1 - lr_decay_ratio)
        if warmup_stable_steps < warmup_steps:
            logger.warning(
                f"The warmup steps should be less than or equal to the warmup-stable steps ({warmup_stable_steps}). "
                f"Consider reducing either the decay ratio ({lr_decay_ratio}) or the warmup steps ({warmup_steps})."
            )
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))
        elif current_step < warmup_stable_steps:
            curr_adjustment = 1.0
        else:
            decay_steps = float(max(1, training_steps - warmup_stable_steps))
            progress = float(current_step - warmup_stable_steps) / decay_steps

            if lr_decay_type == "linear":
                curr_adjustment = 1 - progress
            elif lr_decay_type == "sqrt":
                curr_adjustment = 1 - math.sqrt(progress)
            elif lr_decay_type == "cosine":
                curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            curr_adjustment = lr_min + (1 - lr_min) * curr_adjustment
        return curr_adjustment

    lr_lambda = functools.partial(
        linear_warmup_stable_decay,
        warmup_steps=warmup_steps,
        lr_decay_ratio=lr_decay_ratio,
        lr_decay_type=lr_decay_type,
        lr_min=lr_min,
    )
    return LRSchedulersContainer(optimizers, lr_lambda)
