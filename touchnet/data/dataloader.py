# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from abc import ABC, abstractmethod
from typing import Any, Literal

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from touchnet.data import DataConfig
from touchnet.data.datapipe import (audio_and_metainfo_datapipe,
                                    texttoken_datapipe)
from touchnet.tokenizer.tokenizer import BaseTokenizer
from touchnet.utils.logging import logger


class BaseDataLoader(Stateful, ABC):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in ``Stateful``,
    ``state_dict()`` and ``load_state_dict()``.
    """

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def get_epoch(self):
        ...


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(dataset, batch_size, num_workers=num_workers,
                         pin_memory=pin_memory, prefetch_factor=prefetch_factor)
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            self._rank_id: super().state_dict(),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        super().load_state_dict(state_dict[self._rank_id])

    def get_epoch(self) -> int:
        epoch_for_every_worker = []
        states = super().state_dict()['_snapshot']['_worker_snapshots']
        for k in states.keys():
            epoch_for_every_worker.append(states[k]['dataset_state']['epoch'])
        return max(epoch_for_every_worker)


def build_dataloader(data_config: DataConfig,
                     tokenizer: BaseTokenizer,
                     dp_rank: int, dp_world_size: int,
                     split: Literal['train', 'dev', 'test']) -> BaseDataLoader:
    """Builds a dataloader."""
    data_config = copy.deepcopy(data_config)

    if split != 'train':
        data_config.datalist_shuffling = False
        data_config.dataset_shuffling = False
        data_config.audio_speed_perturb = False
        data_config.audiofeat_spec_aug = False
        data_config.audiofeat_spec_sub = False
        data_config.audiofeat_spec_trim = False
        data_config.audiofeat_dither = 0.0
        if split == 'dev':
            assert data_config.datalist_dev_path, "dev datalist path is not provided"
            data_config.datalist_sharding = False
            data_config.datalist_epoch = 1
            data_config.datalist_path = data_config.datalist_dev_path
        elif split == 'test':
            assert data_config.datalist_test_path, "test datalist path is not provided"
            data_config.datalist_epoch = 1
            data_config.datalist_path = data_config.datalist_test_path

    if data_config.datapipe_type == "texttoken":
        datapipe = texttoken_datapipe(data_config, tokenizer,
                                      dp_rank, dp_world_size)
    elif data_config.datapipe_type == "audio+metainfo":
        datapipe = audio_and_metainfo_datapipe(data_config, tokenizer,
                                               dp_rank, dp_world_size)
    else:
        raise NotImplementedError(f"Unsupported datapipe type: {data_config.datapipe_type}.")

    dataloader = ParallelAwareDataloader(
        dataset=datapipe,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=None,
        num_workers=data_config.dataloader_num_workers,
        pin_memory=True,  # TODO(xcsong): Make it configurable
        prefetch_factor=data_config.dataloader_prefetch_factor,
    )
    return dataloader
