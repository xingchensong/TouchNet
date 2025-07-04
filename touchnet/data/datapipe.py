# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import json
from typing import Any, Dict

import numpy
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from touchnet.data import DataConfig
from touchnet.data.dataset import TouchDataset


class LowLevelTouchDatapipe(IterableDataset, Stateful):
    """The high-level interface dataset class

        We have two shuffle stage in the Dataset. The first is global
        shuffle at list level. The second is global shuffle
        at training samples level.
    """

    def __init__(self, config: DataConfig, dp_rank: int, dp_world_size: int):
        super().__init__()
        self.lists = []
        with open(config.datalist_path, "r") as f:
            lists = f.readlines()
            for line in lists:
                line = line.strip().split()
                assert len(line) == 2
                self.lists.append(dict(dir=line[0], datatypes=line[1]))
        self.config = config
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        # Variables for checkpointing
        self.epoch = 0
        self.consumed_lists = 0
        self.consumed_samples = 0

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.epoch = state_dict["epoch"]
        self.consumed_lists = state_dict["consumed_lists"]
        self.consumed_samples = state_dict["consumed_samples"]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "consumed_lists": self.consumed_lists,
            "consumed_samples": self.consumed_samples,
        }

    def __iter__(self):
        while self.epoch < self.config.datalist_epoch:
            list_idxs = list(range(len(self.lists)))

            # 1st shuffle on lists
            if self.config.datalist_shuffling:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                list_idxs = torch.randperm(len(self.lists), generator=g).tolist()

            # 1st sharding on dp ranks
            if self.config.datalist_sharding:
                assert len(list_idxs) > self.dp_world_size, \
                    f"len(list_idxs) = {len(list_idxs)}, it should be larger than dp_world_size = {self.dp_world_size}"
                list_idxs = list_idxs[self.dp_rank::self.dp_world_size]

            # 2nd sharding on dataloader workers
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
                num_workers = 1
            else:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
            if self.config.datalist_epoch > 1:
                assert len(list_idxs) > num_workers, \
                    f"len(list_idxs) = {len(list_idxs)}, it should be larger than num_workers = {num_workers}"
            list_idxs = list_idxs[worker_id::num_workers]

            start_list = self.consumed_lists
            for list_idx in list_idxs[start_list:]:
                _dataset = TouchDataset(self.lists[list_idx]["dir"],
                                        self.config.dataset_mmap,
                                        self.lists[list_idx]["datatypes"])

                # 2nd shuffle on samples
                num_samples = len(_dataset)
                g = torch.Generator()
                g.manual_seed(self.epoch + self.consumed_lists)
                if self.config.dataset_shuffling:
                    sample_idxs = torch.randperm(num_samples, generator=g).tolist()
                else:
                    sample_idxs = list(range(num_samples))

                start_sample = self.consumed_samples
                for sample_idx in sample_idxs[start_sample:]:
                    if self.lists[list_idx]["datatypes"] == "metainfo":
                        # for text pre-training
                        metainfo = _dataset.get(sample_idx, "metainfo")
                        metainfo = metainfo.tobytes().decode('utf-8')
                        metainfo = json.loads(metainfo.strip())
                        metainfo["datatypes"] = "metainfo"
                        yield metainfo
                    elif self.lists[list_idx]["datatypes"] == "texttoken":
                        # for text pre-training
                        texttoken = _dataset.get(sample_idx, "texttoken").tolist()
                        yield dict(input_ids=texttoken, datatypes="texttoken")
                    elif self.lists[list_idx]["datatypes"] == "audio+metainfo":
                        # for audio pre-training OR audio-text alignment
                        """NOTE(xcsong) Example metainfo:
                        {
                            "key": "BAC009S0002W0122",
                            "wav": "/jfs-hdfs/Aishell/train/S0002/BAC009S0002W0122.wav",
                            "txt": "而对楼市成交抑制作用最大的限购",
                            "sample_rate": 16000,
                            "info": {
                                "segments": [
                                    {"start": 0.0, "end": 1.0, "txt": "而对"},
                                    {"start": 1.0, "end": 2.0, "txt": "楼市"},
                                    ...
                                ],
                                ...
                                "xx": "yy"
                            }
                        }
                        """
                        metainfo = _dataset.get(sample_idx, "metainfo")
                        metainfo = metainfo.tobytes().decode('utf-8')
                        metainfo = json.loads(metainfo.strip())
                        offset = 0
                        length = None
                        sample_rate = metainfo["sample_rate"]
                        info = metainfo.get("info", None)
                        if info is not None and self.config.dataset_load_audio_via_segments:
                            # used in audio sft, like asr or tts
                            segments = info.get("segments", None)
                            # TODO(xcsong): Add arg to control segment selection
                            if segments is not None:
                                g = torch.Generator()
                                g.manual_seed(self.epoch + self.consumed_lists + self.consumed_samples)
                                segment = segments[torch.randint(len(segments), (1,), generator=g).item()]
                                start = int(float(segment["start"]) * sample_rate)
                                end = int(float(segment["end"]) * sample_rate)
                                offset = start
                                length = end - start
                                metainfo['txt'] = segment['txt']
                        if self.config.dataset_random_cut_audio:
                            # used in audio pretrain
                            _, total_length = _dataset.get_idx(sample_idx, "audio")
                            min_length = self.config.dataset_random_cut_audio_min_length_in_ms / 1000.0 * sample_rate
                            max_length = self.config.dataset_random_cut_audio_max_length_in_ms / 1000.0 * sample_rate
                            assert max_length > min_length
                            if total_length > min_length:
                                g = torch.Generator()
                                g.manual_seed(self.epoch + self.consumed_lists + self.consumed_samples)
                                length = torch.randint(
                                    low=int(min_length), high=min(total_length, int(max_length)),
                                    size=(1,), generator=g).item()
                                offset = torch.randint(
                                    low=0, high=max(1, total_length - length),
                                    size=(1,), generator=g).item()
                        audio = _dataset.get(sample_idx, "audio", offset=offset, length=length)
                        audio = audio.astype(numpy.float32) / 32768.0  # normalize to [-1.0, 1.0]
                        metainfo["waveform"] = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
                        metainfo["datatypes"] = "audio+metainfo"
                        yield metainfo
                    else:
                        raise NotImplementedError(f"unsupported datatypes: {self.lists[list_idx]['datatypes']}")
                    self.consumed_samples += 1

                self.consumed_samples = 0
                self.consumed_lists += 1

            # Reset states
            self.consumed_samples = 0
            self.consumed_lists = 0
            self.epoch += 1


class MidLevelTouchDatapipe(IterableDataset, Stateful):
    """
    MidLevelTouchDatapipe class for data processing.
    """

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given functions (self.f).
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return MidLevelTouchDatapipe(self, f, *self.args, **self.kw)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        assert self.source is not None
        self.source.load_state_dict(state_dict)

    def state_dict(self) -> Dict[str, Any]:
        assert self.source is not None
        return self.source.state_dict()
