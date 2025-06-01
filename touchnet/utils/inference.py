# Copyright (c) 2025 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta

import numpy
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from touchnet.bin.make_data import load_audio


@dataclass
class InferenceConfig:
    """Configuration object for inference"""

    _argument_group_name = "inference"

    model_path: str = field(
        default=None,
        metadata={
            "help": ("hf-style model dir."),
        },
    )
    model_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": ("parameters dtype"),
            "choices": [
                "bfloat16",
                "float32",
            ],
        },
    )
    instruct: str = field(
        default="Generate the transcription, description and caption in Chinese:",
        metadata={
            "help": (""),
        },
    )
    data_list: str = field(
        default=None,
        metadata={
            "help": ("each line contains a json dict"),
        },
    )
    output_dir: str = field(
        default="./exp",
        metadata={
            "help": ("dir to save result"),
        },
    )
    batch_size: int = field(
        default=12,
        metadata={
            "help": ("batch size (per-device) for inference"),
        },
    )
    num_workers: int = field(
        default=8,
        metadata={
            "help": ("workers for dataloader"),
        },
    )
    prefetch: int = field(
        default=8,
        metadata={
            "help": ("prefetch for dataloader"),
        },
    )


class AudioDataset(Dataset):
    """
    PyTorch Dataset for loading audio files with metadata.

    Each data sample is a JSON dict containing audio file path and metadata.
    Audio files are loaded and normalized to float32 range [-1, 1].
    """

    def __init__(self, data_list):
        self.data = []

        with open(data_list, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data[idx]
        try:
            audio = load_audio(info["wav"])
            audio = audio.astype(numpy.float32) / 32768.0
            return info, audio
        except Exception as e:
            print(f"Error loading audio file {info['wav']}: {e}")
            return None, None


def collate_fn(batch):
    infos = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    return infos, audios


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Inference on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=30000)
    )
    return world_size, local_rank, rank
