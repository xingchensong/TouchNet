# -*- coding: utf-8 -*-
# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)
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

import torch

from touchnet.data import DataConfig, functions
from touchnet.data.datapipe import LowLevelTouchDatapipe, MidLevelTouchDatapipe
from touchnet.tokenizer.tokenizer import BaseTokenizer


def batch_text(data, config: DataConfig, tokenizer: BaseTokenizer):
    """ Feeding the data into buffer for training.
        We generate attention_mask inside Model.forward().
        Memory assumption:
            dataset_batchsize = 16
            dataset_text_seqlen = 131072 (2^17)
        Then we got:
            input_ids: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
            labels: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
            position_ids: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
                example position_ids for packed sequence:
                    [[0, 1, 2, 0, 1, 2, 0],
                     [0, 1, 0, 1, 2, 0, 1]]
            attention_mask: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
                example attention_mask for packed sequence: start from 1, ignore_idx == 0
                    [[1, 1, 1, 2, 2, 2, 0],
                     [1, 1, 2, 2, 2, 3, 3]]
            sentence_lens: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
                example sentence_lens for packed sequence:
                    [[3, 3, 3, 3, 3, 3, 0],
                     [2, 2, 3, 3, 3, 2, 2]]
        So the total memory cost is around 80MB.
    """
    buffer = {
        "input_ids": torch.zeros([config.dataset_batchsize,
                                  config.dataset_text_seqlen], dtype=torch.int64) + tokenizer.pad,
        "inputs_embeds": None,
        "labels": torch.zeros([config.dataset_batchsize,
                               config.dataset_text_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
        "position_ids": torch.zeros([config.dataset_batchsize,
                                     config.dataset_text_seqlen], dtype=torch.int64),
        "attention_mask": torch.zeros([config.dataset_batchsize,
                                       config.dataset_text_seqlen], dtype=torch.int64),
        "sentence_lens": torch.ones([config.dataset_batchsize,
                                     config.dataset_text_seqlen], dtype=torch.int64),
        "num_sentence": 0,
    }
    cur_batch_idx = 0
    cur_text_idx = 0
    cur_sentence_idx = 1
    for sample in data:
        text_len = len(sample['input_ids']) + 1  # +1 for sos/eos
        if cur_batch_idx == config.dataset_batchsize - 1:
            if cur_text_idx + text_len > config.dataset_text_seqlen:
                yield buffer
                # reset buffer for next batch
                buffer = {
                    "input_ids": torch.zeros([config.dataset_batchsize,
                                              config.dataset_text_seqlen], dtype=torch.int64) + tokenizer.pad,
                    "inputs_embeds": None,
                    "labels": torch.zeros([config.dataset_batchsize,
                                           config.dataset_text_seqlen], dtype=torch.int64) - 100,
                    "position_ids": torch.zeros([config.dataset_batchsize,
                                                 config.dataset_text_seqlen], dtype=torch.int64),
                    "attention_mask": torch.zeros([config.dataset_batchsize,
                                                   config.dataset_text_seqlen], dtype=torch.int64),
                    "sentence_lens": torch.ones([config.dataset_batchsize,
                                                 config.dataset_text_seqlen], dtype=torch.int64),
                    "num_sentence": 0,
                }
                cur_batch_idx = 0
                cur_text_idx = 0
                cur_sentence_idx = 1
        else:
            if cur_text_idx + text_len > config.dataset_text_seqlen:
                cur_batch_idx += 1
                cur_text_idx = 0
                cur_sentence_idx = 1
        buffer["input_ids"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = \
            torch.tensor([tokenizer.bos] + sample['input_ids'], dtype=torch.int64)
        buffer["labels"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = \
            torch.tensor(sample['input_ids'] + [tokenizer.eos], dtype=torch.int64)
        buffer["position_ids"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = torch.arange(
            0, text_len, dtype=torch.int64)
        buffer["attention_mask"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = cur_sentence_idx
        buffer["sentence_lens"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = text_len
        buffer["num_sentence"] += 1
        cur_text_idx += text_len
        cur_sentence_idx += 1
    if (not config.dataloader_drop_last_batch) and (cur_text_idx > 0 or cur_batch_idx > 0):
        yield buffer


def causal_lm_datapipe(
    data_config: DataConfig,
    tokenizer: BaseTokenizer,
    dp_rank: int, dp_world_size: int,
):
    """Construct datapipe for causal language modeling.

    Args:
        data_config: Configuration for data processing
        tokenizer: Tokenizer for text processing
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size

    Returns:
        MidLevelTouchDatapipe: Configured datapipe for causal LM training
    """
    datapipe = LowLevelTouchDatapipe(data_config, dp_rank, dp_world_size)

    datapipe = MidLevelTouchDatapipe(datapipe, functions.filter_samples, data_config)
    datapipe = MidLevelTouchDatapipe(datapipe, batch_text, data_config, tokenizer)
    return datapipe
