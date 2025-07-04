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
from torch.nn.utils.rnn import pad_sequence

from touchnet.data import DataConfig, functions
from touchnet.data.datapipe import LowLevelTouchDatapipe, MidLevelTouchDatapipe
from touchnet.tokenizer.tokenizer import BaseTokenizer, BestRQTokenizer


def batch_audio_packed(data, config: DataConfig, tokenizer: BestRQTokenizer):
    """ Feeding the data into buffer for training.
        We generate attention_mask inside Model.forward().
        Memory assumption:
            dataset_batchsize = 16
            dataset_audio_seqlen = 131072 (2^17)
            audiofeat_stack_length = 4
            audiofeat_num_mel_bins = 128
        Then we got:
            input_features: [16, 131072, 128 * 4] = 16 * 131072 * 512 * 4Bytes / 1024 / 1024 = 4096MB
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
        So the total memory cost is around 4140MB.
    """
    buffer = {
        "input_ids": None,
        "input_features": torch.zeros([config.dataset_batchsize, config.dataset_audio_seqlen,
                                      config.audiofeat_num_mel_bins * config.audiofeat_stack_length],
                                      dtype=torch.float32),
        "labels": torch.zeros([config.dataset_batchsize,
                               config.dataset_audio_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
        "position_ids": torch.zeros([config.dataset_batchsize,
                                     config.dataset_audio_seqlen], dtype=torch.int64),
        "attention_mask": torch.zeros([config.dataset_batchsize,
                                       config.dataset_audio_seqlen], dtype=torch.int64),
        "sentence_lens": torch.ones([config.dataset_batchsize,
                                     config.dataset_audio_seqlen], dtype=torch.int64),
        "num_sentence": 0,
    }
    cur_batch_idx = 0
    cur_audio_idx = 0
    cur_sentence_idx = 1
    for sample in data:
        audio_len = sample['audiofeat'].size(0)
        if audio_len > config.dataset_audio_seqlen:
            continue
        if cur_batch_idx == config.dataset_batchsize - 1:
            if cur_audio_idx + audio_len > config.dataset_audio_seqlen:
                buffer['shift_labels'] = buffer['labels']
                yield buffer
                # reset buffer for next batch
                buffer = {
                    "input_ids": None,
                    "input_features": torch.zeros([config.dataset_batchsize, config.dataset_audio_seqlen,
                                                  config.audiofeat_num_mel_bins * config.audiofeat_stack_length],
                                                  dtype=torch.float32),
                    "labels": torch.zeros([config.dataset_batchsize,
                                           config.dataset_audio_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
                    "position_ids": torch.zeros([config.dataset_batchsize,
                                                 config.dataset_audio_seqlen], dtype=torch.int64),
                    "attention_mask": torch.zeros([config.dataset_batchsize,
                                                   config.dataset_audio_seqlen], dtype=torch.int64),
                    "sentence_lens": torch.ones([config.dataset_batchsize,
                                                 config.dataset_audio_seqlen], dtype=torch.int64),
                    "num_sentence": 0,
                }
                cur_batch_idx = 0
                cur_audio_idx = 0
                cur_sentence_idx = 1
        else:
            if cur_audio_idx + audio_len > config.dataset_audio_seqlen:
                cur_batch_idx += 1
                cur_audio_idx = 0
                cur_sentence_idx = 1
        labels = tokenizer.tokenize(sample['audiofeat'])
        assert len(labels) == audio_len
        buffer["input_features"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = sample['audiofeat']
        buffer["labels"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = \
            torch.tensor(labels[1:] + [-100], dtype=torch.int64)  # just ignore the last output
        buffer["position_ids"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = torch.arange(
            0, audio_len, dtype=torch.int64)
        buffer["attention_mask"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = cur_sentence_idx
        buffer["sentence_lens"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = audio_len
        buffer["num_sentence"] += 1
        cur_audio_idx += audio_len
        cur_sentence_idx += 1
    if (not config.dataloader_drop_last_batch) and (cur_batch_idx > 0 or cur_audio_idx > 0):
        buffer['shift_labels'] = buffer['labels']
        yield buffer


def batch_pairaudio_pairtext_packed(data, config: DataConfig, tokenizer: BaseTokenizer):
    """ Feeding the data into buffer for training.
        We generate attention_mask inside Model.forward().
        Memory assumption:
            dataset_batchsize = 16
            dataset_audio_seqlen = 131072 (2^17)
            dataset_text_seqlen = 131072 (2^17)
            audiofeat_stack_length = 4
            audiofeat_num_mel_bins = 128
        Then we got:
            input_features: [16, 131072, 128 * 4] = 16 * 131072 * 512 * 4Bytes / 1024 / 1024 = 4096MB
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
        So the total memory cost is around 4176MB.
    """
    # TODO(xcsong): merge audio_seqlen & text_seqlen if force-delay works.
    assert config.dataset_audio_seqlen == config.dataset_text_seqlen
    buffer = {
        "input_ids": torch.zeros([config.dataset_batchsize,
                                  config.dataset_text_seqlen], dtype=torch.int64) + tokenizer.pad,
        "input_features": torch.zeros([config.dataset_batchsize, config.dataset_audio_seqlen,
                                      config.audiofeat_num_mel_bins * config.audiofeat_stack_length],
                                      dtype=torch.float32),
        "labels": torch.zeros([config.dataset_batchsize,
                               config.dataset_text_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
        "position_ids": torch.zeros([config.dataset_batchsize,
                                     config.dataset_audio_seqlen], dtype=torch.int64),
        "attention_mask": torch.zeros([config.dataset_batchsize,
                                       config.dataset_audio_seqlen], dtype=torch.int64),
        "sentence_lens": torch.ones([config.dataset_batchsize,
                                     config.dataset_audio_seqlen], dtype=torch.int64),
        "num_sentence": 0,
    }
    cur_batch_idx = 0
    cur_audio_idx = 0
    cur_sentence_idx = 1
    for sample in data:
        audio_len = sample['audiofeat'].size(0)
        text_len = len(sample['input_ids']) + 1  # +1 for sos/eos
        total_len = audio_len + text_len
        if total_len > config.dataset_audio_seqlen:
            continue
        if cur_batch_idx == config.dataset_batchsize - 1:
            if cur_audio_idx + total_len > config.dataset_audio_seqlen:
                buffer['shift_labels'] = buffer['labels']
                yield buffer
                # reset buffer for next batch
                buffer = {
                    "input_ids": torch.zeros([config.dataset_batchsize,
                                              config.dataset_text_seqlen], dtype=torch.int64) + tokenizer.pad,
                    "input_features": torch.zeros([config.dataset_batchsize, config.dataset_audio_seqlen,
                                                  config.audiofeat_num_mel_bins * config.audiofeat_stack_length],
                                                  dtype=torch.float32),
                    "labels": torch.zeros([config.dataset_batchsize,
                                           config.dataset_text_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
                    "position_ids": torch.zeros([config.dataset_batchsize,
                                                 config.dataset_audio_seqlen], dtype=torch.int64),
                    "attention_mask": torch.zeros([config.dataset_batchsize,
                                                   config.dataset_audio_seqlen], dtype=torch.int64),
                    "sentence_lens": torch.ones([config.dataset_batchsize,
                                                 config.dataset_audio_seqlen], dtype=torch.int64),
                    "num_sentence": 0,
                }
                cur_batch_idx = 0
                cur_audio_idx = 0
                cur_sentence_idx = 1
        else:
            if cur_audio_idx + total_len > config.dataset_audio_seqlen:
                cur_batch_idx += 1
                cur_audio_idx = 0
                cur_sentence_idx = 1
        buffer["input_features"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = sample['audiofeat']
        buffer["input_ids"][cur_batch_idx, cur_audio_idx + total_len - text_len:cur_audio_idx + total_len] = \
            torch.tensor([tokenizer.bos] + sample['input_ids'], dtype=torch.int64)
        buffer["labels"][cur_batch_idx, cur_audio_idx + total_len - text_len:cur_audio_idx + total_len] = \
            torch.tensor(sample['input_ids'] + [tokenizer.eos], dtype=torch.int64)
        buffer["position_ids"][cur_batch_idx, cur_audio_idx:cur_audio_idx + total_len] = torch.arange(
            0, total_len, dtype=torch.int64)
        buffer["attention_mask"][cur_batch_idx, cur_audio_idx:cur_audio_idx + total_len] = cur_sentence_idx
        buffer["sentence_lens"][cur_batch_idx, cur_audio_idx:cur_audio_idx + total_len] = text_len
        buffer["num_sentence"] += 1
        cur_audio_idx += total_len
        cur_sentence_idx += 1
    if (not config.dataloader_drop_last_batch) and (cur_batch_idx > 0 or cur_audio_idx > 0):
        buffer['shift_labels'] = buffer['labels']
        yield buffer


def batch_audio(data, config: DataConfig, tokenizer: BestRQTokenizer):
    """ Non-packed version of batch_audio_packed.
        Uses dynamic batching with pad_sequence for variable length sequences.

        Args:
            data: Iterator of audio samples with 'audiofeat' key
            config: DataConfig containing batch size and other parameters
            tokenizer: BestRQTokenizer for audio tokenization

        Yields:
            Dict containing batched tensors with proper padding
    """
    input_features_buf = []
    labels_buf = []
    sentence_lens_buf = []

    max_len_in_buffer = 0
    for sample in data:
        audio_len = sample['audiofeat'].size(0)
        max_len_in_buffer = max(max_len_in_buffer, audio_len)

        # Skip samples that are too long
        if audio_len > config.dataset_audio_seqlen:
            continue

        # Tokenize audio features
        labels = tokenizer.tokenize(sample['audiofeat'])
        assert len(labels) == audio_len

        # Prepare tensors for this sample
        input_features = sample['audiofeat']  # [audio_len, feat_dim]
        labels_tensor = torch.tensor(labels[1:] + [-100], dtype=torch.int64)  # ignore last output
        sentence_lens = torch.ones(audio_len, dtype=torch.int64) * audio_len

        # Yield batch when we have enough samples
        if (len(input_features_buf) + 1) * max_len_in_buffer > (config.dataset_batchsize * config.dataset_audio_seqlen):
            yield {
                "input_ids": None,
                "input_features": pad_sequence(input_features_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=0.0),
                "labels": pad_sequence(labels_buf,
                                       batch_first=True,
                                       padding_side='right',
                                       padding_value=-100),
                "shift_labels": pad_sequence(labels_buf,
                                             batch_first=True,
                                             padding_side='right',
                                             padding_value=-100),
                "position_ids": None,  # Will be generated in model forward
                "attention_mask": None,  # Will be generated in model forward
                "sentence_lens": pad_sequence(sentence_lens_buf,
                                              batch_first=True,
                                              padding_side='right',
                                              padding_value=1),
                "num_sentence": len(input_features_buf),
            }
            # Reset buffers
            input_features_buf = [input_features]
            labels_buf = [labels_tensor]
            sentence_lens_buf = [sentence_lens]
            max_len_in_buffer = audio_len
        else:
            input_features_buf.append(input_features)
            labels_buf.append(labels_tensor)
            sentence_lens_buf.append(sentence_lens)

    # Yield last batch if not empty and drop_last_batch is False
    if (not config.dataloader_drop_last_batch) and len(input_features_buf) > 0:
        yield {
            "input_ids": None,
            "input_features": pad_sequence(input_features_buf,
                                           batch_first=True,
                                           padding_value=0.0),
            "labels": pad_sequence(labels_buf,
                                   batch_first=True,
                                   padding_value=-100),
            "shift_labels": pad_sequence(labels_buf,
                                         batch_first=True,
                                         padding_value=-100),
            "position_ids": None,  # Will be generated in model forward
            "attention_mask": None,  # Will be generated in model forward
            "sentence_lens": pad_sequence(sentence_lens_buf,
                                          batch_first=True,
                                          padding_value=1),
            "num_sentence": len(input_features_buf),
        }


def batch_pairaudio_pairtext(data, config: DataConfig, tokenizer: BaseTokenizer):
    """ Non-packed version of batch_pairaudio_pairtext_packed.
        Uses dynamic batching with pad_sequence for variable length sequences.

        Args:
            data: Iterator of samples with 'audiofeat' and 'input_ids' keys
            config: DataConfig containing batch size and other parameters
            tokenizer: BaseTokenizer for text tokenization

        Yields:
            Dict containing batched tensors with proper padding
    """
    input_ids_buf = []
    input_features_buf = []
    labels_buf = []
    attention_mask_buf = []
    sentence_lens_buf = []

    assert config.dataset_audio_seqlen == config.dataset_text_seqlen

    max_len_in_buffer = 0
    for sample in data:
        audio_len = sample['audiofeat'].size(0)
        text_len = len(sample['input_ids'])
        total_len = audio_len + text_len + 1  # +1 for bos/eos
        max_len_in_buffer = max(max_len_in_buffer, total_len)

        # Skip samples that are too long
        if total_len > config.dataset_audio_seqlen:
            continue

        # Prepare audio features - pad to total_len
        input_features = torch.zeros(total_len, sample['audiofeat'].size(1), dtype=torch.float32)
        input_features[:audio_len] = sample['audiofeat']

        # Prepare input_ids - pad to total_len
        input_ids = torch.zeros(total_len, dtype=torch.int64) + tokenizer.pad
        input_ids[audio_len:] = torch.tensor([tokenizer.bos] + sample['input_ids'], dtype=torch.int64)
        # Prepare labels
        labels = torch.zeros(total_len, dtype=torch.int64) - 100  # ignore_idx = -100
        labels[audio_len:] = torch.tensor(sample['input_ids'] + [tokenizer.eos], dtype=torch.int64)

        # Prepare attention mask (1 for valid positions, 0 for padding)
        attention_mask = torch.ones(total_len, dtype=torch.int64)

        # Sentence length for this sample (text part only)
        sentence_lens = torch.ones(total_len, dtype=torch.int64) * text_len

        # Yield batch when we have enough samples
        if (len(input_features_buf) + 1) * max_len_in_buffer > (config.dataset_batchsize * config.dataset_audio_seqlen):
            yield {
                "input_ids": pad_sequence(input_ids_buf,
                                          batch_first=True,
                                          padding_side='right',
                                          padding_value=tokenizer.pad),
                "input_features": pad_sequence(input_features_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=0.0),
                "labels": pad_sequence(labels_buf,
                                       batch_first=True,
                                       padding_side='right',
                                       padding_value=-100),
                "shift_labels": pad_sequence(labels_buf,
                                             batch_first=True,
                                             padding_side='right',
                                             padding_value=-100),
                "position_ids": None,  # Will be generated in model forward
                "attention_mask": pad_sequence(attention_mask_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=0),
                "sentence_lens": pad_sequence(sentence_lens_buf,
                                              batch_first=True,
                                              padding_side='right',
                                              padding_value=1),
                "num_sentence": len(input_features_buf),
            }
            # Reset buffers
            input_features_buf = [input_features]
            input_ids_buf = [input_ids]
            labels_buf = [labels]
            attention_mask_buf = [attention_mask]
            sentence_lens_buf = [sentence_lens]
            max_len_in_buffer = total_len
        else:
            input_features_buf.append(input_features)
            input_ids_buf.append(input_ids)
            labels_buf.append(labels)
            attention_mask_buf.append(attention_mask)
            sentence_lens_buf.append(sentence_lens)

    # Yield last batch if not empty and drop_last_batch is False
    if (not config.dataloader_drop_last_batch) and len(input_features_buf) > 0:
        yield {
            "input_ids": pad_sequence(input_ids_buf,
                                      batch_first=True,
                                      padding_side='right',
                                      padding_value=tokenizer.pad),
            "input_features": pad_sequence(input_features_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=0.0),
            "labels": pad_sequence(labels_buf,
                                   batch_first=True,
                                   padding_side='right',
                                   padding_value=-100),
            "shift_labels": pad_sequence(labels_buf,
                                         batch_first=True,
                                         padding_side='right',
                                         padding_value=-100),
            "position_ids": None,  # Will be generated in model forward
            "attention_mask": pad_sequence(attention_mask_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=0),
            "sentence_lens": pad_sequence(sentence_lens_buf,
                                          batch_first=True,
                                          padding_side='right',
                                          padding_value=1),
            "num_sentence": len(input_features_buf),
        }


def touch_audio_datapipe(
    data_config: DataConfig,
    tokenizer: BaseTokenizer,
    dp_rank: int, dp_world_size: int,
):
    """ Construct datapipe from configs for touch audio training.

    Args:
        data_config: Configuration for data processing
        tokenizer: Tokenizer for text processing
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size

    Returns:
        MidLevelTouchDatapipe: Configured datapipe for touch audio training
    """
    datapipe = LowLevelTouchDatapipe(data_config, dp_rank, dp_world_size)

    if not isinstance(tokenizer, BestRQTokenizer):
        datapipe = MidLevelTouchDatapipe(datapipe, functions.text_tokenize, tokenizer)

    datapipe = MidLevelTouchDatapipe(datapipe, functions.filter_samples, data_config)
    datapipe = MidLevelTouchDatapipe(datapipe, functions.audio_resample, data_config)

    # wav-level augment
    if data_config.audio_speed_perturb:
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audio_speed_perturb, data_config)

    if data_config.audio_feat_type == 'fbank':
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audio_compute_fbank, data_config)
    elif data_config.audio_feat_type == 'mfcc':
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audio_compute_mfcc, data_config)
    elif data_config.audio_feat_type == 'log_mel_spectrogram':
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audio_compute_log_mel_spectrogram,
                                         data_config)

    # feat-level augment
    if data_config.audiofeat_spec_aug:
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audiofeat_spec_aug, data_config)
    if data_config.audiofeat_spec_sub:
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audiofeat_spec_sub, data_config)
    if data_config.audiofeat_spec_trim:
        datapipe = MidLevelTouchDatapipe(datapipe, functions.audiofeat_spec_trim, data_config)

    # feat-level stack & stride
    datapipe = MidLevelTouchDatapipe(datapipe, functions.audiofeat_stack, data_config)

    if isinstance(tokenizer, BestRQTokenizer):
        # audio pretrain
        datapipe = MidLevelTouchDatapipe(datapipe, batch_audio, data_config,
                                         tokenizer)
    else:
        # audio sft, like asr or tts
        if data_config.dataset_enable_pack:
            datapipe = MidLevelTouchDatapipe(datapipe, batch_pairaudio_pairtext_packed, data_config,
                                             tokenizer)
        else:
            datapipe = MidLevelTouchDatapipe(datapipe, batch_pairaudio_pairtext, data_config,
                                             tokenizer)
    return datapipe
