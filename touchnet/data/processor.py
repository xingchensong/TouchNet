# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2025 WeNet Community. Xingchen Song(sxc19@tsinghua.org.cn)
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

import math
import random

import librosa
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from touchnet.data import DataConfig
from touchnet.tokenizer.tokenizer import BaseTokenizer

torchaudio.utils.sox_utils.set_buffer_size(16500)


def text_tokenize(data, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Dict[{key, waveform, txt, sample_rate}]

        Returns:
            Dict[{key, waveform, input_ids, sample_rate}]
    """
    for sample in data:
        assert 'txt' in sample
        # NOTE(xcsong): add bos/eos in batch_xxx()
        input_ids = tokenizer.tokenize(sample['txt'], add_special_tokens=False)
        sample['input_ids'] = input_ids
        yield sample


def filter(data, config: DataConfig):
    """ Filter sample according to feature and label length
        Inplace operation.
    """
    for sample in data:
        if 'input_ids' in sample:
            num_tokens = len(sample['input_ids'])
            if num_tokens < config.text_min_length_in_tokens_for_filter:
                continue
            if num_tokens > config.text_max_length_in_tokens_for_filter:
                continue
        if 'waveform' in sample:
            assert 'sample_rate' in sample
            # sample['waveform'] is torch.Tensor with shape [1, T], we have 1000ms each second
            duration = sample['waveform'].size(1) / sample['sample_rate'] * 1000.0
            if config.audio_speed_perturb:
                duration *= max(config.audio_speed_perturb_speeds)
            if duration < config.audio_min_length_in_ms_for_filter:
                continue
            if duration > config.audio_max_length_in_ms_for_filter:
                continue
            if 'input_ids' in sample:
                num_tokens = len(sample['input_ids'])
                if duration > 1e-7:
                    if num_tokens / (duration / 10) < config.min_text_audio_ratio:
                        continue
                    if num_tokens / (duration / 10) > config.max_text_audio_ratio:
                        continue
        yield sample


def audio_resample(data, config: DataConfig):
    """ Resample data.
        Inplace operation.
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'waveform' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        if sample_rate != config.audio_resample_rate:
            sample['sample_rate'] = config.audio_resample_rate
            sample['waveform'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=config.audio_resample_rate)(waveform)
        yield sample


def audio_speed_perturb(data, config: DataConfig):
    """ Apply speed perturb to the data.
        Inplace operation.
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'waveform' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        speed = random.choice(config.audio_speed_perturb_speeds)
        if speed != 1.0:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['waveform'] = waveform
        yield sample


def audio_compute_fbank(data, config: DataConfig):
    """ Extract fbank
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'waveform' in sample
        assert 'input_ids' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(waveform,
                          num_mel_bins=config.audiofeat_num_mel_bins,
                          frame_length=config.audiofeat_frame_length,
                          frame_shift=config.audiofeat_frame_shift,
                          dither=config.audiofeat_dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        sample['audiofeat'] = mat
        yield sample


def audio_compute_mfcc(data, config: DataConfig):
    """ Extract mfcc
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'waveform' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        waveform = waveform * (1 << 15)
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=config.audiofeat_num_mel_bins,
                         frame_length=config.audiofeat_frame_length,
                         frame_shift=config.audiofeat_frame_shift,
                         dither=config.audiofeat_dither,
                         num_ceps=config.audiofeat_num_ceps,
                         high_freq=config.audiofeat_high_freq,
                         low_freq=config.audiofeat_low_freq,
                         sample_frequency=sample_rate)
        sample['audiofeat'] = mat
        yield sample


def audio_compute_log_mel_spectrogram(data, config: DataConfig):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'waveform' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['waveform'].squeeze(0)  # (channel=1, sample) -> (sample,)
        if config.audiofeat_padding > 0:
            waveform = F.pad(waveform, (0, config.audiofeat_padding))
        window = torch.hann_window(config.audiofeat_n_fft)
        stft = torch.stft(waveform,
                          config.audiofeat_n_fft,
                          config.audiofeat_hop_length,
                          window=window,
                          return_complex=True)
        magnitudes = stft[..., :-1].abs()**2

        filters = torch.from_numpy(
            librosa.filters.mel(sr=sample_rate,
                                n_fft=config.audiofeat_n_fft,
                                n_mels=config.audiofeat_num_mel_bins))
        mel_spec = filters @ magnitudes

        # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        sample['audiofeat'] = log_spec.transpose(0, 1)
        yield sample


def audiofeat_spec_aug(data, config: DataConfig):
    """ Do spec augmentation
        Inplace operation
    """
    for sample in data:
        assert 'audiofeat' in sample
        x = sample['audiofeat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(config.audiofeat_spec_aug_num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, config.audiofeat_spec_aug_max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(config.audiofeat_spec_aug_num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, config.audiofeat_spec_aug_max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['audiofeat'] = y
        yield sample


def audiofeat_spec_sub(data, config: DataConfig):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]
    """
    for sample in data:
        assert 'audiofeat' in sample
        x = sample['audiofeat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(config.audiofeat_spec_sub_num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, config.audiofeat_spec_sub_max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['audiofeat'] = y
        yield sample


def audiofeat_spec_trim(data, config: DataConfig):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]
    """
    for sample in data:
        assert 'audiofeat' in sample
        x = sample['audiofeat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, config.audiofeat_spec_trim_max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['audiofeat'] = y
        yield sample


def audiofeat_stack(data, config: DataConfig):
    """ Stack audio features.
        lfr stands for low frame rate.
        NOTE(xcsong): this function is copied from
            https://github.com/modelscope/FunASR/blob/main/funasr/frontends/wav_frontend.py#L58-L74
    """
    for sample in data:
        assert 'audiofeat' in sample
        inputs = sample['audiofeat']  # (T, D)
        T = inputs.shape[0]
        T_lfr = int(math.ceil(T / config.audiofeat_stride_length))
        left_padding = inputs[0].repeat((config.audiofeat_stack_length - 1) // 2, 1)
        inputs = torch.vstack((left_padding, inputs))
        T = T + (config.audiofeat_stack_length - 1) // 2
        feat_dim = inputs.shape[-1]
        strides = (config.audiofeat_stride_length * feat_dim, 1)
        sizes = (T_lfr, config.audiofeat_stack_length * feat_dim)
        last_idx = (T - config.audiofeat_stack_length) // config.audiofeat_stride_length + 1
        num_padding = config.audiofeat_stack_length - (T - last_idx * config.audiofeat_stride_length)
        if num_padding > 0:
            num_padding = (2 * config.audiofeat_stack_length - 2 * T
                           + (T_lfr - 1 + last_idx) * config.audiofeat_stride_length) / 2 * (T_lfr - last_idx)
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
        outputs = inputs.as_strided(sizes, strides)
        if config.audiofeat_normalize:
            outputs = (outputs - outputs.mean(dim=-1, keepdim=True)) / \
                      (outputs.std(dim=-1, keepdim=True) + 1e-5)
        sample["audiofeat"] = outputs.clone().type(torch.float32)  # [T // stride, D * stack]
        yield sample


def batch_pairaudio_pairtext(data, config: DataConfig, tokenizer: BaseTokenizer):
    """ Feeding the data into buffer for training.
        We generate attention_mask inside Model.forward().
        Memory assumption:
            dataset_batchsize = 16
            dataset_audio_seqlen = 131072 (2^17)
            dataset_text_seqlen = 131072 (2^17)
            audiofeat_stack_length = 4
            audiofeat_num_mel_bins = 128
        Then we got:
            inputs_embeds: [16, 131072, 128 * 4] = 16 * 131072 * 512 * 4Bytes / 1024 / 1024 = 4096MB
            input_ids: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
            labels: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
            position_ids: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
                example position_ids for packed sequence:
                    [[0, 1, 2, 0, 1, 2, 0],
                     [0, 1, 0, 1, 2, 0, 1]]
            sentence_ids: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
                example sentence_ids for packed sequence: start from 1, ignore_idx == 0
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
        "inputs_embeds": torch.zeros([config.dataset_batchsize, config.dataset_audio_seqlen,
                                      config.audiofeat_num_mel_bins * config.audiofeat_stack_length],
                                     dtype=torch.float32),
        "labels": torch.zeros([config.dataset_batchsize,
                               config.dataset_text_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
        "position_ids": torch.zeros([config.dataset_batchsize,
                                     config.dataset_audio_seqlen], dtype=torch.int64),
        "sentence_ids": torch.zeros([config.dataset_batchsize,
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
                yield buffer
                # reset buffer for next batch
                buffer = {
                    "input_ids": torch.zeros([config.dataset_batchsize,
                                              config.dataset_text_seqlen], dtype=torch.int64) + tokenizer.pad,
                    "inputs_embeds": torch.zeros([config.dataset_batchsize, config.dataset_audio_seqlen,
                                                  config.audiofeat_num_mel_bins * config.audiofeat_stack_length],
                                                 dtype=torch.float32),
                    "labels": torch.zeros([config.dataset_batchsize,
                                           config.dataset_text_seqlen], dtype=torch.int64) - 100,  # ignore_idx = -100
                    "position_ids": torch.zeros([config.dataset_batchsize,
                                                 config.dataset_audio_seqlen], dtype=torch.int64),
                    "sentence_ids": torch.zeros([config.dataset_batchsize,
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
        buffer["inputs_embeds"][cur_batch_idx, cur_audio_idx:cur_audio_idx + audio_len] = sample['audiofeat']
        buffer["input_ids"][cur_batch_idx, cur_audio_idx + total_len - text_len:cur_audio_idx + total_len] = \
            torch.tensor([tokenizer.bos] + sample['input_ids'], dtype=torch.int64)
        buffer["labels"][cur_batch_idx, cur_audio_idx + total_len - text_len:cur_audio_idx + total_len] = \
            torch.tensor(sample['input_ids'] + [tokenizer.eos], dtype=torch.int64)
        buffer["position_ids"][cur_batch_idx, cur_audio_idx:cur_audio_idx + total_len] = torch.arange(
            0, total_len, dtype=torch.int64)
        buffer["sentence_ids"][cur_batch_idx, cur_audio_idx:cur_audio_idx + total_len] = cur_sentence_idx
        buffer["sentence_lens"][cur_batch_idx, cur_audio_idx:cur_audio_idx + total_len] = text_len
        buffer["num_sentence"] += 1
        cur_audio_idx += total_len
        cur_sentence_idx += 1
    if cur_batch_idx > 0 or cur_audio_idx > 0:
        yield buffer


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
            sentence_ids: [16, 131072] = 16 * 131072 * 8Bytes / 1024 / 1024 = 16MB
                example sentence_ids for packed sequence: start from 1, ignore_idx == 0
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
        "sentence_ids": torch.zeros([config.dataset_batchsize,
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
                    "sentence_ids": torch.zeros([config.dataset_batchsize,
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
        buffer["sentence_ids"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = cur_sentence_idx
        buffer["sentence_lens"][cur_batch_idx, cur_text_idx:cur_text_idx + text_len] = text_len
        buffer["num_sentence"] += 1
        cur_text_idx += text_len
        cur_sentence_idx += 1
    if cur_text_idx > 0 or cur_batch_idx > 0:
        yield buffer
