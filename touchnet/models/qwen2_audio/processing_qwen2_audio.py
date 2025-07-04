# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from transformers.models.qwen2_audio.processing_qwen2_audio import \
    Qwen2AudioProcessor

from touchnet.data import DataConfig
from touchnet.data.datapipe import LowLevelTouchDatapipe, MidLevelTouchDatapipe
from touchnet.utils.logging import logger

QWEN2_AUDIO_TEMPLATE_FOR_S2T = "<|audio_bos|><|AUDIO|><|audio_eos|><|INSTRUCT|>"


def dynamic_batch(data, config: DataConfig, processor: Qwen2AudioProcessor):
    """
    Dynamic batching function for Qwen2Audio training data.

    Processes audio waveforms and instruction-response pairs into model input
    features with proper padding and attention masks.

    Args:
        data: Iterator of training samples with waveform, instruct, and response
        config: DataConfig containing batching and filtering parameters
        processor: Qwen2AudioProcessor for tokenization and feature extraction

    Yields:
        Dict containing batched tensors ready for model training
    """

    input_ids_buf, attention_mask_buf, labels_buf, sentence_lens_buf = [], [], [], []
    input_features_buf, feature_attention_mask_buf = [], []
    longest_length = 0
    for sample in data:
        assert 'waveform' in sample

        # NOTE(xcsong): if instruct or response is not in sample, we assume it is an asr task
        if 'instruct' not in sample:
            sample['instruct'] = "Generate the transcription:"
        if 'response' not in sample:
            if 'txt' in sample:
                sample['response'] = sample['txt']
            else:
                logger.info(f"txt not in sample, skip this sample {sample}")
                continue

        audio = sample['waveform'].squeeze(0).numpy()
        sample['instruct'] = QWEN2_AUDIO_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", sample['instruct'])

        # NOTE(xcsong): Qwen2Audio add a pooling layer on top of whisper encoder,
        #               results in 1/4 subsample:
        # Example prompt (a 4 sencond audio with caption `glass is breaking`):
        #   text_inputs['input_ids'], torch.Size([1, 109]), <eos> not included
        #   text_inputs['attention_mask'], torch.Size([1, 109]), padding mask, all ones
        #   audio_inputs['input_features'], torch.Size([1, 128, 3000 or longer]), whisper encoder input
        #   audio_inputs['attention_mask'], torch.Size([1, 3000 or longer]), whisper encoder input

        # NOTE(xcsong): we add truncation=False to support long audio, for audio less than 30s,
        #               feature_extractor will pad to 30s as usual, for audio longer than 30s,
        #               feature_extractor will keep it unchanged.
        audio_inputs = processor.feature_extractor(
            audio,
            sampling_rate=processor.feature_extractor.sampling_rate,
            truncation=False,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt"
        )
        # NOTE(xcsong): WhisperFeatureExtractor has bug, it will return a wrong feature_attention_mask
        #               when audio is longer than 30s, so we use a all-ones tensor instead.
        feature_attention_mask = audio_inputs.pop('attention_mask').squeeze(0)
        if audio_inputs['input_features'].shape[-1] > 3000:
            feature_attention_mask = torch.ones(audio_inputs['input_features'].shape[-1],
                                                dtype=feature_attention_mask.dtype,
                                                device=feature_attention_mask.device)
        text = sample['instruct']
        audio_length = feature_attention_mask.sum().long()
        input_length = (audio_length - 1) // 2 + 1
        num_audio_tokens_expanded = (input_length - 2) // 2 + 1
        expanded_audio_token = "<|AUDIO|>" * int(num_audio_tokens_expanded.item())
        expanded_text = text.replace("<|AUDIO|>", expanded_audio_token, 1)
        text_inputs = processor.tokenizer(
            expanded_text,
            padding=False,
            return_tensors="pt"
        )

        if audio_length * 10 > config.audio_max_length_in_ms_for_filter:
            logger.info(f"audio_length: {audio_length * 10}ms > config.audio_max_length_in_ms_for_filter: {config.audio_max_length_in_ms_for_filter}ms, skip this sample")  # noqa
            continue

        input_features = audio_inputs['input_features'].squeeze(0).transpose(0, 1)  # [3000 or longer, 128]
        prompt_ids = text_inputs['input_ids'].squeeze(0)  # [seq_len,]
        response_ids = torch.tensor(
            processor.tokenizer(sample['response'], add_special_tokens=False).input_ids,
            dtype=prompt_ids.dtype,
        )
        eos = torch.tensor([processor.tokenizer.eos_token_id], dtype=prompt_ids.dtype)

        input_ids = torch.cat((prompt_ids, response_ids))
        labels = torch.cat((torch.zeros_like(prompt_ids[1:]) - 100, response_ids, eos))  # ignore_idx == -100
        sentence_lens = torch.zeros_like(labels) + response_ids.size(0) + 1  # exclude prompt, + 1 for eos

        new_sample_length = input_ids.size(0)
        if new_sample_length < config.text_min_length_in_tokens_for_filter:
            logger.info(f"sample_length: {new_sample_length} < config.text_min_length_in_tokens_for_filter: {config.text_min_length_in_tokens_for_filter}, skip this sample")  # noqa
            continue
        if new_sample_length > config.text_max_length_in_tokens_for_filter:
            logger.info(f"sample_length: {new_sample_length} > config.text_max_length_in_tokens_for_filter: {config.text_max_length_in_tokens_for_filter}, skip this sample")  # noqa
            continue

        longest_length = max(longest_length, new_sample_length)
        size_after_padding = longest_length * (len(input_ids_buf) + 1)
        if size_after_padding > (config.dataset_batchsize * config.dataset_text_seqlen):
            # NOTE(xcsong): right padding for training, left padding for inference
            yield {
                "input_ids": pad_sequence(input_ids_buf,
                                          batch_first=True,
                                          padding_side='right',
                                          padding_value=processor.tokenizer.pad_token_id),
                "attention_mask": pad_sequence(attention_mask_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=0),
                "labels": pad_sequence(labels_buf,
                                       batch_first=True,
                                       padding_side='right',
                                       padding_value=-100),
                "shift_labels": pad_sequence(labels_buf,
                                             batch_first=True,
                                             padding_side='right',
                                             padding_value=-100),
                "input_features": pad_sequence(input_features_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=0).transpose(1, 2),  # [B, 128, T]
                "feature_attention_mask": pad_sequence(feature_attention_mask_buf,
                                                       batch_first=True,
                                                       padding_side='right',
                                                       padding_value=0),
                "num_sentence": len(input_ids_buf),
                "sentence_lens": pad_sequence(sentence_lens_buf,
                                              batch_first=True,
                                              padding_side='right',
                                              padding_value=1),  # 1 for avoid dividing zero
            }
            input_ids_buf, attention_mask_buf, labels_buf = [input_ids], [torch.ones_like(labels)], [labels]
            input_features_buf, feature_attention_mask_buf = [input_features], [feature_attention_mask]
            sentence_lens_buf = [sentence_lens]
            longest_length = new_sample_length
        else:
            input_ids_buf.append(input_ids)
            attention_mask_buf.append(torch.ones_like(labels))
            labels_buf.append(labels)
            input_features_buf.append(input_features)
            feature_attention_mask_buf.append(feature_attention_mask)
            sentence_lens_buf.append(sentence_lens)
    # last batch
    if (not config.dataloader_drop_last_batch) and (len(input_ids_buf) > 0):
        yield {
            "input_ids": pad_sequence(input_ids_buf,
                                      batch_first=True,
                                      padding_side='right',
                                      padding_value=processor.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=0),
            "labels": pad_sequence(labels_buf,
                                   batch_first=True,
                                   padding_side='right',
                                   padding_value=-100),
            "shift_labels": pad_sequence(labels_buf,
                                         batch_first=True,
                                         padding_side='right',
                                         padding_value=-100),
            "input_features": pad_sequence(input_features_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=0).transpose(1, 2),  # [B, 128, T]
            "feature_attention_mask": pad_sequence(feature_attention_mask_buf,
                                                   batch_first=True,
                                                   padding_side='right',
                                                   padding_value=0),
            "num_sentence": len(input_ids_buf),
            "sentence_lens": pad_sequence(sentence_lens_buf,
                                          batch_first=True,
                                          padding_side='right',
                                          padding_value=1),  # 1 for avoid dividing zero
        }


def qwen2_audio_datapipe(
    data_config: DataConfig,
    dp_rank: int, dp_world_size: int,
):
    """ Construct datapipe from configs
    """
    # NOTE(xcsong): It will automatically init tokenizer to tokenize text
    processor = transformers.AutoProcessor.from_pretrained(
        data_config.processor_model,
        trust_remote_code=True
    )
    datapipe = LowLevelTouchDatapipe(data_config, dp_rank, dp_world_size)
    datapipe = MidLevelTouchDatapipe(datapipe, dynamic_batch, data_config, processor)

    return datapipe
