# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from transformers.models.qwen2_audio.processing_qwen2_audio import \
    Qwen2AudioProcessor

from touchnet.data import DataConfig
from touchnet.data.datapipe import LowLevelTouchDatapipe, MidLevelTouchDatapipe

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
        assert 'instruct' in sample  # i.e., Generate the transcription
        assert 'response' in sample  # i.e., Hello World
        audio = sample['waveform'].squeeze(0).numpy()
        sample['instruct'] = QWEN2_AUDIO_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", sample['instruct'])
        # NOTE(xcsong): processor will automatically expand prompt to include instruct_ids and audio_ids
        prompt = processor(text=sample['instruct'],
                           audio=audio,
                           sampling_rate=processor.feature_extractor.sampling_rate,
                           return_tensors="pt")
        # NOTE(xcsong): Qwen2Audio add a pooling layer on top of whisper encoder,
        #               results in 1/4 subsample:
        # Example prompt (a 4 sencond audio with caption `glass is breaking`):
        #   prompt['input_ids'], torch.Size([1, 109]), <eos> not included
        #   prompt['attention_mask'], torch.Size([1, 109]), padding mask, all ones
        #   prompt['input_features'], torch.Size([1, 128, 3000]), whisper encoder input, always padding to 30s
        #   prompt['feature_attention_mask'], torch.Size([1, 3000]), whisper encoder input
        input_features = prompt['input_features'].squeeze(0)
        feature_attention_mask = prompt['feature_attention_mask'].squeeze(0)
        prompt_ids = prompt['input_ids'].squeeze(0)
        response_ids = torch.tensor(
            processor.tokenizer(sample['response'], add_special_tokens=False).input_ids,
            dtype=prompt['input_ids'].dtype,
        )
        eos = torch.tensor([processor.tokenizer.eos_token_id], dtype=prompt['input_ids'].dtype)

        input_ids = torch.cat((prompt_ids, response_ids))
        labels = torch.cat((torch.zeros_like(prompt_ids[1:]) - 100, response_ids, eos))  # ignore_idx == -100
        sentence_lens = torch.zeros_like(labels) + response_ids.size(0) + 1  # exclude prompt, + 1 for eos

        new_sample_length = input_ids.size(0)
        if new_sample_length < config.text_min_length_in_tokens_for_filter:
            continue
        if new_sample_length > config.text_max_length_in_tokens_for_filter:
            continue

        longest_length = max(longest_length, new_sample_length)
        size_after_padding = longest_length * (len(input_ids_buf) + 1)
        if size_after_padding > (config.dataset_batchsize * config.dataset_text_seqlen):
            # NOTE(xcsong): set paddingside to left to align with original impl, see
            #   https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L785
            yield {
                "input_ids": pad_sequence(input_ids_buf,
                                          batch_first=True,
                                          padding_side='left',
                                          padding_value=processor.tokenizer.pad_token_id),
                "attention_mask": pad_sequence(attention_mask_buf,
                                               batch_first=True,
                                               padding_side='left',
                                               padding_value=0),
                "labels": pad_sequence(labels_buf,
                                       batch_first=True,
                                       padding_side='left',
                                       padding_value=-100),
                "input_features": pad_sequence(input_features_buf,
                                               batch_first=True,
                                               padding_side='left',
                                               padding_value=0),
                "feature_attention_mask": pad_sequence(feature_attention_mask_buf,
                                                       batch_first=True,
                                                       padding_side='left',
                                                       padding_value=0),
                "num_sentence": len(input_ids_buf),
                "sentence_lens": pad_sequence(sentence_lens_buf,
                                              batch_first=True,
                                              padding_side='left',
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
                                      padding_side='left',
                                      padding_value=processor.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask_buf,
                                           batch_first=True,
                                           padding_side='left',
                                           padding_value=0),
            "labels": pad_sequence(labels_buf,
                                   batch_first=True,
                                   padding_side='left',
                                   padding_value=-100),
            "input_features": pad_sequence(input_features_buf,
                                           batch_first=True,
                                           padding_side='left',
                                           padding_value=0),
            "feature_attention_mask": pad_sequence(feature_attention_mask_buf,
                                                   batch_first=True,
                                                   padding_side='left',
                                                   padding_value=0),
            "num_sentence": len(input_ids_buf),
            "sentence_lens": pad_sequence(sentence_lens_buf,
                                          batch_first=True,
                                          padding_side='left',
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
