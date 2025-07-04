# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperFeatureExtractor

from touchnet.data import DataConfig
from touchnet.data.datapipe import LowLevelTouchDatapipe, MidLevelTouchDatapipe
from touchnet.tokenizer.tokenizer import BaseTokenizer

"""
    NOTE(xcsong): Here are kimi extra tokens for reference:
        ExtraTokens(
            msg_end=map_fn("<|im_msg_end|>"),  # 151645
            media_begin=map_fn("<|im_media_begin|>"),  # 151661
            media_end=map_fn("<|im_media_end|>"),  # 151663
            kimia_text_blank=map_fn("<|im_kimia_text_blank|>"),  # 151666
            kimia_text_eos=map_fn("<|im_kimia_text_eos|>"),  # 151667
            kimia_user_msg_start=map_fn("<|im_kimia_user_msg_start|>"),  # 151670
            kimia_assistant_msg_start=map_fn("<|im_kimia_assistant_msg_start|>"),  # 151671
            kimia_speech_ct_id=map_fn("<|im_kimia_speech_ct_id|>"),  # 151675
            kimia_speech_ctd_id=map_fn("<|im_kimia_speech_ctd_id|>"),  # 151676
        )

    NOTE(xcsong): Chat template reference:
        https://github.com/MoonshotAI/Kimi-Audio/blob/master/kimia_infer/api/prompt_manager.py

    NOTE(xcsong): When grouping batches for the mixed tasks of Speech2Text and Text2Speech, it should be noted that a batch cannot contain only one task.  # noqa
        Otherwise, an error will occur in the forward process when enabling FSDP. For example, if some cards only have S2T data input and some cards only  # noqa
        have T2S data input, when the forward process reaches the T2S module, an all_gather will be triggered, and the cards with only S2T data will hang.  # noqa
        So we need to ensure that each batch contains at least one S2T and one T2S task.
"""
KIMI_TEXT_TEMPLATE_FOR_S2T = "<|im_kimia_user_msg_start|><|INSTRUCT|><|im_kimia_text_blank|><|AUDIO|><|im_kimia_text_blank|><|im_kimia_text_blank|><|im_kimia_text_blank|><|im_kimia_text_blank|>"  # noqa
KIMI_AUDIO_TEMPLATE_FOR_S2T = "<|im_kimia_text_blank|><|INSTRUCT|><|im_media_begin|><|AUDIO|><|im_media_end|><|im_kimia_speech_ct_id|><|im_msg_end|><|im_kimia_assistant_msg_start|>"  # noqa


def dynamic_batch(
    data, config: DataConfig,
    processor: WhisperFeatureExtractor,
    tokenizer: BaseTokenizer,
):
    """
    Dynamic batching function for KimiAudio training data.

    Processes audio waveforms and instruction-response pairs into model input
    features with proper padding and attention masks.

    Args:
        data: Iterator of training samples with waveform, instruct, and response
        config: DataConfig containing batching and filtering parameters
        processor: WhisperFeatureExtractor for tokenization and feature extraction
        tokenizer: BaseTokenizer for tokenization

    Yields:
        Dict containing batched tensors ready for model training
    """
    text_input_ids_buf, attention_mask_buf, labels_buf, sentence_lens_buf = [], [], [], []
    audio_input_ids_buf = []
    whisper_input_features_buf, whisper_attention_mask_buf = [], []
    longest_length = 0
    for sample in data:
        assert 'waveform' in sample
        # NOTE(xcsong): if instruct or response is not in sample, we assume it is an asr task
        if 'instruct' not in sample:
            sample['instruct'] = "Generate the transcription"
        if 'response' not in sample:
            assert 'txt' in sample
            sample['response'] = sample['txt']
        audio = sample['waveform'].squeeze(0).numpy()
        whisper_features = processor(audio, sampling_rate=processor.sampling_rate,
                                     return_attention_mask=True, return_tensors="pt",
                                     padding="max_length")  # pad to 30s
        # NOTE(xcsong): KimiAudio add a stack&stride layer on top of whisper encoder,
        #               results in 1/8 subsample:
        # Example whisper_features (a 4 sencond audio with caption `glass is breaking`):
        #   whisper_features['input_features'], torch.Size([1, 128, 3000]), whisper encoder input, always padding to 30s
        #   whisper_features['attention_mask'], torch.Size([1, 3000]), whisper encoder input, 0 for padding part
        input_features = whisper_features['input_features'].squeeze(0)
        feature_attention_mask = whisper_features['attention_mask'].squeeze(0)
        num_audio_tokens = feature_attention_mask[::2][::4].sum()  # 1/2 conv subsample, 1/4 pool subsample
        instruct_token_ids = tokenizer.tokenize(sample['instruct'], add_special_tokens=False)
        response_token_ids = tokenizer.tokenize(sample['response'], add_special_tokens=False)
        # NOTE(xcsong): expand *_input to include task_ids and audio_ids
        text_prompt = KIMI_TEXT_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", sample['instruct'])
        text_prompt = text_prompt.replace("<|AUDIO|>", "<|im_kimia_text_blank|>" * num_audio_tokens)
        audio_prompt = KIMI_AUDIO_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", "<|im_kimia_text_blank|>" * len(instruct_token_ids))
        audio_prompt = audio_prompt.replace("<|AUDIO|>", "<|im_kimia_text_blank|>" * num_audio_tokens)
        text_prompt_ids = torch.tensor(
            tokenizer.tokenize(text_prompt, add_special_tokens=False),
            dtype=torch.int64,
        )
        audio_prompt_ids = torch.tensor(
            tokenizer.tokenize(audio_prompt, add_special_tokens=False),
            dtype=torch.int64,
        )
        text_response_ids = torch.tensor(
            response_token_ids,
            dtype=torch.int64,
        )
        audio_response_ids = torch.tensor(
            tokenizer.tokenize("<|im_kimia_text_blank|>" * len(response_token_ids), add_special_tokens=False),
            dtype=torch.int64,
        )
        eos = torch.tensor(
            tokenizer.tokenize("<|im_kimia_text_eos|>", add_special_tokens=False),
            dtype=torch.int64
        )
        # NOTE(xcsong): "<|im_kimia_text_eos|>" is different from kimi_tokenizer.eos_token ("[EOS]"), this is to align with
        #   original inference code: https://github.com/MoonshotAI/Kimi-Audio/blob/master/kimia_infer/api/kimia.py#L319

        assert len(text_prompt_ids) == len(audio_prompt_ids), f"{len(text_prompt_ids)}, {len(audio_prompt_ids)}\n{text_prompt_ids}\n{audio_prompt_ids}"  # noqa
        assert len(text_response_ids) == len(audio_response_ids), f"{len(text_response_ids)}, {len(audio_response_ids)}\n{text_response_ids}\n{audio_response_ids}"  # noqa

        text_input_ids = torch.cat((text_prompt_ids, text_response_ids))
        audio_input_ids = torch.cat((audio_prompt_ids, audio_response_ids))
        labels = torch.cat((torch.zeros_like(text_prompt_ids[1:]) - 100, text_response_ids, eos))  # ignore_idx == -100
        sentence_lens = torch.zeros_like(labels) + text_response_ids.size(0) + 1  # exclude prompt, + 1 for eos

        new_sample_length = text_input_ids.size(0)
        if new_sample_length < config.text_min_length_in_tokens_for_filter:
            continue
        if new_sample_length > config.text_max_length_in_tokens_for_filter:
            continue

        longest_length = max(longest_length, new_sample_length)
        size_after_padding = longest_length * (len(text_input_ids_buf) + 1)
        if size_after_padding > (config.dataset_batchsize * config.dataset_text_seqlen):
            # NOTE(xcsong): set paddingside to left to align with original impl, see
            #   https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L785
            yield {
                "text_input_ids": pad_sequence(text_input_ids_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=tokenizer.pad),
                "audio_input_ids": pad_sequence(audio_input_ids_buf,
                                                batch_first=True,
                                                padding_side='right',
                                                padding_value=tokenizer.pad),
                "attention_mask": pad_sequence(attention_mask_buf,
                                               batch_first=True,
                                               padding_side='right',
                                               padding_value=0),
                "labels": pad_sequence(labels_buf,
                                       batch_first=True,
                                       padding_side='right',
                                       padding_value=-100),
                "whisper_input_features": pad_sequence(whisper_input_features_buf,
                                                       batch_first=True,
                                                       padding_side='right',
                                                       padding_value=0),
                "whisper_attention_mask": pad_sequence(whisper_attention_mask_buf,
                                                       batch_first=True,
                                                       padding_side='right',
                                                       padding_value=0),
                "num_sentence": len(text_input_ids_buf),
                "sentence_lens": pad_sequence(sentence_lens_buf,
                                              batch_first=True,
                                              padding_side='right',
                                              padding_value=1),  # 1 for avoid dividing zero
            }
            text_input_ids_buf, attention_mask_buf, labels_buf = [text_input_ids], [torch.ones_like(labels)], [labels]
            audio_input_ids_buf = [audio_input_ids]
            whisper_input_features_buf, whisper_attention_mask_buf = [input_features], [feature_attention_mask]
            sentence_lens_buf = [sentence_lens]
            longest_length = new_sample_length
        else:
            text_input_ids_buf.append(text_input_ids)
            audio_input_ids_buf.append(audio_input_ids)
            attention_mask_buf.append(torch.ones_like(labels))
            labels_buf.append(labels)
            whisper_input_features_buf.append(input_features)
            whisper_attention_mask_buf.append(feature_attention_mask)
            sentence_lens_buf.append(sentence_lens)
    # last batch
    if (not config.dataloader_drop_last_batch) and (len(text_input_ids_buf) > 0):
        yield {
            "text_input_ids": pad_sequence(text_input_ids_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=tokenizer.pad),
            "audio_input_ids": pad_sequence(audio_input_ids_buf,
                                            batch_first=True,
                                            padding_side='right',
                                            padding_value=tokenizer.pad),
            "attention_mask": pad_sequence(attention_mask_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=0),
            "labels": pad_sequence(labels_buf,
                                   batch_first=True,
                                   padding_side='right',
                                   padding_value=-100),
            "whisper_input_features": pad_sequence(whisper_input_features_buf,
                                                   batch_first=True,
                                                   padding_side='right',
                                                   padding_value=0),
            "whisper_attention_mask": pad_sequence(whisper_attention_mask_buf,
                                                   batch_first=True,
                                                   padding_side='right',
                                                   padding_value=0),
            "num_sentence": len(text_input_ids_buf),
            "sentence_lens": pad_sequence(sentence_lens_buf,
                                          batch_first=True,
                                          padding_side='right',
                                          padding_value=1),  # 1 for avoid dividing zero
        }


def kimi_audio_datapipe(
    data_config: DataConfig,
    tokenizer: BaseTokenizer,
    dp_rank: int, dp_world_size: int,
):
    """ Construct datapipe from configs
    """
    processor = WhisperFeatureExtractor.from_pretrained(
        data_config.processor_model,
    )
    datapipe = LowLevelTouchDatapipe(data_config, dp_rank, dp_world_size)
    datapipe = MidLevelTouchDatapipe(datapipe, dynamic_batch, data_config,
                                     processor, tokenizer)

    return datapipe
