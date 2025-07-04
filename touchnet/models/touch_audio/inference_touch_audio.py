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
""" Example Usage
gpu:

torchrun --nproc_per_node=8 --nnodes=1 \
     --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    touchnet/models/touch_audio/inference_touch_audio.py \
        --model_path "" \
        --instruct "" \
        --data_list xxx.list \
        --output_dir "" \
        --batch_size 12

"""

import json
import os

import numpy
import torch
import torch.distributed as dist
from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                       apply_liger_kernel_to_qwen2)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.hf_argparser import HfArgumentParser

import touchnet  # noqa
from touchnet.data import DataConfig, functions
from touchnet.models.touch_audio.configuration_touch_audio import \
    TouchAudioConfig
from touchnet.utils.distributed import TORCH_DTYPE_MAP
from touchnet.utils.inference import (AudioDataset, InferenceConfig,
                                      collate_fn, init_distributed)


def feature_extraction(audios: list[numpy.ndarray], data_config: DataConfig, special_tokens: dict):
    feature_func = None
    if data_config.audio_feat_type == 'fbank':
        feature_func = functions.audio_compute_fbank
    elif data_config.audio_feat_type == 'mfcc':
        feature_func = functions.audio_compute_mfcc
    elif data_config.audio_feat_type == 'log_mel_spectrogram':
        feature_func = functions.audio_compute_log_mel_spectrogram
    else:
        raise ValueError(f"Unsupported audio feature type: {data_config.audio_feat_type}")

    ids, feats, masks, positions = [], [], [], []
    for audio in audios:
        sample = next(feature_func([{'sample_rate': 16000, 'waveform': torch.from_numpy(audio).unsqueeze(0)}], data_config))
        sample = next(functions.audiofeat_stack([sample], data_config))
        # feat [T, D] => [T + 1, D]
        feat = torch.cat([
            sample['audiofeat'],
            torch.zeros([1, sample['audiofeat'].size(1)], dtype=sample['audiofeat'].dtype)
        ], dim=0)
        feats.append(feat)
        masks.append(torch.ones(feat.size(0), dtype=torch.int64))
        positions.append(torch.arange(0, feat.size(0), dtype=torch.int64))
        ids.append(torch.tensor([special_tokens["pad_token_id"]] * (feat.size(0) - 1) + [special_tokens["bos_token_id"]],
                                dtype=torch.int64))

    input_ids = pad_sequence(ids,
                             batch_first=True,
                             padding_side='left',
                             padding_value=special_tokens["pad_token_id"])
    input_features = pad_sequence(feats,
                                  batch_first=True,
                                  padding_side='left',
                                  padding_value=0)
    attention_mask = pad_sequence(masks,
                                  batch_first=True,
                                  padding_side='left',
                                  padding_value=0)
    position_ids = pad_sequence(positions,
                                batch_first=True,
                                padding_side='left',
                                padding_value=0)

    return {
        "input_ids": input_ids,
        "input_features": input_features,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


if __name__ == "__main__":
    parser = HfArgumentParser([InferenceConfig])
    (args,) = parser.parse_args_into_dataclasses()
    os.makedirs(args.output_dir, exist_ok=True)
    parser_data = HfArgumentParser([DataConfig])
    (data_config,) = parser_data.parse_json_file(f"{args.model_path}/../../data_config.json")
    model_config = TouchAudioConfig.from_json_file(f"{args.model_path}/../../model_config.json")
    special_tokens = {}
    if hasattr(model_config, 'bos_token'):
        special_tokens["bos_token"] = model_config.bos_token
    if hasattr(model_config, 'eos_token'):
        special_tokens["eos_token"] = model_config.eos_token
    if hasattr(model_config, 'pad_token'):
        special_tokens["pad_token"] = model_config.pad_token

    if args.inference_enable_liger_kernel:
        # 1. monkey patch the forward function to Qwen2Model
        apply_liger_kernel_to_qwen2()
        # 2. monkey patch the forward function to LlamaModel
        apply_liger_kernel_to_llama()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        **special_tokens,
    )

    assert (torch.cuda.is_available())
    world_size, local_rank, rank = init_distributed()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True
    ).cuda().eval().to(TORCH_DTYPE_MAP[args.model_dtype])

    dataset = AudioDataset(args.data_list)

    sampler = DistributedSampler(dataset,
                                 num_replicas=world_size,
                                 rank=rank)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch,
                            collate_fn=collate_fn)

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    writer = open(f"{args.output_dir}/part_{rank + 1}_of_{world_size}", "w")
    audio_bos_id = tokenizer("<|audio_bos|>", add_special_tokens=False).input_ids[0]
    audio_eos_id = tokenizer("<|audio_eos|>", add_special_tokens=False).input_ids[0]
    audio_id = tokenizer("<|AUDIO|>", add_special_tokens=False).input_ids[0]
    print(f"pad_token_id: {tokenizer.pad_token_id}, bos_token_id: {tokenizer.bos_token_id}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"audio_bos_id: {audio_bos_id}, audio_eos_id: {audio_eos_id}, audio_id: {audio_id}")
    special_tokens = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "audio_bos_id": audio_bos_id,
        "audio_eos_id": audio_eos_id,
        "audio_id": audio_id,
    }
    for infos, audios in dataloader:
        inputs = feature_extraction(audios, data_config, special_tokens)
        for k in inputs.keys():
            inputs[k] = inputs[k].cuda()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPE_MAP[args.model_dtype]):
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                do_sample=False,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.5,
                max_new_tokens=256,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_beams=1,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = generated_ids[:, inputs["input_features"].size(1):]
        response = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        for i, _ in enumerate(infos):
            writer.write(
                json.dumps({
                    "label": json.dumps(infos[i], ensure_ascii=False),
                    "predict": response[i],
                }, ensure_ascii=False) + "\n"
            )
        if rank == 0:
            progress_bar.update(world_size * len(infos))

    if rank == 0:
        progress_bar.close()
    writer.close()
    dist.barrier()
    dist.destroy_process_group()
