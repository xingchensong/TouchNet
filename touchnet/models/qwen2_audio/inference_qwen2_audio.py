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
    touchnet/models/qwen2_audio/inference_qwen2_audio.py \
        --model_path "" \
        --instruct "" \
        --data_list xxx.list \
        --output_dir "" \
        --batch_size 12

"""

import gc
import json
import os

import torch
import torch.distributed as dist
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (AutoProcessor, Qwen2AudioEncoder,
                          Qwen2AudioForConditionalGeneration)
from transformers.hf_argparser import HfArgumentParser

from touchnet.models.qwen2_audio import forward, forward_audio_tower
from touchnet.models.qwen2_audio.processing_qwen2_audio import \
    QWEN2_AUDIO_TEMPLATE_FOR_S2T
from touchnet.utils.distributed import TORCH_DTYPE_MAP
from touchnet.utils.inference import (AudioDataset, InferenceConfig,
                                      collate_fn, init_distributed)

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceConfig])
    (args,) = parser.parse_args_into_dataclasses()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.inference_enable_liger_kernel:
        # 1. monkey patch the forward function to Qwen2Model
        apply_liger_kernel_to_qwen2()
        # 2. monkey patch the forward function of Qwen2AudioEncoder
        Qwen2AudioEncoder.forward = forward_audio_tower
        # 3. monkey patch the forward function to Qwen2AudioForConditionalGeneration
        Qwen2AudioForConditionalGeneration.forward = forward

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    assert (torch.cuda.is_available())
    world_size, local_rank, rank = init_distributed()

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
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
    for infos, audios in dataloader:
        input_ids_buf, attention_mask_buf = [], []
        input_features_buf, feature_attention_mask_buf = [], []
        for i, audio in enumerate(audios):
            audio_inputs = processor.feature_extractor(
                audio,
                sampling_rate=processor.feature_extractor.sampling_rate,
                truncation=False if args.inference_enable_liger_kernel else True,
                return_attention_mask=True,
                padding="max_length",
                return_tensors="pt"
            )
            feature_attention_mask = audio_inputs.pop('attention_mask').squeeze(0)
            if audio_inputs['input_features'].shape[-1] > 3000:
                feature_attention_mask = torch.ones(audio_inputs['input_features'].shape[-1],
                                                    dtype=feature_attention_mask.dtype,
                                                    device=feature_attention_mask.device)
            text = QWEN2_AUDIO_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", args.instruct)
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
            input_ids_buf.append(text_inputs['input_ids'].squeeze(0))  # [seq_len,]
            attention_mask_buf.append(text_inputs['attention_mask'].squeeze(0))  # [seq_len,]
            input_features_buf.append(audio_inputs['input_features'].squeeze(0).transpose(0, 1))  # [3000 or longer, 128]
            feature_attention_mask_buf.append(feature_attention_mask)

        inputs = {
            "input_ids": pad_sequence(input_ids_buf,
                                      batch_first=True,
                                      padding_side='left',
                                      padding_value=processor.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask_buf,
                                           batch_first=True,
                                           padding_side='left',
                                           padding_value=0),
            "input_features": pad_sequence(input_features_buf,
                                           batch_first=True,
                                           padding_side='right',
                                           padding_value=0).transpose(1, 2),  # [B, 128, T]
            "feature_attention_mask": pad_sequence(feature_attention_mask_buf,
                                                   batch_first=True,
                                                   padding_side='right',
                                                   padding_value=0),
        }

        max_length = args.max_length
        if inputs["input_ids"].shape[1] > max_length:
            print(f"input_ids.shape[1] > max_length: {inputs['input_ids'].shape[1]} > {max_length}, skip")
            continue

        for k in inputs.keys():
            inputs[k] = inputs[k].cuda()
        with torch.no_grad():
            generated_ids = model.generate(**inputs,
                                           max_length=max_length,
                                           use_cache=True)
        generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        for i, k in enumerate(infos):
            writer.write(
                json.dumps({
                    "label": json.dumps(infos[i], ensure_ascii=False),
                    "predict": response[i],
                }, ensure_ascii=False) + "\n"
            )
        del inputs
        # clear cache
        torch.cuda.empty_cache()
        gc.collect()
        if rank == 0:
            progress_bar.update(world_size * len(infos))

    if rank == 0:
        progress_bar.close()
    writer.close()
    dist.barrier()
    dist.destroy_process_group()
