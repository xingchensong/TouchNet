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

import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.hf_argparser import HfArgumentParser

from touchnet.models.qwen2_audio.processing_qwen2_audio import \
    QWEN2_AUDIO_TEMPLATE_FOR_S2T
from touchnet.utils.distributed import TORCH_DTYPE_MAP
from touchnet.utils.inference import (AudioDataset, InferenceConfig,
                                      collate_fn, init_distributed)

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceConfig])
    (args,) = parser.parse_args_into_dataclasses()
    os.makedirs(args.output_dir, exist_ok=True)
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
        prompts = [QWEN2_AUDIO_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", args.instruct)] * len(infos)
        inputs = processor(
            text=prompts,
            audio=audios,
            return_tensors="pt",
            sampling_rate=processor.feature_extractor.sampling_rate,
            padding=True
        )
        for k in inputs.keys():
            inputs[k] = inputs[k].cuda()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=2048, use_cache=True)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
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
        if rank == 0:
            progress_bar.update(world_size * len(infos))

    if rank == 0:
        progress_bar.close()
    writer.close()
    dist.barrier()
    dist.destroy_process_group()
