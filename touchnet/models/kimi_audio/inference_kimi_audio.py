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
    touchnet/models/kimi_audio/inference_kimi_audio.py \
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
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers.hf_argparser import HfArgumentParser

from touchnet.models.kimi_audio.modeling_kimi_audio import \
    MoonshotKimiaForCausalLM
from touchnet.models.kimi_audio.processing_kimi_audio import (
    KIMI_AUDIO_TEMPLATE_FOR_S2T, KIMI_TEXT_TEMPLATE_FOR_S2T)
from touchnet.utils.distributed import TORCH_DTYPE_MAP
from touchnet.utils.inference import (AudioDataset, InferenceConfig,
                                      collate_fn, init_distributed)

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceConfig])
    (args,) = parser.parse_args_into_dataclasses()
    os.makedirs(args.output_dir, exist_ok=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    assert (torch.cuda.is_available())
    world_size, local_rank, rank = init_distributed()

    model = MoonshotKimiaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=TORCH_DTYPE_MAP[args.model_dtype],
        trust_remote_code=True
    ).cuda().eval()
    # NOTE(xcsong): Always keep speech_tokenizer & speech_encoder in fp32
    # model.speech_tokenizer = model.speech_tokenizer.to(TORCH_DTYPE_MAP["float32"])
    # model.speech_encoder = model.speech_encoder.to(TORCH_DTYPE_MAP["float32"])
    # model.model = model.model.to(TORCH_DTYPE_MAP[args.model_dtype])
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
        whisper_features = feature_extractor(
            audios,
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=feature_extractor.sampling_rate,
            padding="max_length",
        )
        num_audio_tokens_list = whisper_features['attention_mask'][:, ::2][:, ::4].sum(dim=1).tolist()  # [batch_size]

        text_prompts = []
        audio_prompts = []
        task_token_ids = tokenizer(args.instruct, add_special_tokens=False).input_ids
        for num_audio_tokens in num_audio_tokens_list:
            text_prompt = KIMI_TEXT_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", args.instruct)
            text_prompt = text_prompt.replace("<|AUDIO|>", "<|im_kimia_text_blank|>" * num_audio_tokens)
            text_prompts.append(text_prompt)
            audio_prompt = KIMI_AUDIO_TEMPLATE_FOR_S2T.replace("<|INSTRUCT|>", "<|im_kimia_text_blank|>" * len(task_token_ids))
            audio_prompt = audio_prompt.replace("<|AUDIO|>", "<|im_kimia_text_blank|>" * num_audio_tokens)
            audio_prompts.append(audio_prompt)

        text_prompt_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(tokenizer(p, add_special_tokens=False).input_ids, dtype=torch.int64) for p in text_prompts],
            batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left',
        ).cuda()
        audio_prompt_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(tokenizer(p, add_special_tokens=False).input_ids, dtype=torch.int64) for p in audio_prompts],
            batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left',
        ).cuda()

        assert (audio_prompt_ids == 151661).any(), "<|im_media_begin|> not found"
        assert (audio_prompt_ids == 151663).any(), "<|im_media_end|> not found"
        assert text_prompt_ids.size() == audio_prompt_ids.size(), f"{text_prompt_ids.size()}, {audio_prompt_ids.size()}"

        for k in whisper_features.keys():
            whisper_features[k] = whisper_features[k].cuda()

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=TORCH_DTYPE_MAP[args.model_dtype]):
                generated_wav_ids, generated_text_ids = model.generate(
                    text_input_ids=text_prompt_ids,
                    audio_input_ids=audio_prompt_ids,
                    whisper_input_features=whisper_features['input_features'],
                    whisper_attention_mask=whisper_features['attention_mask'],
                    max_new_tokens=2048,
                )
        response = [tokenizer.detokenize(text) for text in generated_text_ids]
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
