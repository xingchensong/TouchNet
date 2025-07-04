# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)

import json
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
import transformers
from transformers import AutoModelForCausalLM
from transformers.hf_argparser import HfArgumentParser

import touchnet  # noqa
from touchnet.bin import CkptConverterConfig
from touchnet.utils.logging import init_logger, logger

MODEL_DICT = {
    "qwen2_audio": transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioForConditionalGeneration,
}


@torch.inference_mode()
def convert_hf_weights(args: CkptConverterConfig):
    logger.info(f"Loading model from {args.huggingface_model}")
    if args.model_type in MODEL_DICT:
        # for specific hf model
        model = MODEL_DICT[args.model_type].from_pretrained(args.huggingface_model)
    else:
        # for custom model that has beed registered in AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.huggingface_model)

    logger.info(f"huggingface model:\n{model}")
    state_dict = model.state_dict()

    if args.model_type == "touch_audio":
        prefix = "language_model"
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[f"{prefix}.{k}"] = v
        assert args.training_model_config_path is not None, "training_model_config_path is required for touch_audio"
        with open(args.training_model_config_path, "r") as f:
            model_config = json.load(f)
        projector = torch.nn.Linear(model_config["audio_config"]["input_size"],
                                    model_config["text_config"]["hidden_size"], bias=False)
        state_dict["projector.weight"] = projector.weight
        state_dict["projector.bias"] = None
        logger.warning(f"we append `language_model` to the params_name of {model.config.model_type}, and add a projector to the state_dict for proper touch_audio initialization !!")  # noqa
    else:
        # if you want to finetune a standard huggingface model, we should do nothing here
        pass

    checkpoint = Path(os.path.join(args.ckpt_dir, "checkpoint", "step-0"))
    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


if __name__ == "__main__":
    parser = HfArgumentParser([CkptConverterConfig])
    args = parser.parse_args_into_dataclasses()[0]
    init_logger(f"{args.ckpt_dir}/touchnet_convert_hf_to_dcp.log")
    convert_hf_weights(args)
