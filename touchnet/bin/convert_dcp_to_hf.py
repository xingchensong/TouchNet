# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)

import io
import os
import tempfile
from datetime import timedelta

import torch
import torch.serialization
import transformers
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import (AutoConfig, AutoFeatureExtractor,
                          AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          GenerationConfig)
from transformers.hf_argparser import HfArgumentParser

import touchnet  # noqa
from touchnet.bin import CkptConverterConfig
from touchnet.utils.logging import init_logger, logger

MODEL_DICT = {
    "qwen2_audio": transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioForConditionalGeneration,
}


@torch.inference_mode()
def save_pretrained(args: CkptConverterConfig):
    logger.info(f"Loading the config from {args.config}")
    config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.tokenizer_model)

    ckpt_hf_dir = os.path.join(args.ckpt_dir, "checkpoint_hf", f"step-{args.step}")
    logger.info(f"Saving the config to {ckpt_hf_dir}")
    config.save_pretrained(ckpt_hf_dir)
    logger.info(f"Loading the tokenizer from {args.tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {ckpt_hf_dir}")
    tokenizer.save_pretrained(ckpt_hf_dir)

    with tempfile.TemporaryDirectory(prefix="mytmp_", dir=args.tmp_dir) as tmpdir:
        checkpoint = os.path.join(args.ckpt_dir, f'checkpoint/step-{args.step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        if args.model_type in MODEL_DICT:
            # for specific hf model
            model = MODEL_DICT[args.model_type](config)
        else:
            # for custom model that has beed registered in AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        model.generation_config = generation_config

        logger.info(f"Model:\n{model}")
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])

        # torch.load now with default weights_only=True will work
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['model']
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            raise ValueError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        logger.info(f"Saving the model to {ckpt_hf_dir}")
        model.save_pretrained(ckpt_hf_dir)

        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(args.tokenizer_model)
            processor_component_to_save = feature_extractor
        except Exception:
            try:
                processor = AutoProcessor.from_pretrained(args.tokenizer_model)
                processor_component_to_save = processor
            except Exception:
                processor_component_to_save = None

        if processor_component_to_save:
            processor_component_to_save.save_pretrained(ckpt_hf_dir)


if __name__ == "__main__":
    parser = HfArgumentParser([CkptConverterConfig])
    args = parser.parse_args_into_dataclasses()[0]
    init_logger(f"{args.ckpt_dir}/touchnet_convert_dcp_to_hf.log")
    save_pretrained(args)
