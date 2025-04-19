# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)

import io
import os
import tempfile
from datetime import timedelta

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.hf_argparser import HfArgumentParser

import touchnet  # noqa
from touchnet.bin import CkptConverterConfig
from touchnet.utils.logging import init_logger, logger


@torch.inference_mode()
def save_pretrained(args: CkptConverterConfig):
    logger.info(f"Loading the config from {args.config}")
    config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)

    ckpt_hf_dir = os.path.join(args.ckpt_dir, "checkpoint_hf", f"step-{args.step}")
    logger.info(f"Saving the config to {ckpt_hf_dir}")
    config.save_pretrained(ckpt_hf_dir)
    logger.info(f"Loading the tokenizer from {args.tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {ckpt_hf_dir}")
    tokenizer.save_pretrained(ckpt_hf_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(args.ckpt_dir, f'checkpoint/step-{args.step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu',
                                         weights_only=True)['model'])

        logger.info(f"Saving the model to {ckpt_hf_dir}")
        model.save_pretrained(ckpt_hf_dir)


if __name__ == "__main__":
    parser = HfArgumentParser([CkptConverterConfig])
    args = parser.parse_args_into_dataclasses()[0]
    init_logger()
    save_pretrained(args)
