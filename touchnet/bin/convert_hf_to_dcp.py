# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM
from transformers.hf_argparser import HfArgumentParser

import touchnet  # noqa
from touchnet.bin import CkptConverterConfig
from touchnet.utils.logging import init_logger, logger


@torch.inference_mode()
def convert_hf_weights(args: CkptConverterConfig):
    logger.info(f"Loading model from {args.huggingface_model}")
    model = AutoModelForCausalLM.from_pretrained(args.huggingface_model)
    logger.info(model)
    state_dict = model.state_dict()

    checkpoint = Path(os.path.join(args.ckpt_dir, "checkpoint", "step-0"))
    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


if __name__ == "__main__":
    parser = HfArgumentParser([CkptConverterConfig])
    args = parser.parse_args_into_dataclasses()[0]
    init_logger()
    convert_hf_weights(args)
