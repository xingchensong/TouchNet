# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
#               2025, Xingchen Song(sxc19@tsinghua.org.cn)
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger()


def init_logger(log_file=None):
    logger.setLevel(logging.INFO)

    rank = int(os.environ.get('RANK', 0))
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - cuda{rank} - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None and rank == 0:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"The logs will be output to both the console and the file: {log_file}")

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""
