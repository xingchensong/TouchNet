# -*- coding: utf-8 -*-
# Copyright (c) 2025, Xingchen Song(sxc19@tsinghua.org.cn)

import torch


def _cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor,
    reduction: str = "none", ignore_index: int = -100
) -> torch.Tensor:
    """Common cross-entropy loss function for compilation wrapping.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz, num_tokens, vocab_size)``.
    The CE and upcasting can be compiled together for better memory footprint.

    """
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1),
        reduction=reduction, ignore_index=ignore_index,
    )  # (bs * seq_len // cp,)


COMPILED_LOSSES = {
    "ce": torch.compile(_cross_entropy_loss),
}
