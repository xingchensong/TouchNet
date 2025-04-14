from typing import Tuple

import torch
from torch.distributed.tensor import DTensor, Replicate

from touchnet.loss import COMPILED_LOSSES


def cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor,
    sentence_lens: torch.Tensor, num_sentence: int,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Common cross-entropy loss function for Transformer models training.

    Args:
        pred (Tensor): Prediction tensors (B, Lmax // cp, Vocab // tp) if pred.to_local()
                       else (B, Lmax // cp, Vocab) if pred.full_tensor()
        labels (LongTensor): Target label tensors (B, Lmax // cp).
        ignore_index (int): Ignore label id.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loss value.
    """

    if isinstance(pred, DTensor):
        # NOTE(xcsong): make sentence_lens distributed to work with DTensor-style Loss
        sentence_lens = DTensor.from_local(
            sentence_lens, pred.device_mesh, [Replicate()], run_check=False
        )  # (bs, seq_len // cp)
    batch_size = pred.size(0)
    num_tokens = (labels != ignore_index).sum().item()
    loss = COMPILED_LOSSES["ce"](
        pred, labels, "none", ignore_index
    )  # (bs * seq_len // cp,)
    # NOTE(xcsong): per-sample loss for backward while per-token loss for logging.
    loss_per_token = loss.sum()
    if loss_per_token > 1e-6 and num_tokens > 0:
        loss_per_token = (loss.sum() / num_tokens)  # (1,)
    else:
        loss_per_token = torch.zeros_like(loss_per_token)
    loss_per_sample = loss.reshape(batch_size, -1)  # (bs, seq_len // cp)
    # 1. reduce loss over sentence
    loss_per_sample = torch.sum(loss_per_sample / sentence_lens, dim=-1)  # (bs,)
    # 2. reduce loss over global-batch
    loss_per_sample = torch.sum(loss_per_sample) / num_sentence  # (1,)
    return loss_per_sample, loss_per_token
