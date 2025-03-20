from typing import Tuple

import torch
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoConfig, AutoModelForCausalLM

from touchnet.data.dataloader import build_dataloader
from touchnet.models.llama.parallelize_llama import parallelize_llama
from touchnet.models.llama.pipeline_llama import pipeline_llama
from touchnet.tokenizer.tokenizer import build_tokenizer
from touchnet.utils.metrics import build_metrics_processor
from touchnet.utils.optimizer import build_lr_schedulers, build_optimizers
from touchnet.utils.train_spec import TrainSpec, register_train_spec


def cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor,
    sentence_lens: torch.Tensor, num_sentence: int,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Common cross-entropy loss function for Transformer models training."""
    if isinstance(pred[0], DTensor):
        # NOTE(xcsong): make sentence_lens distributed to work with DTensor-style Loss
        sentence_lens = DTensor.from_local(
            sentence_lens, pred[0].device_mesh, [Replicate()], run_check=False
        )  # (bs, seq_len // cp)
    batch_size = pred[0].size(0)
    num_tokens = (labels != ignore_index).sum().item()
    # logits.shape = pred[0].shape = (bs, seq_len // cp, vocab_size // tp)
    loss = torch.nn.functional.cross_entropy(
        pred[0].flatten(0, 1).float(), labels.flatten(0, 1),
        reduction="none", ignore_index=ignore_index,
    )  # (bs * seq_len // cp,)
    # NOTE(xcsong): per-sample loss for backward while per-token loss for logging.
    loss_per_token = (loss.sum() / num_tokens + 1e-6)  # (1,)
    loss_per_sample = loss.reshape(batch_size, -1)  # (bs, seq_len // cp)
    # 1. reduce loss over sentence
    loss_per_sample = torch.sum(loss_per_sample / sentence_lens, dim=-1)  # (bs,)
    # 2. reduce loss over global-batch
    loss_per_sample = torch.sum(loss_per_sample) / num_sentence  # (1,)
    return loss_per_sample, loss_per_token


register_train_spec(
    TrainSpec(
        name="llama",
        model_cls=AutoModelForCausalLM,
        config_cls=AutoConfig,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        loss_fn=cross_entropy_loss,
        build_metrics_processor_fn=build_metrics_processor,
    )
)
