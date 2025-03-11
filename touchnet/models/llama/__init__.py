import torch
from transformers import AutoConfig, AutoModelForCausalLM

from touchnet.data.dataloader import build_dataloader
from touchnet.models.llama.parallelize_llama import parallelize_llama
from touchnet.models.llama.pipeline_llama import pipeline_llama
from touchnet.tokenizer.tokenizer import build_tokenizer
from touchnet.utils.metrics import build_metrics_processor
from touchnet.utils.optimizer import build_lr_schedulers, build_optimizers
from touchnet.utils.train_spec import TrainSpec, register_train_spec


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred[0].flatten(0, 1).float(), labels.flatten(0, 1)
    )


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
