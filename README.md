<div align="center">

![TouchNet](./assets/Touchnet_16_9.jpg)

# 👆 TouchNet [WIP]

#### A PyTorch native N-D parallel library for large-scale multimodal LLM (text/audio) training

[![integration tests](https://github.com/xingchensong/TouchNet/actions/workflows/unit_test_cpu.yaml/badge.svg?branch=main)](https://github.com/xingchensong/TouchNet/actions/workflows/unit_test_cpu.yaml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](docs/)
[![license](https://img.shields.io/badge/license-Apache_2-lightgrey.svg)](./LICENSE)

</div>

## Overview

`👆 touchnet` is highly motivated by `torchtitan`. Both of them are clean, minimal codebases for large-scale LLM training using native PyTorch. The main goal that differentiates `👆 touchnet` from `torchtitan` is that `👆 touchnet` focuses on multimodal LLM training where special data pipelines and model structures are needed. Please note that `👆 touchnet` is currently in a pre-release state and under extensive development.

Our guiding principles when building `👆 touchnet` are:

1. ⚡️ Blazing-fast checkpointable data loader with modular preprocessing and ​**​fully random access​**​ for large scale **multimodal** data
    - [[New Storage Format]](https://github.com/xingchensong/TouchNet/blob/main/docs/data.md) optimized for random access on sequentially saved tar files
    - Efficient [[Sequence Packing]](https://huggingface.co/blog/sirluk/llm-sequence-packing) powered by [[Flex Attention]](https://pytorch.org/docs/main/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)
2. 🤗 Native integration with `transformers` models while get rid of structured trainer classes (e.g., [[PyTorch-Lightning]](https://github.com/Lightning-AI/pytorch-lightning) or [[HuggingFace Trainer]](https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/trainer#transformers.Trainer))
    - Only reuse model definitions in `transformers` and leave other parts untouched
    - Entire training logic exposed in a single file [[touchnet/bin/train.py]](https://github.com/xingchensong/TouchNet/blob/main/touchnet/bin/train.py), everything is under your control
3. 🛠️ Built-in profilers (CPU/GPU/memory) with flight recorder diagnostics.
    - [[Nsys-like Profiler]](https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md) to get optimization recommendations
    - [[Memory Monitor]](https://pytorch.org/blog/understanding-gpu-memory-1/) to debug OOM errors and improve memory usage
4. 🎯 N-D parallelism enabled through **PyTorch native API** and minimal lines of model code changes.
    - [[FSDP2]](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), [why FSDP1 -> FSDP2?](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
    - [[Tensor Parallel]](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), [[Context Parallel]](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082), [[Pipeline Parallel]](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420) (PP WIP🚧), [[Distributed Checkpoint]](https://pytorch.org/docs/stable/distributed.checkpoint.html)
5. ✨ Intuitive API design for rapid adoption & customization in minutes.
    - Supported tasks: [[text/pretrain]](https://github.com/xingchensong/TouchNet/tree/main/examples/text/pretrain), [[audio/pretrain]](https://github.com/xingchensong/TouchNet/tree/main/examples/audio/pretrain), [[audio/sft/asr]](https://github.com/xingchensong/TouchNet/tree/main/examples/audio/sft/asr), more tasks coming soon
    - Supported models: [[Llama]](https://github.com/xingchensong/TouchNet/tree/main/touchnet/models/llama), more models coming soon


## Quick Glance at 👆 TouchNet

<div align="center">

https://github.com/user-attachments/assets/9e530ad6-2d8d-41b4-9223-8ad7c838e6e4

Loss, Accuracy, Memory, Throughput, TFLOPs, and MFU logged via both stdout and Tensorboard.

https://github.com/user-attachments/assets/dc089589-a355-4abc-a2b3-5e0f768b89a0

Detailed CPU/GPU profiling that can be visualized in Tensorboard. Enjoy your optimization journey ~

https://github.com/user-attachments/assets/10cbf4ce-5f96-4699-b4f4-72c88ce89802

Memory profiling identifies GPU memory allocation patterns to guide tuning strategies.

</div>

## TODO

- [ ] support audio/sft/tts
- [ ] support MoE
- [ ] support vision/pretrain vision/sft
- [ ] support text/sft

## Installation

```sh
# NOTE(xcsong): Ensure that the linux system's glibc version is greater than or equal to 2.17 (see `ldd --version`)
#               (for example, Ubuntu 22.04 and later versions).
conda create -n touchnet python=3.10
conda activate touchnet
conda install -c conda-forge sox ffmpeg -y
# install cuda12.6.3+cudnnn9.5.1.17, be aware to change `prefix` to your path.
bash install_cuda_cudnn.sh
# install the most recent PyTorch to use the latest features of parallelism. recommended torch>=2.7.0
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
pip install .
```

## Citation

```txt
@misc{touchnet,
  title={TouchNet: A PyTorch native N-D parallel library for large-scale multimodal LLM (text/audio) training},
  author={Xingchen Song},
  year={2025},
  url={https://github.com/xingchensong/TouchNet},
}
```

## Acknowledge

1. This repo is highly motivated by [torchtitan](https://github.com/pytorch/torchtitan) and we borrowed a lot of code from it.
2. This repo also benefits from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [WeNet](https://github.com/wenet-e2e/wenet), [flame](https://github.com/fla-org/flame).

Thanks for their wonderful works.
