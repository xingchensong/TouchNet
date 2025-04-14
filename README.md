<div align="center">

# TouchNet [WIP]

#### A PyTorch native 4-D parallel library for large-scale multimodel LLM (text/audio/video) training

[![integration tests](https://github.com/xingchensong/TouchNet/actions/workflows/unit_test_cpu.yaml/badge.svg?branch=main)](https://github.com/xingchensong/TouchNet/actions/workflows/unit_test_cpu.yaml?query=branch%3Amain)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](docs/)
[![license](https://img.shields.io/badge/license-Apache_2-lightgrey.svg)](./LICENSE)

</div>

## Overview

`touchnet` is a clean, minimal codebase for large-scale Multimodal LLM training using native PyTorch. It is currently in a pre-release state and under extensive development.

Our guiding principles when building `touchnet`:

- ⚡️ Blazing-fast checkpointable data loader with modular preprocessing & ​**​full random access​**​ for ultra-large scale multimodal data
- 🤗 Native integration with `transformers` ecosystem.
- 🛠️ Built-in profilers (CPU/GPU/memory) with flight recorder diagnostics.
- 🎯 4-D parallelism enabled through minimal lines of model code changes.
- ✨ Intuitive API design for rapid adoption & customization in minutes.

## Installation

```sh
# NOTE(xcsong): Ensure that the system's glibc version is greater than or equal to 2.17 (see `ldd --version`)
#               (for example, Ubuntu 20.04 and later versions).
conda create -n touchnet python=3.10  # megatron_core requires python>=3.10
conda activate touchnet
conda install -c conda-forge git vim zsh shellcheck tmux cmake nodejs ruby gawk ctags sox ffmpeg -y
conda install -c conda-forge gcc=11.4.0 gxx=11.4.0 libstdcxx-devel_linux-64=11.4.0 -y  # transformer_engine requires gcc>=11.4.0
# install cuda12.6.3+cudnnn9.5.1.17
bash install_cuda_cudnn.sh
# install python packages
pip install pynvim neovim jedi autopep8 cpplint pylint isort cmakelint cmake-format flake8 gpustat nvitop
pip install torch==2.6.0+cu126 torchaudio==2.6.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Citation

```
@misc{touchnet,
  title={TouchNet: A PyTorch native 4-D parallel library for large-scale multimodal LLM (text/audio/video) training},
  author={Xingchen Song},
  year={2025},
  url={https://github.com/xingchensong/TouchNet},
}
```

## Acknowledge

1. This project is highly motivated by [torchtitan](https://github.com/pytorch/torchtitan) and we borrowed a lot of code from it.

Thanks for their wonderful works.
