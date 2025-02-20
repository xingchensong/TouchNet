# TouchNet

```sh
# NOTE(xcsong): Ensure that the system's glibc version is greater than or equal to 2.17 (see `ldd --version`)
#               (for example, Ubuntu 20.04 and later versions).
conda create -n touchnet python=3.10  # megatron_core requires python>=3.10
conda activate touchnet
conda install -c conda-forge git vim zsh shellcheck tmux cmake nodejs ruby gawk ctags sox -y
conda install -c conda-forge gcc=11.4.0 gxx=11.4.0 libstdcxx-devel_linux-64=11.4.0 -y  # transformer_engine requires gcc>=11.4.0
# install cuda12.6.3+cudnnn9.5.1.17
bash install_cuda_cudnn.sh
# install python packages
pip install pynvim neovim jedi autopep8 cpplint pylint isort cmakelint cmake-format flake8 gpustat nvitop
pip install torch==2.6.0+cu126 torchaudio==2.6.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

- [ ] dataset (& UT)
- [ ] model (& UT)
- [ ] train (& UT)
