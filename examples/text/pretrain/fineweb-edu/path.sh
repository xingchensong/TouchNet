cuda_prefix=/usr/local
cache_prefix=/mnt/user-ssd/songxingchen/share

. ./parse_options.sh || exit 1;

if [ ! -d "${cuda_prefix}" ]; then
    echo "Error: CUDA_HOME directory does not exist: ${cuda_prefix}"
    exit 1
fi
if [ ! -d "${cache_prefix}" ]; then
    echo "Error: cache_prefix directory does not exist: ${cache_prefix}"
    exit 1
fi

# cuda related
export CUDA_HOME=${cuda_prefix}/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib:/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDAToolkit_ROOT_DIR=$CUDA_HOME
export CUDAToolkit_ROOT=$CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS
export CXXFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CXXFLAGS
export LDFLAGS=-L$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib:/usr/lib64:$LDFLAGS
export CUDAToolkit_TARGET_DIR=$CUDA_HOME/targets/x86_64-linux

# python related
export TOUCHNET_DIR=$PWD/../../../..
export PATH=$PWD:$PATH
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../../:$PYTHONPATH

# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1800000000
# export NCCL_LAUNCH_TIMEOUT=6000000000000
# export NCCL_SOCKET_TIMEOUT=3000000000000

# torch related
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # see https://github.com/pytorch/torchtitan/blob/main/docs/composability.md#setting-torch_nccl_avoid_record_streams1-for-tp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME=${cache_prefix}/xdg

# huggingface related
export HF_HOME=${cache_prefix}/huggingface
export NUMBA_CACHE_DIR=${cache_prefix}/numba
export MPLCONFIGDIR=${cache_prefix}/matplotlib

echo "$0: CUDA_HOME: ${CUDA_HOME}"
echo "$0: HF_HOME: ${HF_HOME}"
echo "$0: TOUCHNET_DIR: ${TOUCHNET_DIR}"
echo "$0: XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
echo "$0: NUMBA_CACHE_DIR: ${NUMBA_CACHE_DIR}"
echo "$0: MPLCONFIGDIR: ${MPLCONFIGDIR}"
