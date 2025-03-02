# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1800000000
# export NCCL_LAUNCH_TIMEOUT=6000000000000
# export NCCL_SOCKET_TIMEOUT=3000000000000
export HF_HOME=/bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CACHE_DIR=/bucket/output/jfs-hdfs/user/xingchen.song/share/cache
export MPLCONFIGDIR=/bucket/output/jfs-hdfs/user/xingchen.song/share/cache/matplotlib
export XDG_CACHE_HOME=/bucket/output/jfs-hdfs/user/xingchen.song/share/cache/xdg

# cuda related
unset LD_LIBRARY_PATH
prefix=/bucket/output/jfs-hdfs/user/xingchen.song/tools/cuda
cuda_version=12.6.3
driver_version=560.35.05
cudnn_version=9.5.1.17
export CUDA_HOME=${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version}
export PATH=$CUDA_HOME/bin:$PATH
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
export CXXFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS
export LDFLAGS=-L$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib:/usr/lib64:$LDFLAGS
export CUDAToolkit_TARGET_DIR=$CUDA_HOME/targets/x86_64-linux

# python related
export TOUCHNET_DIR=$PWD/../..
export PATH=$PWD:$PATH
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH
