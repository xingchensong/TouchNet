. ./path.sh || exit 1

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
echo "Parsed device_ids: ${device_ids[@]}"

stage=0
stop_stage=1

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

job_id=2026

train_set=data/train
dev_set=data/dev
test_sets="data/test"

train_config=config/debug.json
dir=exp/debug

tensorboard_dir=tensorboard
num_workers=6
prefetch=6

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    touchnet/bin/train.py \
        --
fi
