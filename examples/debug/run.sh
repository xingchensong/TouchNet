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

stage=2
stop_stage=2

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

job_id=2026

train_set=wikitext-2-v1.train.repeat10000
dev_set=wikitext-2-v1.validation
test_sets=wikitext-2-v1.test

train_config=config/debug.json
dir=exp/debug

tensorboard_dir=tensorboard
num_workers=6
prefetch=6

. ./parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for x in ${train_set} ${dev_set} ${test_sets}; do
    if [ ! -d "data/${x}/data.list" ]; then
      echo "$0: data/${x}/data.list does not exist. generate dataset."
      mkdir -p data/${x}
      python touchnet/bin/make_data.py \
          --save_dir "data/${x}" \
          --jsonl_path /bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface/datasets/converted_jsonl_xcsong/${x}.jsonl \
          --tokenizer_model "/bucket/output/jfs-hdfs/user/xingchen.song/share/modelscope/Llama-3.2-1B-Instruct" \
          --tokenizer_type "HuggingFaceTokenizer" \
          --num_utt_per_shard 2000 \
          --num_workers 16 \
          --datatypes 'texttoken'
    fi
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$0: create seed checkpoint for offline initialization"
  torchrun --nnodes=1 --nproc_per_node=1 \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    touchnet/bin/train.py \
      --tokenizer_model "/bucket/output/jfs-hdfs/user/xingchen.song/share/modelscope/Llama-3.2-1B-Instruct" \
      --tokenizer_type "HuggingFaceTokenizer" \
      --datalist_path "data/${train_set}/data.list" \
      --training_model_name "llama" \
      --training_model_config_path "config/debug.json" \
      --training_print_args true \
      --training_trace_dump_folder "exp/debug" \
      --training_enable_ckpt true \
      --training_create_seed_ckpt true
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$0: starting training"
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  # export TORCH_LOGS="+dynamo"
  # export TORCHDYNAMO_VERBOSE=1
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
           --local-ranks-filter "0,4" \
    touchnet/bin/train.py \
      --tokenizer_model "/bucket/output/jfs-hdfs/user/xingchen.song/share/modelscope/Llama-3.2-1B-Instruct" \
      --tokenizer_type "HuggingFaceTokenizer" \
      --datalist_path "data/${train_set}/data.list" \
      --datalist_sharding true \
      --datalist_shuffling true \
      --dataset_shuffling true \
      --dataset_mmap true \
      --dataset_batchsize 1 \
      --dataset_text_seqlen 4096 \
      --text_max_length_in_tokens_for_filter 4096 \
      --text_min_length_in_tokens_for_filter 1 \
      --dataloader_num_workers 6 \
      --dataloader_prefetch_factor 6 \
      --training_description "debug only" \
      --training_model_name "llama" \
      --training_model_config_path "config/debug.json" \
      --training_print_args true \
      --training_trace_dump_folder "exp/debug5_1B_1x4k_fullac_dp8_sdpa_rerun_rerun" \
      --training_fsdp_reshard_after_forward "default" \
      --training_context_parallel_degree 1 \
      --training_context_parallel_rotate_method "allgather" \
      --training_tensor_parallel_degree 1 \
      --training_enable_loss_parallel false \
      --training_pipeline_parallel_degree 1 \
      --training_pipeline_parallel_schedule "1F1B" \
      --training_enable_ckpt true \
      --training_ckpt_load_step -1 \
      --training_ckpt_interval 100 \
      --training_log_freq 1 \
      --training_enable_tensorboard true \
      --training_save_tb_folder "tensorboard" \
      --training_tb_rank_0_only true \
      --training_mixed_precision_param "bfloat16" \
      --training_mixed_precision_reduce "float32" \
      --training_compile true \
      --training_enable_compiled_autograd false \
      --training_gc_freq 500 \
      --training_seed 2025 \
      --training_deterministic false \
      --training_max_norm 1.0 \
      --training_activation_checkpoint_mode "full" \
      --training_activation_checkpoint_selective_ac_option "op" \
      --training_enable_profiling false \
      --training_profiling_traces_folder "profile_traces" \
      --training_profiling_freq 100 \
      --training_profiling_enable_memory_snapshot true \
      --training_profiling_save_memory_snapshot_folder "memory_snapshot" \
      --optimizer_name "AdamW" \
      --optimizer_lr 8e-4 \
      --optimizer_impl "fused" \
      --lr_scheduler_steps 10000 \
      --lr_scheduler_warmup_steps 200 \
      --lr_scheduler_decay_type "linear" \
      --lr_scheduler_lr_min 0.0
fi
