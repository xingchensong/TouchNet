# NOTE(xcsong): change xx_prefix and xx_version to ur setup
cache_prefix=/mnt/user-ssd/songxingchen/share
cuda_prefix=/usr/local
pretrained_weight_dir="/mnt/user-ssd/songxingchen/share/modelscope/Llama-3.2-1B-Instruct"

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
echo "$0: CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=1
stop_stage=2

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

job_id=2026

hf_data_repo="Salesforce/wikitext"
hf_data_name="wikitext-103-v1"

train_set=train
dev_set=validation
test_sets=test

param_dtype="bfloat16"

seed=2025
model_config=config/debug.json
exp_id="debug5_1B_1x16384_fullac_cp4_tp1_dp2_pp1_flex_packloss_tieemb_fromscratch_fixROPEbug_dev_wiki0.1B_infinite"
cp=$(echo $exp_id | grep -oP 'cp\d+' | grep -oP '\d+')
tp=$(echo $exp_id | grep -oP 'tp\d+' | grep -oP '\d+')
dp=$(echo $exp_id | grep -oP 'dp\d+' | grep -oP '\d+')
pp=$(echo $exp_id | grep -oP 'pp\d+' | grep -oP '\d+')
bs=$(echo $exp_id | grep -oP '\d+x\d+' | grep -oP '\d+' | head -n 1)
max_seq_len=$(echo $exp_id | grep -oP '\d+x\d+' | grep -oP '\d+' | tail -n 1)
echo "$0: ${exp_id}: cp=${cp}, tp=${tp}, dp=${dp}, pp=${pp}, bs=${bs}, max_seq_len=${max_seq_len}"

tensorboard_dir=tensorboard
num_workers=6
prefetch=6

. ./parse_options.sh || exit 1;
. ./path.sh --cache_prefix ${cache_prefix} \
            --cuda_prefix ${cuda_prefix} || exit 1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "$0: stage -1: Data Download"
  python download_wikitext.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for x in ${train_set} ${dev_set} ${test_sets}; do
    if [ ! -f "data/${x}/data.list" ]; then
      echo "$0: data/${x}/data.list does not exist. generate dataset."
      mkdir -p data/${x}
      python touchnet/bin/make_data.py \
          --save_dir "data/${x}" \
          --jsonl_path ${HF_HOME}/datasets/converted_jsonl_for_touchnet/${hf_data_repo}/${hf_data_name}/${x}.jsonl \
          --tokenizer_model "${pretrained_weight_dir}" \
          --tokenizer_type "HuggingFaceTokenizer" \
          --num_utt_per_shard 2000 \
          --num_workers 16 \
          --datatypes 'texttoken'
    fi
  done
fi

if [[ $exp_id == *"fromseed"* ]] && [ ${stop_stage} -ge 2 ]; then
  pretrained_weight_dir=""
  stage=1
  stop_stage=2
fi

if [[ $exp_id == *"fromscratch"* ]] && [ ${stop_stage} -ge 2 ]; then
  stage=2
  stop_stage=2
fi

if [[ $exp_id == *"frompretrain"* ]] && [ ${stop_stage} -ge 2 ]; then
  stage=1
  stop_stage=2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$0: Stage 1: create seed checkpoint for offline initialization"
  rm -rf "exp/${exp_id}"
  mkdir -p "exp/${exp_id}"
  torchrun --nnodes=1 --nproc_per_node=1 \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    touchnet/bin/train.py \
      --tokenizer_model "${pretrained_weight_dir}" \
      --tokenizer_type "HuggingFaceTokenizer" \
      --datalist_path "data/${train_set}/data.list" \
      --training_seed "${seed}" \
      --training_model_name "llama" \
      --training_model_config_path "${model_config}" \
      --training_model_pretrained_weight_dir "${pretrained_weight_dir}" \
      --training_print_args true \
      --training_trace_dump_folder "exp/${exp_id}" \
      --training_enable_ckpt true \
      --training_create_seed_ckpt true
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$0: Stage 2: start training"
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  # export TORCH_LOGS="+dynamo"
  # export TORCHDYNAMO_VERBOSE=1
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
           --local-ranks-filter "0,1,2,3,4,5,6,7" \
    touchnet/bin/train.py \
      --tokenizer_model "${pretrained_weight_dir}" \
      --tokenizer_type "HuggingFaceTokenizer" \
      --datalist_path "data/${train_set}/data.list" \
      --datalist_dev_path "data/${dev_set}/data.list" \
      --datalist_sharding true \
      --datalist_epoch 10000 \
      --datalist_shuffling true \
      --dataset_shuffling true \
      --dataset_mmap true \
      --dataset_batchsize ${bs} \
      --dataset_text_seqlen ${max_seq_len} \
      --text_max_length_in_tokens_for_filter $(expr $max_seq_len - 2) \
      --text_min_length_in_tokens_for_filter 1 \
      --dataloader_num_workers ${num_workers} \
      --dataloader_prefetch_factor ${prefetch} \
      --training_description "debug only" \
      --training_seed "${seed}" \
      --training_model_name "llama" \
      --training_model_config_path "${model_config}" \
      --training_print_args true \
      --training_trace_dump_folder "exp/${exp_id}" \
      --training_fsdp_reshard_after_forward "default" \
      --training_context_parallel_degree ${cp} \
      --training_context_parallel_rotate_method "allgather" \
      --training_tensor_parallel_degree ${tp} \
      --training_enable_loss_parallel true \
      --training_pipeline_parallel_degree ${pp} \
      --training_pipeline_parallel_schedule "1F1B" \
      --training_enable_ckpt true \
      --training_ckpt_load_step -1 \
      --training_ckpt_interval 100 \
      --training_ckpt_keep_latest_k 2 \
      --training_log_freq 1 \
      --training_enable_tensorboard true \
      --training_save_tb_folder "tensorboard" \
      --training_tb_rank_0_only true \
      --training_mixed_precision_param "${param_dtype}" \
      --training_mixed_precision_reduce "float32" \
      --training_compile true \
      --training_enable_compiled_autograd false \
      --training_gc_freq 500 \
      --training_deterministic false \
      --training_max_norm 1.0 \
      --training_activation_checkpoint_mode "full" \
      --training_activation_checkpoint_selective_ac_option "op" \
      --training_enable_profiling true \
      --training_profiling_traces_folder "profile_traces" \
      --training_profiling_freq 100 \
      --training_profiling_keep_first_k 10 \
      --training_enable_memory_snapshot true \
      --training_memory_snapshot_folder "memory_snapshot" \
      --optimizer_name "AdamW" \
      --optimizer_lr 8e-4 \
      --optimizer_impl "fused" \
      --lr_scheduler_steps 10000 \
      --lr_scheduler_warmup_steps 200 \
      --lr_scheduler_decay_type "linear" \
      --lr_scheduler_lr_min 0.0
fi
