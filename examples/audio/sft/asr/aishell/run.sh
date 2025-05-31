#!/bin/bash

# NOTE(xcsong): change xx_prefix and xx_version to ur setup
cache_prefix=/mnt/user-ssd/songxingchen/share
cuda_prefix=/usr/local
pretrained_weight_dir=""  # for fromscratch training
# pretrained_weight_dir="/mnt/user-ssd/songxingchen/share/modelscope/Llama-3.2-1B-Instruct"  # for continue pretrain
pretrained_tokenizer_dir="/mnt/user-ssd/songxingchen/share/modelscope/Llama-3.2-1B-Instruct"

if [ "${pretrained_weight_dir}" != "" ]; then
  exp_suffix="frompretrain"
else
  exp_suffix="fromscratch"
fi

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

hf_data_repo="AISHELL/AISHELL-1"
hf_data_name="default"

train_set=train
dev_set=dev
test_sets=test

param_dtype="bfloat16"
seed=2025
model_config=Llama-3_2
tensorboard_dir=tensorboard
num_workers=12
prefetch=12

. ./parse_options.sh || exit 1;
. ./path.sh --cache_prefix ${cache_prefix} \
            --cuda_prefix ${cuda_prefix} || exit 1

# exp_id="aishell_1x4096_fullac_cp1_tp1_dp8_pp1_stack7_stride6_flex_packloss_mid_ar_std0.02_acc_normpreproc_wp2k_total10k_addpad_${model_config}_${exp_suffix}"
exp_id="aishell_1x16384_fullac_cp2_tp2_dp2_pp1_stack7_stride6_flex_packloss_mid_ar_std0.02_acc_normpreproc_wp2k_total10k_addpad_${model_config}_${exp_suffix}"
cp=$(echo $exp_id | grep -oP 'cp\d+' | grep -oP '\d+')
tp=$(echo $exp_id | grep -oP 'tp\d+' | grep -oP '\d+')
dp=$(echo $exp_id | grep -oP 'dp\d+' | grep -oP '\d+')
pp=$(echo $exp_id | grep -oP 'pp\d+' | grep -oP '\d+')
stack=$(echo $exp_id | grep -oP 'stack\d+' | grep -oP '\d+')
stride=$(echo $exp_id | grep -oP 'stride\d+' | grep -oP '\d+')
bs=$(echo $exp_id | grep -oP '\d+x\d+' | grep -oP '\d+' | head -n 1)
max_seq_len=$(echo $exp_id | grep -oP '\d+x\d+' | grep -oP '\d+' | tail -n 1)
echo "$0: ${exp_id}: cp=${cp}, tp=${tp}, dp=${dp}, pp=${pp}, stack=${stack}, stride=${stride}, bs=${bs}, max_seq_len=${max_seq_len}"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "$0: stage -1: Data Download"
  python download_aishell.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for x in ${train_set} ${dev_set} ${test_sets}; do
    if [ ! -f "data/${x}/data.list" ]; then
      echo "$0: data/${x}/data.list does not exist. generate dataset."
      mkdir -p data/${x}
      python touchnet/bin/make_data.py \
          --save_dir "data/${x}" \
          --jsonl_path "/mnt/user-ssd/songxingchen/workspace/wenet/examples/aishell/s0/data/${x}/data.list" \
          --num_utt_per_shard 2000 \
          --num_workers 16 \
          --datatypes "audio+metainfo"
    fi
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ "${pretrained_weight_dir}" != "" ]; then
  echo "$0: Stage 1: create seed checkpoint for offline initialization"
  rm -rf "exp/${exp_id}"
  mkdir -p "exp/${exp_id}"
  python touchnet/bin/convert_hf_to_dcp.py \
    --ckpt_dir "exp/${exp_id}" \
    --huggingface_model "${pretrained_weight_dir}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$0: Stage 2: start training"
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  # export TORCH_LOGS="+dynamo"
  # export TORCHDYNAMO_VERBOSE=1
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
           --local-ranks-filter "0,4" \
    touchnet/bin/train.py \
      --tokenizer_model "${pretrained_tokenizer_dir}" \
      --tokenizer_type "HuggingFaceTokenizer" \
      --datapipe_type "touch_audio" \
      --datalist_path "data/${train_set}/data.list" \
      --datalist_dev_path "data/${dev_set}/data.list" \
      --datalist_sharding true \
      --datalist_epoch 10000 \
      --datalist_shuffling true \
      --dataset_shuffling true \
      --dataset_mmap true \
      --dataset_batchsize ${bs} \
      --dataset_audio_seqlen ${max_seq_len} \
      --dataset_text_seqlen ${max_seq_len} \
      --audio_max_length_in_ms_for_filter $(expr $max_seq_len \* $stride \* 10 - 200) \
      --audio_min_length_in_ms_for_filter 200 \
      --text_max_length_in_tokens_for_filter $(expr $max_seq_len - 1) \
      --text_min_length_in_tokens_for_filter 1 \
      --max_text_audio_ratio 1.0 \
      --min_text_audio_ratio 0.0005 \
      --audio_resample_rate 16000 \
      --audio_speed_perturb true \
      --audio_feat_type "fbank" \
      --audiofeat_spec_aug true \
      --audiofeat_spec_aug_num_t_mask 2 \
      --audiofeat_spec_aug_num_f_mask 2 \
      --audiofeat_spec_aug_max_t 50 \
      --audiofeat_spec_aug_max_f 10 \
      --audiofeat_spec_sub true \
      --audiofeat_spec_sub_num_t_sub 3 \
      --audiofeat_spec_sub_max_t 30 \
      --audiofeat_spec_trim false \
      --audiofeat_spec_trim_max_t 20 \
      --audiofeat_num_mel_bins 80 \
      --audiofeat_frame_length 25 \
      --audiofeat_frame_shift 10 \
      --audiofeat_dither 0.0 \
      --audiofeat_stack_length ${stack} \
      --audiofeat_stride_length ${stride} \
      --audiofeat_normalize true \
      --dataloader_num_workers ${num_workers} \
      --dataloader_prefetch_factor ${prefetch} \
      --training_description "aishell asr" \
      --training_seed "${seed}" \
      --training_model_name "llama.asr" \
      --training_model_config_path "config/${model_config}.json" \
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
      --training_max_norm 5.0 \
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
      --lr_scheduler_warmup_steps 2000 \
      --lr_scheduler_decay_type "linear" \
      --lr_scheduler_lr_min 0.0
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$0: Stage 3: convert dcp to huggingface-format"
  python touchnet/bin/convert_dcp_to_hf.py \
    --ckpt_dir "exp/${exp_id}" \
    --step 10000 \
    --config "config/${model_config}.json" \
    --tokenizer_model "${pretrained_tokenizer_dir}"
fi
