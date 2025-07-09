#!/bin/bash

# NOTE(xcsong): change xx_prefix and xx_version to ur setup
cache_prefix=/mnt/user-ssd/songxingchen/share
cuda_prefix=/usr/local

# NOTE(xcsong): Qwen2-Audio-7B  https://modelscope.cn/models/Qwen/Qwen2-Audio-7B
# pretrained_weight_dir="${cache_prefix}/modelscope/Qwen2-Audio-7B"  # for fintuning
pretrained_weight_dir=""  # for fromscratch training
pretrained_tokenizer_dir="${cache_prefix}/modelscope/Qwen2-Audio-7B"
pretrained_processor_dir="${cache_prefix}/modelscope/Qwen2-Audio-7B"

# NOTE(xcsong): Kimi-Audio-7B-Instruct  https://www.modelscope.cn/models/xingchensong/Kimi-Audio-7B-Instruct-with-Tokenizer-Encoder
# pretrained_weight_dir="${cache_prefix}/modelscope/Kimi-Audio-7B-Instruct-with-Tokenizer-Encoder"
# pretrained_tokenizer_dir="${cache_prefix}/modelscope/Kimi-Audio-7B-Instruct-with-Tokenizer-Encoder"
# pretrained_processor_dir="${cache_prefix}/modelscope/Kimi-Audio-7B-Instruct-with-Tokenizer-Encoder"

# NOTE(xcsong): Kimi-Audio-7B  https://www.modelscope.cn/models/xingchensong/Kimi-Audio-7B-with-Tokenizer-Encoder
# pretrained_weight_dir="${cache_prefix}/modelscope/Kimi-Audio-7B-with-Tokenizer-Encoder"
# pretrained_tokenizer_dir="${cache_prefix}/modelscope/Kimi-Audio-7B-with-Tokenizer-Encoder"
# pretrained_processor_dir="${cache_prefix}/modelscope/Kimi-Audio-7B-with-Tokenizer-Encoder"


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

job_id=2026

hf_data_repo="wenet-e2e/wenetspeech"
hf_data_name="default"

train_set=train_l
dev_set=dev
test_sets="test_net test_meeting"

param_dtype="bfloat16"
seed=2025
tensorboard_dir=tensorboard
num_workers=12
prefetch=12
activation_checkpoint_mode="full"
audio_max_length_in_ms_for_filter=30000  # 30s
liger=false
compile=true

if [[ "${pretrained_tokenizer_dir}" == *"Qwen2-Audio-7B"* ]]; then
  bs=2
  max_seq_len=8192
  model_type="qwen2_audio"
  model_config="Qwen2-Audio-7B"
  pack=false
  if [[ "${exp_suffix}" == "frompretrain" ]]; then
    num_nodes=1  # NOTE(xcsong): for sft, 1 node with 8 gpus (80GB memory) is enough
    HOST_NODE_ADDR="localhost:0"
    lr=2e-5
    lr_scheduler_steps=30000
    lr_scheduler_warmup_steps=2000
  elif [[ "${exp_suffix}" == "fromscratch" ]]; then
    num_nodes=4  # NOTE(xcsong): for from scratch training, we need 4 nodes with 8 gpus per node (80GB memory)
    HOST_NODE_ADDR="xx.xx.xx.xx:9901"  # NOTE(xcsong): change to your master ip, https://pytorch.org/docs/stable/elastic/run.html
    lr=2e-4
    lr_scheduler_steps=30000
    lr_scheduler_warmup_steps=2000
  fi
elif [[ "${pretrained_tokenizer_dir}" == *"Kimi-Audio-7B"* ]]; then
  bs=1
  max_seq_len=8192
  model_type="kimi_audio"
  if [[ "${pretrained_tokenizer_dir}" == *"Kimi-Audio-7B-Instruct"* ]]; then
    model_config="Kimi-Audio-7B-Instruct"
  else
    model_config="Kimi-Audio-7B"
  fi
  pack=false
  if [[ "${exp_suffix}" == "frompretrain" ]]; then
    num_nodes=4
    HOST_NODE_ADDR="xx.xx.xx.xx:9901"  # NOTE(xcsong): change to your master ip, https://pytorch.org/docs/stable/elastic/run.html
    lr=2e-5
    lr_scheduler_steps=30000
    lr_scheduler_warmup_steps=2000
  elif [[ "${exp_suffix}" == "fromscratch" ]]; then
    echo "fromscratch is not supported for Kimi-Audio"
    exit 1
  fi
else
  num_nodes=4
  HOST_NODE_ADDR="xx.xx.xx.xx:9901"  # NOTE(xcsong): change to your master ip, https://pytorch.org/docs/stable/elastic/run.html
  bs=2
  max_seq_len=8192
  model_type="touch_audio"
  model_config="Touch-Audio-7B"
  stack=13
  stride=12
  pack=true
  echo "TODO(xcsong): recipe for Touch-Audio-7B"
  exit 1
fi

datapipe_type=${model_type}
checkpoint_step=${lr_scheduler_steps}

. ./parse_options.sh || exit 1;
. ./path.sh --cache_prefix ${cache_prefix} \
            --cuda_prefix ${cuda_prefix} || exit 1

git config --global --add safe.directory $(realpath ../../../../../)
commit=$(git rev-parse HEAD | cut -c 1-7)
exp_id="nodes${num_nodes}_wenetspeech_${bs}x${max_seq_len}_cp1_tp1_dp8_pp1_lr${lr}_wp${lr_scheduler_warmup_steps}_total${lr_scheduler_steps}_${model_config}_filter${audio_max_length_in_ms_for_filter}_${exp_suffix}_${commit}_ac${activation_checkpoint_mode}_liger${liger}"

cp=$(echo $exp_id | grep -oP 'cp\d+' | grep -oP '\d+')
tp=$(echo $exp_id | grep -oP 'tp\d+' | grep -oP '\d+')
dp=$(echo $exp_id | grep -oP 'dp\d+' | grep -oP '\d+')
pp=$(echo $exp_id | grep -oP 'pp\d+' | grep -oP '\d+')

echo "================================================"
echo "$0: exp_id: ${exp_id}"
echo "$0: chosen_model=${model_config}, activation_checkpoint_mode=${activation_checkpoint_mode}"
echo "$0: num_nodes=${num_nodes}, cp=${cp}, tp=${tp}, dp=${dp}, pp=${pp}, bs=${bs}, max_seq_len=${max_seq_len}, liger=${liger}"
echo "================================================"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "================================================"
  echo "$0: stage -1: Data Download"
  echo "================================================"
  python download_wenetspeech.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for x in ${train_set} ${dev_set} ${test_sets}; do
    if [ ! -f "data/${x}/data.list" ]; then
      echo "================================================"
      echo "$0: data/${x}/data.list does not exist. generate dataset."
      echo "================================================"
      mkdir -p data/${x}
      python touchnet/bin/make_data.py \
          --save_dir "data/${x}" \
          --jsonl_path "/mnt/user-ssd/songxingchen/workspace/wenet/examples/wenetspeech/s0/data/${x}/data.list" \
          --num_utt_per_shard 2000 \
          --num_workers 64 \
          --datatypes "audio+metainfo"
    fi
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ "${pretrained_weight_dir}" != "" ]; then
  echo "================================================"
  echo "$0: Stage 1: create seed checkpoint for offline initialization"
  echo "================================================"
  mkdir -p "exp/${exp_id}"
  python touchnet/bin/convert_hf_to_dcp.py \
    --ckpt_dir "exp/${exp_id}" \
    --model_type "${model_type}" \
    --training_model_config_path "config/${model_config}.json" \
    --huggingface_model "${pretrained_weight_dir}"
  cp "config/${model_config}.json" "exp/${exp_id}/model_config.json"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "================================================"
  echo "$0: Stage 2: start training"
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  echo "================================================"
  # export TORCH_LOGS="+dynamo"
  # export TORCHDYNAMO_VERBOSE=1
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
           --local-ranks-filter "0,1,2,3,4,5,6,7" \
    touchnet/bin/train.py \
      --tokenizer_model "${pretrained_tokenizer_dir}" \
      --tokenizer_type "HuggingFaceTokenizer" \
      --processor_model "${pretrained_processor_dir}" \
      --datapipe_type "${datapipe_type}" \
      --datalist_path "data/${train_set}/data.list" \
      --datalist_dev_path "data/${dev_set}/data.list" \
      --datalist_sharding true \
      --datalist_epoch 1000 \
      --datalist_shuffling true \
      --dataset_enable_pack ${pack} \
      --dataset_shuffling true \
      --dataset_mmap true \
      --dataset_batchsize ${bs} \
      --dataset_audio_seqlen ${max_seq_len} \
      --dataset_text_seqlen ${max_seq_len} \
      --audio_max_length_in_ms_for_filter ${audio_max_length_in_ms_for_filter} \
      --audio_min_length_in_ms_for_filter 200 \
      --text_max_length_in_tokens_for_filter $(expr $max_seq_len - 1) \
      --text_min_length_in_tokens_for_filter 1 \
      --dataloader_num_workers ${num_workers} \
      --dataloader_prefetch_factor ${prefetch} \
      --training_init_timeout_seconds 300 \
      --training_description "wenetspeech asr, ${model_type}" \
      --training_seed "${seed}" \
      --training_model_name "${model_type}" \
      --training_model_config_path "config/${model_config}.json" \
      --training_print_args true \
      --training_trace_dump_folder "exp/${exp_id}" \
      --training_fsdp_reshard_after_forward "default" \
      --training_data_parallel_replicate_degree ${num_nodes} \
      --training_context_parallel_degree ${cp} \
      --training_context_parallel_rotate_method "allgather" \
      --training_tensor_parallel_degree ${tp} \
      --training_enable_loss_parallel true \
      --training_pipeline_parallel_degree ${pp} \
      --training_pipeline_parallel_schedule "1F1B" \
      --training_enable_ckpt true \
      --training_ckpt_load_step -1 \
      --training_ckpt_interval 2000 \
      --training_ckpt_keep_latest_k 2 \
      --training_log_freq 100 \
      --training_enable_tensorboard true \
      --training_save_tb_folder "tensorboard" \
      --training_tb_rank_0_only true \
      --training_mixed_precision_param "${param_dtype}" \
      --training_mixed_precision_reduce "float32" \
      --training_compile ${compile} \
      --training_enable_liger_kernel ${liger} \
      --training_enable_compiled_autograd false \
      --training_gc_freq 1000 \
      --training_deterministic false \
      --training_max_norm 5.0 \
      --training_activation_checkpoint_mode "${activation_checkpoint_mode}" \
      --training_activation_checkpoint_selective_ac_option "op" \
      --training_enable_profiling true \
      --training_profiling_traces_folder "profile_traces" \
      --training_profiling_freq 100 \
      --training_profiling_keep_first_k 2 \
      --training_enable_memory_snapshot true \
      --training_memory_snapshot_folder "memory_snapshot" \
      --optimizer_name "AdamW" \
      --optimizer_lr ${lr} \
      --optimizer_impl "fused" \
      --lr_scheduler_steps ${lr_scheduler_steps} \
      --lr_scheduler_warmup_steps ${lr_scheduler_warmup_steps} \
      --lr_scheduler_decay_type "linear" \
      $(if [ "${model_type}" = "touch_audio" ]; then
        echo "--lr_scheduler_lr_min 0.0 \
              --max_text_audio_ratio 1.0 \
              --min_text_audio_ratio 0.0005 \
              --audio_resample_rate 16000 \
              --audio_speed_perturb true \
              --audio_feat_type fbank \
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
              --audiofeat_normalize true"
      else
        echo "--lr_scheduler_lr_min 0.0"
      fi)
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "================================================"
  echo "$0: Stage 3: convert dcp to huggingface-format"
  echo "================================================"
  python touchnet/bin/convert_dcp_to_hf.py \
    --ckpt_dir "exp/${exp_id}" \
    --step "${checkpoint_step}" \
    --config "exp/${exp_id}/model_config.json" \
    --model_type "${model_type}" \
    --tokenizer_model "${pretrained_tokenizer_dir}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  if [[ "${exp_id}" == *"Kimi-Audio-7B"* ]]; then
    dtypes="float32"
  else
    dtypes="bfloat16"
  fi

  for model_dtype in ${dtypes}; do
    if [ "${model_dtype}" = "bfloat16" ]; then
      batch_size=16
    elif [ "${model_dtype}" = "float32" ]; then
      batch_size=1
    else
      echo "Unsupported model_dtype: ${model_dtype}"
      exit 1
    fi

    for data_type in ${test_sets}; do
      if [ "${model_type}" = "touch_audio" ]; then
        instruct=""
      else
        instruct="Generate the transcription:"
      fi
      model_path="exp/${exp_id}/checkpoint_hf/step-${checkpoint_step}"
      output_dir="${model_path}/inference_result/${data_type}.${model_dtype}"

      echo "================================================"
      echo "$0: data_type: ${data_type}"
      echo "$0: model_dtype: ${model_dtype}"
      echo "$0: batch_size: ${batch_size}"
      echo "$0: model_path: ${model_path}"
      echo "$0: output_dir: ${output_dir}"
      echo "$0: instruct: ${instruct}"
      echo "================================================"

      torchrun --nproc_per_node=8 --nnodes=1 \
               --rdzv_id=2025 --rdzv_backend="c10d" --rdzv_endpoint="localhost:8899" \
               --local-ranks-filter "0" \
          touchnet/models/${model_type}/inference_${model_type}.py \
              --model_path "${model_path}" \
              --model_dtype "${model_dtype}" \
              --instruct "${instruct}" \
              --data_list data/${data_type}/data.list.raw \
              --output_dir "${output_dir}" \
              --batch_size ${batch_size} \
              --inference_enable_liger_kernel ${liger} \
              --num_workers 16 \
              --prefetch 8

      cat ${output_dir}/part* > ${output_dir}/final.jsonl
      python local/extract_trans_and_pred.py --jsonl "${output_dir}/final.jsonl"

      # NOTE(xcsong): we use SPEECHIO-style wer calculator
      rm -f ${output_dir}/ref.txt
      echo "$0 --> Normalizing REF text ..."
      python touchnet/bin/textnorm_zh.py --format=ark \
          --to_upper --to_banjiao --remove_fillers --remove_erhua \
          ${output_dir}/trans.txt ${output_dir}/ref.txt

      rm -f ${output_dir}/rec.txt
      echo "$0 --> Normalizing HYP text ..."
      # add "--cc_mode=t2s" option if charset is traditional
      # (e.g. whisper & google USM model)
      python touchnet/bin/textnorm_zh.py --format=ark \
        --to_upper --to_banjiao --remove_fillers --remove_erhua \
        ${output_dir}/raw_rec.txt ${output_dir}/rec.txt
      grep -v $'\t$' ${output_dir}/rec.txt > ${output_dir}/rec_non_empty.txt

      tokenizer=char
      python touchnet/bin/error_rate_zh \
          --tokenizer ${tokenizer} \
          --ref ${output_dir}/ref.txt \
          --hyp ${output_dir}/rec_non_empty.txt \
          ${output_dir}/DETAILS.txt | tee ${output_dir}/RESULTS.txt
    done
  done
fi
