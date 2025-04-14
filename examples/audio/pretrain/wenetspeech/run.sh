# NOTE(xcsong): change xx_prefix and xx_version to ur setup
cache_prefix=/bucket/output/jfs-hdfs/user/xingchen.song/share
cuda_prefix=/bucket/output/jfs-hdfs/user/xingchen.song/tools/cuda
cuda_version=12.6.3
driver_version=560.35.05
cudnn_version=9.5.1.17
pretrained_weight_dir="/bucket/output/jfs-hdfs/user/xingchen.song/share/modelscope/Llama-3.2-1B-Instruct"
pretrained_tokenizer_dir="/bucket/output/jfs-hdfs/user/xingchen.song/share/modelscope/Llama-3.2-1B-Instruct"
# pretrained_tokenizer_dir="/bucket/output/jfs-hdfs/user/mengtao.xing-halo/workspace/cosyvoice_git/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN"


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

hf_data_repo="wenet-e2e/wenetspeech"
hf_data_name="default"

train_set=train_l
dev_set=dev
test_sets="test_net test_meeting"

param_dtype="bfloat16"

seed=2026
model_config=config/Llama-3.2.json
exp_id="wenetspeech_1x8192_fullac_cp1_tp1_dp8_pp1_stack5_stride4_flex_packloss_fromscratch_mid_ar_std0.02_acc_normpreproc_wp2k_addpad_cb1024_emb16_pretrain"
cp=$(echo $exp_id | grep -oP 'cp\d+' | grep -oP '\d+')
tp=$(echo $exp_id | grep -oP 'tp\d+' | grep -oP '\d+')
dp=$(echo $exp_id | grep -oP 'dp\d+' | grep -oP '\d+')
pp=$(echo $exp_id | grep -oP 'pp\d+' | grep -oP '\d+')
stack=$(echo $exp_id | grep -oP 'stack\d+' | grep -oP '\d+')
stride=$(echo $exp_id | grep -oP 'stride\d+' | grep -oP '\d+')
bs=$(echo $exp_id | grep -oP '\d+x\d+' | grep -oP '\d+' | head -n 1)
max_seq_len=$(echo $exp_id | grep -oP '\d+x\d+' | grep -oP '\d+' | tail -n 1)
echo "$0: ${exp_id}: cp=${cp}, tp=${tp}, dp=${dp}, pp=${pp}, stack=${stack}, stride=${stride}, bs=${bs}, max_seq_len=${max_seq_len}"

tensorboard_dir=tensorboard
num_workers=6
prefetch=6
num_mel_bins=80

. ./parse_options.sh || exit 1;
. ./path.sh --cache_prefix ${cache_prefix} \
            --cuda_prefix ${cuda_prefix} \
            --cuda_version ${cuda_version} \
            --driver_version ${driver_version} \
            --cudnn_version ${cudnn_version} || exit 1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "$0: stage -1: Data Download"
  python download_wenetspeech.py
fi

# TODO(xcsong): character based chinese tokenizer? like CosyVoice2.
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for x in ${train_set} ${dev_set} ${test_sets}; do
    if [ ! -f "data/${x}/data.list" ]; then
      echo "$0: data/${x}/data.list does not exist. generate dataset."
      mkdir -p data/${x}
      python touchnet/bin/make_data.py \
          --save_dir "data/${x}" \
          --jsonl_path "/bucket/output/jfs-hdfs/user/Archive/ASR/testset/universal_scienceTrainTest/wenetspeech.raw/list/${x}.data.list.raw.fix" \
          --num_utt_per_shard 2000 \
          --num_workers 32 \
          --datatypes "audio+metainfo"
    fi
  done
  exit
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
      --tokenizer_type "BestRQTokenizer" \
      --tokenizer_bestrq_vocab_size 1024 \
      --tokenizer_bestrq_input_size $(expr $stack \* $num_mel_bins) \
      --tokenizer_bestrq_emb_size 16 \
      --tokenizer_bestrq_init_seed ${seed} \
      --tokenizer_bestrq_init_method "default" \
      --datalist_path "data/${train_set}/data.list" \
      --training_seed "${seed}" \
      --training_model_name "llama.asr" \
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
  # FIXME(xcsong): Where to apply specaug ??? before quantize or after quantize
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
           --local-ranks-filter "0" \
    touchnet/bin/train.py \
      --tokenizer_type "BestRQTokenizer" \
      --tokenizer_bestrq_vocab_size 1024 \
      --tokenizer_bestrq_input_size $(expr $stack \* $num_mel_bins) \
      --tokenizer_bestrq_emb_size 16 \
      --tokenizer_bestrq_init_seed ${seed} \
      --tokenizer_bestrq_init_method "default" \
      --datapipe_type "audio+metainfo" \
      --datalist_path "data/${train_set}/data.list" \
      --datalist_dev_path "data/${dev_set}/data.list" \
      --datalist_sharding true \
      --datalist_epoch 10000 \
      --datalist_shuffling true \
      --dataset_random_cut_audio false \
      --dataset_random_cut_audio_min_length_in_ms 5000 \
      --dataset_random_cut_audio_max_length_in_ms 3600000 \
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
      --audiofeat_spec_aug false \
      --audiofeat_spec_aug_num_t_mask 2 \
      --audiofeat_spec_aug_num_f_mask 2 \
      --audiofeat_spec_aug_max_t 50 \
      --audiofeat_spec_aug_max_f 10 \
      --audiofeat_spec_sub false \
      --audiofeat_spec_sub_num_t_sub 3 \
      --audiofeat_spec_sub_max_t 30 \
      --audiofeat_spec_trim false \
      --audiofeat_spec_trim_max_t 20 \
      --audiofeat_num_mel_bins ${num_mel_bins} \
      --audiofeat_frame_length 25 \
      --audiofeat_frame_shift 10 \
      --audiofeat_dither 0.0 \
      --audiofeat_stack_length ${stack} \
      --audiofeat_stride_length ${stride} \
      --audiofeat_normalize true \
      --dataloader_num_workers ${num_workers} \
      --dataloader_prefetch_factor ${prefetch} \
      --training_description "wenetspeech ssl" \
      --training_seed "${seed}" \
      --training_model_name "llama.asr" \
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
      --training_ckpt_interval 2000 \
      --training_ckpt_keep_latest_k 2 \
      --training_log_freq 1 \
      --training_enable_tensorboard true \
      --training_save_tb_folder "tensorboard" \
      --training_tb_rank_0_only true \
      --training_mixed_precision_param "${param_dtype}" \
      --training_mixed_precision_reduce "float32" \
      --training_compile true \
      --training_enable_compiled_autograd false \
      --training_gc_freq 1000 \
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
      --lr_scheduler_steps 260000 \
      --lr_scheduler_warmup_steps 2000 \
      --lr_scheduler_decay_type "linear" \
      --lr_scheduler_lr_min 0.0
fi
