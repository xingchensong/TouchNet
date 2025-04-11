import os

from datasets import DownloadConfig, load_dataset

hf_data_repo = "HuggingFaceFW/fineweb-edu"
hf_data_name = "default"

download_config = DownloadConfig(
    num_proc=12,
    max_retries=1200,
)

# English only
signal = 1
while signal:
    try:
        # 9.74TB, 1.3T tokens
        data = load_dataset(
            f"{hf_data_repo}",
            name=f"{hf_data_name}",
            split="train",
            download_config=download_config
        )
        signal = 0
    except Exception as ex:
        pass

HF_HOME = os.environ.get("HF_HOME", "/bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface")
prefix = f"{HF_HOME}/datasets/converted_jsonl_for_touchnet"

key = "train"
# num_samples: 1426200851
print(f"num_samples of {hf_data_repo}/{hf_data_name}[{key}]: {len(data)}")
num_bytes = data.info.splits[key].num_bytes
# shard data for every 10GB
num_shards = num_bytes // (10 * 1024 * 1024 * 1024) + 1
# num_shards: 681
print(f"num_shards of {hf_data_repo}/{hf_data_name}[{key}]: {num_shards}")
for i in range(num_shards):
    data.shard(num_shards, i, writer_batch_size=100000).to_json(
        path_or_buf=f"{prefix}/{hf_data_repo}/{hf_data_name}/{key}-{i:05d}-of-{num_shards:05d}.jsonl",
        batch_size=100000,
        num_proc=16,
    )
