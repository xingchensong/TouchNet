import os

from datasets import load_dataset

HF_HOME = os.environ.get("HF_HOME", "/bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface")
prefix = f"{HF_HOME}/datasets/converted_jsonl_for_touchnet"
hf_data_repo = "Salesforce/wikitext"

hf_data_name = "wikitext-103-v1"
# 548.05 MB, 103M tokens, 0.103B tokens
train_data = load_dataset(f"{hf_data_repo}", f"{hf_data_name}", trust_remote_code=True)
for k in train_data.keys():
    print(f"len of {hf_data_name}[{k}]: {len(train_data[k])}")
    train_data[k].to_json(
        path_or_buf=f"{prefix}/{hf_data_repo}/{hf_data_name}/{k}.jsonl",
        batch_size=64,
        num_proc=32,
    )

hf_data_name = "wikitext-2-v1"
# 13.34 MB, 2M tokens, 0.002B tokens
train_data = load_dataset(f"{hf_data_repo}", f"{hf_data_name}", trust_remote_code=True)
for k in train_data.keys():
    print(f"len of {hf_data_name}[{k}]: {len(train_data[k])}")
    train_data[k].to_json(
        path_or_buf=f"{prefix}/{hf_data_repo}/{hf_data_name}/{k}.jsonl",
        batch_size=64,
        num_proc=32,
    )
