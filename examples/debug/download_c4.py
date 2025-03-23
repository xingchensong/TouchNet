from datasets import DownloadConfig, load_dataset

download_config = DownloadConfig(
    num_proc=12,
    max_retries=1200,
)

# English only
signal = 1
while signal:
    try:
        # 305GB, 156B tokens
        # ref: https://mp.weixin.qq.com/s?__biz=MjM5ODExNDA2MA==&mid=2449950449&idx=1&sn=dcafccb19ef913e905a5b6479a570fe3&chksm=b13c4092864bc9846be7a8bcb2f90d83e40ec5bd6c7e45c8c01f97d884f69213f93c586c9d6c#rd  # noqa
        datas = load_dataset("allenai/c4", "en", download_config=download_config)
        # multilingual (mC4): 9.7TB (108 subsets, one per language), ~6T tokens
        # datas = load_dataset("allenai/c4", "multilingual", download_config=download_config)
        signal = 0
    except Exception as ex:
        pass

prefix = "/bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface/datasets/converted_jsonl_xcsong"

for key in datas.keys():
    # 'train': 364868892
    # 'validation': 364608
    data = datas[key]
    print(f"num_samples of allenai.c4.en[{key}]: {len(data)}")
    num_bytes = data.info.splits[key].num_bytes
    # shard data for every 10GB
    num_shards = num_bytes // (10 * 1024 * 1024 * 1024 * 8) + 1
    print(f"num_shards of allenai.c4.en[{key}]: {num_shards}")
    for i in range(num_shards):
        data.shard(num_shards, i).to_json(
            path_or_buf=f"{prefix}/allenai-c4/en/{key}-{i:05d}-of-{num_shards:05d}.jsonl",
            batch_size=64,
            num_proc=32,
        )
