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
        en = load_dataset("allenai/c4", "en", download_config=download_config)
        # multilingual (mC4): 9.7TB (108 subsets, one per language), ~6T tokens
        # en = load_dataset("allenai/c4", "multilingual", download_config=download_config)
        signal = 0
    except Exception as ex:
        pass
