from datasets import DownloadConfig, load_dataset

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
            "HuggingFaceFW/fineweb-edu",
            name="default",
            split="train",
            download_config=download_config
        )
        signal = 0
    except Exception as ex:
        pass
