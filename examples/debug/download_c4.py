from datasets import DownloadConfig, load_dataset

download_config = DownloadConfig(
    num_proc=12,
    max_retries=1200,
)

# English only
en = load_dataset("allenai/c4", "en", download_config=download_config)
