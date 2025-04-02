from datasets import DownloadConfig, load_dataset

hf_data_repo = "AISHELL/AISHELL-1"
hf_data_name = "default"

download_config = DownloadConfig(
    num_proc=12,
    max_retries=1200,
)

signal = 1
while signal:
    try:
        datas = load_dataset(f"{hf_data_repo}", f"{hf_data_name}", download_config=download_config)
        signal = 0
    except Exception as ex:
        pass
