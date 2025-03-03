from datasets import load_dataset

prefix = "/bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface/datasets/converted_jsonl_xcsong"
train_data = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', trust_remote_code=True)
for k in train_data.keys():
    print(f"len of wikitext-103-v1[{k}]: {len(train_data[k])}")
    train_data[k].to_json(
        path_or_buf=f"{prefix}/wikitext-103-v1.{k}.jsonl",
        batch_size=64,
        num_proc=32,
    )
train_data = load_dataset('Salesforce/wikitext', 'wikitext-2-v1', trust_remote_code=True)
for k in train_data.keys():
    print(f"len of wikitext-2-v1[{k}]: {len(train_data[k])}")
    train_data[k].to_json(
        path_or_buf=f"{prefix}/wikitext-2-v1.{k}.jsonl",
        batch_size=64,
        num_proc=32,
    )
