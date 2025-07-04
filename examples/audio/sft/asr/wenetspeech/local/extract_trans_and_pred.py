import argparse
import json
import os

from tqdm import tqdm


def main(jsonl_path):
    out_dir = os.path.dirname(jsonl_path)
    trans_path = os.path.join(out_dir, 'trans.txt')
    raw_rec_path = os.path.join(out_dir, 'raw_rec.txt')

    with open(jsonl_path, 'r', encoding='utf-8') as fin, \
         open(trans_path, 'w', encoding='utf-8') as ftrans, \
         open(raw_rec_path, 'w', encoding='utf-8') as fraw:
        all_keys = []
        for line in tqdm(fin.readlines()):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                label = json.loads(result['label'])
                key = label['key']
                txt = label['txt']
                predict = result['predict']
                # 有些predict可能是字符串，有些是json字符串
                if isinstance(predict, str):
                    try:
                        predict = json.loads(predict)
                    except Exception:
                        pass
                # 如果predict是字典，优先取transcription，否则直接str(predict)
                if isinstance(predict, dict) and 'transcription' in predict:
                    pred_txt = predict['transcription']
                else:
                    pred_txt = str(predict)
                if key not in all_keys:
                    all_keys.append(key)
                    ftrans.write(f"{key} {txt}\n")
                    if len(pred_txt.replace(' ', '')) > 0:
                        fraw.write(f"{key} {pred_txt}\n")
            except Exception as e:
                print(f"[WARN] 跳过异常行: {e}")
                continue
    print(f"已生成: {trans_path} 和 {raw_rec_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从jsonl文件提取trans.txt和raw_rec.txt')
    parser.add_argument('--jsonl', type=str, required=True, help='输入的jsonl文件路径')
    args = parser.parse_args()
    main(args.jsonl)
