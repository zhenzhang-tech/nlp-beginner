"""构造 ChnSentiCorp 为 parquet，替代 data/download.py。

原 download.py 的数据源是 Google Drive（国内不可达）。
本脚本改为从 GitHub 镜像下载 ChnSentiCorp_htl_all.csv，
规整列名后划分 train/validation/test 并保存为 parquet。
"""
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV = Path("ChnSentiCorp_htl_all.csv")

URLS = [
    "https://cdn.jsdelivr.net/gh/SophonPlus/ChineseNlpCorpus@master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
]

if not CSV.exists():
    ok = False
    for url in URLS:
        print("下载:", url)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                CSV.write_bytes(r.content)
                print("  成功,", len(r.content), "字节")
                ok = True
                break
            print("  状态码", r.status_code)
        except Exception as e:
            print("  异常:", e)
    if not ok:
        sys.exit("下载失败，请检查网络后重试")

df = pd.read_csv(CSV)
df = df.rename(columns={"review": "text"})
df = df[["text", "label"]]
df["label"] = df["label"].astype(int)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

df.iloc[:n_train].to_parquet(DATA_DIR / "train.parquet", index=False)
df.iloc[n_train:n_train + n_val].to_parquet(DATA_DIR / "validation.parquet", index=False)
df.iloc[n_train + n_val:].to_parquet(DATA_DIR / "test.parquet", index=False)

print("完成: train", n_train, "/ validation", n_val, "/ test", n - n_train - n_val)
print("已保存到", DATA_DIR)
