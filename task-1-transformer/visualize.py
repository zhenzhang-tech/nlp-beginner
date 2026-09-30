"""M5：画注意力热图，观察模型在"看"什么。"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.model import load_for_eval

# 中文字体（避免热图里中文显示成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

MAX_VIS_LEN = 40   # 可视化时截断长度（太长热图看不清）

# 1. 加载训练好的模型
model, tokenize_fn = load_for_eval("ckpt/best.pt")
model.eval()

# 2. 选三个样本：正面、负面、最长句
val = pd.read_parquet("data/validation.parquet")
pos = val[val["label"] == 1].iloc[0]
neg = val[val["label"] == 0].iloc[0]
longest = val.loc[val["text"].str.len().idxmax()]

samples = [
    ("正面", pos["text"], int(pos["label"])),
    ("负面", neg["text"], int(neg["label"])),
    ("长句", longest["text"], int(longest["label"])),
]

# 3. 逐样本画热图（取第一层、平均所有 head）
for name, text, label in samples:
    text = text[:MAX_VIS_LEN]
    ids = tokenize_fn(text).unsqueeze(0)          # (1, T)
    with torch.no_grad():
        logits = model(ids)
        pred = logits.argmax(dim=-1).item()

    attn = model.blocks[0].attn.last_attn[0].cpu().numpy()  # (T, T)
    tokens = list(text)
    T = len(tokens)
    attn = attn[:T, :T]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(attn, cmap="Blues")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_title(f"{name}样本  真实={label}  预测={pred}")
    plt.colorbar(im)
    plt.tight_layout()
    out = FIG_DIR / f"attn_{name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已保存 {out}（长度 {T}）")

print("完成，请打开 figures/ 目录查看 3 张热图")
