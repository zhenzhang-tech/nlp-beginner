"""任务一：训练 TransformerClassifier 做中文情感分类。"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model import TransformerClassifier, build_vocab, UNK_TOKEN

# ---- 超参（可调）----
CONFIG = {
    "d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 512,
    "num_classes": 2, "max_len": 128, "dropout": 0.1,
}
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 5
SEED = 42
MAX_LEN = CONFIG["max_len"]

DATA_DIR = Path("data")
CKPT_PATH = Path("ckpt") / "best.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(text, vocab):
    """一句话 -> id 列表（超长截断）"""
    return [vocab.get(c, vocab[UNK_TOKEN]) for c in text][:MAX_LEN]


def make_batch(texts, labels, vocab):
    """把一批文本编码并 padding 到本批最大长度。返回 ids, mask, labels"""
    ids_list = [encode(t, vocab) for t in texts]
    L = max(len(ids) for ids in ids_list)
    B = len(ids_list)
    ids = torch.zeros(B, L, dtype=torch.long)     # 0 = [PAD]
    mask = torch.ones(B, L, dtype=torch.bool)     # True = padding
    for i, ids_i in enumerate(ids_list):
        ids[i, :len(ids_i)] = torch.tensor(ids_i, dtype=torch.long)
        mask[i, :len(ids_i)] = False
    labels = torch.tensor(labels, dtype=torch.long)
    return ids, mask, labels


def iterate_batches(df, vocab, batch_size, shuffle):
    """把 DataFrame 切成一个个 batch（生成器）"""
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    idx = list(range(len(texts)))
    if shuffle:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield make_batch([texts[i] for i in batch_idx],
                         [labels[i] for i in batch_idx], vocab)


def evaluate(model, df, vocab, batch_size):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, mask, labels in iterate_batches(df, vocab, batch_size, shuffle=False):
            logits = model(ids, padding_mask=mask)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    set_seed(SEED)
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "validation.parquet")

    vocab = build_vocab(train_df["text"].tolist())
    print(f"vocab 大小: {len(vocab)}")

    model = TransformerClassifier(vocab_size=len(vocab), **CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for ids, mask, labels in iterate_batches(train_df, vocab, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()                 # 清空上一轮的梯度
            logits = model(ids, padding_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()                       # 反向传播：算梯度
            optimizer.step()                      # 按梯度更新参数
            total_loss += loss.item()
            n_batch += 1

        acc = evaluate(model, val_df, vocab, BATCH_SIZE)
        print(f"epoch {epoch}: loss={total_loss / n_batch:.4f}, val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": CONFIG,
            }, CKPT_PATH)
            print(f"  -> 保存 best.pt (val_acc={best_acc:.4f})")

    print(f"\n训练完成，best val_acc = {best_acc:.4f}（目标 >= 0.80）")


if __name__ == "__main__":
    main()