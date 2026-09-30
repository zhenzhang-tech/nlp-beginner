"""M4：在唐诗上预训练 mini-GPT。"""
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

# ---- 超参 ----
BLOCK_SIZE = 128
BATCH_SIZE = 32
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
MAX_SEQ_LEN = 128         # 和 BLOCK_SIZE 保持一致，训练/推理上下文一致
LR = 1e-4
EPOCHS = 30
DROPOUT = 0.1
N_BATCHES = 100
SEED = 42

DATA_DIR = Path("data")
CKPT_DIR = Path("ckpt")
CKPT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def make_batches(ids, block_size, batch_size, n_batches):
    """随机采样 (x, y) 窗口：y 是 x 右移一位（next-token 目标）。"""
    for _ in range(n_batches):
        ix = torch.randint(0, len(ids) - block_size, (batch_size,))
        x = torch.stack([torch.tensor(ids[i:i + block_size]) for i in ix])
        y = torch.stack([torch.tensor(ids[i + 1:i + block_size + 1]) for i in ix])
        yield x, y


@torch.no_grad()
def evaluate_ppl(model, ids, block_size):
    model.eval()
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(ids) - 1), block_size):
        window = ids[i:i + block_size + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long)
        logits = model(chunk)
        nll += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               chunk[:, 1:].reshape(-1), reduction="sum").item()
        n_tok += chunk.size(1) - 1
    return math.exp(nll / n_tok)


def main():
    set_seed(SEED)
    tok = BPETokenizer.from_pretrained(str(CKPT_DIR / "tokenizer.json"))
    vocab_size = tok.vocab_size

    train_ids = tok.encode((DATA_DIR / "train.txt").read_text(encoding="utf-8"))
    dev_ids = tok.encode((DATA_DIR / "dev.txt").read_text(encoding="utf-8"))
    print(f"train tokens: {len(train_ids)}, dev tokens: {len(dev_ids)}, vocab: {vocab_size}")

    model = MiniGPT(vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                    n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN,
                    dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_ppl = float("inf")
    for epoch in range(1, EPOCHS + 1):
        lr = LR * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / EPOCHS))  # cosine 衰减
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        total_loss = 0.0
        n_batch = 0
        for xb, yb in make_batches(train_ids, BLOCK_SIZE, BATCH_SIZE, N_BATCHES):
            logits = model(xb)                          # (B, block_size, V)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        ppl = evaluate_ppl(model, dev_ids, BLOCK_SIZE)
        print(f"epoch {epoch:2d}: loss={total_loss / n_batch:.4f}, dev_ppl={ppl:.2f}")

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"vocab_size": vocab_size, "d_model": D_MODEL,
                           "n_heads": N_HEADS, "n_layers": N_LAYERS,
                           "d_ff": D_FF, "max_seq_len": MAX_SEQ_LEN},
            }, CKPT_DIR / "best.pt")
            print(f"  -> 保存 best.pt (dev_ppl={best_ppl:.2f})")

    print(f"\n训练完成，best dev_ppl = {best_ppl:.2f}（目标 < 50）")


if __name__ == "__main__":
    main()