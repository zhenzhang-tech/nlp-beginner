"""M4：用 causal mask 跑一个 toy 语言模型（next-token 预测）。"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block import TransformerBlock

# ---- 1. 数据：优先用仓库的唐诗，否则自造 ----
def load_text():
    poetry = Path("../poetryFromTang.txt")
    if poetry.exists():
        return poetry.read_text(encoding="utf-8")[:2000]
    return "床前明月光疑是地上霜举头望明月低头思故乡春眠不觉晓处处闻啼鸟夜来风雨声花落知多少" * 20

text = load_text()
chars = sorted(set(text))
vocab = {c: i for i, c in enumerate(chars)}
V = len(vocab)
print(f"字符数: {len(text)}, vocab 大小: {V}")

# ---- 超参 ----
SEQ_LEN = 32
BATCH = 16
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
EPOCHS = 200
LR = 1e-3

# ---- 2. 构造 next-token 数据：x[i] -> y[i] = x[i+1] ----
ids = torch.tensor([vocab[c] for c in text])
xs, ys = [], []
for i in range(0, len(ids) - SEQ_LEN - 1, SEQ_LEN):
    xs.append(ids[i:i + SEQ_LEN])
    ys.append(ids[i + 1:i + SEQ_LEN + 1])
X = torch.stack(xs)   # (N, SEQ_LEN)
Y = torch.stack(ys)   # (N, SEQ_LEN)

# ---- 3. causal mask：上三角 True（位置 i 屏蔽 j>i）----
causal_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(V, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)
        ])
        self.head = nn.Linear(D_MODEL, V)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, mask=causal_mask)   # 关键：传 causal mask
        return self.head(x)                  # (B, T, V)


model = ToyLM()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- 4. 训练 ----
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(X))
    total_loss = 0.0
    n = 0
    for i in range(0, len(X), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X[idx], Y[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}: loss={total_loss / n:.4f}")

# ---- 5. 验证：未来 token 改动不应影响过去 ----
model.eval()
with torch.no_grad():
    xb = X[:1].clone()
    logits1 = model(xb)

    xb2 = xb.clone()
    xb2[0, -1] = (xb2[0, -1] + 1) % V   # 改最后一个（未来）token
    logits2 = model(xb2)

    diff = (logits1[0, :-1] - logits2[0, :-1]).abs().max().item()
    print(f"\n未来 token 改动后，过去位置 logits 的最大变化: {diff}")
    print("≈0 说明 causal mask 正确阻断了未来信息泄漏")
