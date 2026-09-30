import torch
import torch.nn as nn
from src.block import TransformerBlock

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def build_vocab(texts):
    """扫描所有文本，构建字符级 vocab：{字: id}"""
    chars = set()
    for t in texts:
        chars.update(t)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, c in enumerate(sorted(chars), start=2):
        vocab[c] = i
    return vocab


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 num_classes=2, max_len=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        # ① 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        # ② 堆 N 层 block
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # ④ 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids, padding_mask=None):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(ids) + self.position_embedding(pos)

        # attention 的 mask：padding_mask (B,T) → (B,1,1,T)，广播到 (B,H,T,T)
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        for block in self.blocks:
            x = block(x, mask=attn_mask)

        # ③ mean pooling：排除 padding 后求平均
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = x.sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)   # (B, num_classes)


def load_for_eval(ckpt_path):
    """按 eval 契约：返回 (model, tokenize_fn)"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    cfg = ckpt["config"]
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        num_classes=cfg.get("num_classes", 2),
        max_len=cfg.get("max_len", 256),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def tokenize_fn(text):
        ids = [vocab.get(c, vocab[UNK_TOKEN]) for c in text]
        ids = ids[:cfg.get("max_len", 256)]   # 截断超长文本，避免 position 越界（与训练一致）
        return torch.tensor(ids, dtype=torch.long)

    return model, tokenize_fn