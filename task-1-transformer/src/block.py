import torch
import torch.nn as nn
from src.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """一个 encoder block：attention + FFN + 两个残差 + 两个 LayerNorm。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 子层 1：注意力
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)

        # 子层 2：前馈网络
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),    # 扩维
            nn.GELU(),                    # 非线性激活
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),    # 缩回
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Pre-LN：先归一化 → 子层 → 残差相加
        x = x + self.attn(self.ln1(x), mask=mask)   # 残差 1
        x = x + self.ffn(self.ln2(x))               # 残差 2
        return x