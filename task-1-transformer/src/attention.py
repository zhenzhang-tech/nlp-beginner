"""任务一：手写 attention。"""
import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None, return_weights=False):
    """缩放点积注意力。
    此函数的作用：给句子里的每个词算出一个"新表示"，这个新表示 = 其他所有词的内容，按"跟它有多相关"加权求和。
    Q/K/V: (B, H, T, D)
    mask:  布尔张量（可广播到 B, H, T, T），True = 被屏蔽
    返回:  (B, H, T, D)
    """
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ V
    if return_weights:
        return out, attn
    return out


class MultiHeadAttention(nn.Module):
    """多头注意力：把 d_model 拆成 n_heads 个 d_k，各自做 attention 后拼回。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 4 个投影矩阵：Q/K/V 的输入投影 + 输出投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # 1. 线性投影得到 Q/K/V，形状都还是 (B, T, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # TODO-1: 把 Q/K/V 从 (B, T, d_model) 切成 (B, H, T, d_k)
        #   提示：先 .view(B, T, H, d_k)，再 .transpose(1, 2)
        Q = Q.view(B,T,self.n_heads,self.d_k).transpose(1,2)
        K = K.view(B,T,self.n_heads,self.d_k).transpose(1,2)
        V = V.view(B,T,self.n_heads,self.d_k).transpose(1,2)

        # 2. 逐头做 attention（复用你写好的函数）
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, return_weights=True)
        self.last_attn = attn.mean(dim=1)  # 平均所有 head -> (B, T, T)，供可视化

        # TODO-2: 把 out 从 (B, H, T, d_k) 合并回 (B, T, d_model)
        #   提示：先 .transpose(1, 2)，再 .contiguous().view(B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B,T,self.d_model)

        # 3. 输出投影
        return self.W_o(out)
