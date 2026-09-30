"""decoder-only 的 causal multi-head attention（集成 RoPE + KV cache 接口）。"""
import torch
import torch.nn as nn

from src.rope import apply_rope


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, positions, kv_cache=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = apply_rope(Q, positions)      # RoPE 只加 Q/K
        K = apply_rope(K, positions)

        if kv_cache is not None:           # 拼上历史（M3 会用到）
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        T_kv = K.shape[2]
        k_pos = torch.arange(T_kv, device=x.device)
        mask = k_pos[None, :] > positions[:, None]   # (T, T_kv) True=屏蔽未来

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), (K, V)      # 返回新 cache