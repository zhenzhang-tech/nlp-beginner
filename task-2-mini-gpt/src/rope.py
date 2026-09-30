"""RoPE：旋转位置编码。"""
import torch


def apply_rope(x, positions, base=10000.0):
    """对 Q/K 施加旋转位置编码。

    x: (B, H, T, D)
    positions: (T,) 位置索引
    返回: (B, H, T, D)
    """
    B, H, T, D = x.shape
    d = D // 2
    # 频率 theta_i = base^(-2i/D)
    i = torch.arange(d, device=x.device, dtype=x.dtype)
    theta = base ** (-2 * i / D)              # (d,)
    # 每个位置的角度 = 位置 * 频率
    angle = positions.unsqueeze(-1) * theta   # (T, d)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # 相邻两维配对：(x0,x1) (x2,x3) ...
    x_even = x[..., 0::2]   # 偶数位 -> (B, H, T, d)
    x_odd = x[..., 1::2]    # 奇数位 -> (B, H, T, d)

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = x_even_rot
    out[..., 1::2] = x_odd_rot
    return out