"""采样策略：greedy / temperature / top-k / top-p。"""
import torch
import torch.nn.functional as F


def sample_token(logits, temperature=1.0, top_k=None, top_p=None):
    """从 logits 采样下一个 token id。temperature=0 时退化为 greedy。"""
    logits = logits.reshape(-1).float()

    if temperature == 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature

    # top-k：只保留分数最高的 k 个
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.numel())
        threshold = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < threshold,
                             torch.full_like(logits, float("-inf")), logits)

    probs = F.softmax(logits, dim=-1)

    # top-p：按概率降序累加到 p，再重新归一化采样
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        remove = cumsum > top_p
        remove[1:] = remove[:-1].clone()          # 保留第一个超过阈值的
        remove[0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()   # 重新归一化
        idx = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[idx].item())

    return int(torch.multinomial(probs, 1).item())
