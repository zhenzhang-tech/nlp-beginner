"""M3：验证 KV cache 等价性（增量解码 = 全量前向）。"""
import torch
from src.model import MiniGPT

torch.manual_seed(0)
model = MiniGPT(vocab_size=512, d_model=128, n_heads=4, n_layers=2,
                d_ff=512, max_seq_len=128)
model.eval()

ids = torch.randint(0, 512, (1, 8))

with torch.no_grad():
    # 全量前向（一次性喂 8 个 token）
    logits_full = model(ids)

    # 增量解码（逐个 token 喂，带 KV cache）
    cache = None
    logits_inc = []
    for i in range(ids.size(1)):
        out, cache = model(ids[:, i:i + 1], kv_cache=cache, return_cache=True)
        logits_inc.append(out)
    logits_inc = torch.cat(logits_inc, dim=1)

diff = (logits_full - logits_inc).abs().max().item()
print(f"全量 vs 增量的最大 logits 差: {diff}")
print("通过（< 1e-4）" if diff < 1e-4 else "失败")
