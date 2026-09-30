"""decoder-only mini-GPT。"""
import torch
import torch.nn as nn

from src.attention import CausalAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, positions, kv_cache=None):
        h, cache = self.attn(self.ln1(x), positions, kv_cache)   # Pre-LN
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len=256, dropout=0.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.block_size = max_seq_len      # 自检按这个长度切窗
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, max_seq_len, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids, kv_cache=None, return_cache=False):
        B, T = ids.shape
        # 历史长度 = cache 里已有多少 token；position 要接着算（RoPE 关键）
        hist = 0 if kv_cache is None else kv_cache[0][0].shape[2]
        positions = torch.arange(hist, hist + T, device=ids.device)
        x = self.token_emb(ids)

        new_cache = []
        for i, block in enumerate(self.blocks):
            c = None if kv_cache is None else kv_cache[i]
            x, c = block(x, positions, c)
            new_cache.append(c)

        logits = self.lm_head(self.ln_final(x))   # (B, T, vocab_size)
        if return_cache:
            return logits, new_cache
        return logits

    def generate(self, prompt_ids, max_new_tokens, top_k=None, top_p=None,
                 temperature=1.0):
        """自回归生成。prompt_ids: token id 序列（list 或 tensor）。"""
        from src.sampling import sample_token
        self.eval()
        ids = [int(x) for x in prompt_ids][-self.block_size:]
        kv_cache = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if kv_cache is None:
                    x = torch.tensor([ids], dtype=torch.long)
                    logits, kv_cache = self(x, return_cache=True)
                else:
                    x = torch.tensor([[ids[-1]]], dtype=torch.long)
                    logits, kv_cache = self(x, kv_cache=kv_cache, return_cache=True)
                next_id = sample_token(logits[0, -1], temperature, top_k, top_p)
                ids.append(next_id)
        return ids

def load_for_eval(ckpt_path):
    from pathlib import Path
    from src.tokenizer import BPETokenizer
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = MiniGPT(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                    n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                    d_ff=cfg["d_ff"], max_seq_len=cfg["max_seq_len"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = BPETokenizer.from_pretrained(str(ckpt_path.parent / "tokenizer.json"))
    return model, tok