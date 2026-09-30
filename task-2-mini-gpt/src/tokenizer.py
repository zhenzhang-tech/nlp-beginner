"""手写字节级 BPE 分词器。"""
"""
BPE 训练算法（核心）
1. 语料 → UTF-8 字节序列（每个字节 0~255）
2. 重复 (vocab_size - 256) 次：
   a. 统计所有「相邻 token 对」的出现次数
   b. 合并出现最多的那一对 (a, b) → 新 token id
   c. 把这条 merge 记进 merges 表
3. 得到 merges 表（合并顺序就是优先级）
"""

import json
from pathlib import Path


class BPETokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.id_to_byte = {i: bytes([i]) for i in range(256)}  # 256 个基础字节
        self.merges = []          # list of (a, b)，顺序 = 优先级
        self.merge_rank = {}      # (a,b) -> 优先级
        self.merge_id = {}        # (a,b) -> 新 id

    def _build_lookup(self):
        self.merge_rank = {p: r for r, p in enumerate(self.merges)}
        self.merge_id = {p: 256 + r for r, p in enumerate(self.merges)}

    def train(self, text, verbose=False):
        ids = list(text.encode("utf-8"))          # ① 文本转字节
        n_merges = self.vocab_size - 256
        for step in range(n_merges):
            counts = {}
            for i in range(len(ids) - 1):          # ② 统计相邻对频率
                p = (ids[i], ids[i + 1])
                counts[p] = counts.get(p, 0) + 1
            if not counts:
                break
            pair = max(counts, key=counts.get)     # ③ 最高频对
            new_id = 256 + len(self.merges)
            self.merges.append(pair)
            self.id_to_byte[new_id] = self.id_to_byte[pair[0]] + self.id_to_byte[pair[1]]

            new_ids = []                            # ④ 合并序列
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    new_ids.append(new_id); i += 2
                else:
                    new_ids.append(ids[i]); i += 1
            ids = new_ids
        self._build_lookup()

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:                       # 贪心合并优先级最高的对
            best_i, best_rank = None, float("inf")
            for i in range(len(ids) - 1):
                r = self.merge_rank.get((ids[i], ids[i + 1]))
                if r is not None and r < best_rank:
                    best_rank, best_i = r, i
            if best_i is None:
                break
            p = (ids[best_i], ids[best_i + 1])
            ids = ids[:best_i] + [self.merge_id[p]] + ids[best_i + 2:]
        return ids

    def decode(self, ids):
        data = b"".join(self.id_to_byte[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "merges": [list(p) for p in self.merges],
                       "id_to_byte": {str(k): list(v) for k, v in self.id_to_byte.items()}},
                      f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.id_to_byte = {int(k): bytes(v) for k, v in data["id_to_byte"].items()}
        tok._build_lookup()
        return tok