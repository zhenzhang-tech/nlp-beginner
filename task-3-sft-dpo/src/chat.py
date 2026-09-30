"""Qwen chat template 与 loss masking。"""
from pathlib import Path

import torch
from transformers import AutoTokenizer

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B"
        _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return _tokenizer


def format_messages(messages):
    """应用 Qwen 官方模板，末尾追加 assistant 提示。"""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_labels(ids, messages):
    """构造 labels：只对 assistant 内容算 loss，其余（user/system/控制符）为 -100。"""
    tok = _get_tokenizer()
    labels = torch.full_like(ids, -100)
    pos = 0
    for m in messages:
        role = m["role"]
        content = m["content"]
        segment = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        seg_ids = tok(segment, add_special_tokens=False).input_ids
        seg_len = len(seg_ids)
        if role == "assistant":
            head = tok(f"<|im_start|>{role}\n", add_special_tokens=False).input_ids
            content_ids = tok(content, add_special_tokens=False).input_ids
            start = pos + len(head)
            labels[start:start + len(content_ids)] = ids[start:start + len(content_ids)]
        pos += seg_len
    return labels
