"""SFT：在 Qwen2.5-0.5B 上注入 LoRA，用 MOSS 对话数据监督微调。"""
import argparse
import json
import re
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import inject_lora
from src.chat import format_messages, build_labels

MODEL_PATH = "models/Qwen2.5-0.5B"
CKPT_DIR = Path("ckpt/sft")


def _clean_human(s):
    s = s.strip()
    s = re.sub(r'^<\|Human\|>:\s*', '', s)
    s = re.sub(r'<eoh>\s*$', '', s)
    return s.strip()


def _clean_moss(s):
    s = s.strip()
    s = re.sub(r'^<\|MOSS\|>:\s*', '', s)
    return s.strip()


def load_moss_messages(path, limit=None):
    """从 MOSS jsonl(.zip) 读取对话，返回 messages 列表。"""
    msgs_list = []

    def parse_line(line):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return None
        chat = item.get("chat")
        # MOSS 格式：chat 是 dict，键 turn_1/turn_2，值为 {Human, MOSS, ...}
        if isinstance(chat, dict):
            out = []
            meta = item.get("meta_instruction")
            if meta:
                out.append({"role": "system", "content": meta})
            for k in sorted(chat.keys()):
                turn = chat[k]
                if not isinstance(turn, dict):
                    continue
                human = turn.get("Human") or turn.get("human")
                moss = turn.get("MOSS") or turn.get("moss")
                if human:
                    out.append({"role": "user", "content": _clean_human(str(human))})
                if moss:
                    out.append({"role": "assistant", "content": _clean_moss(str(moss))})
            return out if out else None
        # 标准 role 格式：chat 是 list
        if isinstance(chat, list):
            out = []
            for m in chat:
                if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
                    content = m.get("content", m.get("text", ""))
                    if content:
                        out.append({"role": m["role"], "content": content})
            return out if out else None
        return None

    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in [n for n in z.namelist() if n.endswith(".jsonl")]:
                with z.open(n) as f:
                    for line in f:
                        msgs = parse_line(line.decode("utf-8", errors="ignore"))
                        if msgs:
                            msgs_list.append(msgs)
    else:
        with open(p, encoding="utf-8") as f:
            for line in f:
                msgs = parse_line(line)
                if msgs:
                    msgs_list.append(msgs)
    return msgs_list[:limit] if limit else msgs_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/moss-sft/moss-003-sft-no-tools.jsonl.zip")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=args.r, alpha=args.alpha)
    model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA 可训参数: {trainable:,} / {total:,} = {trainable/total:.4%}")

    msgs_list = load_moss_messages(args.data, args.limit)
    print(f"加载 {len(msgs_list)} 条对话")
    if msgs_list:
        print("样例:", json.dumps(msgs_list[0], ensure_ascii=False)[:200])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        for i in range(0, len(msgs_list), args.batch_size):
            batch = msgs_list[i:i + args.batch_size]
            inputs, labels_list = [], []
            for msgs in batch:
                text = format_messages(msgs)
                enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                inputs.append(enc.input_ids[0])
                labels_list.append(build_labels(enc.input_ids[0], msgs))

            max_len = max(len(x) for x in inputs)
            padded_ids = torch.zeros(len(inputs), max_len, dtype=torch.long)
            padded_labels = torch.full((len(inputs), max_len), -100, dtype=torch.long)
            mask = torch.zeros(len(inputs), max_len, dtype=torch.bool)  # True = padding
            for j, (x, lb) in enumerate(zip(inputs, labels_list)):
                L = len(x)
                padded_ids[j, :L] = x
                padded_labels[j, :L] = lb
                mask[j, L:] = True
            padded_ids, padded_labels, mask = padded_ids.to(device), padded_labels.to(device), mask.to(device)

            logits = model(padded_ids, attention_mask=~mask).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   padded_labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / args.grad_accum).backward()
            if (i // args.batch_size + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}: loss={total_loss/n:.4f}")
        print(f"epoch {epoch+1}: avg loss={total_loss/n:.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save({"lora_state_dict": lora_state,
                "config": {"r": args.r, "alpha": args.alpha, "target_modules": ["q_proj", "v_proj"]}},
               CKPT_DIR / "lora.pt")
    print(f"已保存 LoRA 权重到 {CKPT_DIR / 'lora.pt'}")


if __name__ == "__main__":
    main()
