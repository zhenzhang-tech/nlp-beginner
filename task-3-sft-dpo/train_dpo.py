"""DPO：在 SFT 基础上做偏好对齐（chosen vs rejected）。"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import inject_lora
from src.chat import format_messages

MODEL_PATH = "models/Qwen2.5-0.5B"
SFT_CKPT = Path("ckpt/sft/lora.pt")
CKPT_DIR = Path("ckpt/dpo")


def load_dpo_data(path, limit=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = item.get("prompt") or item.get("instruction") or item.get("question")
            chosen = item.get("chosen") or item.get("chosen_response")
            rejected = item.get("rejected") or item.get("rejected_response")
            if prompt and chosen and rejected:
                data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return data[:limit] if limit else data


def to_text(x, tok):
    """把 prompt/chosen/rejected 规整成文本。"""
    if isinstance(x, str):
        return x
    if isinstance(x, list):  # messages 列表
        msgs = [{"role": m["role"], "content": m.get("content", m.get("text", ""))}
                for m in x if isinstance(m, dict)]
        return format_messages(msgs)
    return str(x)


def load_model_with_lora(lora_path=None):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    if lora_path:
        inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
        ckpt = torch.load(lora_path, map_location="cpu")
        model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    return model


def compute_logprob(model, prompt_ids, resp_ids, device):
    ids = torch.cat([torch.tensor(prompt_ids), torch.tensor(resp_ids)]).unsqueeze(0).to(device)
    logits = model(ids).logits                       # (1, T, V)
    start = len(prompt_ids) - 1
    end = len(prompt_ids) + len(resp_ids) - 1
    resp_logits = logits[0, start:end]               # (L_resp, V)
    resp_targets = ids[0, start + 1:end + 1]         # (L_resp,)
    return F.log_softmax(resp_logits, dim=-1).gather(1, resp_targets.unsqueeze(1)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dpo/dpo_data.jsonl")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    policy = load_model_with_lora(SFT_CKPT).to(device)
    policy.train()
    ref = load_model_with_lora(SFT_CKPT).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    data = load_dpo_data(args.data, args.limit)
    print(f"加载 {len(data)} 条 DPO 数据")
    if not data:
        print("[错误] 无有效 DPO 数据，请检查 --data 路径和格式")
        return

    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    for step in range(args.steps):
        item = data[step % len(data)]
        prompt = to_text(item["prompt"], tok)
        chosen = to_text(item["chosen"], tok)
        rejected = to_text(item["rejected"], tok)

        prompt_ids = tok(prompt, add_special_tokens=False).input_ids[:args.max_length]
        chosen_ids = tok(chosen, add_special_tokens=False).input_ids[:args.max_length]
        rejected_ids = tok(rejected, add_special_tokens=False).input_ids[:args.max_length]

        with torch.no_grad():
            ref_chosen = compute_logprob(ref, prompt_ids, chosen_ids, device)
            ref_rejected = compute_logprob(ref, prompt_ids, rejected_ids, device)
        pol_chosen = compute_logprob(policy, prompt_ids, chosen_ids, device)
        pol_rejected = compute_logprob(policy, prompt_ids, rejected_ids, device)

        log_ratio_chosen = pol_chosen - ref_chosen
        log_ratio_rejected = pol_rejected - ref_rejected
        loss = -F.logsigmoid(args.beta * (log_ratio_chosen - log_ratio_rejected))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            margin = (log_ratio_chosen - log_ratio_rejected).item()
            print(f"step {step + 1}: loss={loss.item():.4f}, margin={margin:.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v for k, v in policy.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save({"lora_state_dict": lora_state,
                "config": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}},
               CKPT_DIR / "lora.pt")
    print(f"已保存 DPO LoRA 权重到 {CKPT_DIR / 'lora.pt'}")


if __name__ == "__main__":
    main()
