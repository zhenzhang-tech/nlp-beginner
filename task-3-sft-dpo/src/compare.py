"""对比 base / SFT / DPO 在同一指令上的输出。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import inject_lora

MODEL_PATH = "models/Qwen2.5-0.5B"


def load_model(lora_path=None):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    if lora_path:
        inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
        ckpt = torch.load(lora_path, map_location="cpu")
        model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    model.eval()
    return model


def generate(model, tok, prompt, device, max_new_tokens=80):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    prompts = [
        "请用一句话介绍深度学习。",
        "写一首关于春天的五言诗。",
    ]

    variants = [
        ("base", load_model()),
        ("sft", load_model("ckpt/sft/lora.pt")),
        ("dpo", load_model("ckpt/dpo/lora.pt")),
    ]

    for prompt in prompts:
        print(f"\n=== 指令: {prompt} ===")
        for name, model in variants:
            model.to(device)
            print(f"[{name}] {generate(model, tok, prompt, device)}")


if __name__ == "__main__":
    main()
