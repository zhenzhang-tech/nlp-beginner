"""M5：用不同采样策略生成唐诗，对比多样性与连贯度。"""
from src.model import load_for_eval

model, tok = load_for_eval("ckpt/best.pt")
model.eval()

prompt = "床前明月光"
prompt_ids = tok.encode(prompt)

strategies = [
    ("greedy          ", dict(temperature=0.0)),
    ("temperature=0.8 ", dict(temperature=0.8)),
    ("top-k=10        ", dict(temperature=0.8, top_k=10)),
    ("top-p=0.9       ", dict(temperature=0.8, top_p=0.9)),
]

for name, kwargs in strategies:
    ids = model.generate(prompt_ids, max_new_tokens=24, **kwargs)
    print(f"[{name}] {tok.decode(ids)}")
