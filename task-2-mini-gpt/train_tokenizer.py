"""训练 BPE tokenizer，保存到 ckpt/tokenizer.json。"""
from pathlib import Path
from src.tokenizer import BPETokenizer

text = Path("data/train.txt").read_text(encoding="utf-8")
tok = BPETokenizer(vocab_size=384)
tok.train(text, verbose=True)
tok.save("ckpt/tokenizer.json")

tok2 = BPETokenizer.from_pretrained("ckpt/tokenizer.json")
for s in ["床前明月光", "Hello, world!", "深度学习需要数学基础"]:
    print(repr(s), "->", repr(tok2.decode(tok2.encode(s))))