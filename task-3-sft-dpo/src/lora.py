"""手写 LoRA 低秩注入（不依赖 peft）。"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包装一个 nn.Linear，旁挂低秩分支 A、B：y = Wx + (alpha/r) * B(Ax)。"""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        # 冻结原权重
        base.weight.requires_grad = False
        if base.bias is not None:
            base.bias.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        # A: (r, in)，B: (out, r)，ΔW = B @ A
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = x @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + self.scaling * delta


def inject_lora(model, target_modules, r, alpha):
    """把目标线性层替换为 LoRALinear，并只放开 LoRA 参数。"""
    # 先收集要替换的层（避免遍历时修改）
    replacements = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, child, module))

    for parent, child, module in replacements:
        setattr(parent, child, LoRALinear(module, r, alpha))

    # 冻结全部参数，只放开 LoRA 分支
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    return model


def merge_lora(model):
    """把 LoRA 分支合并回原权重（推理用）。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = module.scaling * (module.lora_B @ module.lora_A)  # (out, in)
            module.base.weight.data += delta
    return model
