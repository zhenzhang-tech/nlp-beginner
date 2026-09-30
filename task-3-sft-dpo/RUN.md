# 任务三运行步骤（云 GPU 实例）

在云实例的 JupyterLab 终端里，按顺序执行。

## 0. 拉取代码

```bash
cd /root/autodl-tmp/nlp-beginner   # 或你 clone 代码的位置
git pull
cd task-3-sft-dpo
```

## 1. 装依赖 + 下载模型

```bash
source /etc/network_turbo            # AutoDL 学术加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python data/download.py              # 下载 Qwen2.5-0.5B 到 models/Qwen2.5-0.5B
```

## 2. 下载 SFT 数据（MOSS）

```bash
huggingface-cli download OpenMOSS-Team/moss-003-sft-data \
  moss-003-sft-no-tools.jsonl.zip --repo-type dataset --local-dir ./data/moss-sft
```

## 3. 跑自检（验证 M1 LoRA + M2 loss masking）

```bash
python eval/run.py
```

预期：`lora_param_count` [通过]、`loss_masking` [通过]、`sft_vs_base` [跳过]。

## 4. SFT 训练

```bash
python train_sft.py --limit 10000
```

产出 `ckpt/sft/lora.pt`。之后再跑一次 `python eval/run.py`，`sft_vs_base` 应变为 [通过]。

## 5. DPO 训练

先下载 DPO 偏好数据（hiyouga/DPO-En-Zh-20k，中英混合）：

```bash
huggingface-cli download hiyouga/DPO-En-Zh-20k --repo-type dataset --local-dir ./data/dpo
```

下载后若得到的是 json（如 dpo_zh.json），把它转成 jsonl 或用下面命令直接训练：

```bash
python train_dpo.py --data <你的dpo数据路径> --steps 200
```

产出 `ckpt/dpo/lora.pt`。

> 注意：`train_dpo.py` 期望每行 JSON 含 `prompt`/`chosen`/`rejected`（或 `instruction`/`chosen`/`rejected`）。若实际字段名不同，改 `train_dpo.py` 里 `load_dpo_data` 的字段映射即可。

## 6. 对比 base / SFT / DPO

```bash
python src/compare.py
```

输出三个模型在同一指令上的回答，附进提交。
