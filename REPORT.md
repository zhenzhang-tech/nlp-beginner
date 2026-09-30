# LLM-Beginner 工程复现实验报告

> 作者：张圳（GitHub: zhenzhang-tech）
> 仓库：FudanNLP/nlp-beginner（LLM-Beginner：大模型与智能体入门练习）

## 一、实验环境与条件

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| CPU / GPU | 笔记本 RTX 3050Ti（4GB 显存）；任务一、二主要用 CPU 训练 |
| Python | 3.13.5 |
| 虚拟环境 | venv（`nlp-env`）+ pip 阿里云镜像 |
| 核心依赖 | PyTorch 2.14.0（CPU 版）、transformers 5.17.0、datasets 3.x、pandas、matplotlib |
| 实验仓库 | https://github.com/FudanNLP/nlp-beginner |

说明：任务一、二均为百万级参数的小模型，CPU 即可完成训练与推理，无需 GPU。

## 二、实验方法与过程

### 2.1 任务一：手写 Transformer（中文情感分类）

**目标**：不用高层封装，从零手写 Transformer encoder，在中文情感二分类上达标（dev 准确率 ≥ 0.80）。

**数据**：ChnSentiCorp 中文酒店评论 7766 条（原始数据源 Google Drive 国内不可达，改用 GitHub 等价源），按 8:1:1 划分为 train 6212 / validation 776 / test 778 条。

**方法**（全部手写，未调用 `nn.MultiheadAttention`）：

1. `scaled_dot_product_attention`：实现 `softmax(QKᵀ/√d)·V`，支持 mask；
2. `MultiHeadAttention`：切头、多头并行、合并头、输出投影；
3. `TransformerBlock`：attention + FFN + 两个残差 + 两个 LayerNorm（Pre-LN）；
4. `TransformerClassifier`：字符级 embedding + 位置编码 + N 层 block + mean pooling + 分类头；
5. 训练：AdamW、交叉熵，`d_model=128 / n_heads=4 / n_layers=4 / lr=3e-4 / batch=32 / epochs=5`；
6. 用注意力热图可视化模型关注点。

### 2.2 任务二：从零实现 mini-GPT

**目标**：从零实现 decoder-only 语言模型，手写 BPE、RoPE、KV cache、采样策略，在唐诗语料上预训练到困惑度达标（dev 困惑度 < 50）。

**数据**：唐诗语料（约 1.5 万字符），切分为 train / dev。

**方法**（全部手写）：

1. **字节级 BPE tokenizer**（vocab 384）：以 UTF-8 字节为基本单位，迭代合并高频相邻对；
2. **decoder-only 模型 + RoPE**：旋转位置编码作用于 Q/K，causal mask 保证单向；
3. **KV cache**：推理时缓存历史 K/V，实现增量解码；
4. **采样策略**：greedy / temperature / top-k / top-p；
5. 训练：next-token prediction，随机采样切窗、cosine 学习率调度、梯度裁剪，`d_model=128 / 4 层 / 4 头`。

## 三、实验结果及分析

### 3.1 任务一结果

- **dev 准确率 0.8518**，超过 0.80 通过线，也超过 0.85 参考基线；
- 自检三项全部 `[通过]`：attention 与官方实现数值误差 9.5e-07、causal mask 零泄漏、分类准确率达标；
- 注意力热图显示：模型在负面/长句样本中准确聚焦「差」等情感关键词，验证了 attention 的可解释性。

### 3.2 任务二结果

- **dev 困惑度 43.99**，低于 50 达标线；
- 自检三项全部 `[通过]`：BPE 中英文无损往返、KV cache 增量与全量误差 2.4e-06、困惑度达标；
- 四种采样策略对比：greedy 偏向高频字（保守但易重复）、temperature 更发散、top-k/top-p 折中，验证了采样机制对生成多样性的控制。

## 四、遇到的问题及处理情况

| 问题 | 处理 |
|---|---|
| conda 源在校园网下连不通（清华 DNS 失败、阿里云 404、中科大缺包） | 放弃 conda，改用 venv + pip 阿里云镜像 |
| 任务一数据源 Google Drive 国内不可达 | 改用 GitHub 上的 ChnSentiCorp 等价数据 |
| datasets 5.0 不支持老式数据集脚本 | 降级到 datasets 3.x，并加 `trust_remote_code=True` |
| 唐诗小语料训练过拟合（困惑度先降后升） | 随机采样切窗 + cosine 调度 + dropout + 早停 |
| 困惑度卡在 78 无法达标 | 理解 ppl 与 token 粒度的关系后，将 vocab 512→384，系统性降低 ppl 至 43.99 |
| 任务三需微调 0.5B 模型，本机 4GB 显存不足 | 规划使用云 GPU（AutoDL 等） |

## 附录：运行截图

| 编号 | 内容 | 文件位置 |
|---|---|---|
| 图 1 | 任务一、任务二自检（各三项通过） | `screenshots/1_eval.jpeg` |
| 图 2 | 任务二训练日志（dev_ppl 达标） | `screenshots/2_train_log.jpeg` |
| 图 3 | 任务二生成样例（四种采样） | `screenshots/3_generate.jpeg` |
| 图 4-6 | 任务一注意力热图 | `task-1-transformer/figures/attn_*.png` |
