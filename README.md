# autodistill — AI Agent 自主蒸馏实验框架

> 参考 [Karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的理念：让 AI Agent 自动探索最优知识蒸馏策略，你睡觉它炼丹。

"以前蒸馏靠人工调参，一组实验一下午。现在 Agent 一晚上跑 50 组，你醒来看 results.tsv 就行。" — autodistill

## 核心理念

Karpathy 的 autoresearch 让 Agent 自主优化 LLM 预训练。autodistill 把同样的模式搬到知识蒸馏：

- Agent 只改一个文件 (`distill.py`)：蒸馏策略、loss 配比、温度、学习率全部可调
- 固定时间预算（默认 15 分钟/轮）：不同策略公平比较
- 自动 keep/discard：val 指标提升就保留，否则回滚
- 人只写 `program.md`：用 Markdown 指挥 Agent 的研究方向

## 规模定位

| Teacher | Student | 硬件 |
|---------|---------|------|
| 大型 LLM（32B~235B） | 4B / 0.8B / 0.6B / BERT 系列 | 多卡 H20 |

种子文本量级不大 → 重点在蒸馏策略优化，而非数据工程。

## 项目结构

```
autodistill/
├── README.md              # 本文件
├── program.md             # Agent 指令（人编写，Agent 遵循）
├── prepare.py             # 数据准备 + 评估（只读，Agent 不改）
├── distill.py             # 蒸馏脚本（Agent 唯一修改的文件）
├── models/                # 模型配置
│   ├── teacher.py         # Teacher 加载/推理
│   └── student.py         # Student 模型定义
├── configs/
│   └── default.yaml       # 默认超参配置
├── pyproject.toml         # 依赖管理
└── results.tsv            # 实验记录（自动生成）
```

## 快速开始

```bash
# 1. 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 准备数据（一次性）
uv run prepare.py --seed-data /path/to/seeds.jsonl --teacher qwen3-235b

# 4. 跑一次基线
uv run distill.py

# 5. 启动 Agent 自主实验
# 把 Claude/Codex/OpenClaw 指向这个 repo，提示：
# "看 program.md，开始蒸馏实验"
```

## 设计哲学

### 与 autoresearch 的对比

| 原则 | autoresearch | autodistill |
|------|-------------|-------------|
| 单文件修改 | Agent 只改 train.py | Agent 只改 distill.py |
| 固定时间预算 | 5 分钟 | 15 分钟（蒸馏需更多时间） |
| 单一指标 | val_bpb（越低越好） | val_metric（按任务选择，见下方） |
| 自动 keep/discard | 提升保留，否则回滚 | 同上 |
| program.md 驱动 | Markdown 指挥 Agent | 同上 |
| results.tsv 日志 | 所有实验记录 | 同上 |
| 永不停止 | Agent 自主循环 | 同上 |

### autodistill 的特有设计

| 设计 | 原因 |
|------|------|
| 多卡支持 | H20 多卡 → accelerate / DeepSpeed，Teacher 推理和 Student 训练可分卡 |
| Teacher 推理缓存 | 大模型推理慢 → 缓存 soft labels 到磁盘，避免重复推理 |
| 多 Student 支持 | 一套框架跑 4B / 0.8B / 0.6B / BERT |
| 多指标评估 | 分类任务用 F1，生成任务用 BLEU/ROUGE，通用 val_loss |
| 时间预算更长 | 蒸馏涉及 Teacher 推理 + Student 训练，15 分钟更合理 |

### 与 post-training 最佳实践对齐

基于 [post-training-notes](https://github.com/tanlon2miracle-ops/post-training-notes) 中梳理的经验：

| 实践 | 来源 | autodistill 中的体现 |
|------|------|---------------------|
| 规则 reward 优于神经网络 reward | DeepSeek-R1 | 评估函数使用确定性指标（val_loss / F1），不依赖 reward model |
| On-policy (1步梯度) 防 collapse | Skywork-OR1, AceReason | 每轮 15 分钟短迭代，快速检测退化并回滚 |
| 蒸馏模型上可继续 RL 提升 | OR1, AceReason | 框架天然支持 distilled student → RL fine-tune 的衔接 |
| Math→Code 分阶段训练有跨域泛化 | AceReason | 支持多阶段 curriculum 实验（在 program.md 中指定） |
| Entropy collapse 是核心风险 | Skywork-OR1 | 输出指标包含 val_loss 趋势监控，Agent 可探索温度调度应对 |

## Agent 可探索的策略

Agent 可以在 `distill.py` 中自由探索：

### 基础蒸馏
- **Logit Distillation**：KL 散度对齐 Teacher/Student 输出分布
- **Feature Distillation**：对齐中间层表示
- **Attention Transfer**：对齐注意力矩阵

### 高级策略（来自 post-training 研究）
- **Progressive Distillation**：逐层渐进式蒸馏（参考 AceReason 的分阶段思路）
- **Curriculum Learning**：先简单样本再难样本，动态过滤已掌握数据（Skywork-OR1 MAGIC 框架的启发）
- **Multi-stage Distillation**：先 logit → 再 feature → 最后 task-specific fine-tune（类比 DeepSeek-R1 的四阶段训练）
- **Top-k Logit Filtering**：只蒸馏 Teacher 的 top-k logits 减少噪声
- **Dynamic Temperature**：训练早期高温（探索）→ 后期低温（利用），对标 Adaptive Entropy Control
- **Loss 配比调度**：hard/soft label loss 权重随训练动态调整

### 数据增强
- **Teacher 生成增强**：用 Teacher 生成额外训练数据（参考 Nemotron 数据集的合成方法论）
- **Augment Ratio Scheduling**：增强数据比例随训练阶段变化

### Student 初始化
- **Random**：随机初始化
- **Teacher Slice**：从 Teacher 截取对应层权重
- **Pretrained**：从预训练权重开始（推荐，参考 AceReason 在蒸馏模型上继续 RL 的实践）

## 评估指标

```
val_metric:        0.8523   # 主指标（任务相关）
val_loss:          1.2345   # 验证集 loss
teacher_agreement: 0.9120   # 与 Teacher 输出一致率
compression_ratio: 58.8x    # Teacher/Student 参数比
throughput_tps:    12345    # Student 推理吞吐量（tokens/sec）
peak_vram_gb:      45.2     # 峰值显存
```

## 超参参考

```python
# === 蒸馏超参 ===
TEMPERATURE = 4.0              # 蒸馏温度（建议探索范围：1.0 ~ 16.0）
ALPHA = 0.7                    # soft label loss 权重（1-ALPHA = hard label）
LEARNING_RATE = 2e-5           # Student 学习率
BATCH_SIZE = 32                # 训练 batch size
GRADIENT_ACCUMULATION = 4      # 梯度累积步数

# === 策略选择 ===
DISTILL_MODE = "logit"         # logit / feature / attention / progressive
TEACHER_LAYERS = "all"         # all / last-4 / adaptive
STUDENT_INIT = "pretrained"    # random / teacher-slice / pretrained

# === 数据增强 ===
AUGMENT_WITH_TEACHER = True    # 用 Teacher 生成额外训练数据
AUGMENT_RATIO = 2.0            # 增强数据比例

# === 高级（来自 post-training 研究） ===
USE_CURRICULUM = False         # 是否启用课程学习
TOPK_LOGITS = None             # top-k 过滤（None = 不过滤）
TEMPERATURE_SCHEDULE = "constant"  # constant / linear_decay / cosine
```

## 实验循环图示

```
autoresearch 模式:           autodistill 模式:
  人 → program.md → Agent      人 → program.md → Agent
  Agent → 改 train.py          Agent → 改 distill.py
  跑 5 min → val_bpb           跑 15 min → val_metric
  keep / discard               keep / discard
  循环到天亮                    循环到天亮

  单卡 / 预训练 / 单模型        多卡 / 蒸馏 / Teacher+Student
```

## 与风控项目的衔接

此框架可直接服务于 [post-traning-project](https://github.com/tanlon2miracle-ops/post-traning-project)：

| autodistill 阶段 | 风控项目阶段 | 说明 |
|-----------------|------------|------|
| Teacher 准备 | Phase 0 数据准备后 | 用 SFT 后的 Qwen2.5-VL-7B+ 作为 Teacher |
| Student 蒸馏 | Phase 3 部署优化 | 蒸馏到 0.6B-4B 小模型，降低推理成本 |
| 自主实验循环 | 持续优化 | Agent 自动搜索最优蒸馏配置 |

**典型流程**：
1. 在 post-traning-project 中完成 SFT/DPO 训练 → 得到风控专用 Teacher
2. 在 autodistill 中以此 Teacher 蒸馏 → 得到轻量级 Student
3. Student 用于线上推理（吞吐高、成本低）

## License

MIT

## 参考

- [Karpathy/autoresearch](https://github.com/karpathy/autoresearch) — 核心思路来源
- [post-training-notes](https://github.com/tanlon2miracle-ops/post-training-notes) — 后训练技术笔记（DeepSeek-R1 / Skywork-OR1 / AceReason / verl 等）
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO + 四阶段训练
- [Skywork-OR1](https://arxiv.org/abs/2505.22312) — MAGIC 框架 + Entropy Collapse
- [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) — 分阶段 RL + 跨域泛化
- [verl](https://github.com/volcengine/verl) — RL 训练框架
- [Llama-Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) — 数据合成方法论
