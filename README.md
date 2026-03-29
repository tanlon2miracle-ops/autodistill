# autodistill

AI Agent 自主蒸馏实验框架。参考 [Karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的思路：让 AI Agent 自动探索最优知识蒸馏策略，你睡觉它炼丹。

> "以前蒸馏靠人工调参，一组实验一下午。现在 Agent 一晚上跑 50 组，你醒来看 results.tsv 就行。" — autodistill

## 核心思路

Karpathy 的 autoresearch 让 Agent 自主优化 LLM 预训练。autodistill 把同样的模式搬到**知识蒸馏**：

- **Agent 只改一个文件** (`distill.py`)：蒸馏策略、loss 配比、温度、学习率全部可调
- **固定时间预算**（默认 15 分钟/轮）：不同策略公平比较
- **自动 keep/discard**：val 指标提升就保留，否则回滚
- **人只写 `program.md`**：用 Markdown 指挥 Agent 的研究方向

## 场景

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

## 设计原则

### 照搬自 autoresearch 的好设计

| 原则 | autoresearch | autodistill |
|------|-------------|-------------|
| **单文件修改** | Agent 只改 `train.py` | Agent 只改 `distill.py` |
| **固定时间预算** | 5 分钟 | 15 分钟（蒸馏需要更多时间） |
| **单一指标** | `val_bpb`（越低越好） | `val_metric`（按任务选择，见下方） |
| **自动 keep/discard** | 提升保留，否则回滚 | 同上 |
| **program.md 驱动** | Markdown 指挥 Agent | 同上 |
| **results.tsv 日志** | 所有实验记录 | 同上 |
| **永不停止** | Agent 自主循环 | 同上 |

### autodistill 的独特设计

| 设计 | 原因 |
|------|------|
| **多卡支持** | H20 多卡 → 用 `accelerate` / DeepSpeed，Teacher 推理和 Student 训练可分卡 |
| **Teacher 推理缓存** | 大模型推理慢 → 缓存 soft labels 到磁盘，避免重复推理 |
| **多 Student 支持** | 一套框架跑 4B / 0.8B / 0.6B / BERT |
| **多指标评估** | 分类任务用 F1，生成任务用 BLEU/ROUGE，通用 val_loss |
| **时间预算更长** | 蒸馏涉及 Teacher 推理 + Student 训练，15 分钟更合理 |

## 支持的蒸馏策略

Agent 可以在 `distill.py` 中自由探索：

- **Logit Distillation**：KL 散度对齐 Teacher/Student 输出分布
- **Feature Distillation**：对齐中间层表示
- **Attention Transfer**：对齐注意力矩阵
- **Progressive Distillation**：逐层渐进式蒸馏
- **Task-specific Distillation**：针对下游任务微调蒸馏
- **Data Augmentation**：用 Teacher 生成更多训练数据
- **Temperature Scheduling**：动态调整蒸馏温度
- **Loss 配比**：hard label loss + soft label loss 的权重
- 任何 Agent 能想到的新策略...

## 指标说明

```
val_metric:        0.8523       # 主指标（任务相关）
val_loss:          1.2345       # 验证集 loss
teacher_agreement: 0.9120       # 与 Teacher 输出一致率
compression_ratio: 58.8x        # Teacher/Student 参数比
throughput_tps:    12345        # Student 推理吞吐量（tokens/sec）
peak_vram_gb:      45.2         # 峰值显存
```

## Agent 可调参数（在 distill.py 中）

```python
# === 蒸馏超参 ===
TEMPERATURE = 4.0              # 蒸馏温度
ALPHA = 0.7                    # soft label loss 权重（1-ALPHA = hard label）
LEARNING_RATE = 2e-5           # Student 学习率
BATCH_SIZE = 32                # 训练 batch size
GRADIENT_ACCUMULATION = 4      # 梯度累积步数

# === 策略选择 ===
DISTILL_MODE = "logit"         # logit / feature / attention / progressive
TEACHER_LAYERS = "all"         # all / last-4 / adaptive
STUDENT_INIT = "random"        # random / teacher-slice / pretrained

# === 数据增强 ===
AUGMENT_WITH_TEACHER = True    # 用 Teacher 生成额外训练数据
AUGMENT_RATIO = 2.0            # 增强数据比例
```

## 与 autoresearch 的对比

```
autoresearch:                    autodistill:
  人 → program.md → Agent          人 → program.md → Agent
  Agent → 改 train.py              Agent → 改 distill.py
  跑 5 min → val_bpb               跑 15 min → val_metric
  keep / discard                    keep / discard
  循环到天亮                         循环到天亮
  
  单卡 / 预训练 / 单模型            多卡 / 蒸馏 / Teacher+Student
```

## License

MIT

## 致谢

- [Karpathy/autoresearch](https://github.com/karpathy/autoresearch) — 核心思路来源
- [nanochat](https://github.com/karpathy/nanochat) — 训练代码基础
