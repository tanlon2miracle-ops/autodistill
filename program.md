# autodistill — Agent 蒸馏实验指令

这是一个让 AI Agent 自主优化知识蒸馏的实验框架。

## Setup

与用户协商完成初始设置：

1. **确定 run tag**：基于日期（如 `mar30`）。分支 `autodistill/<tag>` 必须是新的。
2. **创建分支**：`git checkout -b autodistill/<tag>`
3. **读取关键文件**：
   - `README.md` — 项目背景 + 后训练研究洞察
   - `prepare.py` — 数据准备、评估函数、Teacher 推理缓存（只读，不改）
   - `distill.py` — 你唯一修改的文件：蒸馏策略、超参、loss 设计
   - `models/teacher.py` — Teacher 加载逻辑（只读）
   - `models/student.py` — Student 模型定义（只读）
   - `configs/default.yaml` — 默认配置（只读）
4. **检查数据**：确认 `~/.cache/autodistill/` 下有 Teacher soft labels 缓存和种子数据
5. **初始化 results.tsv**：只写表头
6. **确认后开始实验**

## 实验规则

### 你可以做的
- 修改 `distill.py` — 这是你唯一编辑的文件
- 改蒸馏策略（logit/feature/attention/progressive）
- 改超参（温度、alpha、学习率、batch size）
- 改 loss 函数设计
- 改数据增强策略
- 改 Student 初始化方式
- 改梯度累积、warm-up、scheduler

### 你不能做的
- 修改 `prepare.py`（评估函数是 ground truth）
- 修改 `models/` 下的文件（模型结构固定）
- 安装新依赖
- 修改评估指标的计算方式
- 修改时间预算

### 指标
主指标取决于 Student 类型：
- **生成式 Student**（0.6B-4B）：`val_loss`（越低越好）
- **BERT 系列 Student**：`val_f1`（越高越好）
- 辅助指标：`teacher_agreement`、`throughput_tps`、`peak_vram_gb`

### 时间预算
每轮实验固定 **15 分钟**（wall clock 训练时间，不含 startup/Teacher 缓存加载）。
因为 Teacher soft labels 已预缓存，15 分钟全部用于 Student 训练。

### 简洁性原则（来自 autoresearch）
同等效果下，更简单更好。微小提升 + 大量复杂代码 = 不值得。删代码还能持平 = 好实验。

## Output format

脚本结束后打印：

```
---
val_metric:        0.8523
val_loss:          1.2345
teacher_agreement: 0.9120
training_seconds:  900.1
total_seconds:     925.3
peak_vram_gb:      45.2
throughput_tps:    12345
student_params_M:  800.0
student_name:      qwen3-0.8b
distill_mode:      logit
temperature:       4.0
alpha:             0.7
```

提取关键指标：`grep "^val_metric:\|^val_loss:\|^peak_vram_gb:" run.log`

## Logging results

实验结果记录到 `results.tsv`（制表符分隔）：

```
commit	val_metric	val_loss	vram_gb	status	description
a1b2c3d	0.0000	2.3456	45.2	keep	baseline logit distill T=4.0
b2c3d4e	0.0000	2.1234	45.5	keep	increase alpha to 0.8
c3d4e5f	0.0000	2.4567	45.2	discard	switch to feature distill
d4e5f6g	0.0000	0.0000	0.0	crash	progressive distill OOM
```

## 实验循环

LOOP FOREVER:

1. 查看 git 状态和当前分支/commit
2. 修改 `distill.py`，尝试新的蒸馏策略或超参调整
3. git commit
4. 运行实验：`uv run distill.py > run.log 2>&1`
5. 读取结果：`grep "^val_metric:\|^val_loss:\|^peak_vram_gb:" run.log`
6. 如果 grep 为空 = 崩溃。`tail -n 50 run.log` 查看错误，尝试修复
7. 记录到 results.tsv（不 commit 此文件）
8. 指标提升 → keep（推进分支）
9. 指标持平或变差 → discard（`git reset` 回滚）

### 超时
每轮实验预期 ~15 分钟。超过 25 分钟 → kill 并视为失败。

### 崩溃处理
简单 bug（typo/import）→ 修了重跑。根本性问题 → 记录 crash，继续下一个。

### 永不停止
实验开始后，**不要停下来问人要不要继续**。人可能在睡觉。你是自主的。
15 分钟/轮 = 4 轮/小时 = 约 30 轮过一夜。

## 推荐实验路径（基于 post-training 研究洞察）

以下实验路径基于 [post-training-notes](https://github.com/tanlon2miracle-ops/post-training-notes) 中 7 篇工作的核心发现。**不是必须按顺序执行**，但建议优先验证高概率有效的方向。

### Phase A：Baseline + 基础调参（前 5-8 轮）

1. **Baseline**：默认配置跑一次，建立基线
2. **温度扫描**：T = 1.0 → 2.0 → 4.0 → 8.0 → 16.0
   - 来自 DeepSeek-R1：蒸馏温度对知识转移效率影响最大
3. **Alpha 扫描**：α = 0.5 → 0.7 → 0.9 → 1.0（纯 soft）
4. **学习率扫描**：1e-5 → 2e-5 → 5e-5 → 1e-4

### Phase B：策略创新（8-20 轮）

5. **Top-k Logit Filtering**：只蒸馏 Teacher top-k logits
   - 来自 Skywork-OR1 的发现：噪声 logits 反而干扰学习
6. **Dynamic Temperature Schedule**：
   - 线性衰减：高温开始 → 低温收尾
   - 余弦退火
   - 来自 OR1 的 Adaptive Entropy Control 思路：训练早期需要更多探索
7. **Curriculum Learning**：
   - 按 Teacher confidence 排序样本，先易后难
   - 来自 OR1 和 AceReason：动态过滤已掌握数据提升效率
8. **Loss 混合实验**：
   - KL divergence → MSE → cosine similarity
   - 多任务 loss：蒸馏 + 自回归 + 对比学习
   - 来自 AceReason：多阶段 loss 权重调整可能优于固定配比
9. **Feature Distillation**：
   - 对齐中间层表示（需要确定层映射策略）
   - 来自通用蒸馏文献：logit + feature 混合通常优于单一

### Phase C：高级探索（20+ 轮）

10. **Progressive Distillation**：逐层渐进式蒸馏
    - 来自 DeepSeek-R1 四阶段训练的启发：复杂任务分阶段处理
11. **Teacher 生成增强**：用 Teacher 生成额外训练数据
    - 来自 Nemotron 数据集：合成数据大幅提升多样性
12. **Student 初始化实验**：
    - random vs teacher-slice vs pretrained
    - 来自 AceReason：蒸馏模型上继续 RL 也能大幅提升
13. **Attention Transfer**：对齐 Teacher/Student attention maps
14. **Loss 配比动态调度**：
    - 训练早期 α=0.9（主 soft），后期 α=0.3（主 hard）
    - 或反过来——让 Agent 自行发现最优调度

### 值得注意的陷阱（来自 post-training 研究）

- **Entropy collapse**（来自 Skywork-OR1）：如果 Student 输出分布快速坍缩（val_loss 先降后升），考虑：
  - 降低学习率
  - 提高温度
  - 添加 entropy bonus 到 loss
- **过拟合 Teacher 噪声**：Teacher 不总是对的，α=1.0（纯 soft）可能不如混合
- **显存 OOM**：feature distillation 需要存中间层，显存会飙升。先用小 batch 试
- **KL loss 的双刃剑**（来自 OR1）：大规模训练后 KL 约束可能阻碍性能，可考虑去掉或退火

## 蒸馏超参速查

```python
# distill.py 中可调参数一览
TEMPERATURE = 4.0              # 蒸馏温度（核心参数）
ALPHA = 0.7                    # soft/hard loss 权重
LEARNING_RATE = 2e-5           # Student 学习率
BATCH_SIZE = 32                # 训练 batch
GRADIENT_ACCUMULATION = 4      # 梯度累积
WARMUP_RATIO = 0.1             # warm-up 比例
WEIGHT_DECAY = 0.01            # 权重衰减
MAX_GRAD_NORM = 1.0            # 梯度裁剪
DISTILL_MODE = "logit"         # 蒸馏策略
STUDENT_FROM_PRETRAINED = True # Student 初始化方式
EVAL_EVERY_STEPS = 50          # 评估频率
```
