# autodistill — Agent 蒸馏实验指令

这是一个让 AI Agent 自主优化知识蒸馏的实验框架。

## Setup

与用户协商完成初始设置：

1. **确定 run tag**：基于日期（如 `mar30`）。分支 `autodistill/<tag>` 必须是新的。
2. **创建分支**：`git checkout -b autodistill/<tag>`
3. **读取关键文件**：
   - `README.md` — 项目背景
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

## 蒸馏实验思路参考

Agent 可以沿这些方向探索（但不限于）：

### 基础调参
- 温度 T：1.0 → 2.0 → 4.0 → 8.0 → 16.0
- Alpha（soft/hard 比例）：0.5 → 0.7 → 0.9 → 1.0
- 学习率：1e-5 → 2e-5 → 5e-5 → 1e-4
- Batch size × Gradient accumulation 组合

### 策略创新
- Logit distill + feature distill 混合
- 分阶段蒸馏：先 logit → 再 feature → 最后 fine-tune
- Attention transfer（对齐 Teacher/Student attention maps）
- 只蒸馏 Teacher 的 top-k logits（减少噪声）
- 动态温度：训练早期高温，后期低温
- Curriculum learning：先简单样本，再难样本
- 用 Teacher 生成额外数据做 augmentation

### Student 初始化
- 随机初始化 vs Teacher 层切片 vs 预训练权重
- 不同层的映射策略

### Loss 设计
- KL divergence vs MSE vs cosine similarity
- 多任务 loss：蒸馏 + 自回归 + 对比学习
- Label smoothing 组合
