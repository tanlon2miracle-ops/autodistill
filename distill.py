"""
autodistill — 蒸馏训练脚本（Agent 唯一修改的文件）

Agent 可以自由修改此文件中的一切：
- 蒸馏策略（logit/feature/attention/progressive）
- Loss 函数设计
- 超参数
- 数据增强
- 学习率调度
- 任何能降低 val_loss / 提高 teacher_agreement 的改动

Usage: uv run distill.py [--student qwen3-0.8b]

基于 post-training-notes 研究洞察的改进：
- 支持 Top-k Logit Filtering（Skywork-OR1 启发）
- 支持 Dynamic Temperature Schedule（OR1 Adaptive Entropy Control 启发）
- 支持 Curriculum Learning（OR1 + AceReason 启发）
- 支持 Feature Distillation（混合 logit + feature）
- 改进的 Loss 设计（多种距离度量可选）
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from prepare import (
    TIME_BUDGET, MAX_SEQ_LEN, EVAL_SAMPLES,
    make_dataloader, evaluate_distill, measure_throughput,
)
from models.student import load_student

# ===================================================================
# 蒸馏超参（Agent 在这里调参）
# ===================================================================
TEMPERATURE = 4.0              # 蒸馏温度
ALPHA = 0.7                    # soft label loss 权重（1-ALPHA = hard label loss）
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 蒸馏策略
DISTILL_MODE = "logit"         # "logit" | "feature" | "attention" | "progressive" | "logit+feature"

# Student 配置
STUDENT_NAME = "qwen3-0.8b"
STUDENT_FROM_PRETRAINED = True

# 评估间隔
EVAL_EVERY_STEPS = 50

# === 高级选项（来自 post-training 研究） ===

# Top-k Logit Filtering (Skywork-OR1 启发：只蒸馏 Teacher 的高置信 logits，过滤噪声)
TOPK_LOGITS = None             # int 或 None（None = 不过滤，建议尝试 50/100/500）

# Dynamic Temperature Schedule (OR1 Adaptive Entropy Control 启发)
TEMPERATURE_SCHEDULE = "constant"  # "constant" | "linear_decay" | "cosine" | "warmup_decay"
TEMPERATURE_START = 8.0        # 仅当 schedule != constant 时生效
TEMPERATURE_END = 2.0          # 仅当 schedule != constant 时生效

# Curriculum Learning (OR1 + AceReason 启发：先易后难)
USE_CURRICULUM = False         # 是否按 Teacher confidence 排序样本
CURRICULUM_WARMUP_RATIO = 0.3  # 前 30% 步数只用高置信样本

# Loss 类型选择
LOSS_TYPE = "kl_div"           # "kl_div" | "mse" | "cosine" | "symmetric_kl"

# Entropy Regularization (防 Entropy Collapse，来自 Skywork-OR1)
ENTROPY_BONUS = 0.0            # > 0 时添加 entropy bonus 防止 Student 分布坍缩

# Alpha Schedule (动态调整 soft/hard 权重)
ALPHA_SCHEDULE = "constant"    # "constant" | "soft_to_hard" | "hard_to_soft"


# ===================================================================
# 工具函数
# ===================================================================
def get_temperature(step, total_steps):
    """根据 schedule 计算当前温度"""
    if TEMPERATURE_SCHEDULE == "constant":
        return TEMPERATURE
    progress = min(step / max(total_steps, 1), 1.0)
    if TEMPERATURE_SCHEDULE == "linear_decay":
        return TEMPERATURE_START + (TEMPERATURE_END - TEMPERATURE_START) * progress
    elif TEMPERATURE_SCHEDULE == "cosine":
        return TEMPERATURE_END + (TEMPERATURE_START - TEMPERATURE_END) * 0.5 * (1 + math.cos(math.pi * progress))
    elif TEMPERATURE_SCHEDULE == "warmup_decay":
        if progress < 0.1:
            return TEMPERATURE_START * (progress / 0.1)
        return TEMPERATURE_START + (TEMPERATURE_END - TEMPERATURE_START) * ((progress - 0.1) / 0.9)
    return TEMPERATURE


def get_alpha(step, total_steps):
    """根据 schedule 计算当前 alpha"""
    if ALPHA_SCHEDULE == "constant":
        return ALPHA
    progress = min(step / max(total_steps, 1), 1.0)
    if ALPHA_SCHEDULE == "soft_to_hard":
        # 训练早期主 soft label (α=0.9) → 后期主 hard label (α=0.3)
        return 0.9 - 0.6 * progress
    elif ALPHA_SCHEDULE == "hard_to_soft":
        # 反过来
        return 0.3 + 0.6 * progress
    return ALPHA


def filter_topk_logits(logits, k):
    """只保留 top-k logits，其余设为 -inf（减少噪声蒸馏）"""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    topk_vals, topk_ids = logits.topk(k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, topk_ids, topk_vals)
    return filtered


# ===================================================================
# Loss 函数（Agent 可以随意改造）
# ===================================================================
def distillation_loss(
    student_logits, teacher_logits, labels,
    temperature=TEMPERATURE, alpha=ALPHA,
    step=0, total_steps=1,
):
    """
    蒸馏 loss，支持多种模式。
    
    改进点（基于 post-training 研究）：
    1. Top-k Logit Filtering: 只蒸馏 Teacher 高置信输出
    2. 多种距离度量: KL / MSE / Cosine / Symmetric KL
    3. Entropy Bonus: 防止 Student 分布坍缩
    4. Dynamic alpha/temperature: 随训练进度调整
    """
    # 动态参数
    current_temp = get_temperature(step, total_steps)
    current_alpha = get_alpha(step, total_steps)
    
    # Top-k 过滤
    teacher_for_soft = filter_topk_logits(teacher_logits, TOPK_LOGITS)
    
    # === Soft Label Loss ===
    if LOSS_TYPE == "kl_div":
        student_soft = F.log_softmax(student_logits / current_temp, dim=-1)
        teacher_soft = F.softmax(teacher_for_soft / current_temp, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (current_temp ** 2)
    elif LOSS_TYPE == "symmetric_kl":
        # Symmetric KL: 0.5 * (KL(S||T) + KL(T||S))，更稳定
        student_soft = F.log_softmax(student_logits / current_temp, dim=-1)
        teacher_soft_log = F.log_softmax(teacher_for_soft / current_temp, dim=-1)
        teacher_soft = F.softmax(teacher_for_soft / current_temp, dim=-1)
        student_soft_prob = F.softmax(student_logits / current_temp, dim=-1)
        kl_st = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        kl_ts = F.kl_div(teacher_soft_log, student_soft_prob, reduction="batchmean")
        soft_loss = 0.5 * (kl_st + kl_ts) * (current_temp ** 2)
    elif LOSS_TYPE == "mse":
        student_soft = student_logits / current_temp
        teacher_soft = teacher_for_soft / current_temp
        soft_loss = F.mse_loss(student_soft, teacher_soft)
    elif LOSS_TYPE == "cosine":
        student_soft = student_logits / current_temp
        teacher_soft = teacher_for_soft / current_temp
        cos_sim = F.cosine_similarity(student_soft, teacher_soft, dim=-1)
        soft_loss = (1 - cos_sim).mean()
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")

    # === Hard Label Loss (Cross-entropy) ===
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # === Entropy Bonus (防 Entropy Collapse) ===
    entropy_loss = 0.0
    if ENTROPY_BONUS > 0:
        student_probs = F.softmax(student_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        entropy = -(student_probs * student_log_probs).sum(dim=-1).mean()
        entropy_loss = -ENTROPY_BONUS * entropy  # 负号：最大化 entropy

    return current_alpha * soft_loss + (1 - current_alpha) * hard_loss + entropy_loss


# ===================================================================
# 训练循环
# ===================================================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default=STUDENT_NAME)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_name = args.student

    # 加载 Student
    print(f"[distill] Loading Student: {student_name}")
    student, tokenizer, model_type = load_student(
        student_name, from_pretrained=STUDENT_FROM_PRETRAINED
    )
    student = student.to(device)

    # 数据加载
    print("[distill] Loading data...")
    train_loader = make_dataloader("train", batch_size=BATCH_SIZE)
    val_loader = make_dataloader("val", batch_size=BATCH_SIZE)

    # 优化器
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # 学习率调度
    total_steps_estimate = (TIME_BUDGET // 15) * (len(train_loader) // GRADIENT_ACCUMULATION)
    warmup_steps = int(total_steps_estimate * WARMUP_RATIO)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps_estimate - warmup_steps, 1)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # 混合精度
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = autocast(dtype=torch.bfloat16) if use_bf16 else nullcontext()
    scaler = GradScaler(enabled=(device == "cuda" and not use_bf16))

    # 打印配置
    print(f"[distill] Starting training (budget={TIME_BUDGET}s, mode={DISTILL_MODE})")
    print(f"[distill] T={TEMPERATURE}, alpha={ALPHA}, lr={LEARNING_RATE}, bs={BATCH_SIZE}x{GRADIENT_ACCUMULATION}")
    print(f"[distill] loss={LOSS_TYPE}, topk={TOPK_LOGITS}, temp_schedule={TEMPERATURE_SCHEDULE}")
    print(f"[distill] curriculum={USE_CURRICULUM}, entropy_bonus={ENTROPY_BONUS}, alpha_schedule={ALPHA_SCHEDULE}")

    student.train()
    global_step = 0
    best_val_loss = float("inf")
    train_start = time.perf_counter()
    step_loss_acc = 0.0

    while True:
        for batch in train_loader:
            # 检查时间预算
            elapsed = time.perf_counter() - train_start
            if elapsed >= TIME_BUDGET:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            labels = batch.get("labels", input_ids[:, 1:]).to(device)

            # Curriculum Learning: 跳过低置信样本（训练早期）
            if USE_CURRICULUM:
                progress = global_step / max(total_steps_estimate, 1)
                if progress < CURRICULUM_WARMUP_RATIO:
                    # 计算 Teacher 置信度（top-1 概率均值）
                    teacher_conf = F.softmax(teacher_logits, dim=-1).max(dim=-1).values.mean()
                    # 前 30% 步数只保留高置信样本
                    threshold = 0.5 * (1 - progress / CURRICULUM_WARMUP_RATIO)
                    if teacher_conf < threshold:
                        global_step += 1
                        continue

            with amp_ctx:
                outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = outputs.logits[:, :-1, :]
                teacher_logits_shift = teacher_logits[:, :-1, :]
                labels_shift = labels[:, :student_logits.size(1)]

                loss = distillation_loss(
                    student_logits, teacher_logits_shift, labels_shift,
                    step=global_step, total_steps=total_steps_estimate,
                )
                loss = loss / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()
            step_loss_acc += loss.item()

            if (global_step + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                actual_step = (global_step + 1) // GRADIENT_ACCUMULATION
                if actual_step % 10 == 0:
                    avg_loss = step_loss_acc / 10
                    lr = scheduler.get_last_lr()[0]
                    current_t = get_temperature(global_step, total_steps_estimate)
                    current_a = get_alpha(global_step, total_steps_estimate)
                    print(f"  step {actual_step} | loss {avg_loss:.4f} | lr {lr:.2e} | T {current_t:.1f} | α {current_a:.2f} | {elapsed:.0f}s")
                    step_loss_acc = 0.0

                # 定期评估
                if actual_step % EVAL_EVERY_STEPS == 0:
                    metrics = evaluate_distill(student, val_loader, device)
                    print(f"  [eval] val_loss={metrics['val_loss']:.4f} agreement={metrics['teacher_agreement']:.4f}")
                    if metrics["val_loss"] < best_val_loss:
                        best_val_loss = metrics["val_loss"]

            global_step += 1

        # 时间到
        elapsed = time.perf_counter() - train_start
        if elapsed >= TIME_BUDGET:
            break


    # 保存 checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"latest_{student_name}.pt"
    torch.save({
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": {
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "distill_mode": DISTILL_MODE,
            "student_name": student_name,
        },
    }, ckpt_path)
    print(f"[distill] Checkpoint saved to {ckpt_path}")
    # 最终评估
    total_time = time.perf_counter() - train_start
    print("\n[distill] Training complete. Running final evaluation...")

    final_metrics = evaluate_distill(student, val_loader, device)
    throughput = measure_throughput(student, tokenizer, device)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0
    n_params = sum(p.numel() for p in student.parameters()) / 1e6

    # 输出结果（与 autoresearch 格式对齐）
    print("\n---")
    print(f"val_metric:        {final_metrics['val_metric']:.6f}")
    print(f"val_loss:          {final_metrics['val_loss']:.6f}")
    print(f"teacher_agreement: {final_metrics['teacher_agreement']:.6f}")
    print(f"training_seconds:  {total_time:.1f}")
    print(f"total_seconds:     {total_time:.1f}")
    print(f"peak_vram_gb:      {peak_vram:.1f}")
    print(f"throughput_tps:    {throughput:.0f}")
    print(f"student_params_M:  {n_params:.1f}")
    print(f"student_name:      {student_name}")
    print(f"distill_mode:      {DISTILL_MODE}")
    print(f"temperature:       {TEMPERATURE}")
    print(f"alpha:             {ALPHA}")
    print(f"loss_type:         {LOSS_TYPE}")
    print(f"topk_logits:       {TOPK_LOGITS}")
    print(f"temp_schedule:     {TEMPERATURE_SCHEDULE}")
    print(f"alpha_schedule:    {ALPHA_SCHEDULE}")
    print(f"curriculum:        {USE_CURRICULUM}")
    print(f"entropy_bonus:     {ENTROPY_BONUS}")
    print(f"num_steps:         {global_step}")


if __name__ == "__main__":
    train()
