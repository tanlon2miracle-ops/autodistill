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
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import argparse
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
DISTILL_MODE = "logit"         # "logit" | "feature" | "attention" | "progressive"

# Student 配置
STUDENT_NAME = "qwen3-0.8b"
STUDENT_FROM_PRETRAINED = True

# 评估间隔
EVAL_EVERY_STEPS = 50

# ===================================================================
# Loss 函数（Agent 可以随意改造）
# ===================================================================
def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA):
    """
    标准 logit 蒸馏 loss = alpha * KL_div(soft) + (1-alpha) * CE(hard)
    
    Agent 可以修改此函数：
    - 换 KL 为 MSE、cosine similarity
    - 加 feature distillation loss
    - 加 attention transfer loss
    - 加 contrastive loss
    - 动态调 temperature / alpha
    """
    # Soft label loss (KL divergence)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    # Hard label loss (Cross-entropy)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * soft_loss + (1 - alpha) * hard_loss


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
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # 混合精度
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_ctx = autocast(dtype=torch.bfloat16) if use_bf16 else nullcontext()
    scaler = GradScaler(enabled=not use_bf16)

    # 训练
    print(f"[distill] Starting training (budget={TIME_BUDGET}s, mode={DISTILL_MODE})")
    print(f"[distill] T={TEMPERATURE}, alpha={ALPHA}, lr={LEARNING_RATE}, bs={BATCH_SIZE}x{GRADIENT_ACCUMULATION}")

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

            with amp_ctx:
                outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = outputs.logits[:, :-1, :]
                teacher_logits_shift = teacher_logits[:, :-1, :]
                labels_shift = labels[:, :student_logits.size(1)]

                loss = distillation_loss(
                    student_logits, teacher_logits_shift, labels_shift,
                    temperature=TEMPERATURE, alpha=ALPHA,
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
                    print(f"  step {actual_step} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
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

    # 最终评估
    total_time = time.perf_counter() - train_start
    print("\n[distill] Training complete. Running final evaluation...")

    final_metrics = evaluate_distill(student, val_loader, device)
    throughput = measure_throughput(student, tokenizer, device)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
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
    print(f"num_steps:         {global_step}")


if __name__ == "__main__":
    train()
