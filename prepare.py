"""
autodistill — 数据准备 + 评估工具（只读，Agent 不改此文件）

功能：
1. 加载种子数据
2. 调用 Teacher 生成 soft labels 并缓存
3. 提供评估函数
4. 提供数据加载器
"""

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
TIME_BUDGET = 900  # 15 分钟训练时间预算（秒）
CACHE_DIR = Path.home() / ".cache" / "autodistill"
EVAL_SAMPLES = 500  # 验证集评估样本数
MAX_SEQ_LEN = 2048

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class DistillDataset(Dataset):
    """
    加载种子数据 + Teacher soft labels 缓存。
    每个样本包含：input_ids, attention_mask, teacher_logits, labels
    """
    def __init__(self, split="train", max_seq_len=MAX_SEQ_LEN):
        self.split = split
        self.max_seq_len = max_seq_len
        self.data_dir = CACHE_DIR / split

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"数据目录不存在：{self.data_dir}\n"
                f"请先运行：uv run prepare.py --seed-data <path> --teacher <model>"
            )

        self.files = sorted(self.data_dir.glob("*.pt"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"数据目录为空：{self.data_dir}")

        # 加载元信息
        meta_path = CACHE_DIR / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        print(f"[prepare] Loaded {len(self.files)} {split} samples from {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)


def make_dataloader(split="train", batch_size=32, max_seq_len=MAX_SEQ_LEN, num_workers=4):
    """创建数据加载器"""
    dataset = DistillDataset(split=split, max_seq_len=max_seq_len)
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Teacher 推理 + 缓存
# ---------------------------------------------------------------------------
def generate_teacher_cache(
    seed_data_path: str,
    teacher_name: str,
    tokenizer_name: str = None,
    batch_size: int = 4,
    max_seq_len: int = MAX_SEQ_LEN,
    val_ratio: float = 0.1,
):
    """
    用 Teacher 模型对种子数据做推理，缓存 soft labels。
    
    Args:
        seed_data_path: 种子数据路径（jsonl，每行 {"text": "..."} 或 {"input": "...", "output": "..."}）
        teacher_name: Teacher 模型名（HuggingFace 格式，如 Qwen/Qwen3-235B）
        tokenizer_name: Tokenizer 名（默认同 teacher）
        batch_size: Teacher 推理 batch size
        max_seq_len: 最大序列长度
        val_ratio: 验证集比例
    """
    from models.teacher import load_teacher

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "train").mkdir(exist_ok=True)
    (CACHE_DIR / "val").mkdir(exist_ok=True)

    # 检查缓存是否已存在
    cache_hash = hashlib.md5(f"{seed_data_path}:{teacher_name}:{max_seq_len}".encode()).hexdigest()[:8]
    meta_path = CACHE_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("hash") == cache_hash:
            print(f"[prepare] Cache already exists (hash={cache_hash}), skipping Teacher inference")
            return

    print(f"[prepare] Loading Teacher: {teacher_name}")
    teacher, tokenizer = load_teacher(teacher_name, tokenizer_name)

    # 加载种子数据
    print(f"[prepare] Loading seed data: {seed_data_path}")
    samples = []
    with open(seed_data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # 划分 train/val
    n_val = max(1, int(len(samples) * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"[prepare] Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Teacher 推理
    teacher.eval()
    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        print(f"[prepare] Generating Teacher soft labels for {split_name}...")
        for i, sample in enumerate(split_samples):
            text = sample.get("text", sample.get("input", ""))
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
            )
            input_ids = tokens["input_ids"].cuda()
            attention_mask = tokens["attention_mask"].cuda()

            with torch.no_grad():
                outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = outputs.logits.cpu().float()

            # 如果有 label
            labels = None
            if "output" in sample:
                label_tokens = tokenizer(
                    sample["output"],
                    return_tensors="pt",
                    max_length=max_seq_len,
                    truncation=True,
                    padding="max_length",
                )
                labels = label_tokens["input_ids"]

            cache_item = {
                "input_ids": input_ids.cpu(),
                "attention_mask": attention_mask.cpu(),
                "teacher_logits": teacher_logits,
            }
            if labels is not None:
                cache_item["labels"] = labels

            torch.save(cache_item, CACHE_DIR / split_name / f"{i:06d}.pt")

            if (i + 1) % 100 == 0:
                print(f"  [{split_name}] {i+1}/{len(split_samples)}")

    # 保存元信息
    meta = {
        "hash": cache_hash,
        "teacher": teacher_name,
        "seed_data": seed_data_path,
        "max_seq_len": max_seq_len,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[prepare] Done! Cached to {CACHE_DIR}")
    del teacher
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 评估函数（ground truth，Agent 不能修改）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_distill(student, val_loader, device="cuda"):
    """
    评估 Student 模型的蒸馏效果。
    
    Returns:
        dict: {
            "val_loss": float,          # 验证集 cross-entropy loss
            "val_metric": float,        # 主指标（生成式=1-val_loss/baseline，分类=F1）
            "teacher_agreement": float, # Teacher/Student top-1 一致率
        }
    """
    student.eval()
    total_loss = 0.0
    total_agree = 0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)
        labels = batch.get("labels", input_ids[:, 1:]).to(device)

        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits

        # Cross-entropy loss
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, :shift_logits.size(1)].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()

        # Teacher agreement
        teacher_top1 = teacher_logits[:, :-1, :].argmax(dim=-1)
        student_top1 = student_logits[:, :-1, :].argmax(dim=-1)
        mask = (shift_labels != -100)
        agree = ((teacher_top1[:, :mask.size(1)] == student_top1[:, :mask.size(1)]) & mask).sum()
        total_agree += agree.item()
        total_tokens += mask.sum().item()

        n_batches += 1
        if n_batches >= EVAL_SAMPLES // val_loader.batch_size:
            break

    avg_loss = total_loss / max(n_batches, 1)
    agreement = total_agree / max(total_tokens, 1)

    student.train()
    return {
        "val_loss": avg_loss,
        "val_metric": agreement,  # teacher_agreement 作为主指标
        "teacher_agreement": agreement,
    }


@torch.no_grad()
def measure_throughput(student, tokenizer, device="cuda", n_tokens=1000):
    """测量 Student 推理吞吐量"""
    student.eval()
    dummy = torch.randint(0, 1000, (1, min(n_tokens, MAX_SEQ_LEN))).to(device)

    # Warmup
    for _ in range(3):
        student(dummy)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        student(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tps = (10 * dummy.numel()) / elapsed
    student.train()
    return tps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autodistill data preparation")
    parser.add_argument("--seed-data", required=True, help="Path to seed data (jsonl)")
    parser.add_argument("--teacher", required=True, help="Teacher model name (HF format)")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name (default: same as teacher)")
    parser.add_argument("--batch-size", type=int, default=4, help="Teacher inference batch size")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    generate_teacher_cache(
        seed_data_path=args.seed_data,
        teacher_name=args.teacher,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
    )
