"""
autodistill — 独立评估脚本
在实验循环外单独评估某个 checkpoint，生成完整报告。

Usage: uv run validate.py --checkpoint ./checkpoints/best.pt --student qwen3-0.8b
"""

import argparse
import json
import time
import torch
from pathlib import Path

from prepare import make_dataloader, evaluate_distill, measure_throughput
from models.student import load_student


def validate(checkpoint_path: str, student_name: str, output_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 Student
    print(f"[validate] Loading student: {student_name}")
    student, tokenizer, model_type = load_student(student_name, from_pretrained=False)

    # 加载 checkpoint
    print(f"[validate] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        student.load_state_dict(state["model_state_dict"])
    else:
        student.load_state_dict(state)
    student = student.to(device)

    # 评估
    val_loader = make_dataloader("val", batch_size=32)
    print("[validate] Running evaluation...")
    metrics = evaluate_distill(student, val_loader, device)

    # 吞吐量
    throughput = measure_throughput(student, tokenizer, device)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    n_params = sum(p.numel() for p in student.parameters()) / 1e6

    report = {
        "checkpoint": checkpoint_path,
        "student": student_name,
        "params_M": round(n_params, 1),
        "val_loss": round(metrics["val_loss"], 6),
        "val_metric": round(metrics["val_metric"], 6),
        "teacher_agreement": round(metrics["teacher_agreement"], 6),
        "throughput_tps": round(throughput, 0),
        "peak_vram_gb": round(peak_vram, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n=== Validation Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--student", required=True, help="Student model name")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    validate(args.checkpoint, args.student, args.output)
