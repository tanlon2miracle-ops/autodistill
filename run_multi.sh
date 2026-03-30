#!/bin/bash
# autodistill — 多 Student 批量实验
# 对同一份 Teacher 缓存，依次跑不同 Student 的基线
# Usage: bash run_multi.sh

set -e

STUDENTS=("qwen3-4b" "qwen3-0.8b" "qwen3-0.6b")

echo "=== autodistill multi-student sweep ==="
echo "Students: ${STUDENTS[@]}"
echo ""

for student in "${STUDENTS[@]}"; do
    echo "=========================================="
    echo "Running baseline for: $student"
    echo "=========================================="

    uv run distill.py --student "$student" > "run_${student}.log" 2>&1

    echo "Results for $student:"
    grep "^val_metric:\|^val_loss:\|^teacher_agreement:\|^peak_vram_gb:" "run_${student}.log"
    echo ""
done

echo "=== All baselines complete ==="
echo "Check run_*.log for details"
