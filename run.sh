#!/bin/bash
# autodistill — 快速启动脚本
# Usage: bash run.sh [student_name]

set -e

STUDENT=${1:-"qwen3-0.8b"}

echo "=== autodistill ==="
echo "Student: $STUDENT"
echo "Time budget: 15 min per experiment"
echo ""

# 检查缓存
if [ ! -d "$HOME/.cache/autodistill/train" ]; then
    echo "ERROR: Teacher cache not found!"
    echo "Please run first: uv run prepare.py --seed-data <path> --teacher <model>"
    exit 1
fi

# 初始化 results.tsv
if [ ! -f results.tsv ]; then
    printf "commit\tval_metric\tval_loss\tvram_gb\tstatus\tdescription\n" > results.tsv
    echo "Created results.tsv"
fi

# 单次实验
echo "Starting experiment..."
uv run distill.py --student "$STUDENT" > run.log 2>&1

echo ""
echo "=== Results ==="
grep "^val_metric:\|^val_loss:\|^teacher_agreement:\|^peak_vram_gb:\|^throughput_tps:" run.log
