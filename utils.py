"""
autodistill — 工具函数
"""

import subprocess
import time
from pathlib import Path


def get_git_hash(short=True):
    """获取当前 git commit hash"""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "", "HEAD"]
        cmd = [c for c in cmd if c]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def log_result(
    results_file: str,
    commit: str,
    val_metric: float,
    val_loss: float,
    vram_gb: float,
    status: str,
    description: str,
):
    """向 results.tsv 追加一行实验记录"""
    path = Path(results_file)
    if not path.exists():
        with open(path, "w") as f:
            f.write("commit\tval_metric\tval_loss\tvram_gb\tstatus\tdescription\n")

    with open(path, "a") as f:
        f.write(f"{commit}\t{val_metric:.6f}\t{val_loss:.6f}\t{vram_gb:.1f}\t{status}\t{description}\n")


def format_time(seconds: float) -> str:
    """格式化秒数为可读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h{int(m)}m"


def count_parameters(model) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "total_M": total / 1e6,
        "trainable": trainable,
        "trainable_M": trainable / 1e6,
        "frozen": total - trainable,
        "frozen_M": (total - trainable) / 1e6,
    }


def get_gpu_info():
    """获取 GPU 信息"""
    import torch
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "name": props.name,
            "total_memory_gb": round(props.total_mem / (1024**3), 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return info
