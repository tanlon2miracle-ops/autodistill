"""
Teacher 模型加载器（只读，Agent 不改此文件）
支持大型 LLM（32B-235B）通过 HuggingFace + 量化加载。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_teacher(model_name: str, tokenizer_name: str = None):
    """
    加载 Teacher 模型。自动选择量化策略以适配显存。
    
    Args:
        model_name: HuggingFace 模型名，如 "Qwen/Qwen3-235B"
        tokenizer_name: Tokenizer 名，默认同 model_name
    
    Returns:
        (model, tokenizer)
    """
    if tokenizer_name is None:
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 根据模型大小选择加载策略
    print(f"[teacher] Loading {model_name} with auto device_map...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 多卡自动分配
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[teacher] Loaded: {n_params:.1f}B parameters")

    return model, tokenizer
