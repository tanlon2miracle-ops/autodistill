"""
Student 模型加载器（只读，Agent 不改此文件）
支持：Qwen3-4B, Qwen3-0.8B, Qwen3-0.6B, BERT 系列
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)

# 预定义 Student 模型
STUDENT_REGISTRY = {
    # 生成式
    "qwen3-4b": {"hf": "Qwen/Qwen3-4B", "type": "causal"},
    "qwen3-0.8b": {"hf": "Qwen/Qwen3-0.8B", "type": "causal"},
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-0.6B", "type": "causal"},
    # BERT 系列
    "bert-base": {"hf": "bert-base-chinese", "type": "cls"},
    "bert-large": {"hf": "bert-large-chinese", "type": "cls"},
    "roberta-base": {"hf": "hfl/chinese-roberta-wwm-ext", "type": "cls"},
    "roberta-large": {"hf": "hfl/chinese-roberta-wwm-ext-large", "type": "cls"},
}


def load_student(student_name: str, num_labels: int = None, from_pretrained: bool = True):
    """
    加载 Student 模型。
    
    Args:
        student_name: 预定义名（如 "qwen3-0.8b"）或 HuggingFace 路径
        num_labels: 分类任务标签数（仅 BERT 类需要）
        from_pretrained: 是否加载预训练权重（False = 随机初始化）
    
    Returns:
        (model, tokenizer, model_type)  model_type: "causal" or "cls"
    """
    if student_name in STUDENT_REGISTRY:
        info = STUDENT_REGISTRY[student_name]
        hf_name = info["hf"]
        model_type = info["type"]
    else:
        hf_name = student_name
        model_type = "causal"  # 默认生成式

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "causal":
        if from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config)
    else:  # cls
        if from_pretrained:
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_name,
                num_labels=num_labels or 2,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
            config.num_labels = num_labels or 2
            model = AutoModelForSequenceClassification.from_config(config)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[student] Loaded {student_name}: {n_params:.1f}M parameters ({model_type})")

    return model, tokenizer, model_type
