#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Hugging Face Transformers + Trainer 的独立图像分类训练脚本
支持 ViT 模型，从本地文件夹加载数据，完整训练 + 评估
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict
from dataclasses import dataclass, field

# ==================== 1. 配置参数 ====================
@dataclass
class Config:
    """训练配置参数"""
    # 数据路径
    data_dir: str = "./data/imagenette"  # 数据集根目录，下含 train/val 子目录
    output_dir: str = "./vit_finetuned"  # 模型保存路径
    
    # 模型配置
    model_name: str = "google/vit-base-patch16-224"  # 预训练模型名称
    num_classes: int = 10  # 分类数（Imagenette 为 10 类）
    
    # 训练超参数
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1  # warmup 步数占总步数的比例
    max_grad_norm: float = 1.0
    
    # Trainer 配置
    seed: int = 42
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    fp16: bool = True  # 混合精度训练
    gradient_accumulation_steps: int = 1
    
    # 图像预处理尺寸
    image_size: int = 224  # ViT 输入尺寸


# ==================== 2. 数据预处理 ====================
def get_transforms(image_size: int, is_train: bool = True):
    """获取图像预处理变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def load_datasets(config: Config):
    """加载训练集和验证集"""
    train_path = os.path.join(config.data_dir, "train")
    val_path = os.path.join(config.data_dir, "val")
    
    # 检查路径是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据目录不存在: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证数据目录不存在: {val_path}")
    
    # 加载数据集
    train_dataset = ImageFolder(
        train_path,
        transform=get_transforms(config.image_size, is_train=True)
    )
    val_dataset = ImageFolder(
        val_path,
        transform=get_transforms(config.image_size, is_train=False)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"类别映射: {train_dataset.classes}")
    
    return train_dataset, val_dataset


# ==================== 3. 模型加载 ====================
def load_model(config: Config):
    """加载预训练模型并调整分类头"""
    # 加载图像处理器（用于验证预处理一致性）
    image_processor = AutoImageProcessor.from_pretrained(config.model_name)
    
    # 加载模型，修改分类头
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True,  # 允许分类头尺寸不匹配
    )
    
    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model, image_processor


# ==================== 4. 评估指标 ====================
def compute_metrics(eval_pred):
    """计算评估指标（准确率）"""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "eval_accuracy": accuracy}


# ==================== 5. 自定义 Collator（可选）====================
class ImageCollator:
    """将图像和标签整理为 Trainer 需要的格式"""
    def __init__(self, image_processor):
        self.image_processor = image_processor
    
    def __call__(self, batch):
        # batch 是 [(image_tensor, label), ...] 的形式
        images = torch.stack([item[0] for item in batch])  # (B, C, H, W)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {"pixel_values": images, "labels": labels}


# ==================== 6. 主训练流程 ====================
def main():
    # 加载配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 1. 加载数据
    print("=" * 50)
    print("1. 加载数据集...")
    train_dataset, val_dataset = load_datasets(config)
    
    # 2. 加载模型和处理器
    print("\n" + "=" * 50)
    print("2. 加载模型...")
    model, image_processor = load_model(config)
    
    # 3. 创建数据整理器
    data_collator = ImageCollator(image_processor)
    
    # 4. 配置训练参数
    print("\n" + "=" * 50)
    print("3. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=config.seed,
    )
    
    # 打印训练参数
    print(f"训练批次大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"混合精度: {config.fp16}")
    
    # 5. 创建 Trainer
    print("\n" + "=" * 50)
    print("4. 初始化 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 6. 开始训练
    print("\n" + "=" * 50)
    print("5. 开始训练...")
    trainer.train()
    
    # 7. 最终评估
    print("\n" + "=" * 50)
    print("6. 最终评估...")
    eval_results = trainer.evaluate()
    print(f"最终评估结果: {eval_results}")
    
    # 8. 保存最终模型
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    print(f"\n模型已保存至: {config.output_dir}")
    
    return trainer


# ==================== 7. 入口 ====================
if __name__ == "__main__":
    # 推荐使用以下命令运行：
    # python train.py
    
    trainer = main()
    
    # 可选：使用 tensorboard 查看训练曲线
    # tensorboard --logdir ./vit_finetuned/logs