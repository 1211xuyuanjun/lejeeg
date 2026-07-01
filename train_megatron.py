#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Megatron-Core 组件作为 Transformer 主干，保留原有数据加载和编码器，
完全使用 PyTorch 原生训练循环（不依赖 Megatron-LM 的 trainer）。
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from dataclasses import dataclass, field
from typing import Optional, List

# ------------------ 导入您的自定义模块 ------------------
# 假设 data 目录和 encoder.py 在同一级目录下
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)

from data.data_loading import get_train_val_loaders
from encoder import MultiKernelConvEncoder, SpectrogramEncoder

# ------------------ Megatron-Core 导入 ------------------
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
# 如果不想用 RoPE，可以忽略；但为了更贴近原模型，我们保留

# ------------------ 配置类（与您原有保持一致） ------------------
@dataclass
class Qwen3NextConfig:
    input_size: int = 128
    hidden_size: int = 512
    intermediate_size: int = 1280
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 128
    mlp_hidden_size: int = 768
    proj_size: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rope_type: str = "default"
    mrope_section: List[int] = field(default_factory=lambda: [32, 32])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: Optional[int] = None
    initializer_range: float = 0.02
    pad_token_id: Optional[int] = None
    use_cache: bool = False

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    warmup_epochs: int = 1
    epochs: int = 10
    start_factor: float = 0.01
    batch_size: int = 2          # 注意：实际 batch size 由 data_loader 的 batch_size 决定，这里只是备用
    num_workers: int = 8
    max_grad_norm: float = 1.0
    lmab: float = 0.02
    dtype: str = "bfloat16"
    resampling_rate: int = 256
    dataset_list: List[str] = field(default_factory=lambda: ['TUAB'])
    signal_transform: Optional[str] = None
    project_name: str = "JEPA_EEG"
    log_interval: int = 100
    save_dir: str = "./checkpoints"
    save_interval: int = 10

# ------------------ 工具函数 ------------------
def add_time(pos, time_steps):
    """
    将 (batch, channel, 3) 扩展为 (batch, channel, time_steps, 4)
    第四个维度为 [x, y, z, t]
    """
    batch, channel, n = pos.shape
    out = pos.new_empty(batch, channel, time_steps, n + 1)
    out[..., :n] = pos.unsqueeze(2)                          # (b,c,1,n) → broadcast
    out[..., n] = torch.arange(time_steps, device=pos.device) # (t,) → broadcast
    return out

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer"""
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

# ------------------ 使用 Megatron-Core 构建的 Transformer 模型 ------------------
def build_mcore_transformer_config(
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    use_flash_attn: bool = True,
):
    """创建 Megatron-Core 的 TransformerConfig 对象（单卡配置）"""
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=kv_heads,
        tensor_model_parallel_size=1,   # 单卡
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        normalization='RMSNorm',         # 使用 torch 的 LayerNorm（Fused 版本需要 apex）
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        # 开启 bf16
        bf16=True,
        fp16=False,
        # 其他默认
    )

class MCoreTransformerModel(nn.Module):
    """
    使用 Megatron-Core 的 TransformerLayer 构建主干，
    保持与原模型相同的输入输出接口。
    """
    def __init__(
        self,
        mcore_config: TransformerConfig,
        input_dim: int = 128,
        proj_dim: int = 128,
        max_seq_len: int = 32768,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        self.mcore_config = mcore_config
        self.input_dim = input_dim
        self.proj_dim = proj_dim

        # 输入投影
        self.input_proj = nn.Linear(input_dim, mcore_config.hidden_size)

        # 构建 Transformer 层的规格（Spec）
        # 使用标准的 SelfAttention 和 MLP（单卡时内部会用 nn.Linear）
        self_attn_spec = ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.no_mask},  # 双向（若需要因果可改为 causal）
            submodules=SelfAttentionSubmodules(
                linear_qkv=None,  # 将使用默认的 ColumnParallelLinear（单卡退化为 Linear）
                linear_proj=None,
            ),
        )
        mlp_spec = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=None,
                linear_fc2=None,
            ),
        )

        # 创建层列表
        self.layers = nn.ModuleList([
            TransformerLayer(
                config=mcore_config,
                layer_number=i+1,
                layer_spec=ModuleSpec(
                    module=TransformerLayer,
                    submodules={
                        "self_attention": self_attn_spec,
                        "mlp": mlp_spec,
                        # 使用融合的 bias+dropout+add（可选）
                        "self_attn_bda": bias_dropout_add_fused_train,
                        "mlp_bda": bias_dropout_add_fused_train,
                    }
                )
            )
            for i in range(mcore_config.num_layers)
        ])

        # 最终层归一化（使用 FusedLayerNorm，如果可用）
        try:
            self.norm = FusedLayerNorm(mcore_config.hidden_size, eps=1e-6)
        except:
            self.norm = nn.LayerNorm(mcore_config.hidden_size, eps=1e-6)

        # 输出投影（MLP）
        self.proj = nn.Sequential(
            nn.Linear(mcore_config.hidden_size, mcore_config.hidden_size * 2),
            nn.LayerNorm(mcore_config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(mcore_config.hidden_size * 2, proj_dim),
        )

        # 初始化权重
        self.apply(self._init_weights)

        # 可选：添加 RotaryEmbedding（如果不使用，可以注释）
        # 本示例中，我们不在 Transformer 层内使用 RoPE，而是采用绝对位置编码（由 input_proj 添加？）
        # 为了简洁，我们跳过 RoPE，而是使用可学习的位置编码（或不做任何位置编码）
        # 因为原模型使用了 Qwen3VLRotaryEmbedding，但此处为简化，我们省略。
        # 若需要，可以通过传入 rotary_pos_emb 参数给 TransformerLayer.forward

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, FusedLayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs_embeds, position_ids=None):
        """
        inputs_embeds: (B, L, D) 其中 B = batch * N (因为可能 stack 了两个模态)
        position_ids: (n, B*N, L) 形状的 RoPE 位置索引（本实现中忽略）
        """
        # 1. 投影到 hidden_size
        hidden_states = self.input_proj(inputs_embeds)  # (B, L, H)

        # 2. 调整维度顺序：TransformerLayer 需要 (L, B, H)
        hidden_states = hidden_states.transpose(0, 1)   # (L, B, H)

        # 3. 逐层前向（不传入 attention_mask，使用全连接）
        for layer in self.layers:
            # 若需要 RoPE，可以在这里计算 cos/sin 并传入 rotary_pos_emb
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                rotary_pos_emb=None,   # 此处省略
                encoder_output=None,
            )

        # 4. 恢复维度顺序
        hidden_states = hidden_states.transpose(0, 1)   # (B, L, H)
        hidden_states = self.norm(hidden_states)

        # 5. 输出投影
        proj_out = self.proj(hidden_states)             # (B, L, proj_dim)
        return hidden_states, proj_out

# ------------------ 训练函数（与您的原始逻辑基本一致） ------------------
def train_one_epoch(
        model, time_encoder, freq_encoder, train_loader,
        optimizer, scheduler, scaler, sigreg,
        max_grad_norm, epoch, config: TrainingConfig,
):
    model.train()
    time_encoder.train()
    freq_encoder.train()

    pbar = tqdm.tqdm(train_loader, total=len(train_loader))

    for x in pbar:
        with autocast('cuda', dtype=torch.bfloat16):
            eeg, positions = x
            eeg = eeg.to('cuda', non_blocking=True)
            positions = positions.to('cuda', non_blocking=True)

            # 时间分支
            time_embedding = time_encoder(eeg)   # (B, C, T1, input_dim)
            b = eeg.shape[0]

            # 频域分支
            spec = torch.stft(
                eeg.flatten(0, 1),
                n_fft=127,
                hop_length=8,
                win_length=127,
                window=torch.hann_window(127, device='cuda'),
                center=True,
                onesided=True,
                return_complex=True
            )
            spec = spec.reshape(b, -1, *spec.shape[1:])  # (B, C, freq_bins, time_frames)
            magnitude_log = 20 * torch.log10(spec.abs() + 1e-10)
            spec_embedding = freq_encoder(magnitude_log)  # (B, C, T2, input_dim)

            # 处理 positions
            positions = add_time(positions, time_embedding.shape[2])  # (B, C, T1, 4)
            positions = positions.permute(3, 0, 1, 2).flatten(2, 3)   # (4, B, C*T1)
            time_embedding = time_embedding.flatten(1, 2)  # (B, C*T1, D)
            spec_embedding = spec_embedding.flatten(1, 2)  # (B, C*T2, D)

            # 合并两个模态
            inputs_embeds = torch.stack([time_embedding, spec_embedding], dim=1)  # (B, 2, L, D)
            B, N = inputs_embeds.shape[:2]
            inputs_embeds = inputs_embeds.flatten(0, 1)    # (B*2, L, D)

            positions = positions.unsqueeze(2).expand(-1, -1, N, -1).flatten(1, 2)  # (4, B*2, L)

            # 前向模型
            hidden_states, proj = model(
                inputs_embeds=inputs_embeds,
                position_ids=positions,
            )
            # proj: (B*2, L, proj_dim)

            # 重组成 (2, B, L*proj_dim)
            proj = proj.reshape(B, N, -1).transpose(0, 1)  # (N, B, L*proj_dim)

            # 损失计算
            inv_loss = (proj - proj.mean(dim=0, keepdim=True)).square().mean()
            sigreg_loss = sigreg(proj)
            lejepa_loss = sigreg_loss * config.lmab + inv_loss * (1 - config.lmab)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(lejepa_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            lejepa_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        scheduler.step()

        wandb.log({
            "train/epoch": epoch,
            "train/lejepa": lejepa_loss.item(),
            "train/sigreg": sigreg_loss.item(),
            "train/inv": inv_loss.item(),
            "train/lr": optimizer.param_groups[0]['lr'],
        })

    return lejepa_loss, sigreg_loss, inv_loss


def validate(model, time_encoder, freq_encoder, sigreg, test_loader, config: TrainingConfig):
    model.eval()
    time_encoder.eval()
    freq_encoder.eval()

    pbar = tqdm.tqdm(test_loader, total=len(test_loader))

    with torch.inference_mode():
        for x in pbar:
            with autocast('cuda', dtype=torch.bfloat16):
                eeg, positions = x
                eeg = eeg.to('cuda', non_blocking=True)
                positions = positions.to('cuda', non_blocking=True)

                time_embedding = time_encoder(eeg)
                b = eeg.shape[0]
                spec = torch.stft(
                    eeg.flatten(0, 1),
                    n_fft=127,
                    hop_length=8,
                    win_length=127,
                    window=torch.hann_window(127, device='cuda'),
                    center=True,
                    onesided=True,
                    return_complex=True
                )
                spec = spec.reshape(b, -1, *spec.shape[1:])
                magnitude_log = 20 * torch.log10(spec.abs() + 1e-10)
                spec_embedding = freq_encoder(magnitude_log)

                positions = add_time(positions, time_embedding.shape[2])
                positions = positions.permute(3, 0, 1, 2).flatten(2, 3)
                time_embedding = time_embedding.flatten(1, 2)
                spec_embedding = spec_embedding.flatten(1, 2)

                inputs_embeds = torch.stack([time_embedding, spec_embedding], dim=1)
                B, N = inputs_embeds.shape[:2]
                inputs_embeds = inputs_embeds.flatten(0, 1)

                positions = positions.unsqueeze(2).expand(-1, -1, N, -1).flatten(1, 2)

                hidden_states, proj = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=positions,
                )

                proj = proj.reshape(B, N, -1).transpose(0, 1)

                inv_loss = (proj - proj.mean(dim=0, keepdim=True)).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * config.lmab + inv_loss * (1 - config.lmab)

            wandb.log({
                "validate/lejepa": lejepa_loss.item(),
                "validate/sigreg": sigreg_loss.item(),
                "validate/inv": inv_loss.item(),
            })


def train(model, time_encoder, freq_encoder, sigreg, train_loader, val_loader, config: TrainingConfig):
    os.makedirs(config.save_dir, exist_ok=True)
    wandb.init(project=config.project_name, config=config.__dict__)
    torch.manual_seed(42)

    device = torch.device('cuda')
    model.to(device)
    time_encoder.to(device)
    freq_encoder.to(device)
    sigreg.to(device)

    # 可以开启 torch.compile（可选）
    # model = torch.compile(model)

    # 优化器
    lr, weight_decay = config.lr, config.weight_decay
    groups = [
        {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': time_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': freq_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
    ]
    optimizer = torch.optim.AdamW(groups, betas=config.betas, eps=config.eps)

    warmup_steps = config.warmup_epochs * len(train_loader)
    total_steps = config.epochs * len(train_loader)
    s1 = LinearLR(optimizer, start_factor=config.start_factor, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.lr * config.start_factor)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler() if config.dtype != "float32" else None

    for epoch in range(config.epochs):
        lejepa_loss, sigreg_loss, inv_loss = train_one_epoch(
            model=model,
            time_encoder=time_encoder,
            freq_encoder=freq_encoder,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sigreg=sigreg,
            max_grad_norm=config.max_grad_norm,
            epoch=epoch,
            config=config,
        )

        validate(model, time_encoder, freq_encoder, sigreg, val_loader, config)

        checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch}_lejepa_{lejepa_loss:.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    wandb.finish()


def create_default_args():
    """数据加载需要的参数（同您之前）"""
    args = SimpleNamespace()
    args.data = SimpleNamespace()
    args.data.path = "/home/xuyuanjun/leeg/data/data"   # 请修改为您的实际路径
    args.data.subset = "small"
    args.data.loader = SimpleNamespace()
    args.data.loader.num_workers = 16
    args.data.loader.prefetch_factor = 2

    args.preprocessing = SimpleNamespace()
    args.preprocessing.window_duration = 256
    args.preprocessing.clip = 5.0

    args.preprocessing.masking = SimpleNamespace()
    args.preprocessing.masking.use_block = True
    args.preprocessing.masking.masking_window = 128
    args.preprocessing.masking.masking_overlap = 64
    args.preprocessing.masking.ratio = 0.5
    args.preprocessing.masking.radius_spat_mask = 0.0
    args.preprocessing.masking.radius_temp_mask = 3
    args.preprocessing.masking.dropout_ratio = 0.1
    args.preprocessing.masking.dropout_radius = 0.0

    args.trainer = SimpleNamespace()
    args.trainer.batch_size = 64
    args.trainer.n_gpus = 1
    args.trainer.n_nodes = 1

    args.seed = 42
    return args


def main():
    # ---- 模型配置 ----
    model_config = Qwen3NextConfig(
        input_size=128,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=32768,
        proj_size=128,
    )

    # ---- 创建 Megatron-Core 配置 ----
    mcore_config = build_mcore_transformer_config(
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        max_seq_len=model_config.max_position_embeddings,
        use_flash_attn=True,   # 若已安装 flash-attn
    )

    # ---- 构建模型 ----
    model = MCoreTransformerModel(
        mcore_config=mcore_config,
        input_dim=model_config.input_size,
        proj_dim=model_config.proj_size,
        max_seq_len=model_config.max_position_embeddings,
        rotary_base=model_config.rope_theta,
    )

    # ---- 编码器和正则器 ----
    time_encoder = MultiKernelConvEncoder(out_dim=model_config.input_size)
    freq_encoder = SpectrogramEncoder(out_dim=model_config.input_size, freq_bins=64, time_steps=32)
    sigreg = SIGReg()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count}")

    # ---- 数据加载 ----
    args = create_default_args()
    print("Loading data...")
    train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler = get_train_val_loaders(
        args,
        return_val=True,
    )
    print(f"训练集样本数: {len_train}, 训练集批次: {len_train_sampler}")
    print(f"验证集样本数: {len_val}, 验证集批次: {len_val_sampler}")

    # ---- 训练配置 ----
    train_config = TrainingConfig(
        lr=1e-4,
        weight_decay=0.01,
        warmup_epochs=1,
        epochs=10,
        lmab=0.02,
        dtype="bfloat16",
    )

    # ---- 开始训练 ----
    train(model, time_encoder, freq_encoder, sigreg, train_loader, val_loader, train_config)


if __name__ == "__main__":
    main()