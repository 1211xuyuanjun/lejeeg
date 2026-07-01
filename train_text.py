import os
import sys
from types import SimpleNamespace

# 将 data 目录加入搜索路径
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)

from data.data_loading import get_train_val_loaders


def create_default_args():
    """创建 get_train_val_loaders 需要的默认配置参数"""
    args = SimpleNamespace()
    
    # 数据配置
    args.data = SimpleNamespace()
    args.data.path = "/home/xuyuanjun/leeg/data/data"
    args.data.subset = "small"
    args.data.loader = SimpleNamespace()
    args.data.loader.num_workers = 16
    args.data.loader.prefetch_factor = 2
    
    # 预处理配置
    args.preprocessing = SimpleNamespace()
    args.preprocessing.window_duration = 256
    args.preprocessing.clip = 5.0
    
    # 掩码配置
    args.preprocessing.masking = SimpleNamespace()
    args.preprocessing.masking.use_block = True
    args.preprocessing.masking.masking_window = 128
    args.preprocessing.masking.masking_overlap = 64
    args.preprocessing.masking.ratio = 0.5
    args.preprocessing.masking.radius_spat_mask = 0.0
    args.preprocessing.masking.radius_temp_mask = 3
    args.preprocessing.masking.dropout_ratio = 0.1
    args.preprocessing.masking.dropout_radius = 0.0
    
    # 训练器配置
    args.trainer = SimpleNamespace()
    args.trainer.batch_size = 64
    args.trainer.n_gpus = 1
    args.trainer.n_nodes = 1
    
    # 随机种子
    args.seed = 42
    
    return args


import torch
import tqdm
import wandb
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from qwen3 import Qwen3NextModelJepa, Qwen3NextModelJepaCore

from dataclasses import dataclass, field
from typing import Optional, List

from encoder import MultiKernelConvEncoder, SpectrogramEncoder
from megatron.core.transformer.transformer_config import TransformerConfig
from leeg_layer_spec import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


@dataclass
class Qwen3NextConfig(TransformerConfig):
    # === 模型架构 ===
    input_size: int = 128
    mlp_hidden_size: int = 768
    proj_size: int = 128
    
    # === 位置编码 ===
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    mrope_section: List[int] = field(default_factory=lambda: [32, 32])
    


@dataclass
class TrainingConfig:
    # === 优化器 ===
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    
    # === 学习率调度 ===
    warmup_epochs: int = 1
    epochs: int = 10
    start_factor: float = 0.01
    
    # === 训练 ===
    batch_size: int = 2
    num_workers: int = 8
    max_grad_norm: float = 1.0
    lmab: float = 0.02
    
    # === 混合精度 ===
    dtype: str = "bfloat16"
    
    # === 数据 ===
    resampling_rate: int = 256
    dataset_list: List[str] = field(default_factory=lambda: ['TUAB'])
    signal_transform: Optional[str] = None
    
    # === 日志 ===
    project_name: str = "JEPA_EEG"
    log_interval: int = 100
    
    # === 检查点 ===
    save_dir: str = "./checkpoints"
    save_interval: int = 10


def add_time(pos, time_steps):
    """预分配输出张量，用广播赋值替代 expand+cat"""
    batch, channel, n = pos.shape
    out = pos.new_empty(batch, channel, time_steps, n + 1)
    out[..., :n] = pos.unsqueeze(2)                          # (b,c,1,n) → broadcast 到 (b,c,t,n)
    out[..., n] = torch.arange(time_steps, device=pos.device) # (t,) → broadcast 到 (b,c,t)
    return out


class SIGReg(torch.nn.Module):
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


def train_one_epoch(
        model, time_encoder, freq_encoder, train_loader, 
        optimizer, scheduler, scaler, sigreg,
        max_grad_norm, epoch, config: TrainingConfig,
):
    model.train()
    time_encoder.train()
    freq_encoder.train()

    pbar = tqdm.tqdm(train_loader, total=len(train_loader))

    B,N = 64, 2
    inputs_embeds = torch.randn(64, 1024, 128, device='cuda')
    positions = torch.randn(4, 64, 1024, device='cuda')
    for x in pbar:
        with autocast('cuda', dtype=torch.bfloat16):
            # eeg, positions = x
            # eeg = eeg.to('cuda', non_blocking=True)
            # positions = positions.to('cuda', non_blocking=True)

            # print(eeg.shape)
            # time_embedding = time_encoder(eeg)
            # b = eeg.shape[0]
            # spec = torch.stft(
            #     eeg.flatten(0, 1),
            #     n_fft=127,
            #     hop_length=8,
            #     win_length=127,
            #     window=torch.hann_window(127, device='cuda'),
            #     center=True,
            #     onesided=True,
            #     return_complex=True
            # )
            # spec = spec.reshape(b, -1, *spec.shape[1:])
            # magnitude_log = 20 * torch.log10(spec.abs() + 1e-10)
            # spec_embedding = freq_encoder(magnitude_log)
            
            # positions = add_time(positions, time_embedding.shape[2])
            # positions = positions.permute(3, 0, 1, 2).flatten(2, 3)   # (b,c,t,n) → (n,b,c*t)
            # time_embedding = time_embedding.flatten(1, 2)              # (b,c,t,d) → (b,c*t,d)
            # spec_embedding = spec_embedding.flatten(1, 2)              # (b,c,t,d) → (b,c*t,d)

            # inputs_embeds = torch.stack([time_embedding, spec_embedding], dim=1)
            # B, N = inputs_embeds.shape[:2]
            # inputs_embeds = inputs_embeds.flatten(0, 1)                # (b,n,t,d) → (b*n,t,d)
            
            # positions = positions.unsqueeze(2).expand(-1, -1, N, -1).flatten(1, 2)  # (n,b,seq) → (n,b*N,seq)
            
            hidden_states, proj = model(
                inputs_embeds=inputs_embeds,
                position_ids=positions,
            )

            proj = proj.reshape(B, N, -1).transpose(0, 1)  # (b*n,t,d) → (n,b,t*d)
            
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
    
    # 将模型放到 GPU
    device = torch.device('cuda')
    model.to(device)
    time_encoder.to(device)
    freq_encoder.to(device)
    sigreg.to(device)


    # 优化器
    lr, weight_decay = config.lr, config.weight_decay
    groups = [
        {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': time_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': freq_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
    ]
    optimizer = torch.optim.AdamW(groups, betas=config.betas, eps=config.eps)

    # 学习率调度
    warmup_steps = config.warmup_epochs * len(train_loader)
    total_steps = config.epochs * len(train_loader)

    s1 = LinearLR(optimizer, start_factor=config.start_factor, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.lr * config.start_factor)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    # 混合精度
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

def setup_distributed():
    """初始化分布式训练环境"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    dist.barrier()

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def setup_parallel_state():
    """Setup Megatron parallel state."""
    parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,  # 根据您的GPU数量调整
    pipeline_model_parallel_size=1, # 暂时不用流水线并行
    expert_model_parallel_size=1,
    context_parallel_size=2,
)
    model_parallel_cuda_manual_seed(123)

def main():
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    setup_parallel_state()
    
    
    # 模型配置
    model_config = Qwen3NextConfig(
        input_size=128,
        proj_size=128,
        mlp_hidden_size=1536,
        num_layers=16,
        hidden_size=512,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=1536,
        normalization="RMSNorm",
        activation_func=F.silu,
        qk_layernorm = True,
        layernorm_zero_centered_gamma=True,
        attention_output_gate=True,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        layernorm_epsilon=1e-6,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mrope_section=[16,16,16,16]
        # recompute_granularity="full",
        # recompute_method="uniform",
        # recompute_num_layers=1,
    )
    
    # 训练配置
    train_config = TrainingConfig(
        lr=1e-4,
        weight_decay=0.01,
        warmup_epochs=1,
        epochs=10,
        lmab=0.02,
        dtype="bfloat16",
    )
    
    # 实例化模型
    model = Qwen3NextModelJepaCore(config=model_config, spec=get_gpt_layer_with_transformer_engine_spec())
    time_encoder = MultiKernelConvEncoder(out_dim=model_config.input_size)
    freq_encoder = SpectrogramEncoder(out_dim=model_config.input_size, freq_bins=64, time_steps=32)
    sigreg = SIGReg()

    # 在构建模型之后，训练循环之前
    resume_path = "leeg/checkpoints/checkpoint_epoch_2_lejepa_0.0155.pth"
    if os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        model.load_state_dict(torch.load(resume_path))
        print("Checkpoint loaded successfully.")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count}")

    # 加载数据
    args = create_default_args()
    
    print("Loading data...")
    
    train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler = get_train_val_loaders(
        args,
        return_val=True,
    )

    print(f"训练集样本数: {len_train}, 训练集批次: {len_train_sampler}")
    print(f"验证集样本数: {len_val}, 验证集批次: {len_val_sampler}")

    # 训练
    train(model, time_encoder, freq_encoder, sigreg, train_loader, val_loader, train_config)


if __name__ == "__main__":
    main()
