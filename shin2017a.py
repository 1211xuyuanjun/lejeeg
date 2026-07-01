import os
import sys
from types import SimpleNamespace

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
from qwen3 import Qwen3NextModelJepa, Qwen3NextModelJepaCore, Qwen3NextModelJepaShin2017a

import numpy as np
from braindecode.datautil import load_concat_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass, field
from typing import Optional, List

from encoder import MultiKernelConvEncoder, SpectrogramEncoder
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import AutoModel
from leeg_layer_spec import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


@dataclass
class Qwen3NextConfig:
    # === 模型架构 ===
    input_size: int = 128
    hidden_size: int = 512
    intermediate_size: int = 1280
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 128
    mlp_hidden_size: int = 768
    proj_size: int = 128
    
    # === 激活与归一化 ===
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    
    # === 位置编码 ===
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rope_type: str = "default"
    mrope_section: List[int] = field(default_factory=lambda: [32, 32])
    
    # === 注意力 ===
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: Optional[int] = None
    
    # === 初始化 ===
    initializer_range: float = 0.02
    
    # === 其他 ===
    pad_token_id: Optional[int] = None
    use_cache: bool = False

    # === JEPA ===
    proj_hidden_size: int = 2048
    predictor_num_hidden_layers: int = 4
    predictor_hidden_size: int = 512

    n_class: int = 1


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
    ratio: float = 0.5
    
    # === 混合精度 ===
    dtype: str = "bfloat16"
    
    # === 数据 ===
    resampling_rate: int = 256
    dataset_list: List[str] = field(default_factory=lambda: ['TUAB'])
    signal_transform: Optional[str] = None
    
    # === 日志 ===
    project_name: str = "JEPA_EEG_TUNE"
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

def make_mask(batch_size, time_steps, channel_num, mask_ratio):
    # 1. 在时间维随机选 mask_num 个步，生成时间掩码 shape: (t_num,)
    selected_t_idx = torch.randperm(time_steps)[:int(time_steps * mask_ratio)]
    t_mask = torch.zeros(time_steps, dtype=torch.bool)
    t_mask[selected_t_idx] = True

    # 2. 扩展到通道维：每个被选中的时间步，对应 c_num 个通道都为 True
    #    再展平成序列维度 shape: (time_steps * c_num,)
    seq_mask = t_mask.unsqueeze(-1).expand(-1, channel_num).flatten()

    

    # # 1. 批量生成时间维掩码，每个样本严格选 mask_num 个时间步
    # rand_scores = torch.rand(b, t_num, device=x.device)
    # selected_idx = torch.argsort(rand_scores, dim=-1)[:, :mask_num]
    # t_mask_batch = torch.zeros(b, t_num, dtype=torch.bool, device=x.device)
    # t_mask_batch.scatter_(1, selected_idx, True)  # shape: (b, t_num)

    # # 2. 扩展到通道维并展平 shape: (b, t_num * c_num)
    # seq_mask_batch = t_mask_batch.unsqueeze(-1).expand(-1, -1, c_num).flatten(1, 2)
    
    return seq_mask

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
        A = torch.randn(proj.size(-1), 128, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()



class EEGDataset(Dataset):
    def __init__(self, windows_dataset):
        self.ds = windows_dataset
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        X, y, _ = self.ds[idx]
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.tensor(y, dtype=torch.long)
        return X, y


def train_one_epoch(
        model, time_encoder, freq_encoder, train_loader, positions,
        optimizer, scheduler, scaler, sigreg,
        max_grad_norm, epoch, config: TrainingConfig,
):
    model.train()
    freq_encoder.train()

    pbar = tqdm.tqdm(train_loader, total=len(train_loader))
    
    positions_origial = positions.to('cuda', non_blocking=True)

    # B,N = 64, 2
    # inputs_embeds = torch.randn(64, 1024, 128, device='cuda')
    # positions = torch.randn(4, 64, 1024, device='cuda')
    for x in pbar:
        with autocast('cuda', dtype=torch.bfloat16):
            eeg, y = x
            eeg = eeg.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)

            b = eeg.shape[0]
            positions_expand = positions_origial.expand(b, -1, -1)  # (batch_size, channels, 3)
            spec = torch.stft(
                eeg.flatten(0, 1),
                n_fft=127,
                hop_length=32,
                win_length=127,
                window=torch.hann_window(127, device='cuda'),
                center=True,
                onesided=True,
                return_complex=True
            )
            spec = spec.reshape(b, -1, *spec.shape[1:])
            magnitude_log = 20 * torch.log10(spec.abs() + 1e-10)
            spec_embedding = freq_encoder(magnitude_log)
            
            positions_expand = add_time(positions_expand, spec_embedding.shape[2])
            positions_expand = positions_expand.permute(3, 0, 2, 1).flatten(2, 3)   # (b,c,t,n) → (n,b,t*c)
            # positions_expand = positions_expand.permute(3, 0, 1, 2).flatten(2, 3)   # (b,c,t,n) → (n,b,c*t)

            b,c,t,_ = spec_embedding.shape
            spec_embedding = spec_embedding.permute(0, 2, 1, 3).flatten(1, 2)              # (b,c,t,d) → (b,t*c,d)  time block          
            # spec_embedding = spec_embedding.flatten(1, 2)              # (b,c,t,d) → (b,c*t,d)  channel block          
            mask = make_mask(
                batch_size=b,
                time_steps=t,
                channel_num=c,
                mask_ratio=config.ratio
            )
            
            
            logit, cls, cls_proj, hidden_proj, predictor = model(
                inputs_embeds=spec_embedding,
                position_ids=positions_expand,
                mask=mask,
            )

            with torch.no_grad():
                # 计算奇异值（返回前 min(B,D) 个）
                U, S, V = torch.svd(cls)  # S 形状 (min(B,D),)
                # 归一化为概率分布
                p = S / (S.sum() + 1e-6)
                # 计算熵
                entropy = - (p * torch.log(p + 1e-6)).sum()
                eff_rank = torch.exp(entropy)

                # 记录到 wandb（或打印）
                wandb.log({
                    "eff_rank": eff_rank.item(),
                })


            jepa_loss = F.mse_loss(hidden_proj[:, mask, :], predictor[:, mask, :])
            sigreg_loss = sigreg(cls_proj)
            logit_loss = F.cross_entropy(logit, y)

            lejepa_loss = sigreg_loss * config.lmab + jepa_loss * (1 - config.lmab)
            loss = lejepa_loss + logit_loss

        optimizer.zero_grad() # set_none
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
            "train/jepa_loss": jepa_loss.item(),
            "train/logit": logit_loss.item(),
            "train/loss": loss.item(),

            "train/lr": optimizer.param_groups[0]['lr'],
        })

    return lejepa_loss, sigreg_loss, jepa_loss


def validate(model, time_encoder, freq_encoder, sigreg, test_loader, positions, epoch, config: TrainingConfig):
    model.eval()
    time_encoder.eval()
    freq_encoder.eval()

    pbar = tqdm.tqdm(test_loader, total=len(test_loader))


    with torch.inference_mode():
        positions_origial = positions.to('cuda', non_blocking=True)

        # B,N = 64, 2
        # inputs_embeds = torch.randn(64, 1024, 128, device='cuda')
        # positions = torch.randn(4, 64, 1024, device='cuda')
        correct = 0
        total = 0
        all_feats = []
        all_labels = []
        for x in pbar:
            with autocast('cuda', dtype=torch.bfloat16):
                eeg, y = x
                eeg = eeg.to('cuda', non_blocking=True)
                y = y.to('cuda', non_blocking=True)

                b = eeg.shape[0]
                positions_expand = positions_origial.expand(b, -1, -1)  # (batch_size, channels, 3)
                spec = torch.stft(
                    eeg.flatten(0, 1),
                    n_fft=127,
                    hop_length=32,
                    win_length=127,
                    window=torch.hann_window(127, device='cuda'),
                    center=True,
                    onesided=True,
                    return_complex=True
                )
                spec = spec.reshape(b, -1, *spec.shape[1:])
                magnitude_log = 20 * torch.log10(spec.abs() + 1e-10)
                spec_embedding = freq_encoder(magnitude_log)
                
                positions_expand = add_time(positions_expand, spec_embedding.shape[2])
                positions_expand = positions_expand.permute(3, 0, 2, 1).flatten(2, 3)   # (b,c,t,n) → (n,b,t*c)
                # positions_expand = positions_expand.permute(3, 0, 1, 2).flatten(2, 3)   # (b,c,t,n) → (n,b,c*t)

                b,c,t,_ = spec_embedding.shape
                spec_embedding = spec_embedding.permute(0, 2, 1, 3).flatten(1, 2)              # (b,c,t,d) → (b,t*c,d)  time block          
                # spec_embedding = spec_embedding.flatten(1, 2)              # (b,c,t,d) → (b,c*t,d)  channel block          
                mask = make_mask(
                    batch_size=b,
                    time_steps=t,
                    channel_num=c,
                    mask_ratio=config.ratio
                )
                
                
                logit, cls, cls_proj, hidden_proj, predictor = model(
                    inputs_embeds=spec_embedding,
                    position_ids=positions_expand,
                    mask=mask,
                )

                all_feats.append(cls.cpu())
                all_labels.append(y.cpu())

                correct += (logit.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        # print("acc", acc)
        wandb.log({"test/acc": acc, "test/epoch": epoch})

        features = torch.cat(all_feats, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()   

        import umap
        from sklearn.preprocessing import StandardScaler

        # 【可选但推荐】特征标准化，UMAP对特征尺度敏感，标准化后效果更稳定
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 初始化UMAP降维器
        reducer = umap.UMAP(
            n_neighbors=15,      # 局部/全局平衡：越小越侧重局部结构，越大越侧重全局
            min_dist=0.1,        # 簇的紧凑度：越小簇越聚拢，越大越分散
            n_components=2,      # 降到2维
            random_state=42,     # 固定随机种子，保证结果可复现
            n_jobs=-1            # 用满CPU核心，加速计算
        )

        # 执行降维
        embedding_2d = reducer.fit_transform(features_scaled)      
        
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8), dpi=100)

        # 画散点图，按标签着色
        sns.scatterplot(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            hue=labels,           # 按类别上色
            palette="tab20",      # 配色方案，类别多选tab20
            s=15,                 # 点的大小
            alpha=0.7,            # 透明度，避免重叠看不清
            edgecolor=None        # 去掉点的边框，更清爽
        )

        # 图表美化
        plt.title("UMAP Visualization of EEG Features", fontsize=14, pad=15)
        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")  # 图例放外侧
        plt.tight_layout()  # 自动调整布局，防止标签被截断

        # 保存图片 + 显示
        plt.savefig("umap_result.png", dpi=300, bbox_inches="tight")
        plt.show()
        # wandb.log({"test/acc": correct / len(test_loader), "test/epoch": epoch})


def train(model, time_encoder, freq_encoder, sigreg, train_loader, val_loader, positions, config: TrainingConfig):
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

    backbone_params = [
    p for name, p in model.named_parameters() 
    if not name.startswith('classifier')
]
    groups = [
        {'params': backbone_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': time_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': freq_encoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-7},
    ]
    optimizer = torch.optim.AdamW(groups, betas=config.betas, eps=config.eps)

    # 学习率调度
    warmup_steps = config.warmup_epochs * len(train_loader)
    total_steps = config.epochs * len(train_loader)

    s1 = LinearLR(optimizer, start_factor=config.start_factor, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=config.lr * config.start_factor)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    # 混合精度
    scaler = GradScaler() if config.dtype == "float16" else None

    for epoch in range(config.epochs):
        train_one_epoch(
            model=model,
            time_encoder=time_encoder,
            freq_encoder=freq_encoder,
            train_loader=train_loader,
            positions=positions,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sigreg=sigreg,
            max_grad_norm=config.max_grad_norm,
            epoch=epoch,
            config=config,
        )

        validate(model, time_encoder, freq_encoder, sigreg, val_loader, positions, epoch, config)

        # checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch}_lejepa_{lejepa_loss:.4f}.pth")
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f"Checkpoint saved to {checkpoint_path}")

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
    context_parallel_size=1,
)
    model_parallel_cuda_manual_seed(123)

def main():
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    setup_parallel_state()
    
    
    # 模型配置
    model_config = Qwen3NextConfig(
        input_size=256,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=1536,
        max_position_embeddings=32768,
        mrope_section=[16, 16, 16, 16],
        mlp_hidden_size=1536,
        proj_size=64,
        n_class=3,
        use_cache=False,
        proj_hidden_size=2048,
        predictor_num_hidden_layers=4,

    )
    
    # 训练配置
    train_config = TrainingConfig(
        batch_size=32,
        lr=1e-4,
        weight_decay=0.05,
        warmup_epochs=10,
        epochs=100,
        lmab=0.02,
        dtype="bfloat16",
    )
    
    # 实例化模型
    model = Qwen3NextModelJepaShin2017a(config=model_config)
    time_encoder = MultiKernelConvEncoder(out_dim=model_config.input_size)
    freq_encoder = SpectrogramEncoder(out_dim=model_config.input_size, freq_bins=64, time_steps=32)
    sigreg = SIGReg()

    checkpoint_path = "/home/xuyuanjun/leeg/checkpoints/checkpoint_epoch_2_lejepa_0.0155.pth"
    state_dict = torch.load(checkpoint_path)

    # model.load_state_dict(state_dict, strict=False)  # 或 strict=False

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count}")

    # 加载数据

    
    print("Loading data...")
    # 加载两个已保存的数据集
    windows_dataset = load_concat_dataset(path='/home/xuyuanjun/leeg/data/neural/moabb/Alex/EEG_process', preload=True)

    # 验证
    print(f"数据集窗口数: {len(windows_dataset)}")
    first_epochs = windows_dataset.datasets[0].windows
    info = first_epochs.info
    ch_names = np.array([info['chs'][i]['ch_name'] for i in range(len(info['chs']))])
    print(ch_names)

    full_dataset = EEGDataset(windows_dataset)
    # 1. 定义划分比例（例如 80% 训练，20% 验证）
    train_ratio = 0.8
    val_ratio = 0.2
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    val_len = total_len - train_len

    # 2. 使用 random_split 划分（可指定随机种子以保证可复现）
    torch.manual_seed(42)  # 设置随机种子
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,          # 训练集打乱
        pin_memory=True,
        num_workers=16,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,  # 或使用稍大的验证 batch
        shuffle=False,         # 验证集不打乱
        pin_memory=True,
        num_workers=16,
    )
    

    print(f"训练集样本数: {train_len}")
    print(f"验证集样本数: {val_len}")


    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)

    electrode_names = ch_names  # List of electrode names corresponding to the channels in eeg_data

    positions = pos_bank(electrode_names) * 100 # Get positions (channels, 3)

    ## Expand the positions vector to match the batch size 

    # 训练
    train(model, time_encoder, freq_encoder, sigreg, train_loader, val_loader, positions, train_config)


if __name__ == "__main__":
    main()
