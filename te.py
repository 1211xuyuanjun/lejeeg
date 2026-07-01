import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
import torch.distributed as dist
from leeg_layer_spec import get_gpt_layer_with_transformer_engine_spec
from torch.amp import autocast, GradScaler

def setup_distributed():
    """初始化分布式训练环境"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    dist.barrier()

setup_distributed()
rank = dist.get_rank()
world_size = dist.get_world_size()


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1
)

# 2. 配置 MCore Transformer 参数 (类似 Llama/GPT 的配置)
config = TransformerConfig(
    num_layers=12,
    hidden_size=512,
    num_attention_heads=4,
    num_query_groups=2,
    kv_channels=128,
    ffn_hidden_size=1536,
    normalization="RMSNorm",
    qk_layernorm=True,
    layernorm_zero_centered_gamma=True,
    attention_output_gate=True,
    activation_func=F.silu,
    gated_linear_unit=True,
    add_bias_linear=False,
    add_qkv_bias=False,
    layernorm_epsilon=1e-6,
    bf16=True,
    hidden_dropout=0.0,
    attention_dropout=0.0,
)

# 3. 实例化 MCore 的 Transformer 层（高度优化过的算子集合）
encoder = TransformerBlock(
    config=config,
    spec=get_gpt_layer_with_transformer_engine_spec()
).to('cuda')
encoder.train()

# from train_text import Qwen3NextConfig
from qwen3 import Qwen3NextModelJepa
from dataclasses import dataclass
@dataclass
class Qwen3NextConfig:
    input_size=128
    hidden_size=512
    num_hidden_layers=12
    num_attention_heads=4
    num_key_value_heads=2
    head_dim=128
    intermediate_size=1536
    max_position_embeddings=32768
    mrope_section=[16, 16, 16, 16]
    mlp_hidden_size=1536
    proj_size=128
    use_cache=False
    attention_dropout=0.0
    attention_bias=False
    rms_norm_eps=1e-6


model_config = Qwen3NextConfig()
model = Qwen3NextModelJepa(model_config).to('cuda')
model.train()


# 4. 标准的 PyTorch 训练流程
optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

# 模拟数据: (batch=2, seq_len=512, hidden=512)
hidden_states = torch.randn(64, 1024, 512, dtype=torch.bfloat16).cuda()
attention_mask = None # 简单演示，实际可能需要传入因果掩码等
rotary_pos_emb = torch.randn(64, 1024, 1, 128, dtype=torch.bfloat16).cuda()
# hidden_states = torch.randn(64, 1024, 128, dtype=torch.bfloat16).cuda()

# position = torch.randn(4, 64, 1024, dtype=torch.bfloat16).cuda()


import time





# 计算 MFU (需要填入你的真实参数)
# MFU = 实际FLOPS / GPU峰值FLOPS
# 实际FLOPS = 6 * B * S * P * 3 / step_time  (P = 模型参数量, 3=fwd+bwd系数)
# 简化: 用上面的表格对照你的 step_time

# 预热
for step in range(5):
    optimizer.zero_grad()
    
    # 原生前向传播
    # 注意：MCore的TransformerLayer返回 (hidden_states, context, ...)
    with autocast('cuda', dtype=torch.bfloat16):
        output = encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        # output, _ = model(
        #     hidden_states,
        #     position
        # )
        
        # 模拟损失计算
        loss = output.mean()
    loss.backward() # 原生 PyTorch 反向传播，MCore 的算子会自动处理梯度
    
    optimizer.step()
    print(f"Step {step} | Loss: {loss.item()}")
torch.cuda.synchronize()

# 计时
N = 20
start = time.time()
for step in range(N):
    optimizer.zero_grad()
    
    # 原生前向传播
    # 注意：MCore的TransformerLayer返回 (hidden_states, context, ...)
    with autocast('cuda', dtype=torch.bfloat16):
        output = encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        # output, _ = model(
        #     hidden_states,
        #     position
        # )
        
        # 模拟损失计算
        loss = output.mean()
    loss.backward() # 原生 PyTorch 反向传播，MCore 的算子会自动处理梯度
    
    optimizer.step()
    print(f"Step {step} | Loss: {loss.item()}")
torch.cuda.synchronize()

step_time = (time.time() - start) / N
print(f"Step time: {step_time*1000:.1f} ms")

