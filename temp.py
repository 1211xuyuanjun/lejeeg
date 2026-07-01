from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.nn.functional as F
import torch.distributed as dist
from qwen3 import Qwen3NextModelJepa, Qwen3NextModelJepaCore
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_transformer_block_with_experimental_attention_variant_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

config = MLATransformerConfig(
    # MLA
    multi_latent_attention=True,
    q_lora_rank=256, # hidden_size // 4
    kv_lora_rank=256,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
    normalization="RMSNorm",
    rope_type='rope',
    rotary_base= 10000,
    rotary_percent= 1.0,
    rotary_scaling_factor = 1.0,
    mscale = 1.0,
    mscale_all_dim = 1.0,


    # DSA
    experimental_attention_variant = "dsa",
    dsa_indexer_head_dim = 128,
    dsa_indexer_n_heads = 32,
    dsa_indexer_topk = 512,
    dsa_indexer_loss_coeff = 0.001,
    dsa_indexer_use_sparse_loss = True,

    # Transformer
    num_layers=1,
    hidden_size=1024,
    num_attention_heads=4,
    ffn_hidden_size=1536,
    activation_func=F.silu,
    gated_linear_unit=True,
    qk_layernorm=True,
    bf16=True,
    add_bias_linear=False,

)

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

setup_distributed()
rank = dist.get_rank()
world_size = dist.get_world_size()
setup_parallel_state()

spec = get_transformer_block_with_experimental_attention_variant_spec(config=config)
block = TransformerBlock(config=config, spec=spec).cuda()


block = block.to(dtype=torch.bfloat16)

# 3. 准备输入数据 [Seq_Len, Batch_Size, Hidden_Size]
seq_len = 512
batch_size = 2
hidden_size = config.hidden_size
hidden_states = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16, device='cuda')
# 注意：这里不使用 attention_mask，block 会自己根据 attn_mask_type 生成（默认 padding mask）

# 4. 前向传播（如需 AMP，可使用 autocast，但模型已 bf16 则可省略）
output = block(hidden_states, attention_mask=None)

# 5. 计算损失并反向传播
loss = output.mean()
loss.backward()

# 6. 检查梯度
total_norm = 0.0
for p in block.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
if rank == 0:
    print(f"Forward output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    print(f"Gradient norm: {total_norm}")

# 7. 清理
cleanup_distributed()
# @dataclass
# class Qwen3NextConfig(TransformerConfig):
#     # === 模型架构 ===
#     input_size: int = 128
#     mlp_hidden_size: int = 768
#     proj_size: int = 128
    
#     # === 位置编码 ===
#     max_position_embeddings: int = 32768
#     rope_theta: float = 10000.0
#     mrope_section: List[int] = field(default_factory=lambda: [32, 32])

# model_config = Qwen3NextConfig(
#         input_size=128,
#         proj_size=128,
#         mlp_hidden_size=1536,
#         num_layers=24,
#         hidden_size=512,
#         num_attention_heads=4,
#         num_query_groups=2,
#         kv_channels=128,
#         ffn_hidden_size=1536,
#         normalization="RMSNorm",
#         activation_func=F.silu,
#         qk_layernorm = True,
#         layernorm_zero_centered_gamma=True,
#         attention_output_gate=True,
#         gated_linear_unit=True,
#         add_bias_linear=False,
#         add_qkv_bias=False,
#         layernorm_epsilon=1e-6,
#         bf16=True,
#         hidden_dropout=0.0,
#         attention_dropout=0.0,
#         mrope_section=[16,16,16,16]
#         # recompute_granularity="full",
#         # recompute_method="uniform",
#         # recompute_num_layers=1,
#     )
# def setup_distributed():
#     """初始化分布式训练环境"""
#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(dist.get_rank())
#     dist.barrier()

# def cleanup_distributed():
#     """清理分布式训练环境"""
#     dist.destroy_process_group()

# def setup_parallel_state():
#     """Setup Megatron parallel state."""
#     parallel_state.initialize_model_parallel(
#     tensor_model_parallel_size=1,  # 根据您的GPU数量调整
#     pipeline_model_parallel_size=1, # 暂时不用流水线并行
#     expert_model_parallel_size=1,
#     context_parallel_size=1,
# )
#     model_parallel_cuda_manual_seed(123)

# setup_distributed()
# rank = dist.get_rank()
# world_size = dist.get_world_size()
# setup_parallel_state()


# # model = Qwen3NextModelJepaCore(config=model_config, spec=get_gpt_layer_with_transformer_engine_spec())

# model = TransformerBlock(
#         config=model_config,
#         spec=get_gpt_layer_with_transformer_engine_spec()
#     ).to('cuda')
# # @dataclass
# # class Qwen3NextConfig:
# #     # === 模型架构 ===
# #     input_size: int = 128
# #     hidden_size: int = 512
# #     intermediate_size: int = 1280
# #     num_hidden_layers: int = 6
# #     num_attention_heads: int = 4
# #     num_key_value_heads: int = 2
# #     head_dim: int = 128
# #     mlp_hidden_size: int = 768
# #     proj_size: int = 128
    
# #     # === 激活与归一化 ===
# #     hidden_act: str = "silu"
# #     rms_norm_eps: float = 1e-6
    
# #     # === 位置编码 ===
# #     max_position_embeddings: int = 32768
# #     rope_theta: float = 10000.0
# #     rope_type: str = "default"
# #     mrope_section: List[int] = field(default_factory=lambda: [32, 32])
    
# #     # === 注意力 ===
# #     attention_bias: bool = False
# #     attention_dropout: float = 0.0
# #     sliding_window: Optional[int] = None
    
# #     # === 初始化 ===
# #     initializer_range: float = 0.02
    
# #     # === 其他 ===
# #     pad_token_id: Optional[int] = None
# #     use_cache: bool = False

# # # 模型配置
# # model_config = Qwen3NextConfig(
# #     input_size=128,
# #     hidden_size=1024,
# #     num_hidden_layers=12,
# #     num_attention_heads=8,
# #     num_key_value_heads=2,
# #     head_dim=128,
# #     intermediate_size=3072,
# #     max_position_embeddings=32768,
# #     mrope_section=[16, 16, 16, 16],
# #     mlp_hidden_size=3072,
# #     proj_size=128,
# #     use_cache=False,
# # )

    
# # # 在 CPU 上实例化模型，FSDP 会负责将参数移到 GPU 并分片
# # model = Qwen3NextModelJepa(model_config).to('cuda')

# model.train()
# inputs_embeds = torch.randn(64, 64, 512, device='cuda')
# positions = torch.randn(64, 64, 1, 128, device='cuda')
# # inputs_embeds = torch.randn(1024, 64, 128, device='cuda')
# # positions = torch.randn(4, 1024, 64,  device='cuda')

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# for i in range(100):

#     optimizer.zero_grad()
#     output = model(
#         hidden_states=inputs_embeds,
#         attention_mask=None,
#         rotary_pos_emb=positions

#     )
#     loss = F.mse_loss(inputs_embeds, output)

#     # output, proj = model(
#     #     inputs_embeds,
#     #     positions
#     # )

#     # loss = F.mse_loss(inputs_embeds, proj)    
#     # 反向传播 (原生)
#     # MCore 的反向传播会自动在后台处理梯度的 All-Reduce 同步
#     loss.backward()
    
#     # 梯度裁剪 (原生，可选)
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
#     # 优化器更新 (原生)
#     optimizer.step()
        
# cleanup_distributed()



# import timm
# from torchinfo import summary

# model = timm.create_model(
#             "vit_small_patch8_224",
#             pretrained=False,
#             num_classes=512,
#             drop_path_rate=0.1,
#             img_size=128,
#         )

# summary(model, input_size=(256, 3, 128, 128))

