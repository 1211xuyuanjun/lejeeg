import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import transformer_engine.pytorch as te
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config = None,
        device = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor # fix linting for 'register_buffer'

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        rope_init_fn = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.mrope_section = config.mrope_section

    @staticmethod
    def compute_default_rope_parameters(
        config = None,
        device = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = 1e4
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device, dtype=torch.float) / dim)
        )

        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(4, position_ids.shape[1], -1, 1).to(x.device)
        ) # (t,x,y,z)
        position_ids_expanded = position_ids[:, :, None, :].float() # shape=(4, batch_size, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    
    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1,2,3), start=1):
            length = mrope_section[dim] * 4
            idx = slice(offset, length, 4)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        return freqs_t

class Qwen3VLRotaryEmbeddingCore(nn.Module):
    inv_freq: torch.Tensor # fix linting for 'register_buffer'

    def __init__(self, config, device=None):
        super().__init__()
        # self.max_seq_len_cached = config.max_position_embeddings
        # self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # self.rope_type = self.config.rope_type
        rope_init_fn = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.mrope_section = config.mrope_section

    @staticmethod
    def compute_default_rope_parameters(
        config = None,
        device = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = 1e4
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device, dtype=torch.float) / dim)
        )

        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(4, position_ids.shape[1], -1, 1).to(x.device)
        ) # (t,x,y,z)
        position_ids_expanded = position_ids[:, :, None, :].float() # shape=(4, batch_size, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            # freqs = freqs.permute(1,0,2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # only emb to core

        return emb.unsqueeze(dim=2)
    
    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1,2,3), start=1):
            length = mrope_section[dim] * 4
            idx = slice(offset, length, 4)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        return freqs_t




def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,      # (B, H_q, S, D)
    key: torch.Tensor,        # (B, H_kv, S, D)
    value: torch.Tensor,      # (B, H_kv, S, D)
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # GQA: repeat_kv 仍然需要（SDPA 不处理 GQA 广播）
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # SDPA 自动处理 causal mask、softmax、dropout、内存优化
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,           # 支持 additive mask
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,    # 无 mask 时自动因果
        scale=scaling,
    )
    
    # SDPA 输出: (B, H, S, D)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, None  # SDPA 不返回 attn_weights（省内存）




class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = sdpa_attention_forward


        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3NextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # attention_interface = eager_attention_forward # or sdpa
        attention_interface = sdpa_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3NextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3NextModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # causal_mask = create_causal_mask(
        #     config=self.config,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        # TODO: random mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = None # or random mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states
    
    def _init_weights(self, model):
        if isinstance(model, nn.Linear):
            nn.init.kaiming_normal_(model.weight, a="fan_in", nonlinearity="silu")
            nn.init.zeros_(model.bias)
        elif isinstance(model, nn.Embedding):
            nn.init.normal_(model.weight, std=self.config.initializer_range)

class Qwen3NextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.config = config

        nn.init.normal_(self.cls_token, std=config.initializer_range)
        nn.init.normal_(self.mask_token, std=config.initializer_range)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        bool_masked_pos: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        return_strategy: str = "mask_token",
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        batch_size, seq_len = inputs_embeds.shape[:2]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        inputs_embeds = inputs_embeds * (1 - w) + mask_token * w

        inputs_embeds = torch.cat((cls_tokens, inputs_embeds), dim=1)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids) # position_ids_length = vocab_size + 1

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = None # or random mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)

        if return_strategy == "cls_token":
            return self.linear(hidden_states[:, 0]) # self.linear(hidden_states)[:, 0]
        
        hidden_states = hidden_states[:, 1:]
        
        if return_strategy == "mask_token":
            return self.linear(hidden_states[bool_masked_pos])
        else:
            return self.linear(hidden_states)
    
    def _init_weights(self, model):
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(model, nn.Linear):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.bias is not None:
                model.bias.data.zero_()
        
        elif isinstance(model, nn.Embedding):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.padding_idx is not None:
                model.weight.data[model.padding_idx].zero_()
            # self.linear.weight.copy_(self.embed_tokens.weight) 
        
        elif isinstance(model, Qwen3RMSNorm):
            model.weight.data.fill_(1.0)  # ✅ RMSNorm weight=1
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.self_attn.o_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.down_proj.weight.data, layer_id + 1)

class Qwen3NextModelJepa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config=config)
        # Initialize weights and apply final processing

        self.input_proj = nn.Linear(config.input_size, config.hidden_size)

        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_hidden_size),
            nn.LayerNorm(config.mlp_hidden_size),
            nn.GELU(),
            nn.Linear(config.mlp_hidden_size, config.mlp_hidden_size),
            nn.LayerNorm(config.mlp_hidden_size),
            nn.GELU(),
            nn.Linear(config.mlp_hidden_size, config.proj_size),
        ) # qwen3_mlp

        self.config = config

        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
    ):

        batch_size, seq_len, input_size = inputs_embeds.shape
        
        hidden_states = self.input_proj(inputs_embeds)  
        position_embeddings = self.rotary_emb(hidden_states, position_ids) # position_ids_length = vocab_size + 1
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        proj = self.proj(hidden_states)
        
        return hidden_states, proj

    
    def _init_weights(self, model):
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(model, nn.Linear):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.bias is not None:
                model.bias.data.zero_()
        
        elif isinstance(model, nn.Embedding):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.padding_idx is not None:
                model.weight.data[model.padding_idx].zero_()
            # self.linear.weight.copy_(self.embed_tokens.weight) 
        
        elif isinstance(model, Qwen3RMSNorm):
            model.weight.data.fill_(1.0)  # ✅ RMSNorm weight=1
        
        elif isinstance(model, nn.BatchNorm1d):
            model.weight.data.fill_(1.0)

    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.self_attn.o_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.down_proj.weight.data, layer_id + 1)




class MCoreProjection(nn.Module):
    """
    使用 Transformer Engine 加速的 Projection Head
    替代原有的 nn.Sequential(Linear, LayerNorm, GELU, ...)
    """
    def __init__(self, hidden_size, mlp_hidden_size, proj_size):
        super().__init__()
        # 使用 TE 的 Linear 层，原生支持 FP8 和高带宽矩阵乘法
        # te.LayerNorm 和 te.RMSNorm 内部进行了算子融合
        self.fc1 = te.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.norm1 = te.RMSNorm(mlp_hidden_size, eps=1e-6) # 对应 provider.normalization = "RMSNorm"
        self.act1 = te.ops.GELU()                              # TE 优化的 GELU 激活函数
        
        self.fc2 = te.Linear(mlp_hidden_size, mlp_hidden_size, bias=False)
        self.norm2 = te.RMSNorm(mlp_hidden_size, eps=1e-6)
        self.act2 = te.ops.GELU()
        
        self.fc3 = te.Linear(mlp_hidden_size, proj_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.fc3(x)
        return x
    
class Qwen3NextModelJepaCore(nn.Module):
    def __init__(self, config, spec):
        super().__init__()
        self.config = config

        self.rotary_emb = Qwen3VLRotaryEmbeddingCore(config=config)
        # Initialize weights and apply final processing

        self.input_proj = te.Linear(config.input_size, config.hidden_size, bias=False)

        self.block = TransformerBlock(
            config=config,
            spec=spec
        )

        self.proj = MCoreProjection(
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.mlp_hidden_size,
            proj_size=config.proj_size
        )

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):

        hidden_states = self.input_proj(inputs_embeds)  
        position_embeddings = self.rotary_emb(hidden_states, position_ids) # position_ids_length = vocab_size + 1
        # hidden_states = hidden_states.permute(1,0,2)
        hidden_states = self.block(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=position_embeddings,
        )

        # hidden_states = hidden_states.permute(1,0,2)


        proj = self.proj(hidden_states)
        
        return hidden_states, proj




class Qwen3NextModelJepaShin2017aPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.predictor_num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.predictor_hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config=config)
        # Initialize weights and apply final processing

        self.input_proj = nn.Linear(config.proj_size, config.predictor_hidden_size)
        self.output_proj = nn.Linear(config.predictor_hidden_size, config.proj_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.predictor_hidden_size, dtype=torch.bfloat16))
        self.all_states = torch.zeros(1, 1, config.predictor_hidden_size, dtype=torch.bfloat16)

        self.config = config

        nn.init.normal_(self.mask_token, std=config.initializer_range)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        mask: torch.FloatTensor | None = None,
    ):

        _, batch_size, seq_len = position_ids.shape

        hidden_states = self.input_proj(inputs_embeds)
        # 1. 初始化和原始形状一致的空张量
        all_states = torch.empty(batch_size, seq_len, self.config.predictor_hidden_size, 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask_token = self.mask_token.expand(batch_size, int(mask.sum()), -1)

        # 2. 按掩码位置回填两部分
        all_states[:, ~mask, :] = hidden_states  # 回填 context 区域
        all_states[:, mask, :] = mask_token      # 回填 mask 区域
        position_embeddings = self.rotary_emb(all_states, position_ids) # position_ids_length = vocab_size + 1

        for i, decoder_layer in enumerate(self.layers[: self.config.predictor_hidden_size]):

            all_states = decoder_layer(
                all_states,
                position_embeddings=position_embeddings,
            )

        all_states = self.norm(all_states)
        out_states = self.output_proj(all_states)
        
        return out_states

    
    def _init_weights(self, model):
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(model, nn.Linear):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.bias is not None:
                model.bias.data.zero_()
        
        elif isinstance(model, nn.Embedding):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.padding_idx is not None:
                model.weight.data[model.padding_idx].zero_()
            # self.linear.weight.copy_(self.embed_tokens.weight) 
        
        elif isinstance(model, Qwen3RMSNorm):
            model.weight.data.fill_(1.0)  # ✅ RMSNorm weight=1
        
        elif isinstance(model, nn.BatchNorm1d):
            model.weight.data.fill_(1.0)

    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.self_attn.o_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.down_proj.weight.data, layer_id + 1)



class Qwen3NextModelJepaShin2017a(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config=config)
        # Initialize weights and apply final processing

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)


        self.proj = MLP(
            in_channels=config.hidden_size,
            norm_layer=nn.LayerNorm,
            hidden_channels=[config.proj_hidden_size, config.proj_hidden_size, config.proj_size],
            # activation_layer
        )

        self.predictor = Qwen3NextModelJepaShin2017aPredictor(config=config)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.n_class)
        )

        self.config = config

        nn.init.normal_(self.cls_token, std=config.initializer_range)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        mask: list[bool] | None = False,
    ):

        # ------ ALL
        batch_size, seq_len, input_size = inputs_embeds.shape

        hidden_states = self.input_proj(inputs_embeds)  
        position_embeddings = self.rotary_emb(hidden_states, position_ids) # position_ids_length = vocab_size + 1

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        # cls_proj = self.cls_proj(hidden_states[:, 0])#[:, 0] # detach   # self.linear(hidden_states)[:, 0]

        cls = hidden_states.mean(dim=1)
        
        
        cls_proj = self.proj(cls)

        hidden_proj = self.proj(hidden_states.detach())

        # ------ CONTEXT
        context = hidden_states[:, ~mask, :,]
        position_context = position_ids[:, :, ~mask]
        position_embeddings = self.rotary_emb(context, position_context) # position_ids_length = vocab_size + 1

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            context = decoder_layer(
                context,
                position_embeddings=position_embeddings,
            )

        context = self.norm(context)
        context_proj = self.proj(context)
        
        # ------ PREDICTOR
        
        predictor = self.predictor(
            inputs_embeds=context_proj,
            position_ids=position_ids,
            mask=mask,
        )

        # mask_predictor = predictor[mask, :,]

        logit = self.classifier(cls.detach())
        
        return logit, cls,  cls_proj, hidden_proj, predictor

    
    def _init_weights(self, model):
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(model, nn.Linear):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.bias is not None:
                model.bias.data.zero_()
        
        elif isinstance(model, nn.Embedding):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.padding_idx is not None:
                model.weight.data[model.padding_idx].zero_()
            # self.linear.weight.copy_(self.embed_tokens.weight) 
        
        elif isinstance(model, Qwen3RMSNorm):
            model.weight.data.fill_(1.0)  # ✅ RMSNorm weight=1
        
        elif isinstance(model, nn.BatchNorm1d):
            model.weight.data.fill_(1.0)

    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.self_attn.o_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.down_proj.weight.data, layer_id + 1)

class Qwen3NextModelJepaAlex(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config=config)
        # Initialize weights and apply final processing

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)

        self.proj = MLP(
            in_channels=config.hidden_size,
            norm_layer=nn.LayerNorm,
            hidden_channels=[config.proj_hidden_size, config.proj_hidden_size, config.proj_size],
            # activation_layer
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.n_class)
        )

        self.config = config

        nn.init.normal_(self.cls_token, std=config.initializer_range)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        mask: list[bool] | None = False,
    ):

        # ------ ALL
        batch_size, seq_len, input_size = inputs_embeds.shape

        hidden_states = self.input_proj(inputs_embeds)  
        position_embeddings = self.rotary_emb(hidden_states, position_ids) # position_ids_length = vocab_size + 1

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        # cls_proj = self.cls_proj(hidden_states[:, 0])#[:, 0] # detach   # self.linear(hidden_states)[:, 0]

        cls = hidden_states.mean(dim=1)
        
        cls_proj = self.proj(cls)


        logit = self.classifier(cls.detach())
        
        return logit, cls, cls_proj

    
    def _init_weights(self, model):
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(model, nn.Linear):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.bias is not None:
                model.bias.data.zero_()
        
        elif isinstance(model, nn.Embedding):
            model.weight.data.normal_(mean=0.0, std=std)
            if model.padding_idx is not None:
                model.weight.data[model.padding_idx].zero_()
            # self.linear.weight.copy_(self.embed_tokens.weight) 
        
        elif isinstance(model, Qwen3RMSNorm):
            model.weight.data.fill_(1.0)  # ✅ RMSNorm weight=1
        
        elif isinstance(model, nn.BatchNorm1d):
            model.weight.data.fill_(1.0)

    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.self_attn.o_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.down_proj.weight.data, layer_id + 1)




