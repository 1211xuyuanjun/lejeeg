import warnings
from functools import partial
from typing import Optional, Union

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import (
    BackendSpecProvider,
    InferenceSpecProvider,
    LocalSpecProvider,
)
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec_for_backend,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    MlpBuilder,
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.typed_torch import copy_signature, not_none
from megatron.core.utils import is_te_min_version

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEFusedMLP,
        TEFusedMLPWithGroupedLinear,
        TENorm,
    )
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
else:
    TEFusedMLPWithGroupedLinear, TEFusedMLP, TENorm, TESpecProvider = None, None, None, None

try:
    from megatron.core.extensions.kitchen import HAVE_KITCHEN, KitchenSpecProvider

except ImportError:
    HAVE_KITCHEN = False

try:
    import apex  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def get_gpt_layer_with_transformer_engine_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    mla_down_proj_fusion: bool = False,
    use_grouped_gemm_for_dense_mlp: bool = False,
) -> TransformerLayerSubmodules:
    """Use these submodules to use lower-level Transformer Engine modules (required for fp8
    training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MLA. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.
        mla_down_proj_fusion (bool, optional): Enable fused q/kv down-projection and fused input
                                               layernorm when backend supports. Otherwise fall back
                                               to the unfused MLA.

    Returns:
        TransformerLayerSubmodules: TE modules to construct a TransformerLayer

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_kitchen:
        assert HAVE_KITCHEN
        backend: BackendSpecProvider = KitchenSpecProvider(
            fallback=TESpecProvider(),
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )
        if use_te_op_fuser:
            raise AssertionError("use_te_op_fuser not compatible with using kitchen in mlp.")
        if use_te_activation_func:
            raise AssertionError("use_te_activation_func not compatible with using kitchen.")
    else:
        backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
        use_grouped_gemm_for_dense_mlp=use_grouped_gemm_for_dense_mlp,
    )

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )

        if mla_down_proj_fusion:
            fuse_input_layernorm = backend.column_parallel_layer_norm_linear() is not None
            input_layernorm = IdentityOp if fuse_input_layernorm else backend.layer_norm()
            down_proj_linear = (
                backend.column_parallel_layer_norm_linear()
                if fuse_input_layernorm
                else backend.linear()
            )
            return TransformerLayerSubmodules(
                input_layernorm=input_layernorm,
                self_attention=ModuleSpec(
                    module=FusedMLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_qkv_down_proj=down_proj_linear,
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map=(
                    {
                        "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
                        "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
                        "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
                    }
                    if fuse_input_layernorm
                    else {}
                ),
            )
        return TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(has_residual=True),
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=backend.column_parallel_linear(),
                    linear_q_down_proj=backend.linear(),
                    linear_q_up_proj=linear_q_up_proj,
                    linear_kv_down_proj=backend.linear(),
                    linear_kv_up_proj=linear_kv_up_proj,
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm(has_residual=True) if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        )
    else:
        qk_norm = backend.layer_norm(for_qk=True)
        return TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_layer_norm_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm(has_residual=True) if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
            },
        )


@copy_signature(get_gpt_layer_with_transformer_engine_submodules)
def get_gpt_layer_with_transformer_engine_spec(*args, **kwargs) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training)."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=get_gpt_layer_with_transformer_engine_submodules(*args, **kwargs),
    )

def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    use_te_op_fuser: Optional[bool] = False,
) -> MlpBuilder:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )
    if use_te_op_fuser:
        if not is_te_min_version("1.13.0"):
            raise ValueError(
                "Transformer Engine operation-based API requires Transformer Engine 1.13+"
            )
        if num_experts is not None:
            raise ValueError(
                "Transformer Engine operation-based API does not support mixture-of-experts"
            )

    return get_mlp_module_spec_for_backend(
        backend=TESpecProvider() if use_te else LocalSpecProvider(),
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
    )


def get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_te_activation_func: bool = False,
    use_grouped_gemm_for_dense_mlp: bool = False,
) -> MlpBuilder:
    """Helper function to get module spec for MLP/MoE"""

    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func() if use_te_activation_func else None

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        if use_grouped_gemm_for_dense_mlp and use_te_op_fuser:
            module = not_none(TEFusedMLPWithGroupedLinear).as_mlp_submodule
        elif use_te_op_fuser:
            module = not_none(TEFusedMLP).as_mlp_submodule
        else:
            module = MLP.as_mlp_submodule
        if backend.fuse_layernorm_and_linear():
            linear_fc1 = backend.column_parallel_layer_norm_linear()
            assert linear_fc1 is not None
        else:
            linear_fc1 = backend.column_parallel_linear()
        return partial(
            module,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec_for_backend(
            backend=backend,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            use_te_activation_func=use_te_activation_func,
        )
