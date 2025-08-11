from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel

from agentgrid.models.gpt_oss.model import WrappedGptOssBlock
from agentgrid.models.qwen2.block import WrappedQwen2Block
from agentgrid.models.qwen3_moe.block import WrappedQwen3MoeBlock
from agentgrid.models.qwen3.block import WrappedQwen3Block
from agentgrid.models.nemotron import WrappedNemotronBlock
from agentgrid.utils.convert_block import QuantType
from agentgrid.utils.misc import get_size_in_bytes

from hivemind import get_logger

logger = get_logger(__name__)


def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it using BloomConfig. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    if config.torch_dtype not in ("auto", None, torch.float32):
        # If config specifies float32, we override it to the default dtype below
        return config.torch_dtype
    return torch.bfloat16


def get_block_size(
    config: PretrainedConfig,
    location: str,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    quant_type: QuantType = QuantType.NONE,
    eps: float = 0.01,  # eps accounts for ~1% of metainfo for tensor descriptions, quantization tables, etc.
    block_index: int = 0,
    # New parameter to prevent double-counting experts in shared-expert models
    exclude_experts: bool = False,
) -> int:
    """
    Calculates the memory or disk size of a single transformer block.

    Args:
        config: The model's configuration object.
        location: Either "memory" or "disk".
        dtype: The torch dtype for memory calculations.
        quant_type: The quantization type for memory calculations.
        eps: A small fraction to account for metadata overhead.
        block_index: The index of the block to analyze.
        exclude_experts: If True, the calculation will ignore parameters belonging
                         to MoE experts. This is crucial for shared-expert models.
    """
    if location == "memory":
        assert (
            dtype is not None and quant_type is not None
        ), 'get_block_size(..., location="memory") requires to specify dtype and quant_type for calculations'

    with init_empty_weights(include_buffers=False):
        block = get_model_block(config, block_index)
        total_params_counted = 0

        for name, param in block.named_parameters():
            is_expert = ".experts." in name
            is_counted = not (exclude_experts and is_expert)

            if is_counted:
                total_params_counted += param.numel()
        
        n_params = total_params_counted

    if location == "memory":
        if quant_type == QuantType.NONE:
            dtype = resolve_block_dtype(config, dtype)
            bytes_per_value = get_size_in_bytes(dtype)
        elif quant_type == QuantType.INT8:
            bytes_per_value = 1
        elif quant_type == QuantType.NF4:
            bytes_per_value = 4.25 / 8  # Bitness of NF4 with this config (measured empirically)
        else:
            raise ValueError(f"Unsupported quant_type={quant_type}")
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)
    else:
        raise ValueError(f"Unknown location: {location}")

    return round(n_params * bytes_per_value * (1 + eps))

from transformers.utils import is_flash_attn_2_available
def get_model_block(config, layer_idx: int = 0):
    """
    The function to create a model block based on the block class
    kwargs argument **only** is necessary for specific classes, like Mixtral.
    They will not be passed to other block constructors.
    """
    if is_flash_attn_2_available() and config.block_class != WrappedGptOssBlock:
        config._attn_implementation = "flash_attention_2"

    #if config.block_class == WrappedQwen2Block or config.block_class == WrappedQwen3MoeBlock or config.block_class == WrappedNemotronBlock:
    #    return config.block_class(config, layer_idx)
    if config.block_class == WrappedQwen3Block:
        config = PreTrainedModel._autoset_attn_implementation(config)
        return config.block_class(config, layer_idx)
    return config.block_class(config, layer_idx)
