import os
from typing import Optional, Union

from hivemind import get_logger

from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from agentgrid.client.config import ClientConfig
from agentgrid.client.lm_head import LMHeadConfig
from agentgrid.client.ptune import PTuneConfig
from agentgrid.models.qwen3_moe.block import WrappedQwen3MoeBlock

logger = get_logger(__name__)

class DistributedQwen3MoeConfig(Qwen3MoeConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedQwen3MoeBlock
    attn_class = Qwen3MoeAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        dht_prefix: Optional[str] = None,
        **kwargs):
        loading_from_repo = pretrained_model_name_or_path is not None and not os.path.isdir(pretrained_model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(pretrained_model_name_or_path)
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        result = super().from_pretrained(pretrained_model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if config.pad_token_id is None:
            config.pad_token_id = 0
        config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by Agent Grid
        return result