import dataclasses
import os
from typing import Any, Dict, Optional, Union

from hivemind import get_logger

from transformers.utils import is_flash_attn_2_available

from agentgrid.client.config import ClientConfig
from agentgrid.client.lm_head import LMHeadConfig
from agentgrid.client.ptune import PTuneConfig
from agentgrid.models.nemotron.block import WrappedNemotronBlock, BaseNemotronAttention

from agentgrid.models.nemotron.configuration_decilm import DeciLMConfig

logger = get_logger(__name__)


class DistributedNemotronConfig(DeciLMConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedNemotronBlock
    attn_class = BaseNemotronAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike, None],
        *args,
        dht_prefix: Optional[str] = None,
        **kwargs,
    ):
        logger.info("Make sure you follow the Nemotron terms of use: "
                    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/")

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by Agent Grid

        return result