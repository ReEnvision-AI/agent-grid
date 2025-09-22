import os

from hivemind import get_logger

from agentgrid.client.config import ClientConfig
from agentgrid.client.lm_head import LMHeadConfig
from agentgrid.client.ptune import PTuneConfig

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from agentgrid.models.qwen2.block import WrappedQwen2Block

logger = get_logger(__name__)

class DistributedQwen2Config(Qwen2Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedQwen2Block
    attn_class = Qwen2Attention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | os.PathLike | None, *args, dht_prefix: str | None = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
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