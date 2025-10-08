#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from agentgrid.client.config import ClientConfig
from agentgrid.client.lm_head import LMHeadConfig
from agentgrid.client.ptune import PTuneConfig
from agentgrid.models.gpt_oss.block import WrappedGptOssBlock

logger = get_logger(__name__)


class DistributedGptOssConfig(GptOssConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGptOssBlock
    attn_class = GptOssAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike, None],
        *args,
        dht_prefix: Optional[str] = None,
        **kwargs,
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).split("/")[-1]
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.use_cache = True
        return result
