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

from hivemind import get_logger

from agentgrid.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

logger = get_logger(__name__)

class DistributedNemotronHConfig(NemotronHConfig):
    block_class = None  # to be defined later
    attn_class = None  # to be defined later
    block_prefix = "backbone.layers"

    @property
    def num_key_value_groups(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike | None,
        *args,
        dht_prefix: str | None = None,
        **kwargs,
    ):
        logger.info("Make sure you follow the Nemotron-H terms of use: "
                    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/")

        if dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)

        config = result[0] if isinstance(result, tuple) else result
        config.use_cache = True

        return result

