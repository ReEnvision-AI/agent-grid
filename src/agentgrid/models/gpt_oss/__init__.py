#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from agentgrid.models.gpt_oss.block import WrappedGptOssBlock
from agentgrid.models.gpt_oss.config import DistributedGptOssConfig
from agentgrid.models.gpt_oss.model import DistributedGptOssForCausalLM, DistributedGptOssModel
from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGptOssConfig,
    model=DistributedGptOssModel,
    model_for_causal_lm=DistributedGptOssForCausalLM,
)

__all__ = [
    "WrappedGptOssBlock",
    "DistributedGptOssConfig",
    "DistributedGptOssModel",
    "DistributedGptOssForCausalLM",
]
