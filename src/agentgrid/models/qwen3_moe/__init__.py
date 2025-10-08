#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from agentgrid.models.qwen3_moe.block import WrappedQwen3MoeBlock
from agentgrid.models.qwen3_moe.config import DistributedQwen3MoeConfig
from agentgrid.models.qwen3_moe.model import DistributedQwen3MoeModel, DistributedQwen3MoeForCausalLM

from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedQwen3MoeConfig,
    model=DistributedQwen3MoeModel,
    model_for_causal_lm=DistributedQwen3MoeForCausalLM,
)