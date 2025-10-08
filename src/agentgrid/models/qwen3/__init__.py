#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from agentgrid.models.qwen3.block import WrappedQwen3Block
from agentgrid.models.qwen3.config import DistributedQwen3Config
from agentgrid.models.qwen3.model import (
    DistributedQwen3Model,
    DistributedQwen3ForCausalLM,
    DistributedQwen3ForSequenceClassification
)
from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedQwen3Config,
    model=DistributedQwen3Model,
    model_for_causal_lm=DistributedQwen3ForCausalLM,
    model_for_sequence_classification=DistributedQwen3ForSequenceClassification,
)
