#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from agentgrid.models.qwen2.config import DistributedQwen2Config
from agentgrid.models.qwen2.model import DistributedQwen2Model, DistributedQwen2ForCausalLM, DistributedQwen2ForSequenceClassification
from agentgrid.models.qwen2.speculative_model import DistributedQwen2ForSepculativeGeneration

from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedQwen2Config,
    model=DistributedQwen2Model,
    model_for_causal_lm=DistributedQwen2ForCausalLM,
    model_for_speculative=DistributedQwen2ForSepculativeGeneration,
    model_for_sequence_classification=DistributedQwen2ForSequenceClassification,
)