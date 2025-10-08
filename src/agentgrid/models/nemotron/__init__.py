#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from agentgrid.models.nemotron.block import WrappedNemotronBlock
from agentgrid.models.nemotron.config import DistributedNemotronConfig
from agentgrid.models.nemotron.model import DistributedNemotronForCausalLM, DistributedNemotronModel

from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedNemotronConfig,
    model=DistributedNemotronModel,
    model_for_causal_lm=DistributedNemotronForCausalLM
)