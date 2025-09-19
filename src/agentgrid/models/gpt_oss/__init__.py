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
