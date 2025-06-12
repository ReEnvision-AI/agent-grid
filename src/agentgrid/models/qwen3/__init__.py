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
