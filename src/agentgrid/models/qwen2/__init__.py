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