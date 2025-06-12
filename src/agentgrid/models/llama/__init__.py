from agentgrid.models.llama.block import WrappedLlamaBlock
from agentgrid.models.llama.config import DistributedLlamaConfig
from agentgrid.models.llama.model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
)
from agentgrid.models.llama.speculative_model import DistributedLlamaForSpeculativeGeneration
from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_speculative=DistributedLlamaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
