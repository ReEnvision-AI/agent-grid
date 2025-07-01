from agentgrid.models.nemotron.block import WrappedNemotronBlock
from agentgrid.models.nemotron.config import DistributedNemotronConfig
from agentgrid.models.nemotron.model import DistributedNemotronForCausalLM, DistributedNemotronModel

from agentgrid.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedNemotronConfig,
    model=DistributedNemotronModel,
    model_for_causal_lm=DistributedNemotronForCausalLM
)