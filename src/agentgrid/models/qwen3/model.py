from hivemind import DHT, get_logger

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForSequenceClassification, Qwen3Model, Qwen3PreTrainedModel

from agentgrid.client.from_pretrained import FromPretrainedMixin
from agentgrid.client.lm_head import LMHead
from agentgrid.client.ptune import PTuneMixin
from agentgrid.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from agentgrid.client.remote_sequential import RemoteSequential

from agentgrid.models.qwen3.config import DistributedQwen3Config

logger = get_logger(__name__)

class DistributedQwen3Model(FromPretrainedMixin, PTuneMixin, Qwen3Model):

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedQwen3Config

    def __init__(self, config: DistributedQwen3Config, *, dht: DHT | None = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
        self.init_prompts(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: RemotePastKeyValues | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
         # The causal mask will be added on the server-side
        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        output_shape = input_shape + (hidden_states.size(-1),)
       
        hidden_states = self.layers(
            hidden_states,
            prompts=None,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
            
        )

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()

        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None
        )
    
    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm
    
class DistributedQwen3ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, Qwen3ForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedQwen3Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen3Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedQwen3Config

    def __init__(self, config: DistributedQwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = DistributedQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedQwen3Model:  # For compatibility with RemoteGenerationMixin
        return self.model

class DistributedQwen3ForSequenceClassification(FromPretrainedMixin, Qwen3ForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedQwen3Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen3Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedQwen3Config

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedQwen3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def transformer(self) -> DistributedQwen3Model:  # For compatibility with RemoteGenerationMixin
        return self.model