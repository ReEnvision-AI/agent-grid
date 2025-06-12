from hivemind import DHT, get_logger

import torch
import torch.nn as nn
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeForSequenceClassification, Qwen3MoeModel, Qwen3MoePreTrainedModel

from agentgrid.client.from_pretrained import FromPretrainedMixin
from agentgrid.client.lm_head import LMHead
from agentgrid.client.ptune import PTuneMixin
from agentgrid.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from agentgrid.client.remote_sequential import RemoteSequential
from agentgrid.models.qwen3_moe import DistributedQwen3MoeConfig

logger = get_logger(__name__)

class DistributedQwen3MoeModel(FromPretrainedMixin, PTuneMixin, Qwen3MoeModel):
    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedQwen3MoeConfig

    def __init__(self, config:DistributedQwen3MoeConfig, *, dht: DHT | None = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0 # Prevent initialization
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
        return_dict: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> MoeModelOutputWithPast:
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
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
        )

        # Remove prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        past_key_values.update_seen(hidden_states.size(1))

        # Add last hidden state
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
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
    
class DistributedQwen3MoeForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, Qwen3MoeForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedQwen3MoeModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedQwen3MoeModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedQwen3MoeConfig

    def __init__(self, config: DistributedQwen3MoeConfig):
        Qwen3MoePreTrainedModel.__init__(self, config)
        self.model = DistributedQwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head
    
    @property
    def transformer(self) -> DistributedQwen3MoeModel:
        return self.model